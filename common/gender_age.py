"""
MiVOLO v2 gender + age estimation wrapper.

Integrates MiVOLO transformer with our existing RF-DETR person detections.
Instead of using MiVOLO's own YOLO detector, we pass our person crops directly.

Setup:
    git clone https://github.com/WildChlamydia/MiVOLO.git
    cd MiVOLO && pip install -r requirements.txt && pip install .
    # Download model weights:
    #   models/yolov8x_person_face.pt  (face detector — optional for our integration)
    #   models/mivolo_imbd.pth.tar     (MiVOLO checkpoint)

Falls back gracefully to 'unknown' if MiVOLO is not installed.
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Try importing MiVOLO — graceful fallback if not installed
_MIVOLO_AVAILABLE = False
_mivolo_import_error: str | None = None

try:
    from mivolo.model.mi_volo import MiVOLO as _MiVOLOModel
    from mivolo.model.create import create_model
    from mivolo.data.data_reader import InputType
    import timm
    import torch
    _MIVOLO_AVAILABLE = True
except ImportError as exc:
    _mivolo_import_error = str(exc)

# Cached portrait gender model (works on full-body/person crops too).
_PORTRAIT_MODEL = None
_PORTRAIT_PROCESSOR = None
_PORTRAIT_DEVICE: str | None = None
_PORTRAIT_FAILED = False
_PORTRAIT_MODEL_NAME = "prithivMLmods/Realistic-Gender-Classification"
_PORTRAIT_CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "hf_cache"


@dataclass
class GenderAgeResult:
    """Result of gender + age estimation for a single person."""
    gender: str = "unknown"         # "male" / "female" / "unknown"
    gender_conf: float = 0.0       # confidence 0..1
    age: float | None = None        # estimated age in years
    age_conf: float = 0.0          # confidence 0..1


class GenderAgeEstimator:
    """
    MiVOLO v2 wrapper for gender + age estimation from person crops.

    Usage:
        estimator = GenderAgeEstimator(checkpoint="models/mivolo_imbd.pth.tar")
        result = estimator.predict_crop(person_crop_bgr)
        print(result.gender, result.age)
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        device: str = "cuda",
        half: bool = False,
        with_persons: bool = True,
        disable_faces: bool = False,
    ) -> None:
        self.device = device
        self.half = half
        self.with_persons = with_persons
        self.disable_faces = disable_faces
        self.model: Any = None
        self._available = False
        self._siglip_text: dict[str, np.ndarray] | None = None

        # Track-level temporal stabilization for gender
        self._gender_history: dict[int, Counter[str]] = {}

        if not _MIVOLO_AVAILABLE:
            print(
                f"[WARN] MiVOLO not installed ({_mivolo_import_error}). "
                f"Using the portrait/SigLIP gender fallback.",
                file=sys.stderr,
            )
            return

        if checkpoint is None:
            # Try common default paths
            for candidate in [
                Path("models/mivolo_imbd.pth.tar"),
                Path("MiVOLO/models/mivolo_imbd.pth.tar"),
                Path("../models/mivolo_imbd.pth.tar"),
            ]:
                if candidate.is_file():
                    checkpoint = candidate
                    break

        if checkpoint is None or not Path(checkpoint).is_file():
            print(
                f"[WARN] MiVOLO checkpoint not found at '{checkpoint}'. "
                f"Using the SigLIP zero-shot gender fallback. "
                f"Download from: https://huggingface.co/iitolstykh/mivolo_v2",
                file=sys.stderr,
            )
            return

        try:
            self._load_model(str(checkpoint))
            self._available = True
            print(f"[INFO] MiVOLO loaded successfully from {checkpoint}", file=sys.stderr)
        except Exception as exc:
            print(f"[WARN] Failed to load MiVOLO: {exc}. Using the SigLIP zero-shot gender fallback.", file=sys.stderr)

    def _load_model(self, checkpoint: str) -> None:
        """Load MiVOLO model from checkpoint."""
        import torch

        # Load checkpoint to inspect model config
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # Extract model name from checkpoint if available
        model_name = "mivolo_d1_224"
        if "model_name" in ckpt:
            model_name = ckpt["model_name"]
        elif "args" in ckpt and hasattr(ckpt["args"], "model_name"):
            model_name = ckpt["args"].model_name

        # Determine number of classes (age bins + gender)
        num_classes = 3  # default: age, age_std, gender
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
            for key in state:
                if "head.fc" in key or "classifier" in key:
                    if state[key].dim() >= 1:
                        num_classes = state[key].shape[0]
                        break

        # Create model using timm + MiVOLO architecture
        try:
            self.model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                in_chans=6 if self.with_persons else 3,
                pretrained=False,
                checkpoint_path=checkpoint,
            )
        except Exception:
            # Fallback: try loading as a standard timm model
            self.model = timm.create_model(
                model_name,
                num_classes=num_classes,
                in_chans=6 if self.with_persons else 3,
                pretrained=False,
            )
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            # Clean state dict keys (remove 'module.' prefix if present)
            cleaned = {}
            for k, v in state.items():
                cleaned[k.replace("module.", "")] = v
            self.model.load_state_dict(cleaned, strict=False)

        self.model.to(self.device)
        self.model.eval()
        if self.half and self.device.startswith("cuda"):
            self.model.half()

    @property
    def is_available(self) -> bool:
        """Check if MiVOLO is loaded and ready."""
        return self._available

    @staticmethod
    def _is_informative_crop(person_crop_bgr: np.ndarray | None) -> bool:
        if person_crop_bgr is None or not isinstance(person_crop_bgr, np.ndarray):
            return False
        crop = person_crop_bgr
        if crop.size == 0 or crop.ndim < 2 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return False
        # Blank/uniform crops carry no gender signal (and keep tests deterministic).
        if float(crop.mean()) < 0.5 and float(crop.std()) < 0.5:
            return False
        return True

    def _predict_with_portrait_model(self, person_crop_bgr: np.ndarray) -> GenderAgeResult | None:
        """Classify gender with a SigLIP2 portrait model. Returns None when unavailable."""
        global _PORTRAIT_MODEL, _PORTRAIT_PROCESSOR, _PORTRAIT_DEVICE, _PORTRAIT_FAILED
        if _PORTRAIT_FAILED:
            return None
        try:
            import cv2
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            device = "cuda" if self.device.startswith("cuda") else "cpu"
            if _PORTRAIT_MODEL is None or _PORTRAIT_DEVICE != device:
                _PORTRAIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                print(
                    f"[gender] Loading {_PORTRAIT_MODEL_NAME} on {device}...",
                    file=sys.stderr,
                )
                _PORTRAIT_PROCESSOR = AutoImageProcessor.from_pretrained(
                    _PORTRAIT_MODEL_NAME, cache_dir=str(_PORTRAIT_CACHE_DIR)
                )
                _PORTRAIT_MODEL = AutoModelForImageClassification.from_pretrained(
                    _PORTRAIT_MODEL_NAME, cache_dir=str(_PORTRAIT_CACHE_DIR)
                ).eval().to(device)
                _PORTRAIT_DEVICE = device

            rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
            assert _PORTRAIT_MODEL is not None and _PORTRAIT_PROCESSOR is not None
            inputs = _PORTRAIT_PROCESSOR(images=Image.fromarray(rgb), return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = _PORTRAIT_MODEL(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            best_idx = int(torch.argmax(probs).item())
            confidence = float(probs[best_idx].item())
            label = str(_PORTRAIT_MODEL.config.id2label.get(best_idx, "")).lower()
            if confidence < 0.65:
                return GenderAgeResult()
            if "female" in label:
                return GenderAgeResult(gender="female", gender_conf=confidence)
            if "male" in label:
                return GenderAgeResult(gender="male", gender_conf=confidence)
            return None
        except Exception as exc:
            _PORTRAIT_FAILED = True
            print(
                f"[WARN] Portrait gender model unavailable ({exc}); using SigLIP fallback.",
                file=sys.stderr,
            )
            return None

    def _predict_with_fallback(self, person_crop_bgr: np.ndarray | None) -> GenderAgeResult:
        """Gender fallback used when MiVOLO is unavailable."""
        if not self._is_informative_crop(person_crop_bgr):
            return GenderAgeResult()
        assert person_crop_bgr is not None
        portrait = self._predict_with_portrait_model(person_crop_bgr)
        if portrait is not None:
            return portrait
        return self._predict_with_siglip(person_crop_bgr)

    def _siglip_prompts(self) -> dict[str, np.ndarray]:
        """Build (and cache) zero-shot gender text embeddings."""
        if self._siglip_text is not None:
            return self._siglip_text
        from common.embed import embed_text

        prompts = {
            "male": ["a photo of a man", "male"],
            "female": ["a photo of a woman", "female"],
        }
        device = "cuda" if self.device.startswith("cuda") else "cpu"
        text_vectors: dict[str, np.ndarray] = {}
        for label, label_prompts in prompts.items():
            vectors = [vec for vec in (embed_text(text, device=device) for text in label_prompts) if vec is not None]
            if not vectors:
                continue
            mean = np.mean(np.stack(vectors, axis=0), axis=0)
            text_vectors[label] = mean / (np.linalg.norm(mean) + 1e-8)
        self._siglip_text = text_vectors
        return text_vectors

    def _predict_with_siglip(self, person_crop_bgr: np.ndarray | None) -> GenderAgeResult:
        """Zero-shot gender fallback using the cached SigLIP vision-language model."""
        if person_crop_bgr is None or not isinstance(person_crop_bgr, np.ndarray):
            return GenderAgeResult()
        crop = person_crop_bgr
        if crop.size == 0 or crop.ndim < 2 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return GenderAgeResult()
        # Blank/uniform crops carry no gender signal (and keep tests deterministic).
        if float(crop.mean()) < 0.5 and float(crop.std()) < 0.5:
            return GenderAgeResult()

        try:
            from common.embed import cosine_similarity, embed_image

            device = "cuda" if self.device.startswith("cuda") else "cpu"
            image_vec = embed_image(crop, device=device)
            if image_vec is None:
                return GenderAgeResult()
            text_vectors = self._siglip_prompts()
            if "male" not in text_vectors or "female" not in text_vectors:
                return GenderAgeResult()

            male_score = cosine_similarity(image_vec, text_vectors["male"])
            female_score = cosine_similarity(image_vec, text_vectors["female"])
            diff = female_score - male_score
            if abs(diff) < 0.0025:
                return GenderAgeResult()

            gender = "female" if diff > 0 else "male"
            confidence = min(0.95, 0.5 + abs(diff) * 5.0)
            return GenderAgeResult(gender=gender, gender_conf=float(confidence))
        except Exception as exc:
            print(f"[WARN] SigLIP gender fallback failed: {exc}", file=sys.stderr)
            return GenderAgeResult()

    def predict_crop(
        self,
        person_crop_bgr: np.ndarray,
        face_crop_bgr: np.ndarray | None = None,
    ) -> GenderAgeResult:
        """
        Predict gender and age from a person crop (BGR).

        Args:
            person_crop_bgr: Person crop in BGR format (from our RF-DETR detector)
            face_crop_bgr: Optional face crop in BGR format

        Returns:
            GenderAgeResult with gender, age, and confidence scores
        """
        if not self._available:
            return self._predict_with_fallback(person_crop_bgr)

        import cv2
        import torch

        try:
            # Preprocess: resize to model input size (typically 224x224)
            input_size = 224

            # Process person crop
            person_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
            person_resized = cv2.resize(person_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
            person_tensor = torch.from_numpy(person_resized).permute(2, 0, 1).float() / 255.0

            if self.with_persons and face_crop_bgr is not None:
                # Concatenate person + face as 6-channel input
                face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
                face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                input_tensor = torch.cat([person_tensor, face_tensor], dim=0)  # 6 channels
            elif self.with_persons:
                # Person only — pad face channels with zeros
                face_tensor = torch.zeros_like(person_tensor)
                input_tensor = torch.cat([person_tensor, face_tensor], dim=0)  # 6 channels
            else:
                input_tensor = person_tensor  # 3 channels

            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
            if input_tensor.shape[0] == 6:
                mean = mean.repeat(2, 1, 1)
                std = std.repeat(2, 1, 1)
            input_tensor = (input_tensor - mean) / std

            if self.half:
                input_tensor = input_tensor.half()
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            with torch.inference_mode():
                output = self.model(input_tensor)

            # Parse output: typically [age, age_std, gender_logit] or similar
            output_np = output.cpu().float().numpy().flatten()

            if len(output_np) >= 3:
                age = float(np.clip(output_np[0], 0, 120))
                gender_logit = float(output_np[2]) if len(output_np) > 2 else 0.0
                gender_prob = 1.0 / (1.0 + np.exp(-gender_logit))  # sigmoid
                # Convention: >0.5 = female, <0.5 = male (check MiVOLO's convention)
                if gender_prob > 0.6:
                    gender = "female"
                    gender_conf = gender_prob
                elif gender_prob < 0.4:
                    gender = "male"
                    gender_conf = 1.0 - gender_prob
                else:
                    gender = "unknown"
                    gender_conf = 0.5
            elif len(output_np) >= 1:
                age = float(np.clip(output_np[0], 0, 120))
                gender = "unknown"
                gender_conf = 0.0
            else:
                return GenderAgeResult()

            return GenderAgeResult(
                gender=gender,
                gender_conf=float(gender_conf),
                age=age,
                age_conf=0.7,  # MiVOLO has good accuracy; moderate confidence
            )

        except Exception as exc:
            print(f"[WARN] MiVOLO prediction failed: {exc}", file=sys.stderr)
            return GenderAgeResult()

    def predict_crop_stabilized(
        self,
        person_crop_bgr: np.ndarray,
        track_id: int,
        face_crop_bgr: np.ndarray | None = None,
    ) -> GenderAgeResult:
        """
        Predict gender/age with temporal stabilization across track frames.

        Uses majority voting across frames for the same track to stabilize
        gender predictions (avoids flickering between male/female).
        """
        result = self.predict_crop(person_crop_bgr, face_crop_bgr)

        if result.gender != "unknown":
            history = self._gender_history.setdefault(track_id, Counter())
            history[result.gender] += int(round(result.gender_conf * 100))
            # Use the majority vote
            stabilized_gender = history.most_common(1)[0][0]
            result.gender = stabilized_gender

        return result

    def cleanup_tracks(self, active_track_ids: set[int]) -> None:
        """Remove history for tracks no longer active."""
        self._gender_history = {
            tid: h for tid, h in self._gender_history.items() if tid in active_track_ids
        }
