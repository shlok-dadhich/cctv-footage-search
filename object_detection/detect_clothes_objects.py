#!/usr/bin/env python3
"""
object_clothes_detect.py — Production clothing (with dominant LAB colour) + general object detector.
Outputs bounding boxes labeled strictly as: "<colour> <item> <confidence>" (e.g. "red shirt 0.87").

Pip install (verified minimum versions):
    pip install "torch>=2.2.0" "torchvision>=0.17.0" "rfdetr>=1.8.2" "transformers>=4.48.0" "pillow>=10.2.0" "supervision>=0.25.0" "opencv-python>=4.9.0" "numpy>=1.26.0" "tqdm>=4.66.0"
    # Optional for YOLOE-26 single-pass open-vocabulary segmentation mode:
    pip install "ultralytics>=8.4.0"

Model options (--model) & Licenses:
    - hybrid-small     (DEFAULT) : RFDETRSmall (Apache-2.0) + valentinafevu/yolos-fashionpedia (MIT)
    - hybrid-nano      (LIGHTER) : RFDETRNano  (Apache-2.0) + valentinafevu/yolos-fashionpedia (MIT)
    - hybrid-large     (HEAVIER) : RFDETRLarge (Apache-2.0) + valentinafevu/yolos-fashionpedia (MIT)
    - yoloe-26s-seg.pt (LIGHT-OV): Ultralytics YOLOE-26s open-vocab box+mask (AGPL-3.0)
    - grounding-dino   (HEAVY-OV): IDEA-Research/grounding-dino-tiny zero-shot (Apache-2.0)

Hardware notes:
    - Person-gated optimization: in hybrid mode, Fashionpedia only runs on frames where RF-DETR
      detects >=1 person (or when --always-run-fashion is set), doubling speed on non-person frames.
    - Colour pipeline: Central 50% crop (or instance mask) -> YCrCb skin-pixel rejection ->
      CIE LAB achromatic gate + 2-cluster K-Means -> CIE76 Delta-E nearest named colour lookup.
    - Run: python object_clothes_detect.py --source input.mp4 --output out.mp4
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
except ImportError as exc:
    sys.exit(
        f"[FATAL] Missing core dependency: {exc}\n"
        'Install with: pip install "torch>=2.2.0" "opencv-python>=4.9.0" "numpy>=1.26.0" "pillow>=10.2.0" "tqdm>=4.66.0"'
    )

from common.db import CCTVDatabase
from common.device import configure_torch_for_inference, resolve_device
from common.video_io import create_video_writer

FASHIONPEDIA_ID = "valentinafevu/yolos-fashionpedia"
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"

# 27 core Fashionpedia apparel/accessory categories mapped to concise labels
FASHION_SHORT_NAMES: dict[str, str] = {
    "shirt, blouse": "shirt",
    "top, t-shirt, sweatshirt": "t-shirt",
    "sweater": "sweater",
    "cardigan": "cardigan",
    "jacket": "jacket",
    "vest": "vest",
    "pants": "pants",
    "shorts": "shorts",
    "skirt": "skirt",
    "coat": "coat",
    "dress": "dress",
    "jumpsuit": "jumpsuit",
    "cape": "cape",
    "glasses": "glasses",
    "hat": "hat",
    "headband, head covering, hair accessory": "headwear",
    "tie": "tie",
    "glove": "glove",
    "watch": "watch",
    "belt": "belt",
    "leg warmer": "leg warmer",
    "tights, stockings": "tights",
    "sock": "sock",
    "shoe": "shoe",
    "bag, wallet": "bag",
    "scarf": "scarf",
    "umbrella": "umbrella",
}

# Fine garment sub-parts excluded by default to prevent noisy overlapping micro-boxes
FASHION_SUBPARTS: set[str] = {
    "hood", "collar", "lapel", "epaulette", "sleeve", "pocket", "neckline",
    "buckle", "zipper", "applique", "bead", "bow", "flower", "fringe",
    "ribbon", "rivet", "ruffle", "sequin", "tassel",
}

# Excluded from general object output because person_detect.py and vehicle_detect.py handle them
EXCLUDED_PERSON_VEHICLE: set[str] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
}


# Open-vocabulary class list for YOLOE / Grounding DINO modes
OPEN_VOCAB_CLASSES: list[str] = [
    "shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie", "cardigan", "vest",
    "pants", "jeans", "shorts", "skirt", "dress", "jumpsuit", "suit", "shoe",
    "sneakers", "boots", "hat", "glasses", "sunglasses", "tie", "scarf", "belt",
    "watch", "glove", "sock", "backpack", "handbag", "suitcase", "bag", "umbrella",
    "laptop", "cell phone", "keyboard", "mouse", "remote", "tv", "bottle", "cup",
    "wine glass", "bowl", "chair", "couch", "bed", "dining table", "potted plant",
    "book", "clock", "vase", "scissors", "teddy bear", "sports ball",
]

# Reference BGR colours for perceptual CIE LAB centroid lookup and box rendering
NAMED_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "black": (22, 22, 22),
    "dark gray": (75, 75, 75),
    "gray": (140, 140, 140),
    "white": (242, 242, 242),
    "red": (35, 35, 215),
    "orange": (25, 130, 240),
    "yellow": (35, 220, 240),
    "green": (45, 165, 45),
    "cyan": (205, 205, 35),
    "blue": (210, 85, 30),
    "navy": (115, 40, 20),
    "purple": (160, 45, 135),
    "pink": (180, 135, 245),
    "brown": (40, 75, 130),
    "beige": (175, 210, 230),
}

_CHROMATIC_NAMES = [
    k for k in NAMED_COLORS_BGR if k not in ("black", "dark gray", "gray", "white")
]
_CHROMATIC_BGR = np.array([NAMED_COLORS_BGR[k] for k in _CHROMATIC_NAMES], dtype=np.uint8).reshape(-1, 1, 3)
_CHROMATIC_LAB = cv2.cvtColor(_CHROMATIC_BGR, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def estimate_dominant_color_lab(
    frame_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    mask_crop: np.ndarray | None = None,
    reject_skin: bool = True,
) -> str:
    """
    Vectorized dominant colour estimator in CIE LAB space:
      1. Extracts central 50% torso/object region (or exact foreground mask if available).
      2. Downsamples to <=48x48 and rejects exposed skin pixels in YCrCb space.
      3. Uses CIE LAB Lightness (L*) and Chroma (C*) for achromatic gating, then runs
         vectorized 2-cluster K-Means in LAB space and matches the dominant cluster
         to the closest perceptual colour centroid via CIE76 Delta-E.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return "gray"

    if mask_crop is not None and mask_crop.shape == (bh, bw) and np.any(mask_crop):
        roi_bgr = frame_bgr[y1:y2, x1:x2]
        pixels_bgr = roi_bgr[mask_crop > 0]
        if len(pixels_bgr) > 2048:
            step = max(1, len(pixels_bgr) // 2048)
            pixels_bgr = pixels_bgr[::step]
    else:
        # Central 50% crop (25% margin on each side) to strip background edges
        px, py = int(bw * 0.22), int(bh * 0.22)
        cx1, cy1 = x1 + px, y1 + py
        cx2, cy2 = x2 - px, y2 - py
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2
        roi_bgr = frame_bgr[cy1:cy2, cx1:cx2]
        if roi_bgr.size == 0:
            return "gray"
        if roi_bgr.shape[0] > 48 or roi_bgr.shape[1] > 48:
            roi_bgr = cv2.resize(roi_bgr, (48, 48), interpolation=cv2.INTER_AREA)
        pixels_bgr = roi_bgr.reshape(-1, 3)

    if len(pixels_bgr) == 0:
        return "gray"

    # YCrCb skin-pixel filter (removes neck, hands, and face leakage on upper-body garments)
    if reject_skin and len(pixels_bgr) >= 16:
        ycrcb = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        y_val, cr, cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]
        skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (y_val >= 45) & (y_val <= 235)
        non_skin = pixels_bgr[~skin_mask]
        # Keep non-skin pixels if at least 25% of the crop remains
        if len(non_skin) >= max(8, int(0.25 * len(pixels_bgr))):
            pixels_bgr = non_skin

    lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    l_chan = lab[:, 0]  # 0..255 in OpenCV (L* scaled by 255/100)
    a_shift = lab[:, 1] - 128.0
    b_shift = lab[:, 2] - 128.0
    chroma = np.hypot(a_shift, b_shift)

    # Check chromatic vs achromatic dominance
    chromatic_mask = (chroma >= 14.0) & (l_chan >= 35.0) & (l_chan <= 245.0)
    chromatic_ratio = float(np.count_nonzero(chromatic_mask)) / float(len(lab))

    if chromatic_ratio < 0.22:
        med_l = float(np.median(l_chan))
        if med_l < 58.0:
            return "black"
        if med_l < 110.0:
            return "dark gray"
        if med_l > 205.0:
            return "white"
        return "gray"

    chroma_lab = lab[chromatic_mask]
    # Fast vectorized 2-cluster K-Means (5 iterations) in LAB space to isolate primary garment color
    if len(chroma_lab) >= 12:
        q25 = np.percentile(chroma_lab, 25, axis=0)
        q75 = np.percentile(chroma_lab, 75, axis=0)
        c0, c1 = q25.copy(), q75.copy()
        for _ in range(5):
            d0 = np.sum((chroma_lab - c0) ** 2, axis=1)
            d1 = np.sum((chroma_lab - c1) ** 2, axis=1)
            mask0 = d0 <= d1
            if np.any(mask0) and np.any(~mask0):
                c0 = chroma_lab[mask0].mean(axis=0)
                c1 = chroma_lab[~mask0].mean(axis=0)
            else:
                break
        dominant_lab = c0 if np.count_nonzero(mask0) >= np.count_nonzero(~mask0) else c1
    else:
        dominant_lab = np.median(chroma_lab, axis=0)

    # Weighted CIE76 Delta-E (weighting chroma channels a*, b* 1.35x to resist lighting shadows)
    diff = _CHROMATIC_LAB - dominant_lab[None, :]
    dists = np.sqrt((0.75 * diff[:, 0]) ** 2 + (1.35 * diff[:, 1]) ** 2 + (1.35 * diff[:, 2]) ** 2)
    return _CHROMATIC_NAMES[int(np.argmin(dists))]


def agnostic_nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-agnostic greedy NMS to suppress overlapping duplicate garment predictions (e.g., shirt vs t-shirt)."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


class SimpleItemTracker:
    """Lightweight IoU tracker maintaining stable track IDs across frames."""

    def __init__(self, iou_threshold: float = 0.35, max_lost: int = 25) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: dict[int, tuple[np.ndarray, str, int]] = {}

    def update(self, boxes: np.ndarray, item_names: list[str]) -> np.ndarray:
        if len(boxes) == 0:
            self.tracks = {
                tid: (b, name, lost + 1)
                for tid, (b, name, lost) in self.tracks.items()
                if lost + 1 <= self.max_lost
            }
            return np.empty((0,), dtype=np.int32)

        assigned = np.full((len(boxes),), -1, dtype=np.int32)
        used: set[int] = set()
        for i, (box, name) in enumerate(zip(boxes, item_names)):
            best_iou = self.iou_threshold
            best_tid = -1
            for tid, (tbox, tname, _) in self.tracks.items():
                if tid in used or tname != name:
                    continue
                ix1, iy1 = max(box[0], tbox[0]), max(box[1], tbox[1])
                ix2, iy2 = min(box[2], tbox[2]), min(box[3], tbox[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                area_b = max(0.0, tbox[2] - tbox[0]) * max(0.0, tbox[3] - tbox[1])
                iou = inter / max(area_a + area_b - inter, 1e-9)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid != -1:
                assigned[i] = best_tid
                used.add(best_tid)
            else:
                assigned[i] = self.next_id
                self.next_id += 1

        new_tracks: dict[int, tuple[np.ndarray, str, int]] = {}
        for tid, (tbox, tname, lost) in self.tracks.items():
            if tid not in used and lost + 1 <= self.max_lost:
                new_tracks[tid] = (tbox, tname, lost + 1)
        for i, tid in enumerate(assigned):
            new_tracks[int(tid)] = (boxes[i], item_names[i], 0)
        self.tracks = new_tracks
        return assigned


class ObjectClothesPipeline:
    """
    Supports 3 architectures via --model:
      1. hybrid-small / hybrid-nano / hybrid-large (Default: RF-DETR + YOLOS-Fashionpedia, Apache-2.0 + MIT)
      2. yoloe-26n-seg.pt / yoloe-26s-seg.pt / yoloe-26m-seg.pt (Ultralytics YOLOE open-vocab + masks, AGPL-3.0)
      3. grounding-dino (IDEA-Research/grounding-dino-tiny zero-shot transformer, Apache-2.0)
    """

    HYBRID_RFDETR_MAP = {
        "hybrid-nano": "RFDETRNano",
        "hybrid-small": "RFDETRSmall",
        "hybrid-medium": "RFDETRMedium",
        "hybrid-large": "RFDETRLarge",
        "hybrid": "RFDETRSmall",
    }

    def __init__(
        self,
        model_name: str,
        device: str,
        half: bool = False,
        imgsz: int | None = None,
        enable_track: bool = False,
        always_run_fashion: bool = False,
    ) -> None:
        self.model_name = model_name.strip()
        self.device = device
        self.half = bool(half and device.startswith("cuda"))
        self.imgsz = imgsz
        self.enable_track = enable_track
        self.always_run_fashion = always_run_fashion
        self.tracker = SimpleItemTracker() if enable_track else None

        m_low = self.model_name.lower()
        if m_low in self.HYBRID_RFDETR_MAP:
            self.mode = "hybrid"
        elif "grounding-dino" in m_low:
            self.mode = "grounding-dino"
        else:
            self.mode = "yoloe"

        self.rfdetr_model: Any = None
        self.coco_classes: Any = {}
        self.fashion_proc: Any = None
        self.fashion_model: Any = None
        self.gdino_proc: Any = None
        self.gdino_model: Any = None
        self.yoloe_model: Any = None

        self._load_models()

    def _load_models(self) -> None:
        try:
            if self.mode == "hybrid":
                import rfdetr
                from transformers import AutoImageProcessor, AutoModelForObjectDetection

                try:
                    from rfdetr.assets.coco_classes import COCO_CLASSES

                    self.coco_classes = COCO_CLASSES
                except ImportError:
                    from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore[no-redef]

                    self.coco_classes = COCO_CLASSES

                cls_name = self.HYBRID_RFDETR_MAP[self.model_name.lower()]
                model_cls = getattr(rfdetr, cls_name, None) or getattr(rfdetr, "RFDETRSmall")
                try:
                    self.rfdetr_model = model_cls(device=self.device)
                except TypeError:
                    self.rfdetr_model = model_cls()

                if self.device.startswith("cuda") and hasattr(self.rfdetr_model, "to"):
                    self.rfdetr_model.to(self.device)
                dtype = torch.float16 if self.half else torch.float32
                if hasattr(self.rfdetr_model, "inference"):
                    try:
                        self.rfdetr_model.inference(dtype=dtype)
                    except Exception:
                        pass

                self.fashion_proc = AutoImageProcessor.from_pretrained(FASHIONPEDIA_ID)
                self.fashion_model = AutoModelForObjectDetection.from_pretrained(FASHIONPEDIA_ID).to(self.device)
                self.fashion_model.eval()

            elif self.mode == "grounding-dino":
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                model_id = GROUNDING_DINO_ID if self.model_name.lower() == "grounding-dino" else self.model_name
                self.gdino_proc = AutoProcessor.from_pretrained(model_id)
                self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
                self.gdino_model.eval()

            else:
                from ultralytics import YOLOE

                self.yoloe_model = YOLOE(self.model_name)
                self.yoloe_model.to(self.device)
                self.yoloe_model.set_classes(OPEN_VOCAB_CLASSES)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize '{self.model_name}' on '{self.device}': {exc}") from exc

    def _resolve_rfdetr_name(self, detections: Any, idx: int, cid: int) -> str:
        if hasattr(detections, "data") and isinstance(detections.data, dict):
            cnames = detections.data.get("class_name")
            if cnames is not None and len(cnames) > idx:
                return str(cnames[idx]).strip().lower()
        if isinstance(self.coco_classes, dict) and cid in self.coco_classes:
            return str(self.coco_classes[cid]).strip().lower()
        if isinstance(self.coco_classes, (list, tuple)) and 0 <= cid < len(self.coco_classes):
            return str(self.coco_classes[cid]).strip().lower()
        return f"class_{cid}"

    def predict(
        self, frame_bgr: np.ndarray, conf: float, iou: float
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray | None]:
        """Return (boxes[N,4], scores[N], colored_labels[N], colors[N], track_ids[N] or None)."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes_list: list[np.ndarray] = []
        scores_list: list[float] = []
        names_list: list[str] = []
        masks_list: list[np.ndarray | None] = []

        if self.mode == "hybrid":
            with torch.inference_mode():
                rf_det = self.rfdetr_model.predict(frame_rgb, threshold=conf)

            person_detected = False
            if rf_det is not None and len(rf_det) > 0:
                for i, cid in enumerate(rf_det.class_id):
                    cname = self._resolve_rfdetr_name(rf_det, i, int(cid))
                    if cname == "person":
                        person_detected = True
                        continue
                    if cname in EXCLUDED_PERSON_VEHICLE:
                        continue
                    boxes_list.append(np.asarray(rf_det.xyxy[i], dtype=np.float32))
                    scores_list.append(float(rf_det.confidence[i]))
                    names_list.append(cname)
                    masks_list.append(None)

            # Run Fashionpedia only when a person is present in the frame (or --always-run-fashion)
            if person_detected or self.always_run_fashion:
                pil_img = Image.fromarray(frame_rgb)
                inputs = self.fashion_proc(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    if self.half:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.fashion_model(**inputs)
                    else:
                        outputs = self.fashion_model(**inputs)

                target_sizes = torch.tensor([[h, w]], device=self.device)
                f_res = self.fashion_proc.post_process_object_detection(
                    outputs, threshold=conf, target_sizes=target_sizes
                )[0]
                id2label = self.fashion_model.config.id2label

                f_boxes: list[np.ndarray] = []
                f_scores: list[float] = []
                f_names: list[str] = []
                for sc, lid, bx in zip(f_res["scores"], f_res["labels"], f_res["boxes"]):
                    raw_name = str(id2label.get(int(lid), "")).strip().lower()
                    if not raw_name or raw_name in FASHION_SUBPARTS:
                        continue
                    short_name = FASHION_SHORT_NAMES.get(raw_name, raw_name)
                    f_boxes.append(bx.detach().cpu().numpy().astype(np.float32))
                    f_scores.append(float(sc.item()))
                    f_names.append(short_name)

                # Class-agnostic NMS across Fashionpedia predictions to deduplicate shirt vs t-shirt
                if f_boxes:
                    fb_arr = np.vstack(f_boxes)
                    fs_arr = np.asarray(f_scores, dtype=np.float32)
                    keep_f = agnostic_nms_numpy(fb_arr, fs_arr, iou)
                    for idx in keep_f:
                        boxes_list.append(fb_arr[int(idx)])
                        scores_list.append(float(fs_arr[int(idx)]))
                        names_list.append(f_names[int(idx)])
                        masks_list.append(None)

        elif self.mode == "grounding-dino":
            pil_img = Image.fromarray(frame_rgb)
            inputs = self.gdino_proc(
                images=pil_img,
                text=[[f"a {lbl}" for lbl in OPEN_VOCAB_CLASSES]],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                if self.half:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.gdino_model(**inputs)
                else:
                    outputs = self.gdino_model(**inputs)

            res = self.gdino_proc.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                threshold=conf,
                text_threshold=max(0.20, conf - 0.05),
                target_sizes=[(h, w)],
            )[0]
            raw_labels = res.get("text_labels", res.get("labels", []))
            for bx, sc, lbl in zip(res["boxes"], res["scores"], raw_labels):
                clean = str(lbl).strip().lower().removeprefix("a ").removeprefix("an ").strip()
                if not clean or clean in EXCLUDED_PERSON_VEHICLE:
                    continue
                boxes_list.append(bx.detach().cpu().numpy().astype(np.float32))
                scores_list.append(float(sc.item()))
                names_list.append(clean)
                masks_list.append(None)

        else:
            # YOLOE-26 open-vocabulary detection + instance segmentation
            infer_kwargs: dict[str, Any] = {
                "conf": conf,
                "iou": iou,
                "device": self.device,
                "half": self.half,
                "verbose": False,
            }
            if self.imgsz is not None and self.imgsz > 0:
                infer_kwargs["imgsz"] = int(self.imgsz)
            results = self.yoloe_model.predict(frame_bgr, **infer_kwargs)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                b_obj = results[0].boxes
                xyxy_y = b_obj.xyxy.detach().cpu().numpy().astype(np.float32)
                conf_y = b_obj.conf.detach().cpu().numpy().astype(np.float32)
                cls_y = b_obj.cls.int().detach().cpu().numpy().astype(np.int32)
                id2n = results[0].names or {}
                m_obj = results[0].masks
                for i in range(len(xyxy_y)):
                    cname = str(id2n.get(int(cls_y[i]), "object")).strip().lower()
                    if cname in EXCLUDED_PERSON_VEHICLE:
                        continue
                    boxes_list.append(xyxy_y[i])
                    scores_list.append(float(conf_y[i]))
                    names_list.append(cname)
                    if m_obj is not None and m_obj.data is not None and i < len(m_obj.data):
                        full_m = m_obj.data[i].detach().cpu().numpy()
                        if full_m.shape != (h, w):
                            full_m = cv2.resize(full_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy_y[i]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        masks_list.append(full_m[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None)
                    else:
                        masks_list.append(None)

        if not boxes_list:
            if self.tracker is not None:
                self.tracker.update(np.empty((0, 4), dtype=np.float32), [])
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), [], [], None

        boxes_arr = np.vstack(boxes_list).astype(np.float32)
        scores_arr = np.asarray(scores_list, dtype=np.float32)

        # Final cross-detector NMS to prevent duplicate bag/umbrella/backpack boxes
        keep_all = agnostic_nms_numpy(boxes_arr, scores_arr, iou)
        boxes_arr = boxes_arr[keep_all]
        scores_arr = scores_arr[keep_all]
        names_list = [names_list[int(i)] for i in keep_all]
        masks_list = [masks_list[int(i)] for i in keep_all]

        track_ids = self.tracker.update(boxes_arr, names_list) if self.tracker is not None else None

        colored_items: list[str] = []
        color_names: list[str] = []
        for box, item_name, m_crop in zip(boxes_arr, names_list, masks_list):
            cname = estimate_dominant_color_lab(frame_bgr, box, mask_crop=m_crop, reject_skin=True)
            color_names.append(cname)
            colored_items.append(f"{cname} {item_name}".strip())

        return boxes_arr, scores_arr, colored_items, color_names, track_ids


def draw_object_clothes_annotations(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    colored_items: list[str],
    color_names: list[str],
    track_ids: np.ndarray | None,
) -> None:
    """Draw boxes tinted by detected colour with '<colour> <item> <confidence>' labels."""
    h, w = frame_bgr.shape[:2]
    thickness = max(2, int(round(min(w, h) / 450)))
    font_scale = max(0.48, min(w, h) / 1450.0)
    font_thick = max(1, thickness - 1)

    for i, (box, score, c_item, c_name) in enumerate(zip(boxes, scores, colored_items, color_names)):
        color = NAMED_COLORS_BGR.get(c_name, (210, 80, 230))
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (15, 15, 15) if lum > 140 else (250, 250, 250)

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        tid_str = f" #{int(track_ids[i])}" if (track_ids is not None and i < len(track_ids)) else ""
        # Strictly follows required "<colour> <item> <confidence>" format
        label = f"{c_item}{tid_str} {float(score):.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        pad = max(4, thickness * 2)
        bg_y1 = y1 - th - baseline - pad * 2
        if bg_y1 < 0:
            bg_y1 = y1
            bg_y2 = min(h - 1, y1 + th + baseline + pad * 2)
            text_y = min(h - 2, bg_y1 + th + pad)
        else:
            bg_y2 = y1
            text_y = max(th + 2, y1 - baseline - pad)

        bg_x2 = min(w - 1, x1 + tw + pad * 2)
        cv2.rectangle(frame_bgr, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + pad, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thick,
            cv2.LINE_AA,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect clothes with their dominant colour ('<colour> <item> <conf>') plus everyday objects."
    )
    parser.add_argument("--source", "-s", type=Path, required=True, help="Input video file path")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output annotated video path")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid-small",
        help="Model: hybrid-nano | hybrid-small (default) | hybrid-large | yoloe-26s-seg.pt | grounding-dino",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0 < conf < 1)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (0 < iou < 1)")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size (e.g., 640)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cuda | mps | cpu")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on CUDA")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process 1 out of every N frames (default: 1)")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Process only the first N seconds of the input video (default: entire video)",
    )
    parser.add_argument("--show", action="store_true", help="Display live annotated preview window")
    parser.add_argument("--track", action="store_true", help="Enable persistent multi-object tracking IDs")
    parser.add_argument(
        "--always-run-fashion",
        action="store_true",
        help="Run Fashionpedia on every frame even if no person is detected (useful for standalone product videos)",
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--no-db", action="store_true", help="Disable SQLite database persistence")
    parser.add_argument("--camera-id", type=str, default="cam_default", help="Camera identifier")
    parser.add_argument("--video-id", type=str, default=None, help="Custom video ID (default: source file stem)")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        parser.error(f"Input video file not found: {args.source}")
    if not (0.0 < args.conf < 1.0):
        parser.error("--conf must be strictly between 0.0 and 1.0")
    if not (0.0 < args.iou < 1.0):
        parser.error("--iou must be strictly between 0.0 and 1.0")
    if args.vid_stride < 1:
        parser.error("--vid-stride must be >= 1")
    if args.max_duration is not None and args.max_duration <= 0.0:
        parser.error("--max-duration must be greater than 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or args.source.with_name(f"{args.source.stem}_objects_clothes_out.mp4")

    try:
        device = resolve_device(args.device)
        configure_torch_for_inference(device)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {args.source}", file=sys.stderr)
        return 1

    writer: cv2.VideoWriter | None = None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions ({width}x{height}) in {args.source}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(src_fps) or src_fps <= 0.0 or src_fps > 240.0:
            print("[WARN] Invalid or zero FPS metadata detected; defaulting to 25.0 FPS.", file=sys.stderr)
            src_fps = 25.0
        out_fps = max(1.0, src_fps / float(args.vid_stride))

        raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = (raw_total / src_fps) if raw_total > 0 else None
        max_raw_frames = (
            int(np.ceil(args.max_duration * src_fps))
            if args.max_duration is not None
            else None
        )
        if max_raw_frames is not None:
            raw_total = min(raw_total, max_raw_frames) if raw_total > 0 else max_raw_frames
            duration_s = min(duration_s or 0.0, float(args.max_duration))
        total_steps = (raw_total + args.vid_stride - 1) // args.vid_stride if raw_total > 0 else None

        video_id = args.video_id or args.source.stem
        db: CCTVDatabase | None = None
        if not args.no_db:
            try:
                db = CCTVDatabase(args.db)
                db.register_video(
                    video_id=video_id,
                    file_path=str(args.source.resolve()),
                    camera_id=args.camera_id,
                    duration_s=duration_s,
                    fps=src_fps,
                    width=width,
                    height=height,
                )
            except Exception as exc:
                print(f"[WARN] Could not initialize database at {args.db}: {exc}. Continuing without DB.", file=sys.stderr)
                db = None

        pipeline = ObjectClothesPipeline(
            model_name=args.model,
            device=device,
            half=args.half,
            imgsz=args.imgsz,
            enable_track=args.track,
            always_run_fashion=args.always_run_fashion,
        )
        writer, final_output_path = create_video_writer(output_path, out_fps, width, height)

        print(
            f"[INFO] Model: {args.model} ({pipeline.mode}) | Device: {device} (FP16={pipeline.half}) | "
            f"Resolution: {width}x{height} @ {out_fps:.2f} FPS | "
            f"DB: {'Enabled' if db is not None else 'Disabled'}"
        )

        raw_idx = 0
        processed_frames = 0
        frames_with_items = 0
        item_counts: Counter[str] = Counter()
        t0 = time.perf_counter()

        with tqdm(total=total_steps, desc="Detecting clothes & objects", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                raw_idx += 1
                if max_raw_frames is not None and raw_idx > max_raw_frames:
                    break
                if (raw_idx - 1) % args.vid_stride != 0:
                    continue

                t_current_s = float(raw_idx - 1) / src_fps

                boxes, scores, colored_items, color_names, track_ids = pipeline.predict(
                    frame_bgr, conf=args.conf, iou=args.iou
                )
                if len(boxes) > 0:
                    frames_with_items += 1
                    item_counts.update(colored_items)

                    if db is not None:
                        for i, (box, score, item_str, c_name) in enumerate(
                            zip(boxes, scores, colored_items, color_names)
                        ):
                            # item_str is e.g. "red shirt" -> object_type is "shirt"
                            if c_name and item_str.lower().startswith(c_name.lower()):
                                obj_type = item_str[len(c_name):].strip()
                            else:
                                parts = item_str.split(" ", 1)
                                obj_type = parts[1] if len(parts) > 1 else parts[0]
                            tid = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else None
                            db.insert_object(
                                video_id=video_id,
                                track_id=tid,
                                object_type=obj_type,
                                color=c_name,
                                first_seen_s=t_current_s,
                                last_seen_s=t_current_s,
                                frame_count=1,
                                peak_conf=float(score),
                                bbox_json=json.dumps([
                                    float(np.floor(box[0])),
                                    float(np.floor(box[1])),
                                    float(np.ceil(box[2])),
                                    float(np.ceil(box[3])),
                                ]),
                                bbox_time_s=t_current_s,
                                bbox_frame=raw_idx - 1,
                            )

                    draw_object_clothes_annotations(
                        frame_bgr, boxes, scores, colored_items, color_names, track_ids
                    )

                writer.write(frame_bgr)
                processed_frames += 1
                pbar.update(1)
                pbar.set_postfix(items=len(boxes), total=sum(item_counts.values()))

                if db is not None and processed_frames % 100 == 0:
                    db.commit_objects()

                if args.show:
                    try:
                        cv2.imshow("object_clothes_detect", frame_bgr)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            print("\n[INFO] Preview interrupted by user ('q').")
                            break
                    except cv2.error:
                        args.show = False

        if db is not None:
            db.commit_objects()
            db.update_video_status(video_id, "completed")

        elapsed = max(time.perf_counter() - t0, 1e-6)
        if processed_frames == 0:
            raise RuntimeError(f"Video '{args.source}' yielded 0 readable frames.")
        total_boxes = sum(item_counts.values())
        if total_boxes == 0:
            print("[WARN] 0 clothing/object detections across the video. Try --always-run-fashion or lower --conf.")

        top_summary = ", ".join(f"{k}:{v}" for k, v in item_counts.most_common(8)) or "none"
        print(
            f"[DONE] Processed {processed_frames} frames in {elapsed:.2f}s ({processed_frames / elapsed:.2f} FPS) | "
            f"Frames w/ detections: {frames_with_items} | Total: {total_boxes} (Top: {top_summary}) | "
            f"Saved: {final_output_path.resolve()}"
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if db is not None:
            db.update_video_status(video_id, "failed")
        return 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if db is not None:
            db.close()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())