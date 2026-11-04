"""
SigLIP 2 embedding wrapper for text and image encoding.

Provides 768-dimensional L2-normalized vectors for both images (person/vehicle
crops) and text queries, enabling semantic similarity search via cosine distance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_EMBEDDING_DIM = 768  # SigLIP 2 base output dimension
_MODEL_TAG = "google/siglip2-base-patch16-256"


def get_model_tag() -> str:
    """Return the tag written to embeddings.model for every vector we produce."""
    return _MODEL_TAG


def _load_model(device: str = "cpu") -> None:
    """Lazy-load SigLIP 2 model and processor (downloads on first use)."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_name = "google/siglip2-base-patch16-256"
        print(f"[embed] Loading {model_name} on {device}...", file=sys.stderr)
        _PROCESSOR = AutoProcessor.from_pretrained(model_name)
        _MODEL = AutoModel.from_pretrained(model_name).eval().to(device)
        _DEVICE = device
        print("[embed] Model loaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"[embed] WARNING: Failed to load SigLIP 2: {e}", file=sys.stderr)
        _MODEL = None


def is_available() -> bool:
    """Check if embedding model can be loaded (transformers installed)."""
    try:
        from transformers import AutoModel  # noqa: F401
        return True
    except ImportError:
        return False


def get_embedding_dim() -> int:
    """Return the dimensionality of embedding vectors."""
    return _EMBEDDING_DIM


def _extract_feature_tensor(outputs):
    """Get the feature tensor from either tensor or Transformers model output."""
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if hasattr(outputs, "image_embeds"):
        return outputs.image_embeds
    if hasattr(outputs, "text_embeds"):
        return outputs.text_embeds
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def embed_image(image_bgr: np.ndarray, device: str = "cpu") -> Optional[np.ndarray]:
    """Embed a BGR image crop into a 768-d L2-normalized vector. Returns None on failure."""
    _load_model(device)
    if _MODEL is None:
        return None
    try:
        import torch
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = _PROCESSOR(images=pil_img, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            outputs = _MODEL.get_image_features(**inputs)
        vec = _extract_feature_tensor(outputs).cpu().numpy().astype(np.float32).reshape(-1)
        vec = vec / (np.linalg.norm(vec) + 1e-8)  # L2 normalize
        return vec
    except Exception as e:
        print(f"[embed] Image embedding failed: {e}", file=sys.stderr)
        return None


def embed_text(text: str, device: str = "cpu") -> Optional[np.ndarray]:
    """Embed a text query into a 768-d L2-normalized vector. Returns None on failure."""
    _load_model(device)
    if _MODEL is None:
        return None
    try:
        import torch

        inputs = _PROCESSOR(text=[text], return_tensors="pt", padding=True).to(_DEVICE)
        with torch.no_grad():
            outputs = _MODEL.get_text_features(**inputs)
        vec = _extract_feature_tensor(outputs).cpu().numpy().astype(np.float32).reshape(-1)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec
    except Exception as e:
        print(f"[embed] Text embedding failed: {e}", file=sys.stderr)
        return None


def embed_images_batch(
    images_bgr: list[np.ndarray], device: str = "cpu", batch_size: int = 16
) -> list[Optional[np.ndarray]]:
    """Embed a batch of BGR images. Returns list of vectors (None for failures)."""
    _load_model(device)
    if _MODEL is None:
        return [None] * len(images_bgr)
    try:
        import torch
        import cv2
        from PIL import Image

        results: list[Optional[np.ndarray]] = []
        for i in range(0, len(images_bgr), batch_size):
            batch = images_bgr[i : i + batch_size]
            pil_imgs = []
            for img in batch:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_imgs.append(Image.fromarray(rgb))
            inputs = _PROCESSOR(images=pil_imgs, return_tensors="pt", padding=True).to(_DEVICE)
            with torch.no_grad():
                outputs = _MODEL.get_image_features(**inputs)
            vecs = _extract_feature_tensor(outputs).cpu().numpy().astype(np.float32)
            vecs = np.atleast_2d(vecs)
            # L2 normalize each row
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs = vecs / norms
            for v in vecs:
                results.append(v)
        return results
    except Exception as e:
        print(f"[embed] Batch image embedding failed: {e}", file=sys.stderr)
        return [None] * len(images_bgr)


def embed_images_mean(
    images_bgr: list[np.ndarray], device: str = "cpu", batch_size: int = 16
) -> Optional[np.ndarray]:
    """Embed a list of images and return their mean vector, L2-normalized.

    Used for multi-image queries (up to 5 crops): the mean embedding represents
    the group. Returns None when no image in the list could be embedded.
    """
    vectors = [
        v for v in embed_images_batch(images_bgr, device=device, batch_size=batch_size)
        if v is not None
    ]
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors, axis=0), axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def batch_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of row-vectors.

    Both query_vec and matrix rows should be L2-normalized.
    Returns 1-D array of scores in [−1, 1].
    """
    query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Embedding matrix must be 2-D, got shape {vectors.shape}")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError(
            f"Embedding dimensions must match: matrix has {vectors.shape[1]}, "
            f"query has {query.shape[0]}"
        )
    return vectors @ query
