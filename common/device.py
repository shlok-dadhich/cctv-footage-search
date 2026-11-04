"""Shared device resolution and torch configuration."""
from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> str:
    """Resolve compute device: auto | cuda | mps | cpu."""
    req = (requested or "auto").strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS device requested but MPS backend is unavailable.")
    return req


def configure_torch_for_inference(device: str) -> None:
    """Set optimal torch settings for the given device."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment has no usable CUDA runtime."
            )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
