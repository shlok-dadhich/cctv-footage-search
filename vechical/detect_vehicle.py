#!/usr/bin/env python3
"""
vehicle_detect.py — Production real-time road vehicle detection, color, and tracking pipeline.
Detects: car, motorcycle, bus, truck, bicycle (customizable via --classes) and estimates
visible paint color from the central vehicle body region.

Pip install (verified minimum versions):
    pip install "torch>=2.2.0" "torchvision>=0.17.0" "rfdetr>=1.8.2" "supervision>=0.25.0" "opencv-python>=4.9.0" "numpy>=1.26.0" "tqdm>=4.66.0"
    # Optional for Ultralytics YOLO26/YOLO11/RT-DETR weights:
    pip install "ultralytics>=8.4.0"

Model options (--model) & Licenses:
    - rfdetr-medium (DEFAULT) : Roboflow RF-DETR Medium(54.7 COCO mAP, 4.4 ms T4)  | License: Apache-2.0
    - rfdetr-nano   (LIGHTER) : Roboflow RF-DETR Nano  (48.4 COCO mAP, 2.3 ms T4)  | License: Apache-2.0
    - rfdetr-small  (LIGHT)   : Roboflow RF-DETR Small (53.0 COCO mAP, 3.5 ms T4)  | License: Apache-2.0
    - rfdetr-large  (HEAVIER) : Roboflow RF-DETR Large (56.5 COCO mAP, 6.8 ms T4)  | License: Apache-2.0
    - yolo26n.pt / yolo26s.pt / yolo26m.pt / yolo26x.pt : Ultralytics YOLO         | License: AGPL-3.0

Hardware notes:
    - Auto-detects CUDA -> Apple Silicon MPS -> CPU.
    - Pass --half on CUDA for FP16 inference.
    - Run: python vehicle_detect.py --source input.mp4 --output out.mp4

Color notes:
    - Labels combine estimated paint color and vehicle type, for example "red car 0.91".
    - Color is estimated from the pixels inside each detection, not from the detector's
      fixed box color. Track mode temporally stabilizes the estimate.
    - Estimates are less reliable for distant vehicles, heavy occlusion, night footage,
      strong reflections, and boxes dominated by windows, wheels, or background.
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
    from tqdm import tqdm
except ImportError as exc:
    sys.exit(
        f"[FATAL] Missing core dependency: {exc}\n"
        'Install with: pip install "torch>=2.2.0" "opencv-python>=4.9.0" "numpy>=1.26.0" "tqdm>=4.66.0"'
    )

from common.db import CCTVDatabase
from common.device import configure_torch_for_inference, resolve_device
from common.video_io import create_video_writer

# Strictly scoped to the 5 target vehicle classes by default
DEFAULT_VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")

# Standard 0-indexed COCO IDs for Ultralytics YOLO / RT-DETR models
COCO80_NAME_TO_ID = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
}

# Distinct high-contrast BGR colors per vehicle class
VEHICLE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car": (255, 140, 30),        # Dodger Blue
    "motorcycle": (40, 210, 255), # Gold / Amber
    "bus": (180, 60, 230),        # Magenta / Purple
    "truck": (30, 100, 255),      # Deep Orange
    "bicycle": (80, 230, 80),     # Lime Green
    "train": (220, 220, 40),      # Cyan
    "airplane": (230, 180, 230),  # Lavender
    "boat": (255, 200, 100),      # Sky Blue
}

# Paint colors used for the vehicle label. These are separate from
# VEHICLE_COLORS_BGR, which controls bounding-box rendering.
VEHICLE_PAINT_BGR: dict[str, tuple[int, int, int]] = {
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
_PAINT_NAMES = list(VEHICLE_PAINT_BGR)
_PAINT_BGR = np.array([VEHICLE_PAINT_BGR[name] for name in _PAINT_NAMES], dtype=np.uint8)
_PAINT_LAB = cv2.cvtColor(_PAINT_BGR.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
_CHROMATIC_PAINT_NAMES = [name for name in _PAINT_NAMES if name not in {"black", "dark gray", "gray", "white"}]
_CHROMATIC_PAINT_LAB = np.array(
    [VEHICLE_PAINT_BGR[name] for name in _CHROMATIC_PAINT_NAMES], dtype=np.uint8
)
_CHROMATIC_PAINT_LAB = cv2.cvtColor(
    _CHROMATIC_PAINT_LAB.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
).reshape(-1, 3).astype(np.float32)


def ciede2000_distance(lab_values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return CIEDE2000 distances for OpenCV-scaled LAB values."""
    values = lab_values.astype(np.float64)
    ref = reference.astype(np.float64)
    l1, a1, b1 = values[:, 0] * (100.0 / 255.0), values[:, 1] - 128.0, values[:, 2] - 128.0
    l2, a2, b2 = ref[0] * (100.0 / 255.0), ref[1] - 128.0, ref[2] - 128.0
    c1, c2 = np.hypot(a1, b1), float(np.hypot(a2, b2))
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt((c_bar**7) / (c_bar**7 + 25.0**7 + 1e-12)))
    ap1, ap2 = (1.0 + g) * a1, (1.0 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = (np.degrees(np.arctan2(b1, ap1)) % 360.0)
    hp2 = np.degrees(np.arctan2(b2, ap2)) % 360.0
    delta_l = l1 - l2
    delta_c = cp1 - cp2
    delta_h = hp1 - hp2
    delta_h = np.where(np.abs(delta_h) <= 180.0, delta_h, delta_h - np.sign(delta_h) * 360.0)
    delta_big_h = 2.0 * np.sqrt(cp1 * cp2) * np.sin(np.radians(delta_h / 2.0))
    l_bar = (l1 + l2) / 2.0
    c_bar_p = (cp1 + cp2) / 2.0
    h_diff = np.abs(hp1 - hp2)
    h_bar = np.where(
        (cp1 * cp2 == 0.0),
        hp1 + hp2,
        np.where(h_diff <= 180.0, (hp1 + hp2) / 2.0, (hp1 + hp2 + np.where(hp1 + hp2 < 360.0, 360.0, -360.0)) / 2.0),
    )
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -2.0 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25.0**7 + 1e-12)) * np.sin(
        np.radians(60.0 * np.exp(-((h_bar - 275.0) / 25.0) ** 2))
    )
    return np.sqrt(
        (delta_l / sl) ** 2
        + (delta_c / sc) ** 2
        + (delta_big_h / sh) ** 2
        + rt * (delta_c / sc) * (delta_big_h / sh)
    )

# Fallback 91-class 1-indexed COCO lookup for older rfdetr (<1.8.2)
FALLBACK_COCO91 = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
}


def estimate_vehicle_color_with_confidence(
    frame_bgr: np.ndarray, box_xyxy: np.ndarray
) -> tuple[str, float]:
    """Estimate paint color from the central body region of a vehicle box."""
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    width, height = x2 - x1, y2 - y1
    if width < 8 or height < 8:
        return "unknown", 0.0

    # Exclude the outer box where road, sky, wheels, and neighboring objects
    # are most likely to dominate the detector's box.
    left, right = x1 + int(width * 0.18), x2 - int(width * 0.18)
    top, bottom = y1 + int(height * 0.20), y1 + int(height * 0.78)
    roi = frame_bgr[top:bottom, left:right]
    if roi.size == 0:
        return "unknown", 0.0
    roi = cv2.resize(roi, (64, 48), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv_flat = hsv.reshape(-1, 3)
    value = hsv_flat[:, 2]
    saturation = hsv_flat[:, 1]
    lightness = lab[:, 0]

    # Reject black window/road pixels and near-white glare only when there are
    # enough remaining paint pixels; achromatic paint is handled separately.
    valid = (value >= 28) & (lightness >= 22) & (lightness <= 248)
    pixels = lab[valid]
    if len(pixels) < 20:
        pixels = lab
    if len(pixels) == 0:
        return "unknown", 0.0

    median_l = float(np.median(pixels[:, 0]))
    chroma = np.hypot(pixels[:, 1] - 128.0, pixels[:, 2] - 128.0)
    chromatic_ratio = float(np.count_nonzero(chroma >= 13.0)) / float(len(pixels))
    if chromatic_ratio < 0.20:
        if median_l < 58:
            color = "black"
            return color, min(0.99, 0.55 + 0.35 * (1.0 - chromatic_ratio))
        if median_l < 105:
            color = "dark gray"
            return color, min(0.99, 0.55 + 0.35 * (1.0 - chromatic_ratio))
        if median_l < 178:
            color = "gray"
            return color, min(0.99, 0.55 + 0.35 * (1.0 - chromatic_ratio))
        color = "white"
        return color, min(0.99, 0.55 + 0.35 * (1.0 - chromatic_ratio))

    chromatic_pixels = pixels[chroma >= 13.0]
    if len(chromatic_pixels) < 10:
        chromatic_pixels = pixels
    # Median is more stable than the mean under metallic highlights and shadows.
    dominant = np.median(chromatic_pixels, axis=0)
    distances = ciede2000_distance(_CHROMATIC_PAINT_LAB, dominant)
    best = int(np.argmin(distances))
    margin = float(np.partition(distances, 1)[1] - distances[best]) if len(distances) > 1 else 0.0
    confidence = float(np.clip(0.45 + margin / 45.0 + min(chromatic_ratio, 0.8) * 0.2, 0.0, 0.99))
    return _CHROMATIC_PAINT_NAMES[best], confidence


def estimate_vehicle_color(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> str:
    return estimate_vehicle_color_with_confidence(frame_bgr, box_xyxy)[0]


def vehicle_crop_quality(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> float:
    """Score crop sharpness and usable area for temporal evidence weighting."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    area_score = min(1.0, (crop.shape[0] * crop.shape[1]) / 40000.0)
    sharpness_score = float(np.clip(sharpness / 500.0, 0.0, 1.0))
    return float(np.clip(0.35 * area_score + 0.65 * sharpness_score, 0.0, 1.0))


def refine_vehicle_color(
    frame_bgr: np.ndarray, box_xyxy: np.ndarray, scales: tuple[float, ...]
) -> tuple[str, float]:
    """Run a bounded multi-scale, sharpened ROI color refinement pass."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box_xyxy]
    pad_x, pad_y = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown", 0.0

    votes: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    for scale in scales:
        resized = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
        color, confidence = estimate_vehicle_color_with_confidence(
            sharpened, np.array([0, 0, sharpened.shape[1], sharpened.shape[0]], dtype=np.float32)
        )
        votes[color] += 1
        confidence_sum[color] += confidence
    if not votes:
        return "unknown", 0.0
    color = votes.most_common(1)[0][0]
    return color, float(confidence_sum[color] / votes[color])


def class_aware_nms_numpy(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Class-aware greedy NMS in NumPy."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    max_coord = float(np.max(boxes)) + 4096.0
    offsets = class_ids.astype(np.float32) * max_coord
    shifted = boxes + offsets[:, None]
    x1, y1, x2, y2 = shifted[:, 0], shifted[:, 1], shifted[:, 2], shifted[:, 3]
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


class SimpleClassIoUTracker:
    """Class-aware IoU tracker fallback when supervision.ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: dict[int, tuple[np.ndarray, str, int]] = {}

    def update(self, boxes: np.ndarray, names: list[str]) -> np.ndarray:
        if len(boxes) == 0:
            self.tracks = {
                tid: (b, cname, lost + 1)
                for tid, (b, cname, lost) in self.tracks.items()
                if lost + 1 <= self.max_lost
            }
            return np.empty((0,), dtype=np.int32)

        assigned_ids = np.full((len(boxes),), -1, dtype=np.int32)
        used_tracks: set[int] = set()
        for i, (box, cname) in enumerate(zip(boxes, names)):
            best_iou = self.iou_threshold
            best_tid = -1
            for tid, (tbox, tcname, _) in self.tracks.items():
                if tid in used_tracks or tcname != cname:
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
                assigned_ids[i] = best_tid
                used_tracks.add(best_tid)
            else:
                assigned_ids[i] = self.next_id
                self.next_id += 1

        new_tracks: dict[int, tuple[np.ndarray, str, int]] = {}
        for tid, (tbox, tcname, lost) in self.tracks.items():
            if tid not in used_tracks and lost + 1 <= self.max_lost:
                new_tracks[tid] = (tbox, tcname, lost + 1)
        for i, tid in enumerate(assigned_ids):
            new_tracks[int(tid)] = (boxes[i], names[i], 0)
        self.tracks = new_tracks
        return assigned_ids


class VehicleDetector:
    """Unified Vehicle Detector supporting RF-DETR (Apache-2.0) and Ultralytics YOLO/RT-DETR."""

    RFDETR_MAP = {
        "rfdetr-nano": "RFDETRNano",
        "rfdetr-small": "RFDETRSmall",
        "rfdetr-medium": "RFDETRMedium",
        "rfdetr-large": "RFDETRLarge",
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    }

    def __init__(
        self,
        model_name: str,
        target_classes: set[str],
        device: str,
        half: bool = False,
        imgsz: int | None = None,
        enable_track: bool = False,
        fps: float = 25.0,
    ) -> None:
        self.model_name = model_name.strip()
        self.target_classes = {c.strip().lower() for c in target_classes}
        self.device = device
        self.half = bool(half and device.startswith("cuda"))
        self.imgsz = imgsz
        self.enable_track = enable_track
        self.backend = "rfdetr" if self.model_name.lower() in self.RFDETR_MAP else "ultralytics"
        self.model: Any = None
        self.coco_classes: Any = FALLBACK_COCO91
        self.sv_tracker: Any = None
        self.fallback_tracker = SimpleClassIoUTracker() if enable_track else None
        self.color_history: dict[int, Counter[str]] = {}
        self.type_history: dict[int, Counter[str]] = {}

        if enable_track and self.backend == "rfdetr":
            try:
                import supervision as sv

                self.sv_tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
            except Exception:
                self.sv_tracker = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            if self.backend == "rfdetr":
                import rfdetr

                try:
                    from rfdetr.assets.coco_classes import COCO_CLASSES

                    self.coco_classes = COCO_CLASSES
                except ImportError:
                    try:
                        from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore[no-redef]

                        self.coco_classes = COCO_CLASSES
                    except ImportError:
                        self.coco_classes = FALLBACK_COCO91

                cls_name = self.RFDETR_MAP[self.model_name.lower()]
                model_cls = getattr(rfdetr, cls_name, None)
                if model_cls is None:
                    raise RuntimeError(f"rfdetr package does not expose '{cls_name}'. Upgrade: pip install -U rfdetr")

                init_kwargs: dict[str, Any] = {"device": self.device}
                if self.imgsz is not None and self.imgsz > 0:
                    init_kwargs["resolution"] = int(self.imgsz)
                try:
                    self.model = model_cls(**init_kwargs)
                except TypeError:
                    init_kwargs.pop("resolution", None)
                    try:
                        self.model = model_cls(**init_kwargs)
                    except TypeError:
                        self.model = model_cls()

                if self.device.startswith("cuda") and hasattr(self.model, "to"):
                    self.model.to(self.device)
                dtype = torch.float16 if self.half else torch.float32
                if hasattr(self.model, "inference"):
                    try:
                        self.model.inference(dtype=dtype)
                    except Exception:
                        pass
                elif hasattr(self.model, "optimize_for_inference"):
                    try:
                        self.model.optimize_for_inference(dtype=dtype)
                    except Exception:
                        pass
            else:
                from ultralytics import YOLO

                self.model = YOLO(self.model_name)
                self.model.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load/download model '{self.model_name}' on device '{self.device}': {exc}"
            ) from exc

    def _resolve_rfdetr_class_name(self, detections: Any, idx: int, cid: int) -> str:
        """Resolve class name safely across rfdetr >=1.8.2 (0-based) and <1.8.2 (1-based) schemas."""
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
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[float], np.ndarray | None]:
        """Return boxes, scores, class names, paint colors/confidences, and track IDs."""
        if self.backend == "rfdetr":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                detections = self.model.predict(frame_rgb, threshold=conf)
            if detections is None or len(detections) == 0:
                empty_b = np.empty((0, 4), dtype=np.float32)
                empty_s = np.empty((0,), dtype=np.float32)
                if self.fallback_tracker is not None and self.sv_tracker is None:
                    self.fallback_tracker.update(empty_b, [])
                return empty_b, empty_s, [], [], [], None

            xyxy = np.asarray(detections.xyxy, dtype=np.float32)
            scores = np.asarray(detections.confidence, dtype=np.float32)
            cids = np.asarray(detections.class_id, dtype=np.int32)

            keep_idx: list[int] = []
            names: list[str] = []
            for i, cid in enumerate(cids):
                cname = self._resolve_rfdetr_class_name(detections, i, int(cid))
                if cname in self.target_classes:
                    keep_idx.append(i)
                    names.append(cname)

            if not keep_idx:
                empty_b = np.empty((0, 4), dtype=np.float32)
                empty_s = np.empty((0,), dtype=np.float32)
                if self.fallback_tracker is not None and self.sv_tracker is None:
                    self.fallback_tracker.update(empty_b, [])
                return empty_b, empty_s, [], [], [], None

            keep_arr = np.array(keep_idx, dtype=np.int64)
            xyxy = xyxy[keep_arr]
            scores = scores[keep_arr]
            cids = cids[keep_arr]

            if 0.0 < iou < 1.0 and len(xyxy) > 1:
                nms_idx = class_aware_nms_numpy(xyxy, scores, cids, iou)
                xyxy = xyxy[nms_idx]
                scores = scores[nms_idx]
                names = [names[int(k)] for k in nms_idx]
            else:
                nms_idx = np.arange(len(xyxy), dtype=np.int64)

            track_ids: np.ndarray | None = None
            if self.enable_track:
                if self.sv_tracker is not None:
                    try:
                        sub_det = detections[keep_arr][nms_idx]
                        tracked = self.sv_tracker.update_with_detections(sub_det)
                        xyxy = np.asarray(tracked.xyxy, dtype=np.float32)
                        scores = np.asarray(tracked.confidence, dtype=np.float32)
                        names = [
                            self._resolve_rfdetr_class_name(tracked, j, int(cid))
                            for j, cid in enumerate(tracked.class_id)
                        ]
                        if tracked.tracker_id is not None:
                            track_ids = np.asarray(tracked.tracker_id, dtype=np.int32)
                    except Exception:
                        track_ids = self.fallback_tracker.update(xyxy, names) if self.fallback_tracker else None
                elif self.fallback_tracker is not None:
                    track_ids = self.fallback_tracker.update(xyxy, names)
            track_ids = track_ids if track_ids is not None else (
                self.fallback_tracker.update(xyxy, names) if self.fallback_tracker is not None else None
            )
            color_results = [estimate_vehicle_color_with_confidence(frame_bgr, box) for box in xyxy]
            colors = [result[0] for result in color_results]
            color_confs = [result[1] for result in color_results]
            names = self._stabilize_types(names, scores, xyxy, track_ids, frame_bgr)
            return xyxy, scores, names, self._stabilize_colors(
                colors, track_ids, color_confs, xyxy, frame_bgr
            ), color_confs, track_ids

        # Ultralytics YOLO / RT-DETR branch
        target_ids = [COCO80_NAME_TO_ID[c] for c in self.target_classes if c in COCO80_NAME_TO_ID]
        infer_kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "device": self.device,
            "half": self.half,
            "verbose": False,
        }
        if target_ids:
            infer_kwargs["classes"] = sorted(target_ids)
        if self.imgsz is not None and self.imgsz > 0:
            infer_kwargs["imgsz"] = int(self.imgsz)

        if self.enable_track:
            results = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **infer_kwargs)
        else:
            results = self.model.predict(frame_bgr, **infer_kwargs)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), [], [], [], None

        boxes_obj = results[0].boxes
        xyxy = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
        cls_ids = boxes_obj.cls.int().detach().cpu().numpy().astype(np.int32)
        id2name = results[0].names or {}
        names = [str(id2name.get(int(cid), f"class_{int(cid)}")).lower() for cid in cls_ids]
        track_ids = (
            boxes_obj.id.int().detach().cpu().numpy().astype(np.int32)
            if (self.enable_track and boxes_obj.id is not None)
            else None
        )
        color_results = [estimate_vehicle_color_with_confidence(frame_bgr, box) for box in xyxy]
        colors = [result[0] for result in color_results]
        color_confs = [result[1] for result in color_results]
        names = self._stabilize_types(names, scores, xyxy, track_ids, frame_bgr)
        return xyxy, scores, names, self._stabilize_colors(
            colors, track_ids, color_confs, xyxy, frame_bgr
        ), color_confs, track_ids

    def _stabilize_types(
        self,
        names: list[str],
        scores: np.ndarray,
        boxes: np.ndarray,
        track_ids: np.ndarray | None,
        frame_bgr: np.ndarray,
    ) -> list[str]:
        if track_ids is None:
            return names
        active_ids = {int(track_id) for track_id in track_ids}
        self.type_history = {
            track_id: history for track_id, history in self.type_history.items() if track_id in active_ids
        }
        stabilized: list[str] = []
        for name, score, box, track_id in zip(names, scores, boxes, track_ids):
            tid = int(track_id)
            quality = vehicle_crop_quality(frame_bgr, box)
            history = self.type_history.setdefault(tid, Counter())
            history[name] += max(0.05, float(score) * (0.35 + quality))
            stabilized.append(history.most_common(1)[0][0])
        return stabilized

    def _stabilize_colors(
        self,
        colors: list[str],
        track_ids: np.ndarray | None,
        confidences: list[float],
        boxes: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> list[str]:
        if track_ids is None:
            return colors
        stabilized: list[str] = []
        active_ids = {int(track_id) for track_id in track_ids}
        self.color_history = {
            track_id: history for track_id, history in self.color_history.items() if track_id in active_ids
        }
        for color, confidence, box, track_id in zip(colors, confidences, boxes, track_ids):
            tid = int(track_id)
            history = self.color_history.setdefault(tid, Counter())
            history[color] += max(0.05, float(confidence) * (0.35 + vehicle_crop_quality(frame_bgr, box)))
            stabilized.append(history.most_common(1)[0][0])
        return stabilized


def draw_vehicle_annotations(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    colors: list[str],
    track_ids: np.ndarray | None,
) -> None:
    """Draw resolution-scaled per-class bounding boxes and filled label backgrounds in-place."""
    h, w = frame_bgr.shape[:2]
    thickness = max(2, int(round(min(w, h) / 450)))
    font_scale = max(0.5, min(w, h) / 1400.0)
    font_thick = max(1, thickness - 1)

    for i, (box, score, cname, paint_color) in enumerate(zip(boxes, scores, names, colors)):
        color = VEHICLE_COLORS_BGR.get(cname, (40, 180, 255))
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (15, 15, 15) if lum > 140 else (250, 250, 250)

        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        tid_str = f" #{int(track_ids[i])}" if (track_ids is not None and i < len(track_ids)) else ""
        label = f"{paint_color} {cname}{tid_str} {float(score):.2f}"

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
        description="Detect vehicles (car, motorcycle, bus, truck, bicycle) in a video and save an annotated MP4."
    )
    parser.add_argument("--source", "-s", type=str, required=True, help="Video path, image path, camera index, or stream URL")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output annotated video path")
    parser.add_argument("--json-output", type=Path, default=None, help="JSONL detection output path")
    parser.add_argument(
        "--model",
        type=str,
        default="rfdetr-medium",
        help="Model: rfdetr-nano | rfdetr-small | rfdetr-medium (default) | rfdetr-large | yolo26n.pt | yolo26s.pt",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0 < conf < 1)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (0 < iou < 1)")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size (e.g., 640)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cuda | mps | cpu")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on CUDA")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process 1 out of every N frames (default: 1)")
    parser.add_argument("--show", action="store_true", help="Display live annotated preview window")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack multi-object tracking IDs")
    parser.add_argument("--refine-threshold", type=float, default=0.60, help="Refine low-confidence color estimates")
    parser.add_argument(
        "--refine-scales", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0],
        help="ROI scales for the refinement pass",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_VEHICLE_CLASSES),
        help="Vehicle classes to detect (default: car motorcycle bus truck bicycle)",
    )
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--no-db", action="store_true", help="Disable SQLite database persistence")
    parser.add_argument("--camera-id", type=str, default="cam_default", help="Camera identifier")
    parser.add_argument("--video-id", type=str, default=None, help="Custom video ID (default: source file stem)")
    args = parser.parse_args(argv)

    if not args.source.isdigit() and not args.source.startswith(("rtsp://", "http://", "https://")):
        if not Path(args.source).is_file():
            parser.error(f"Input source not found: {args.source}")
    if not (0.0 < args.conf < 1.0):
        parser.error("--conf must be strictly between 0.0 and 1.0")
    if not (0.0 < args.iou < 1.0):
        parser.error("--iou must be strictly between 0.0 and 1.0")
    if args.vid_stride < 1:
        parser.error("--vid-stride must be >= 1")
    if not (0.0 < args.refine_threshold < 1.0):
        parser.error("--refine-threshold must be strictly between 0.0 and 1.0")
    if any(scale <= 0.0 for scale in args.refine_scales):
        parser.error("--refine-scales values must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = args.output or source_path.with_name(f"{source_path.stem}_vehicle_out.mp4")
    image_mode = source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if image_mode and args.output is None:
        output_path = source_path.with_name(f"{source_path.stem}_vehicle_out{source_path.suffix}")
    json_output = args.json_output or output_path.with_suffix(".jsonl")

    try:
        device = resolve_device(args.device)
        configure_torch_for_inference(device)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    capture_source: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(capture_source)
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
        total_steps = (raw_total + args.vid_stride - 1) // args.vid_stride if raw_total > 0 else None

        video_id = args.video_id or source_path.stem
        db: CCTVDatabase | None = None
        if not image_mode and not args.no_db:
            try:
                db = CCTVDatabase(args.db)
                db.register_video(
                    video_id=video_id,
                    file_path=str(source_path.resolve()),
                    camera_id=args.camera_id,
                    duration_s=duration_s,
                    fps=src_fps,
                    width=width,
                    height=height,
                )
            except Exception as exc:
                print(f"[WARN] Could not initialize database at {args.db}: {exc}. Continuing without DB.", file=sys.stderr)
                db = None

        detector = VehicleDetector(
            model_name=args.model,
            target_classes=set(args.classes),
            device=device,
            half=args.half,
            imgsz=args.imgsz,
            enable_track=args.track,
            fps=out_fps,
        )
        if image_mode:
            final_output_path = output_path
        else:
            writer, final_output_path = create_video_writer(output_path, out_fps, width, height)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_file = json_output.open("w", encoding="utf-8")

        print(
            f"[INFO] Model: {args.model} ({detector.backend}) | Classes: {sorted(detector.target_classes)} | "
            f"Device: {device} (FP16={detector.half}) | Resolution: {width}x{height} @ {out_fps:.2f} FPS | "
            f"DB: {'Enabled' if db is not None else 'Disabled'}"
        )

        raw_idx = 0
        processed_frames = 0
        frames_with_vehicles = 0
        class_counts: Counter[str] = Counter()
        t0 = time.perf_counter()

        with tqdm(total=total_steps, desc="Detecting vehicles", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                raw_idx += 1
                if (raw_idx - 1) % args.vid_stride != 0:
                    continue

                t_current_s = float(raw_idx - 1) / src_fps

                boxes, scores, names, colors, color_confs, track_ids = detector.predict(
                    frame_bgr, conf=args.conf, iou=args.iou
                )
                if len(boxes) > 0:
                    frames_with_vehicles += 1
                    class_counts.update(names)
                    for i, (box, score, name, color, color_conf) in enumerate(
                        zip(boxes, scores, names, colors, color_confs)
                    ):
                        refined = False
                        if float(score) < args.refine_threshold or color_conf < args.refine_threshold:
                            refined_color, refined_conf = refine_vehicle_color(
                                frame_bgr, box, tuple(args.refine_scales)
                            )
                            if refined_conf > color_conf:
                                color, color_conf = refined_color, refined_conf
                                colors[i] = color
                            refined = True
                        uncertain = float(score) < args.refine_threshold or color_conf < args.refine_threshold
                        track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                        record = {
                            "frame_id": raw_idx,
                            "track_id": track_id,
                            "bbox": [round(float(value), 2) for value in box],
                            "vehicle_type": name,
                            "type_conf": round(float(score), 4),
                            "color": color,
                            "color_conf": round(float(color_conf), 4),
                            "refined": refined,
                            "uncertain": uncertain,
                        }
                        json_file.write(json.dumps(record, separators=(",", ":")) + "\n")

                        if db is not None and track_id is not None:
                            db.upsert_vehicle(
                                video_id=video_id,
                                track_id=track_id,
                                vehicle_type=name,
                                color=color,
                                color_conf=float(color_conf),
                                first_seen_s=t_current_s,
                                last_seen_s=t_current_s,
                                frame_count=1,
                                peak_conf=float(score),
                                bbox_json=json.dumps([round(float(v), 2) for v in box]),
                                bbox_time_s=t_current_s,
                                bbox_frame=raw_idx - 1,
                            )

                    draw_vehicle_annotations(frame_bgr, boxes, scores, names, colors, track_ids)

                if image_mode:
                    if not cv2.imwrite(str(final_output_path), frame_bgr):
                        raise RuntimeError(f"Failed to write annotated image: {final_output_path}")
                else:
                    writer.write(frame_bgr)
                processed_frames += 1
                pbar.update(1)
                pbar.set_postfix(vehicles=len(boxes), total=sum(class_counts.values()))

                if db is not None and processed_frames % 100 == 0:
                    db.commit_vehicles()

                if args.show:
                    try:
                        cv2.imshow("vehicle_detect", frame_bgr)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            print("\n[INFO] Preview interrupted by user ('q').")
                            break
                    except cv2.error:
                        args.show = False

        if db is not None:
            db.commit_vehicles()
            db.update_video_status(video_id, "completed")

        elapsed = max(time.perf_counter() - t0, 1e-6)
        if processed_frames == 0:
            raise RuntimeError(f"Video '{args.source}' yielded 0 readable frames.")
        total_boxes = sum(class_counts.values())
        if total_boxes == 0:
            print("[WARN] 0 vehicle detections across the entire video. Consider lowering --conf.")

        breakdown = ", ".join(f"{k}:{v}" for k, v in class_counts.most_common()) or "none"
        print(
            f"[DONE] Processed {processed_frames} frames in {elapsed:.2f}s ({processed_frames / elapsed:.2f} FPS) | "
            f"Frames w/ vehicles: {frames_with_vehicles} | Total: {total_boxes} ({breakdown}) | "
            f"Saved: {final_output_path.resolve()}"
        )
        json_file.flush()
        json_file.close()
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
        if "json_file" in locals() and not json_file.closed:
            json_file.close()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())