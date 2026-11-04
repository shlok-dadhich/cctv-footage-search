#!/usr/bin/env python3
"""
person_detect.py — Production real-time person detection and tracking pipeline.

Pip install (verified minimum versions):
    pip install "torch>=2.2.0" "torchvision>=0.17.0" "rfdetr>=1.8.2" "supervision>=0.25.0" "opencv-python>=4.9.0" "numpy>=1.26.0" "tqdm>=4.66.0"
    # Optional for Ultralytics YOLO26/YOLO11/RT-DETR weights:
    pip install "ultralytics>=8.4.0"

Model options (--model) & Licenses:
    - rfdetr-small  (DEFAULT) : Roboflow RF-DETR Small (53.0 COCO mAP, 3.5 ms T4)  | License: Apache-2.0
    - rfdetr-nano   (LIGHTER) : Roboflow RF-DETR Nano  (48.4 COCO mAP, 2.3 ms T4)  | License: Apache-2.0
    - rfdetr-medium (MEDIUM)  : Roboflow RF-DETR Medium(54.7 COCO mAP, 4.4 ms T4)  | License: Apache-2.0
    - rfdetr-large  (HEAVIER) : Roboflow RF-DETR Large (56.5 COCO mAP, 6.8 ms T4)  | License: Apache-2.0
    - yolo26n.pt / yolo26s.pt / yolo26l.pt / yolo11s.pt : Ultralytics YOLO         | License: AGPL-3.0

Hardware notes:
    - Auto-detects CUDA -> Apple Silicon MPS -> CPU.
    - Pass --half on CUDA for FP16 inference.
    - Run: python person_detect.py --source input.mp4 --output out.mp4
"""
from __future__ import annotations

import argparse
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

from common.color import estimate_person_colors
from common.db import CCTVDatabase
from common.device import configure_torch_for_inference, resolve_device
from common.gender_age import GenderAgeEstimator, GenderAgeResult
from common.video_io import create_video_writer

PERSON_COLOR_BGR = (60, 220, 80)  # High-visibility green in BGR


def nms_boxes_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Pure NumPy greedy NMS for additional IoU filtering when requested."""
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
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


class SimpleIoUTracker:
    """Lightweight zero-dependency Kalman/IoU fallback tracker if supervision.ByteTrack is unavailable."""

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks: dict[int, tuple[np.ndarray, int]] = {}  # track_id -> (box, lost_frames)

    def update(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            self.tracks = {
                tid: (b, lost + 1) for tid, (b, lost) in self.tracks.items() if lost + 1 <= self.max_lost
            }
            return np.empty((0,), dtype=np.int32)

        assigned_ids = np.full((len(boxes),), -1, dtype=np.int32)
        used_tracks: set[int] = set()

        for i, box in enumerate(boxes):
            best_iou = self.iou_threshold
            best_tid = -1
            for tid, (tbox, _) in self.tracks.items():
                if tid in used_tracks:
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

        new_tracks: dict[int, tuple[np.ndarray, int]] = {}
        for tid, (tbox, lost) in self.tracks.items():
            if tid not in used_tracks and lost + 1 <= self.max_lost:
                new_tracks[tid] = (tbox, lost + 1)
        for i, tid in enumerate(assigned_ids):
            new_tracks[int(tid)] = (boxes[i], 0)
        self.tracks = new_tracks
        return assigned_ids


class PersonDetector:
    """Unified Person Detector supporting RF-DETR (Apache-2.0) and Ultralytics YOLO/RT-DETR."""

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
        device: str,
        half: bool = False,
        imgsz: int | None = None,
        enable_track: bool = False,
        fps: float = 25.0,
    ) -> None:
        self.model_name = model_name.strip()
        self.device = device
        self.half = bool(half and device.startswith("cuda"))
        self.imgsz = imgsz
        self.enable_track = enable_track
        self.backend = "rfdetr" if self.model_name.lower() in self.RFDETR_MAP else "ultralytics"
        self.model: Any = None
        self.coco_classes: Any = {}
        self.sv_tracker: Any = None
        self.fallback_tracker = SimpleIoUTracker() if enable_track else None

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
                except ImportError:
                    from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore[no-redef]
                self.coco_classes = COCO_CLASSES

                cls_name = self.RFDETR_MAP[self.model_name.lower()]
                model_cls = getattr(rfdetr, cls_name, None)
                if model_cls is None:
                    raise RuntimeError(f"rfdetr package does not expose '{cls_name}'. Upgrade: pip install -U rfdetr")

                init_kwargs: dict[str, Any] = {"device": self.device}
                if self.imgsz is not None and self.imgsz > 0:
                    # RF-DETR requires resolution divisible by patch size (typically 32/56); pass only when valid
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

    def _is_person_rfdetr(self, detections: Any, idx: int, cid: int) -> bool:
        """Schema-agnostic person check across rfdetr 0-indexed (>=1.8.2) and 1-indexed (<1.8.2) checkpoints."""
        if hasattr(detections, "data") and isinstance(detections.data, dict):
            cnames = detections.data.get("class_name")
            if cnames is not None and len(cnames) > idx:
                return str(cnames[idx]).strip().lower() == "person"
        if isinstance(self.coco_classes, dict) and cid in self.coco_classes:
            return str(self.coco_classes[cid]).strip().lower() == "person"
        if isinstance(self.coco_classes, (list, tuple)) and 0 <= cid < len(self.coco_classes):
            return str(self.coco_classes[cid]).strip().lower() == "person"
        return cid == 0

    def predict(
        self, frame_bgr: np.ndarray, conf: float, iou: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (boxes_xyxy[N,4], scores[N], track_ids[N] or None)."""
        if self.backend == "rfdetr":
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                detections = self.model.predict(frame_rgb, threshold=conf)
            if detections is None or len(detections) == 0:
                empty_b = np.empty((0, 4), dtype=np.float32)
                empty_s = np.empty((0,), dtype=np.float32)
                if self.fallback_tracker is not None and self.sv_tracker is None:
                    self.fallback_tracker.update(empty_b)
                return empty_b, empty_s, None

            xyxy = np.asarray(detections.xyxy, dtype=np.float32)
            scores = np.asarray(detections.confidence, dtype=np.float32)
            cids = np.asarray(detections.class_id, dtype=np.int32)

            keep = [i for i, cid in enumerate(cids) if self._is_person_rfdetr(detections, i, int(cid))]
            if not keep:
                empty_b = np.empty((0, 4), dtype=np.float32)
                empty_s = np.empty((0,), dtype=np.float32)
                if self.fallback_tracker is not None and self.sv_tracker is None:
                    self.fallback_tracker.update(empty_b)
                return empty_b, empty_s, None

            keep_arr = np.array(keep, dtype=np.int64)
            xyxy = xyxy[keep_arr]
            scores = scores[keep_arr]

            if 0.0 < iou < 1.0 and len(xyxy) > 1:
                nms_idx = nms_boxes_numpy(xyxy, scores, iou)
                xyxy = xyxy[nms_idx]
                scores = scores[nms_idx]
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
                        if tracked.tracker_id is not None:
                            track_ids = np.asarray(tracked.tracker_id, dtype=np.int32)
                    except Exception:
                        track_ids = self.fallback_tracker.update(xyxy) if self.fallback_tracker else None
                elif self.fallback_tracker is not None:
                    track_ids = self.fallback_tracker.update(xyxy)
            return xyxy, scores, track_ids

        # Ultralytics YOLO / RT-DETR branch (COCO class 0 == person)
        infer_kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "classes": [0],
            "device": self.device,
            "half": self.half,
            "verbose": False,
        }
        if self.imgsz is not None and self.imgsz > 0:
            infer_kwargs["imgsz"] = int(self.imgsz)

        if self.enable_track:
            results = self.model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", **infer_kwargs)
        else:
            results = self.model.predict(frame_bgr, **infer_kwargs)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), None

        boxes_obj = results[0].boxes
        xyxy = boxes_obj.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes_obj.conf.detach().cpu().numpy().astype(np.float32)
        track_ids = (
            boxes_obj.id.int().detach().cpu().numpy().astype(np.int32)
            if (self.enable_track and boxes_obj.id is not None)
            else None
        )
        return xyxy, scores, track_ids


def draw_annotations(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray | None,
    color: tuple[int, int, int] = PERSON_COLOR_BGR,
    labels_extra: list[str] | None = None,
) -> None:
    """Draw resolution-scaled bounding boxes and filled high-contrast label backgrounds in-place."""
    h, w = frame_bgr.shape[:2]
    thickness = max(2, int(round(min(w, h) / 450)))
    font_scale = max(0.5, min(w, h) / 1400.0)
    font_thick = max(1, thickness - 1)

    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (15, 15, 15) if lum > 140 else (250, 250, 250)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        tid_str = f" #{int(track_ids[i])}" if (track_ids is not None and i < len(track_ids)) else ""
        extra_str = f" | {labels_extra[i]}" if (labels_extra is not None and i < len(labels_extra) and labels_extra[i]) else ""
        label = f"person{tid_str}{extra_str} {float(score):.2f}"

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
        description="Detect persons in a video and save an annotated video (Apache-2.0 RF-DETR default)."
    )
    parser.add_argument("--source", "-s", type=Path, required=True, help="Input video file path")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output annotated video path")
    parser.add_argument(
        "--model",
        type=str,
        default="rfdetr-small",
        help="Model: rfdetr-nano | rfdetr-small (default) | rfdetr-medium | rfdetr-large | yolo26n.pt | yolo26s.pt",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0 < conf < 1)")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold (0 < iou < 1)")
    parser.add_argument("--imgsz", type=int, default=None, help="Optional inference image size (e.g., 640)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto | cuda | mps | cpu")
    parser.add_argument("--half", action="store_true", help="Enable FP16 inference on CUDA")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process 1 out of every N frames (default: 1)")
    parser.add_argument("--show", action="store_true", help="Display live annotated preview window")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack multi-object tracking IDs")
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--no-db", action="store_true", help="Disable SQLite database persistence")
    parser.add_argument("--camera-id", type=str, default="cam_default", help="Camera identifier")
    parser.add_argument("--video-id", type=str, default=None, help="Custom video ID (default: source file stem)")
    parser.add_argument("--mivolo-checkpoint", type=Path, default=None, help="Path to MiVOLO gender/age model checkpoint")
    args = parser.parse_args(argv)

    if not args.source.is_file():
        parser.error(f"Input video file not found: {args.source}")
    if not (0.0 < args.conf < 1.0):
        parser.error("--conf must be strictly between 0.0 and 1.0")
    if not (0.0 < args.iou < 1.0):
        parser.error("--iou must be strictly between 0.0 and 1.0")
    if args.vid_stride < 1:
        parser.error("--vid-stride must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or args.source.with_name(f"{args.source.stem}_person_out.mp4")
    video_id = args.video_id or args.source.stem

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
    db: CCTVDatabase | None = None
    if not args.no_db:
        try:
            db = CCTVDatabase(args.db)
        except Exception as exc:
            print(f"[WARN] Could not initialize database at {args.db}: {exc}. Continuing without DB.", file=sys.stderr)
            db = None

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

        if db is not None:
            db.register_video(
                video_id=video_id,
                file_path=str(args.source.resolve()),
                camera_id=args.camera_id,
                duration_s=duration_s,
                fps=src_fps,
                width=width,
                height=height,
            )

        detector = PersonDetector(
            model_name=args.model,
            device=device,
            half=args.half,
            imgsz=args.imgsz,
            enable_track=args.track,
            fps=out_fps,
        )

        gender_estimator = GenderAgeEstimator(
            checkpoint=args.mivolo_checkpoint,
            device=device,
            half=args.half,
            with_persons=True,
        )

        writer, final_output_path = create_video_writer(output_path, out_fps, width, height)

        print(
            f"[INFO] Model: {args.model} ({detector.backend}) | Device: {device} (FP16={detector.half}) | "
            f"Resolution: {width}x{height} @ {out_fps:.2f} FPS | "
            f"MiVOLO Gender: {'Enabled' if gender_estimator.is_available else 'Fallback'} | "
            f"DB: {'Enabled' if db is not None else 'Disabled'}"
        )

        raw_idx = 0
        processed_frames = 0
        frames_with_person = 0
        total_detections = 0
        t0 = time.perf_counter()

        with tqdm(total=total_steps, desc="Detecting persons", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                raw_idx += 1
                if (raw_idx - 1) % args.vid_stride != 0:
                    continue

                t_current_s = float(raw_idx - 1) / src_fps

                boxes, scores, track_ids = detector.predict(frame_bgr, conf=args.conf, iou=args.iou)
                labels_extra: list[str] = []

                if len(boxes) > 0:
                    frames_with_person += 1
                    total_detections += len(boxes)

                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        tid = int(track_ids[i]) if (track_ids is not None and i < len(track_ids)) else (i + 1)
                        bx1, by1, bx2, by2 = [int(round(float(v))) for v in box]
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(width, bx2), min(height, by2)

                        person_crop = frame_bgr[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else np.empty((0, 0, 3), dtype=np.uint8)
                        ga_res = (
                            gender_estimator.predict_crop_stabilized(person_crop, track_id=tid)
                            if (person_crop.size > 0 and tid > 0)
                            else GenderAgeResult()
                        )
                        upper_col, lower_col = estimate_person_colors(frame_bgr, box)

                        extra_parts: list[str] = []
                        if ga_res.gender != "unknown":
                            age_str = f" {int(ga_res.age)}y" if ga_res.age else ""
                            extra_parts.append(f"{ga_res.gender}{age_str}")
                        if upper_col != "unknown":
                            extra_parts.append(f"top:{upper_col}")
                        if lower_col != "unknown":
                            extra_parts.append(f"bot:{lower_col}")
                        labels_extra.append(" | ".join(extra_parts))

                        if db is not None:
                            bbox_json_str = json.dumps([round(float(v), 2) for v in box])
                            db.upsert_person(
                                video_id=video_id,
                                track_id=tid,
                                gender=ga_res.gender,
                                age_est=ga_res.age,
                                upper_color=upper_col,
                                lower_color=lower_col,
                                first_seen_s=t_current_s,
                                last_seen_s=t_current_s,
                                frame_count=1,
                                peak_conf=float(score),
                                bbox_json=bbox_json_str,
                            )

                    draw_annotations(frame_bgr, boxes, scores, track_ids, labels_extra=labels_extra)

                writer.write(frame_bgr)
                processed_frames += 1
                pbar.update(1)
                pbar.set_postfix(persons=len(boxes), total=total_detections)

                # Periodic DB commit every 100 processed frames
                if db is not None and processed_frames % 100 == 0:
                    db.commit_persons()

                if args.show:
                    try:
                        cv2.imshow("person_detect", frame_bgr)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            print("\n[INFO] Preview interrupted by user ('q').")
                            break
                    except cv2.error:
                        args.show = False

        if db is not None:
            db.commit_persons()
            db.update_video_status(video_id, "completed")

        elapsed = max(time.perf_counter() - t0, 1e-6)
        if processed_frames == 0:
            raise RuntimeError(f"Video '{args.source}' yielded 0 readable frames.")
        if total_detections == 0:
            print("[WARN] 0 person detections across the entire video. Consider lowering --conf.")

        print(
            f"[DONE] Processed {processed_frames} frames in {elapsed:.2f}s ({processed_frames / elapsed:.2f} FPS) | "
            f"Frames w/ person: {frames_with_person} | Total boxes: {total_detections} | "
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