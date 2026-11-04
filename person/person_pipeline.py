#!/usr/bin/env python3
"""Person detection, tracking, pose/activity analytics, and database persistence."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
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
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        f"[FATAL] Missing dependency: {exc}. "
        'Install with: pip install "ultralytics>=8.4.0" opencv-python numpy torch tqdm'
    ) from exc

from common.color import estimate_person_colors
from common.db import CCTVDatabase
from common.device import configure_torch_for_inference, resolve_device
from common.gender_age import GenderAgeEstimator, GenderAgeResult


def load_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read metadata JSON '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Metadata JSON must be an object mapping track IDs to labels.")
    return {str(key): str(value) for key, value in payload.items()}


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class FacePresence:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"Could not load face detector cascade: {cascade_path}")

    def detect(self, crop: np.ndarray) -> bool:
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        return len(faces) > 0


class TrackAnalytics:
    def __init__(self, history_size: int = 12) -> None:
        self.centers: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def update(
        self,
        track_id: int,
        box: tuple[int, int, int, int],
        keypoints: np.ndarray | None,
        face_present: bool,
        fps: float,
    ) -> tuple[str, str, str]:
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        history = self.centers[track_id]
        history.append(center)
        movement = 0.0
        if len(history) >= 2:
            movement = float(np.linalg.norm(np.subtract(history[-1], history[-2]))) * fps
        activity = "moving" if movement >= 18.0 else "stationary"

        pose = "unknown"
        orientation = "unknown"
        if keypoints is not None and len(keypoints) >= 17:
            visible = keypoints[:, 2] >= 0.35
            left_shoulder, right_shoulder = keypoints[5], keypoints[6]
            left_hip, right_hip = keypoints[11], keypoints[12]
            if visible[[5, 6, 11, 12]].all():
                shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2)
                hip_y = float((left_hip[1] + right_hip[1]) / 2)
                knee_y = float((keypoints[13, 1] + keypoints[14, 1]) / 2)
                pose = "sitting" if knee_y < hip_y + max(12.0, (hip_y - shoulder_y) * 0.55) else "standing"
            if face_present:
                orientation = "front"
            elif visible[[5, 6]].all():
                orientation = "side_or_back"
        return pose, orientation, activity


def draw_label(frame: np.ndarray, box: tuple[int, int, int, int], label: str) -> None:
    x1, y1, _, _ = box
    font = max(0.5, min(frame.shape[:2]) / 1400.0)
    thickness = max(1, int(round(font * 2)))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thickness)
    top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(frame, (x1, top), (min(frame.shape[1] - 1, x1 + tw + 10), y1), (40, 190, 80), -1)
    cv2.putText(frame, label, (x1 + 5, max(th + 2, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                font, (10, 10, 10), thickness, cv2.LINE_AA)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU person detection and non-identifying analytics.")
    parser.add_argument("--source", "-s", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model.")
    parser.add_argument("--device", default="auto", help="auto | cuda:0 | cpu")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--track", action="store_true", help="Accepted for compatibility; tracking is always enabled.")
    parser.add_argument("--crops", type=Path, default=None, help="Export one crop per track per interval.")
    parser.add_argument("--crop-every", type=int, default=15, help="Crop interval in processed frames.")
    parser.add_argument("--metadata-json", type=Path, default=None,
                        help="Consented user labels, e.g. {\"1\": \"employee-A\"}.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--no-db", action="store_true", help="Disable SQLite database persistence")
    parser.add_argument("--camera-id", type=str, default="cam_default", help="Camera identifier")
    parser.add_argument("--video-id", type=str, default=None, help="Custom video ID (default: source file stem)")
    parser.add_argument("--mivolo-checkpoint", type=Path, default=None, help="Path to MiVOLO model checkpoint")
    args = parser.parse_args(argv)
    if not args.source.is_file():
        parser.error(f"Input video not found: {args.source}")
    if args.vid_stride < 1 or args.crop_every < 1:
        parser.error("--vid-stride and --crop-every must be >= 1")
    if not 0.0 < args.conf < 1.0 or not 0.0 < args.iou < 1.0:
        parser.error("--conf and --iou must be between 0 and 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or args.source.with_name(f"{args.source.stem}_person_analytics_out.mp4")
    video_id = args.video_id or args.source.stem

    try:
        device = resolve_device(args.device)
        configure_torch_for_inference(device)
        metadata = load_metadata(args.metadata_json)
        model = YOLO(args.model)
        model.to(device)
        face_detector = FacePresence()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    gender_estimator = GenderAgeEstimator(
        checkpoint=args.mivolo_checkpoint,
        device=device,
        half=args.half,
        with_persons=True,
    )

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.source}", file=sys.stderr)
        return 1
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps if np.isfinite(fps) and fps > 0 else 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.vid_stride, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_path}")
    if args.crops:
        args.crops.mkdir(parents=True, exist_ok=True)

    db: CCTVDatabase | None = None
    if not args.no_db:
        try:
            db = CCTVDatabase(args.db)
            raw_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            duration_s = (raw_cnt / fps) if raw_cnt else None
            db.register_video(
                video_id=video_id,
                file_path=str(args.source.resolve()),
                camera_id=args.camera_id,
                duration_s=duration_s,
                fps=fps,
                width=width,
                height=height,
            )
        except Exception as exc:
            print(f"[WARN] Could not initialize database at {args.db}: {exc}. Continuing without DB.", file=sys.stderr)
            db = None

    analytics = TrackAnalytics()
    frame_index = processed = 0
    started = time.perf_counter()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    print(
        f"[INFO] Model: {args.model} | Device: {device} | Resolution: {width}x{height} @ {fps / args.vid_stride:.2f} FPS | "
        f"MiVOLO Gender: {'Enabled' if gender_estimator.is_available else 'Fallback'} | "
        f"DB: {'Enabled' if db is not None else 'Disabled'}"
    )

    try:
        with tqdm(total=total, desc="Person analytics", unit="frame", dynamic_ncols=True) as progress:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                if (frame_index - 1) % args.vid_stride:
                    continue

                t_current_s = float(frame_index - 1) / fps

                result = model.track(
                    frame, persist=True, classes=[0], conf=args.conf, iou=args.iou,
                    imgsz=args.imgsz, device=device, half=bool(args.half and device.startswith("cuda")),
                    tracker="bytetrack.yaml", verbose=False,
                )[0]
                boxes = result.boxes
                keypoints = result.keypoints
                count = 0
                if boxes is not None and len(boxes):
                    ids = boxes.id.int().cpu().numpy() if boxes.id is not None else np.arange(len(boxes)) + 1
                    for index, track_id in enumerate(ids):
                        box = clamp_box(boxes.xyxy[index].cpu().numpy(), width, height)
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box
                        face_present = face_detector.detect(frame[y1:y2, x1:x2])
                        kp = keypoints.data[index].cpu().numpy() if keypoints is not None else None
                        pose, orientation, activity = analytics.update(
                            int(track_id), box, kp, face_present, fps / args.vid_stride
                        )

                        person_crop = frame[y1:y2, x1:x2]
                        ga_res = (
                            gender_estimator.predict_crop_stabilized(person_crop, track_id=int(track_id))
                            if (person_crop.size > 0 and int(track_id) > 0)
                            else GenderAgeResult()
                        )
                        box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
                        upper_col, lower_col = estimate_person_colors(frame, box_arr)

                        if args.crops and processed % args.crop_every == 0:
                            track_dir = args.crops / f"track_{int(track_id):04d}"
                            track_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(track_dir / f"frame_{frame_index:07d}.jpg"), person_crop)

                        if db is not None:
                            db.upsert_person(
                                video_id=video_id,
                                track_id=int(track_id),
                                gender=ga_res.gender,
                                age_est=ga_res.age,
                                activity=activity,
                                pose=pose,
                                orientation=orientation,
                                upper_color=upper_col,
                                lower_color=lower_col,
                                first_seen_s=t_current_s,
                                last_seen_s=t_current_s,
                                frame_count=1,
                                peak_conf=float(boxes.conf[index]),
                                bbox_json=json.dumps([x1, y1, x2, y2]),
                                bbox_time_s=t_current_s,
                                bbox_frame=frame_index - 1,
                            )

                        user_label = f" | {metadata[str(int(track_id))]}" if str(int(track_id)) in metadata else ""
                        gender_tag = f" | {ga_res.gender}" if ga_res.gender != "unknown" else ""
                        color_tag = f" | {upper_col}/{lower_col}" if (upper_col != "unknown" or lower_col != "unknown") else ""
                        label = f"person #{int(track_id)}{gender_tag}{color_tag} | {'face' if face_present else 'no-face'} | {pose} | {activity}{user_label}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 190, 80), 2)
                        draw_label(frame, box, label)
                        count += 1

                writer.write(frame)
                processed += 1
                progress.update(args.vid_stride)
                progress.set_postfix(persons=count)

                if db is not None and processed % 100 == 0:
                    db.commit_persons()

                if args.show:
                    cv2.imshow("person analytics", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if db is not None:
            db.commit_persons()
            db.update_video_status(video_id, "completed")

    finally:
        cap.release()
        writer.release()
        if db is not None:
            db.close()
        if args.show:
            cv2.destroyAllWindows()
    elapsed = max(time.perf_counter() - started, 1e-6)
    print(f"[DONE] Device: {device} | Processed: {processed} frames | FPS: {processed / elapsed:.2f} | Saved: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
