#!/usr/bin/env python3
"""
ingest.py — Unified Single-Pass CCTV Video Ingest Orchestrator.

Optimized two-phase ingestion engine:
Phase A: Single-pass video decoding at configured stride (e.g. 5 FPS = 5x speedup)
         running Person + Vehicle + Object tracking, MiVOLO gender/age,
         LAB color estimation, and virtual gate line crossing events.
Phase B: Sequential VRAM scheduling — unloads detection models from GPU,
         loads SigLIP 2, and batch-embeds representative crops into vector database.

Usage:
    # Full ingest with 5-frame stride (5x speedup) + SigLIP 2 vector embedding:
    python ingest.py --source "data/raw/cam1.mp4" --db "outputs/cctv.db" --camera-id "cam1" --vid-stride 5 --embed

    # Fast ingest without vector embeddings:
    python ingest.py --source "data/raw/cam1.mp4" --vid-stride 5 --no-embed

    # Ingest with virtual gate tripwire (e.g. horizontal line across middle):
    python ingest.py --source "data/raw/cam1.mp4" --gate-line "main_gate:0,540,1920,540" --vid-stride 5
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# Ensure project root on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.cameras import camera_tz_name
from common.color import estimate_person_colors
from common.db import CCTVDatabase
from common.device import configure_torch_for_inference, resolve_device
from common.events import VirtualGateLine
from common.gender_age import GenderAgeEstimator
from common.video_io import create_video_writer, probe_creation_time
from vechical.detect_vehicle import DEFAULT_VEHICLE_CLASSES


def resolve_start_epoch(
    source_path: str,
    start_time_arg: str | None,
    duration_s: float | None,
    tz_name: str = "Asia/Kolkata",
) -> tuple[float | None, str]:
    """Resolve videos.start_epoch (UTC epoch s) with a strict fallback chain.

    Order: --start-time arg (interpreted in camera tz) -> container
    creation_time metadata via PyAV -> file mtime minus duration -> None.
    Returns (start_epoch, start_source) with start_source one of
    'arg' | 'metadata' | 'mtime' | 'unknown'.
    """
    if start_time_arg:
        text = start_time_arg.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                naive = datetime.strptime(text, fmt)
                return naive.replace(tzinfo=ZoneInfo(tz_name)).timestamp(), "arg"
            except ValueError:
                continue
        print(
            f"[WARN] Could not parse --start-time '{start_time_arg}' "
            "(expected 'YYYY-MM-DD HH:MM[:SS]'); falling back to file metadata.",
            file=sys.stderr,
        )
    creation = probe_creation_time(source_path)
    if creation is not None:
        return float(creation), "metadata"
    try:
        duration = float(duration_s) if duration_s is not None and duration_s > 0 else get_media_duration(source_path)
        mtime = Path(source_path).stat().st_mtime
        if duration > 0:
            return mtime - duration, "mtime"
    except (OSError, ValueError):
        pass
    return None, "unknown"


def get_media_duration(source_path: str) -> float:
    """Best-effort duration probe (OpenCV) for the mtime fallback."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def parse_gate_line(spec: str) -> Optional[VirtualGateLine]:
    """Parse 'name:x1,y1,x2,y2' string into a VirtualGateLine instance."""
    if not spec or not spec.strip():
        return None
    try:
        parts = spec.strip().split(":")
        name = parts[0] if len(parts) > 1 else "gate"
        coords_str = parts[1] if len(parts) > 1 else parts[0]
        coords = [float(v.strip()) for v in coords_str.split(",")]
        if len(coords) != 4:
            raise ValueError("Expected 4 coordinates: x1,y1,x2,y2")
        p1 = (coords[0], coords[1])
        p2 = (coords[2], coords[3])
        return VirtualGateLine(name=name, p1=p1, p2=p2, refractory_sec=2.0)
    except Exception as exc:
        print(f"[WARN] Failed to parse gate line spec '{spec}': {exc}", file=sys.stderr)
        return None


def run_phase_a_ingest(
    source_path: str,
    db: CCTVDatabase,
    video_id: str,
    camera_id: str,
    device: str,
    vid_stride: int = 5,
    conf: float = 0.35,
    iou: float = 0.50,
    mivolo_checkpoint: Optional[Path] = None,
    gate_line: Optional[VirtualGateLine] = None,
    save_crops: bool = True,
    crops_dir: Path = Path("outputs/crops"),
    skip_person: bool = False,
    skip_vehicle: bool = False,
    skip_objects: bool = False,
    output_video: Optional[Path] = None,
    start_time_arg: Optional[str] = None,
    tz_name: Optional[str] = None,
    person_model: str = "rfdetr-small",
    vehicle_model: str = "rfdetr-medium",
    object_model: str = "hybrid-small",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Phase A: Single-pass video decoding, tracking, and relational DB indexing."""
    print(f"\n{'='*70}")
    print(f"[PHASE A] Ingest & Tracking: {source_path}")
    print(f"Device: {device} | Stride: {vid_stride} (Every {vid_stride}th frame)")
    print(f"{'='*70}\n")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (total_frames / fps) if total_frames > 0 else 0.0

    # Effective FPS for trackers after striding
    effective_fps = max(1.0, fps / vid_stride)

    print(f"[VIDEO] Dimensions: {width}x{height} | Source FPS: {fps:.1f} | Effective Tracker FPS: {effective_fps:.1f}")
    print(f"[VIDEO] Total Frames: {total_frames} | Duration: {duration_s:.1f}s")

    # Resolve wall-clock start of footage (arg -> container metadata -> mtime - duration -> unknown)
    if tz_name is None:
        tz_name = camera_tz_name(camera_id)
    start_epoch, start_source = resolve_start_epoch(source_path, start_time_arg, duration_s, tz_name)
    if start_epoch is not None:
        start_local = datetime.fromtimestamp(start_epoch, ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TIME] Start epoch: {start_local} ({tz_name}) | source: {start_source}")
    else:
        print("[TIME] No reliable start time found — absolute-time search disabled for this video.")

    # Register video in DB
    db.register_video(
        video_id=video_id,
        file_path=str(Path(source_path).resolve()),
        camera_id=camera_id,
        duration_s=duration_s,
        fps=fps,
        width=width,
        height=height,
        start_epoch=start_epoch,
        start_source=start_source,
        tz=tz_name,
    )

    # ── Initialize Models ──
    person_detector = None
    gender_estimator = None
    if not skip_person:
        try:
            from person._detect_person import PersonDetector
            print("[INIT] Loading PersonDetector (RF-DETR / YOLO)...")
            person_detector = PersonDetector(
                model_name=person_model,
                device=device,
                enable_track=True,
                fps=effective_fps,
            )
            gender_estimator = GenderAgeEstimator(
                checkpoint=mivolo_checkpoint,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] Failed to load person detector: {e}", file=sys.stderr)

    vehicle_detector = None
    if not skip_vehicle:
        try:
            from vechical.detect_vehicle import VehicleDetector
            print("[INIT] Loading VehicleDetector (RF-DETR / YOLO)...")
            vehicle_detector = VehicleDetector(
                model_name=vehicle_model,
                target_classes=set(DEFAULT_VEHICLE_CLASSES),
                device=device,
                enable_track=True,
                fps=effective_fps,
            )
        except Exception as e:
            print(f"[WARN] Failed to load vehicle detector: {e}", file=sys.stderr)

    clothes_pipeline = None
    if not skip_objects:
        try:
            from object_detection.detect_clothes_objects import ObjectClothesPipeline
            print("[INIT] Loading ObjectClothesPipeline (Hybrid)...")
            clothes_pipeline = ObjectClothesPipeline(
                model_name=object_model,
                device=device,
                enable_track=True,
            )
        except Exception as e:
            print(f"[WARN] Failed to load clothes/object pipeline: {e}", file=sys.stderr)

    # Video writer for annotated preview
    writer = None
    if output_video:
        writer, out_final = create_video_writer(output_video, effective_fps, width, height)
        print(f"[PREVIEW] Writing annotated video to: {out_final}")

    # Track best crop per entity: (entity_type, track_id) -> (confidence, crop_bgr, timestamp_s, bbox)
    best_crops: dict[tuple[str, int], tuple[float, np.ndarray, float, list[float]]] = {}

    frame_idx = 0
    processed_count = 0
    t_start = time.perf_counter()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Ingesting", unit="frame")
    except ImportError:
        pbar = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            current_frame_idx = frame_idx
            frame_idx += 1
            if pbar:
                pbar.update(1)

            # Apply frame striding
            if current_frame_idx % vid_stride != 0:
                continue

            processed_count += 1
            if progress_cb is not None and (processed_count == 1 or processed_count % 10 == 0):
                progress_cb(processed_count, max(1, math.ceil(total_frames / max(1, vid_stride))))
            t_curr_s = current_frame_idx / fps

            annotated_frame = frame_bgr.copy() if writer else None

            # 1. ── Person Detection & Tracking ──
            p_boxes = np.empty((0, 4), dtype=np.float32)
            if person_detector is not None:
                try:
                    p_boxes, p_scores, p_track_ids = person_detector.predict(frame_bgr, conf=conf, iou=iou)
                    for i in range(len(p_boxes)):
                        box = p_boxes[i]
                        sc = float(p_scores[i])
                        tid = int(p_track_ids[i]) if (p_track_ids is not None and p_track_ids[i] is not None) else None
                        if tid is None or tid < 0:
                            continue

                        # Crops for gender and color
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame_bgr[y1:y2, x1:x2]
                            ga_res = gender_estimator.predict_crop_stabilized(crop, tid) if gender_estimator else None
                            upper_col, lower_col = estimate_person_colors(frame_bgr, box)

                            # Save best crop
                            crop_key = ("person", tid)
                            if crop_key not in best_crops or sc > best_crops[crop_key][0]:
                                best_crops[crop_key] = (sc, crop.copy(), t_curr_s, [float(v) for v in box])

                            db.upsert_person(
                                video_id=video_id,
                                track_id=tid,
                                gender=ga_res.gender if ga_res else "unknown",
                                age_est=ga_res.age if ga_res else None,
                                upper_color=upper_col,
                                lower_color=lower_col,
                                first_seen_s=t_curr_s,
                                last_seen_s=t_curr_s,
                                frame_count=1,
                                peak_conf=sc,
                                bbox_json=json.dumps([x1, y1, x2, y2]),
                                bbox_time_s=t_curr_s,
                                bbox_frame=current_frame_idx,
                            )

                            # Check virtual gate crossing
                            if gate_line:
                                crossing = gate_line.update(tid, "person", box, t_curr_s, sc)
                                if crossing:
                                    db.insert_event(
                                        video_id=video_id,
                                        event_type=crossing.event_type,
                                        entity_type="person",
                                        track_id=tid,
                                        line_name=gate_line.name,
                                        timestamp_s=t_curr_s,
                                        confidence=sc,
                                        bbox_json=json.dumps([x1, y1, x2, y2]),
                                    )

                            if annotated_frame is not None:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    annotated_frame,
                                    f"Person #{tid} {upper_col}",
                                    (x1, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )
                except Exception as e:
                    pass

            # 2. ── Vehicle Detection & Tracking ──
            if vehicle_detector is not None:
                try:
                    v_boxes, v_scores, v_names, v_colors, v_cconfs, v_track_ids = vehicle_detector.predict(
                        frame_bgr, conf=conf, iou=iou
                    )
                    for i in range(len(v_boxes)):
                        v_box = v_boxes[i]
                        v_sc = float(v_scores[i])
                        v_tid = int(v_track_ids[i]) if (v_track_ids is not None and v_track_ids[i] is not None) else None
                        if v_tid is None or v_tid < 0:
                            continue

                        v_type = v_names[i]
                        v_col = v_colors[i]
                        v_cconf = float(v_cconfs[i])

                        vx1, vy1, vx2, vy2 = [int(v) for v in v_box]
                        vx1, vy1 = max(0, vx1), max(0, vy1)
                        vx2, vy2 = min(width, vx2), min(height, vy2)
                        if vx2 > vx1 and vy2 > vy1:
                            v_crop = frame_bgr[vy1:vy2, vx1:vx2]
                            v_crop_key = ("vehicle", v_tid)
                            if v_crop_key not in best_crops or v_sc > best_crops[v_crop_key][0]:
                                best_crops[v_crop_key] = (v_sc, v_crop.copy(), t_curr_s, [float(v) for v in v_box])

                            db.upsert_vehicle(
                                video_id=video_id,
                                track_id=v_tid,
                                vehicle_type=v_type,
                                color=v_col,
                                color_conf=v_cconf,
                                first_seen_s=t_curr_s,
                                last_seen_s=t_curr_s,
                                frame_count=1,
                                peak_conf=v_sc,
                                bbox_json=json.dumps([vx1, vy1, vx2, vy2]),
                                bbox_time_s=t_curr_s,
                                bbox_frame=current_frame_idx,
                            )

                            if gate_line:
                                v_crossing = gate_line.update(v_tid, "vehicle", v_box, t_curr_s, v_sc)
                                if v_crossing:
                                    db.insert_event(
                                        video_id=video_id,
                                        event_type=v_crossing.event_type,
                                        entity_type="vehicle",
                                        track_id=v_tid,
                                        line_name=gate_line.name,
                                        timestamp_s=t_curr_s,
                                        confidence=v_sc,
                                        bbox_json=json.dumps([vx1, vy1, vx2, vy2]),
                                    )

                            if annotated_frame is not None:
                                cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (255, 128, 0), 2)
                                cv2.putText(
                                    annotated_frame,
                                    f"{v_col} {v_type} #{v_tid}",
                                    (vx1, max(15, vy1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 128, 0),
                                    1,
                                )
                except Exception as e:
                    pass

            # 3. ── Person-Gated Clothing & Object Detection ──
            # Only run if persons were detected in this frame (person-gated optimization)
            if clothes_pipeline is not None and len(p_boxes) > 0:
                try:
                    o_boxes, o_scores, o_labels, o_colors, o_track_ids = clothes_pipeline.predict(
                        frame_bgr, conf=conf, iou=iou
                    )
                    for i in range(len(o_boxes)):
                        obox = o_boxes[i]
                        osc = float(o_scores[i])
                        otid = int(o_track_ids[i]) if (o_track_ids is not None and o_track_ids[i] is not None) else None
                        ocol = o_colors[i]
                        olbl = o_labels[i]

                        ox1, oy1 = int(np.floor(obox[0])), int(np.floor(obox[1]))
                        ox2, oy2 = int(np.ceil(obox[2])), int(np.ceil(obox[3]))
                        ox1, oy1 = max(0, ox1), max(0, oy1)
                        ox2, oy2 = min(width, ox2), min(height, oy2)
                        if ox2 > ox1 and oy2 > oy1:
                            db.insert_object(
                                video_id=video_id,
                                track_id=otid,
                                object_type=olbl,
                                color=ocol,
                                first_seen_s=t_curr_s,
                                last_seen_s=t_curr_s,
                                frame_count=1,
                                peak_conf=osc,
                                bbox_json=json.dumps([ox1, oy1, ox2, oy2]),
                                bbox_time_s=t_curr_s,
                                bbox_frame=current_frame_idx,
                            )
                except Exception as e:
                    pass

            if writer and annotated_frame is not None:
                writer.write(annotated_frame)

            # Periodic batch commit every 50 processed frames
            if processed_count % 50 == 0:
                db.commit_persons()
                db.commit_vehicles()
                db.commit_objects()
                db.commit_events()

    finally:
        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()

    # Final DB commits
    db.commit_persons()
    db.commit_vehicles()
    db.commit_objects()
    db.commit_events()
    db.update_video_status(video_id, "indexed")

    # Save best crop images to disk & DB
    saved_crop_records = []
    if save_crops and best_crops:
        for (etype, tid), (sc, crop_img, t_s, bbox) in best_crops.items():
            subdir = crops_dir / etype / video_id
            subdir.mkdir(parents=True, exist_ok=True)
            crop_path = subdir / f"{etype}_{tid}.jpg"
            cv2.imwrite(str(crop_path), crop_img)

            db.save_crop_record(
                video_id=video_id,
                entity_type=etype,
                entity_id=tid,
                track_id=tid,
                timestamp_s=t_s,
                crop_path=str(crop_path.resolve()),
            )
            if etype in {"person", "vehicle"}:
                db.update_representative_bbox(
                    video_id=video_id,
                    entity_type=etype,
                    track_id=tid,
                    bbox_json=json.dumps([float(v) for v in bbox]),
                    timestamp_s=t_s,
                    frame_index=int(round(t_s * fps)),
                )
            saved_crop_records.append({
                "entity_type": etype,
                "track_id": tid,
                "crop_img": crop_img,
                "crop_path": str(crop_path.resolve()),
                "timestamp_s": t_s,
            })
        db.commit()

    t_elapsed = time.perf_counter() - t_start
    ingest_fps = processed_count / t_elapsed if t_elapsed > 0 else 0.0

    print(f"\n[PHASE A COMPLETE] Processed {processed_count} frames in {t_elapsed:.1f}s ({ingest_fps:.1f} FPS)")
    print(f"Saved {len(saved_crop_records)} representative crop images.")

    return {
        "video_id": video_id,
        "processed_frames": processed_count,
        "elapsed_s": t_elapsed,
        "ingest_fps": ingest_fps,
        "saved_crops": saved_crop_records,
    }


def run_phase_b_embed(
    db: CCTVDatabase,
    video_id: str,
    saved_crops: list[dict[str, Any]],
    device: str,
    batch_size: int = 16,
) -> int:
    """Phase B: Sequential VRAM scheduling — vector embed crops using SigLIP 2."""
    print(f"\n{'='*70}")
    print(f"[PHASE B] SigLIP 2 Vector Embedding")
    print(f"Crops to vectorize: {len(saved_crops)} | Device: {device}")
    print(f"{'='*70}\n")

    if not saved_crops:
        print("[PHASE B] No crops to embed.")
        return 0

    # Explicit memory cleanup before loading transformer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        from common.embed import embed_images_batch, is_available
        if not is_available():
            print("[WARN] transformers not installed. Skipping vector embeddings.", file=sys.stderr)
            return 0
    except ImportError:
        print("[WARN] common.embed module not available. Skipping.", file=sys.stderr)
        return 0

    count = 0
    for i in range(0, len(saved_crops), batch_size):
        batch = saved_crops[i : i + batch_size]
        imgs = [item["crop_img"] for item in batch]
        embeddings = embed_images_batch(imgs, device=device, batch_size=batch_size)

        for item, emb in zip(batch, embeddings):
            if emb is not None:
                db.store_embedding(
                    video_id=video_id,
                    entity_type=item["entity_type"],
                    entity_id=item["track_id"],
                    track_id=item["track_id"],
                    timestamp_s=item["timestamp_s"],
                    embedding=emb.tobytes(),
                    crop_path=item["crop_path"],
                )
                count += 1

        db.commit()
        print(f"  Embedded {min(i + batch_size, len(saved_crops))}/{len(saved_crops)} crops...")

    print(f"[PHASE B COMPLETE] Successfully created {count} SigLIP 2 vector embeddings in DB.")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Single-Pass CCTV Ingest Orchestrator (Phase A Tracking + Phase B Vectorization)."
    )
    parser.add_argument("--source", "-s", type=str, required=True, help="Video file path or stream URL")
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--camera-id", type=str, default="cam_default", help="Camera identifier")
    parser.add_argument("--tz", type=str, default=None, help="IANA camera timezone (default: config/cameras.json, else Asia/Kolkata)")
    parser.add_argument("--start-time", type=str, default=None, help="Wall-clock start of footage 'YYYY-MM-DD HH:MM[:SS]' in camera tz (highest-priority start source)")
    parser.add_argument("--video-id", type=str, default=None, help="Unique video ID (defaults to file stem)")
    parser.add_argument("--vid-stride", type=int, default=5, help="Frame stride (default: 5 = 5x faster processing)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, cpu")
    parser.add_argument("--embed", action="store_true", default=True, help="Run Phase B SigLIP 2 embedding (default: True)")
    parser.add_argument("--no-embed", dest="embed", action="store_false", help="Skip Phase B vector embedding")
    parser.add_argument("--batch-size", type=int, default=16, help="Phase B embedding batch size")
    parser.add_argument("--mivolo-checkpoint", type=Path, default=None, help="Path to MiVOLO v2 checkpoint")
    parser.add_argument("--gate-line", type=str, default=None, help="Virtual gate tripwire spec: 'name:x1,y1,x2,y2'")
    parser.add_argument("--output-video", "-o", type=Path, default=None, help="Optional output annotated video")
    parser.add_argument("--skip-person", action="store_true", help="Skip person detection")
    parser.add_argument("--skip-vehicle", action="store_true", help="Skip vehicle detection")
    parser.add_argument("--skip-objects", action="store_true", help="Skip clothes/object detection")
    parser.add_argument("--person-model", default="rfdetr-small", help="Person model (rfdetr-* or yolo*.pt)")
    parser.add_argument("--vehicle-model", default="rfdetr-medium", help="Vehicle model (rfdetr-* or yolo*.pt)")
    parser.add_argument("--object-model", default="hybrid-small", help="Object/clothing model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_p = Path(args.source)
    video_id = args.video_id or (source_p.stem if source_p.is_file() else "video_stream")

    device = resolve_device(args.device)
    configure_torch_for_inference(device)

    db = CCTVDatabase(args.db)
    gate_line = parse_gate_line(args.gate_line) if args.gate_line else None

    try:
        # Phase A: Single-pass Ingest & Tracking
        res_a = run_phase_a_ingest(
            source_path=args.source,
            db=db,
            video_id=video_id,
            camera_id=args.camera_id,
            device=device,
            vid_stride=args.vid_stride,
            conf=args.conf,
            iou=args.iou,
            mivolo_checkpoint=args.mivolo_checkpoint,
            gate_line=gate_line,
            skip_person=args.skip_person,
            skip_vehicle=args.skip_vehicle,
            skip_objects=args.skip_objects,
            output_video=args.output_video,
            start_time_arg=args.start_time,
            tz_name=args.tz,
            person_model=args.person_model,
            vehicle_model=args.vehicle_model,
            object_model=args.object_model,
        )

        # Phase B: SigLIP 2 Vector Embedding
        if args.embed and res_a["saved_crops"]:
            run_phase_b_embed(
                db=db,
                video_id=video_id,
                saved_crops=res_a["saved_crops"],
                device=device,
                batch_size=args.batch_size,
            )

        # Final Summary
        summary = db.get_video_summary(video_id)
        print(f"\n{'='*70}")
        print(f"INGESTION COMPLETE: {video_id}")
        print(f"Persons: {summary['persons']} | Vehicles: {summary['vehicles']} | Objects: {summary['objects']}")
        print(f"Total Embeddings in DB: {db.count_embeddings()}")
        print(f"{'='*70}\n")
        return 0

    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
