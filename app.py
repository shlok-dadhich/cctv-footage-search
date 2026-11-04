#!/usr/bin/env python3
"""
app.py — CCTV Intelligence System — Streamlit Frontend

A unified web UI for searching tracked persons, vehicles, and objects.
Natural language queries are contextually vectorized via SigLIP 2 and matched
against the database using a hybrid approach (SQL filtering + vector similarity).
Provides side-by-side view of query event snapshot image and ±30s video footage clip.

Run:
    streamlit run app.py
    streamlit run app.py -- --db outputs/cctv.db
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

# ── Ensure project root on sys.path ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from common.db import CCTVDatabase
from common.time_query import DEFAULT_TZ, load_time_config
from common.video_io import cut_clip, ensure_browser_playable_video, extract_frame
from query_norm import detect_language, normalize_query
from search import (
    ParsedQuery,
    _clean_object_type,
    _deduplicate_track_results,
    execute_search,
    format_time_s,
    parse_natural_language_query,
)

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="CCTV footage search",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────
DEFAULT_DB = ROOT_DIR / "outputs" / "cctv.db"
CLIP_DIR = ROOT_DIR / "outputs" / "clips"
CROPS_DIR = ROOT_DIR / "outputs" / "crops"

DEFAULT_VISIBLE_RESULTS = 7
VISIBLE_RESULTS_STEP = 10

UPLOAD_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
MAX_UPLOAD_MB = 500
MAX_UPLOAD_SECONDS_CPU = 300
MAX_UPLOAD_SECONDS_GPU = 1800
_INGEST_LOCK = threading.Lock()


# ── Session Upload Helpers ───────────────────────────────────────────

def _upload_state() -> dict[str, Any]:
    return dict(st.session_state.get("upload_state") or {})


def _active_upload_state() -> dict[str, Any]:
    if st.session_state.get("source_mode") == "upload":
        return _upload_state()
    return {}


def _writable_session_root() -> Path:
    """Find a writable directory for per-browser-session uploads."""
    cached = st.session_state.get("session_root")
    if cached and Path(cached).is_dir():
        return Path(cached)

    candidates: list[Path] = []
    try:
        candidates.append(Path(tempfile.gettempdir()) / "cctv_web_sessions")
    except Exception:
        pass
    candidates.append(ROOT_DIR / "outputs" / "sessions")

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / f".write_{uuid.uuid4().hex}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            st.session_state["session_root"] = str(base)
            return base
        except Exception:
            continue
    raise RuntimeError("No writable directory is available for uploaded videos.")


def _session_upload_root() -> Path:
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
        st.session_state["session_id"] = session_id
    return _writable_session_root() / session_id


def _sanitize_name(value: str, max_len: int = 40) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in value).strip("_")
    return (clean or "video")[:max_len]


def _crop_search_dirs() -> list[Path]:
    state = _active_upload_state()
    dirs: list[Path] = []
    if state.get("crops_dir"):
        dirs.append(Path(state["crops_dir"]))
    dirs.append(CROPS_DIR)
    return dirs


def _clip_search_dirs() -> list[Path]:
    state = _active_upload_state()
    dirs: list[Path] = []
    if state.get("clips_dir"):
        dirs.append(Path(state["clips_dir"]))
    dirs.append(CLIP_DIR)
    return dirs


def _active_clips_dir() -> Path:
    state = _active_upload_state()
    if state.get("clips_dir"):
        return Path(state["clips_dir"])
    return CLIP_DIR


def _index_uploaded_video(
    uploaded: Any,
    camera_id: str,
    vid_stride: int,
    detect_objects: bool,
    start_time: str,
) -> None:
    """Save and index one uploaded video into this browser session only."""
    if not _INGEST_LOCK.acquire(blocking=False):
        st.warning("Another video is being indexed right now. Please wait and try again.")
        return

    upload_root = _session_upload_root() / uuid.uuid4().hex[:8]
    uploads_dir = upload_root / "uploads"
    crops_dir = upload_root / "crops"
    clips_dir = upload_root / "clips"
    db_path = upload_root / "cctv.db"
    db = None
    failed = False
    try:
        for directory in (uploads_dir, crops_dir, clips_dir):
            directory.mkdir(parents=True, exist_ok=True)

        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {f".{ext}" for ext in UPLOAD_VIDEO_TYPES}:
            raise ValueError("Unsupported video format. Use MP4, MOV, AVI, MKV, WEBM, or M4V.")
        if getattr(uploaded, "size", 0) and uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"Video is larger than {MAX_UPLOAD_MB} MB. Please upload a shorter clip.")

        video_id = f"upload_{_sanitize_name(Path(uploaded.name).stem, 32)}_{uuid.uuid4().hex[:6]}"
        source_path = uploads_dir / f"{video_id}{suffix}"

        with st.status("Indexing uploaded video", expanded=True) as status:
            status.write("Saving video...")
            uploaded.seek(0)
            with open(source_path, "wb") as handle:
                while True:
                    chunk = uploaded.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            duration = get_video_duration(str(source_path))
            from common.device import resolve_device

            device = resolve_device("auto")
            max_seconds = MAX_UPLOAD_SECONDS_GPU if device.startswith("cuda") else MAX_UPLOAD_SECONDS_CPU
            if duration > max_seconds:
                raise ValueError(
                    f"Video is {duration:.0f}s long. The hosted app supports up to "
                    f"{max_seconds}s per clip; trim it or run the local GPU app."
                )

            fast_cpu = not device.startswith("cuda")
            if fast_cpu and detect_objects:
                status.write("CPU mode: clothing/object detection is skipped for speed.")
                detect_objects = False

            status.write(f"Video: {duration:.0f}s | device: {device.upper()}")

            yolo_root = _writable_session_root() / ".ultralytics"
            yolo_root.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(yolo_root)
            yolo_weight = yolo_root / "yolo26n.pt"

            db = CCTVDatabase(db_path)

            progress = st.progress(0.0)

            def on_progress(done: int, total: int) -> None:
                progress.progress(min(1.0, max(0.0, done / max(1, total))))

            from ingest import run_phase_a_ingest, run_phase_b_embed

            status.write("Detecting, tracking, and saving snapshots...")
            result = run_phase_a_ingest(
                source_path=str(source_path),
                db=db,
                video_id=video_id,
                camera_id=_sanitize_name(camera_id or "upload", 40),
                device=device,
                vid_stride=max(1, int(vid_stride)),
                person_model=str(yolo_weight) if fast_cpu else "rfdetr-small",
                vehicle_model=str(yolo_weight) if fast_cpu else "rfdetr-medium",
                skip_objects=not detect_objects,
                save_crops=True,
                crops_dir=crops_dir,
                start_time_arg=start_time.strip() or None,
                progress_cb=on_progress,
            )

            status.write("Building SigLIP embeddings...")
            embedded = 0
            if result["saved_crops"]:
                embedded = run_phase_b_embed(
                    db=db,
                    video_id=video_id,
                    saved_crops=result["saved_crops"],
                    device=device,
                    batch_size=8 if fast_cpu else 16,
                )

            summary = db.get_video_summary(video_id)
            try:
                db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                pass
            progress.empty()
            status.update(label="Video indexed", state="complete", expanded=False)

        st.session_state["upload_state"] = {
            "video_id": video_id,
            "name": uploaded.name,
            "db_path": str(db_path),
            "source_path": str(source_path),
            "crops_dir": str(crops_dir),
            "clips_dir": str(clips_dir),
            "camera_id": camera_id or "upload",
            "duration_s": duration,
            "persons": summary["persons"],
            "vehicles": summary["vehicles"],
            "objects": summary["objects"],
            "embeddings": embedded,
        }
        st.session_state["source_mode"] = "upload"
    except Exception as exc:  # noqa: BLE001
        st.error(f"Indexing failed: {exc}")
        failed = True
    finally:
        if db is not None:
            db.close()
        if failed:
            shutil.rmtree(upload_root, ignore_errors=True)
        _INGEST_LOCK.release()


# ── Database & Helpers ───────────────────────────────────────────────

def get_db_path() -> Path:
    """Get the database path from the active source or CLI args/default."""
    state = _active_upload_state()
    if state.get("db_path"):
        return Path(state["db_path"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
        return args.db if args.db.is_absolute() else ROOT_DIR / args.db
    except SystemExit:
        return DEFAULT_DB


@st.cache_resource
def get_database(db_path: str, read_only: bool = False) -> CCTVDatabase:
    """Open database connection (cached for session)."""
    return CCTVDatabase(db_path, read_only=read_only)


def numpy_to_pil(bgr_img: np.ndarray):
    """Convert BGR numpy array to RGB for Streamlit display."""
    from PIL import Image
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def improve_snapshot_detail(bgr_img: np.ndarray) -> np.ndarray:
    """Apply a restrained unsharp mask without changing detection geometry."""
    blurred = cv2.GaussianBlur(bgr_img, (0, 0), 1.1)
    return cv2.addWeighted(bgr_img, 1.35, blurred, -0.35, 0)


def upscale_crop_for_display(crop: np.ndarray, min_width: int = 360) -> np.ndarray:
    """Enlarge small detections for inspection while retaining original pixels."""
    if crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    if width >= min_width:
        return improve_snapshot_detail(crop)
    scale = min_width / max(1, width)
    resized = cv2.resize(
        crop,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return improve_snapshot_detail(resized)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    if not video_path or not Path(video_path).is_file():
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frames / fps
    return 0.0


def extract_crop_from_video(
    video_path: str,
    bbox_json: str | None,
    timestamp_s: float | None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Extract a cropped region from video at timestamp using bbox."""
    if not video_path or not Path(video_path).is_file() or not bbox_json:
        return None
    try:
        bbox = json.loads(bbox_json)
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        else:
            return None
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        if frame_index is not None and frame_index >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        elif timestamp_s is not None and timestamp_s > 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(timestamp_s * fps)))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
    finally:
        cap.release()


def extract_annotated_frame(
    video_path: str,
    bbox_json: str | None,
    timestamp_s: float | None,
    label: str = "",
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Extract full frame at timestamp with bounding box drawn."""
    if not video_path or not Path(video_path).is_file():
        return None

    bbox_list = None
    if bbox_json:
        try:
            parsed = json.loads(bbox_json)
            if isinstance(parsed, dict):
                bbox_list = [parsed.get("x1", 0), parsed.get("y1", 0), parsed.get("x2", 0), parsed.get("y2", 0)]
            elif isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
                bbox_list = [float(v) for v in parsed[:4]]
        except Exception:
            bbox_list = None

    ts = timestamp_s if (timestamp_s is not None and timestamp_s >= 0) else 0.0
    return extract_frame(video_path, ts, bbox=bbox_list, label=label, frame_index=frame_index)


def _load_saved_crop(entity: str, video_id: str, track_id: int | str) -> np.ndarray | None:
    """Load the representative crop saved during ingest (matches the stored bbox)."""
    if track_id in (None, "—", ""):
        return None
    for crops_dir in _crop_search_dirs():
        for ext in ("jpg", "png", "jpeg"):
            path = crops_dir / entity / video_id / f"{entity}_{track_id}.{ext}"
            if path.is_file():
                return cv2.imread(str(path))
    return None


def _find_saved_clip(entity: str, video_id: str, track_id: int | str) -> Path | None:
    """Return an already-cut clip for a track, if one was generated during ingest."""
    if track_id in (None, "—", ""):
        return None
    safe_video_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_id)
    prefix = f"{entity}_{safe_video_id}_trk{track_id}_"
    candidates: list[Path] = []
    for clips_dir in _clip_search_dirs():
        candidates.extend(clips_dir.glob(f"{prefix}*.mp4"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1024:
            return path
    return None


def extract_event_clip(
    video_path: str,
    event_time_s: float,
    pad_s: float = 30.0,
    clip_id: str = "clip",
) -> Path | None:
    """Extract ±pad_s seconds around an event from the source video."""
    if not video_path or not Path(video_path).is_file():
        return None

    duration = get_video_duration(video_path)
    start_s = max(0.0, event_time_s - pad_s)
    end_s = min(duration, event_time_s + pad_s) if duration > 0 else event_time_s + pad_s

    clips_dir = _active_clips_dir()
    clips_dir.mkdir(parents=True, exist_ok=True)
    clean_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
    output_path = clips_dir / f"{clean_id}_{int(start_s)}s_to_{int(end_s)}s.mp4"

    # Return cached if already extracted and non-empty (and make old clips playable).
    if output_path.is_file():
        if output_path.stat().st_size <= 1024:
            output_path.unlink(missing_ok=True)
        else:
            return ensure_browser_playable_video(output_path, max_height=480)

    try:
        final_path = cut_clip(video_path, start_s, end_s, output_path, max_height=480)
        return ensure_browser_playable_video(final_path, max_height=480)
    except Exception as e:
        st.error(f"Clip extraction failed: {e}")
        return None


# ── Vector & Hybrid Search ───────────────────────────────────────────

def vector_search(
    db: CCTVDatabase, query_text: str, video_id: str | None = None, top_k: int = 20
) -> list[dict[str, Any]]:
    """Perform semantic vector search using SigLIP 2 embeddings."""
    try:
        from common.embed import batch_cosine_similarity, embed_text
    except ImportError:
        return []

    emb_count = db.count_embeddings()
    if emb_count == 0:
        return []

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    query_vec = embed_text(query_text, device=device)
    if query_vec is None:
        return []

    emb_rows = db.get_all_embeddings(video_id=video_id)
    if not emb_rows:
        return []

    valid_rows = []
    vectors = []
    query_dim = query_vec.reshape(-1).shape[0]
    for row in emb_rows:
        try:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.shape[0] == query_dim:
                vectors.append(vec)
                valid_rows.append(row)
        except Exception:
            continue

    if not vectors:
        return []

    matrix = np.stack(vectors)
    scores = batch_cosine_similarity(query_vec, matrix)

    results = []
    for i, row in enumerate(valid_rows):
        results.append({
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "track_id": row["track_id"],
            "timestamp_s": row["timestamp_s"],
            "video_id": row["video_id"],
            "crop_path": row.get("crop_path"),
            "similarity": float(scores[i]),
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def _result_relevance_score(result: dict[str, Any], parsed: ParsedQuery, similarity: float) -> float:
    """Score how well a result matches the parsed query.

    The target entity type dominates so a query like "black car" always puts
    vehicles first. Structured attribute matches come next, and semantic
    similarity is only a tie-breaker/tie-adder. Video identity never affects rank.
    """
    score = 0.0
    entity = result.get("entity")

    if parsed.target_entity:
        if entity == parsed.target_entity:
            score += 100.0
    else:
        score += 10.0

    result_color = (result.get("color") or "").lower()
    if parsed.color:
        if result_color == parsed.color:
            score += 35.0
        elif result_color and parsed.color in result_color:
            score += 25.0
        elif result_color:
            score -= 8.0
        else:
            score -= 2.0

    if parsed.vehicle_type:
        value = (result.get("type") or "").lower()
        if value == parsed.vehicle_type:
            score += 30.0
        elif value:
            score -= 4.0

    if parsed.gender:
        value = (result.get("gender") or "").lower()
        if value == parsed.gender:
            score += 30.0
        elif value and value != "unknown":
            score -= 4.0

    if parsed.object_type:
        value = (result.get("type") or "").lower()
        if parsed.object_type in value or value in parsed.object_type:
            score += 30.0
        elif value:
            score -= 4.0

    score += max(0.0, float(similarity)) * 60.0
    score += float(result.get("peak_conf") or 0.0) * 3.0
    frames = result.get("frame_count") or 0
    if frames:
        score += min(3.0, math.log1p(float(frames)) / 2.0)
    return score


def _matches_query_constraints(result: dict[str, Any], parsed: ParsedQuery) -> bool:
    """Return True only when a result satisfies every structured query attribute.

    Used to keep semantic-only candidates from leaking in results that look
    relevant to the vector model but contradict the typed query (wrong entity
    class, wrong colour, wrong vehicle type, etc.).
    """
    if parsed.target_entity and result.get("entity") != parsed.target_entity:
        return False

    if parsed.color:
        color = (result.get("color") or "").lower()
        if not color:
            color = (result.get("upper_color") or "").lower()
        if parsed.color not in color:
            return False

    if parsed.vehicle_type:
        if (result.get("type") or "").lower() != parsed.vehicle_type:
            return False

    if parsed.gender:
        if (result.get("gender") or "").lower() != parsed.gender:
            return False

    if parsed.object_type:
        value = (result.get("type") or "").lower()
        if parsed.object_type not in value and value not in parsed.object_type:
            return False

    return True


def hybrid_search(
    db: CCTVDatabase,
    query_text: str,
    video_id: str | None = None,
    top_k: int = 50,
    anchor_ts: float | None = None,
) -> list[dict[str, Any]]:
    """Combined SQL structured filter + SigLIP 2 contextual vector similarity.

    Results are ranked by query relevance: the requested entity type first,
    then attribute matches, then semantic similarity. When the query names an
    entity (e.g. "blue car" → vehicle) results from other classes are excluded
    entirely. Source video never influences rank ordering. Natural-language time
    phrases are resolved against `anchor_ts` (end of indexed footage when available).
    """
    parsed = parse_natural_language_query(query_text, anchor_ts=anchor_ts)
    sql_results = execute_search(
        db=db,
        query=parsed,
        video_id=video_id,
    )

    for r in sql_results:
        r["search_method"] = "Structured Match"
        r["similarity"] = 0.0

    # Vector search over all indexed embeddings.
    vec_results = vector_search(db, query_text, video_id=video_id, top_k=max(top_k * 2, 60))

    # Keep the best embedding per entity, dropping duplicate vectors.
    vec_best: dict[tuple, dict[str, Any]] = {}
    for vr in vec_results:
        key = (vr["entity_type"], vr["video_id"], vr["track_id"], vr["entity_id"])
        if key not in vec_best or vr["similarity"] > vec_best[key]["similarity"]:
            vec_best[key] = vr

    # Annotate SQL matches that also matched vector similarity.
    vec_by_track = {
        key[:3]: value
        for key, value in vec_best.items()
    }
    for r in sql_results:
        key = (r.get("entity"), r.get("video_id"), r.get("track_id"))
        best = vec_by_track.get(key)
        if best is not None:
            r["similarity"] = best["similarity"]
            r["search_method"] = "Hybrid Match"

    # Add semantic-only matches only when they are consistent with the typed
    # query. When the query names an entity ("blue car"), results from other
    # classes are never shown; strictly matching attributes are also required.
    existing_keys = {(r.get("entity"), r.get("video_id"), r.get("track_id")) for r in sql_results}
    target = parsed.target_entity
    vector_only: list[tuple[dict[str, Any], float]] = []
    for vr in sorted(vec_best.values(), key=lambda item: item["similarity"], reverse=True):
        if target is not None and vr["entity_type"] != target:
            continue
        identity = (vr["entity_type"], vr["video_id"], vr["track_id"])
        if identity in existing_keys:
            continue
        detail = _fetch_entity_detail(
            db,
            vr["entity_type"],
            vr["entity_id"],
            video_id=vr["video_id"],
            track_id=vr["track_id"],
        )
        if not detail or not _matches_query_constraints(detail, parsed):
            continue
        detail["similarity"] = vr["similarity"]
        detail["search_method"] = "Semantic Match"
        vector_only.append((detail, vr["similarity"]))
        existing_keys.add(identity)

    results = _deduplicate_track_results(sql_results + [detail for detail, _ in vector_only])
    for r in results:
        r["match_score"] = _result_relevance_score(r, parsed, r.get("similarity", 0.0))

    # Pure relevance ordering — no video priority, no match-method priority.
    results.sort(
        key=lambda x: (
            -float(x.get("match_score", 0.0)),
            -float(x.get("similarity", 0.0)),
            -float(x.get("peak_conf", 0.0) or 0.0),
            float(x.get("first_seen_s") or 0.0),
        )
    )
    if video_id is not None:
        results = [result for result in results if result.get("video_id") == video_id]
    return results[:top_k]


def _attribute_match_bonus(result: dict[str, Any], parsed: ParsedQuery | None) -> float:
    """Small text-attribute nudge for image+text results.

    Appearance similarity stays dominant; structured attributes parsed from the
    typed text (colour/type/gender) only break close calls and demote obvious
    mismatches. Returns a value in roughly [-0.03, +0.05].
    """
    if parsed is None:
        return 0.0

    bonus = 0.0
    color = (result.get("color") or "").lower()
    if parsed.color:
        if color == parsed.color:
            bonus += 0.03
        elif color and parsed.color in color:
            bonus += 0.02
        elif color:
            bonus -= 0.01

    if parsed.vehicle_type:
        value = (result.get("type") or result.get("vehicle_type") or "").lower()
        if value == parsed.vehicle_type:
            bonus += 0.02
        elif value:
            bonus -= 0.005

    if parsed.gender:
        value = (result.get("gender") or "").lower()
        if value == parsed.gender:
            bonus += 0.02
        elif value and value != "unknown":
            bonus -= 0.005

    if parsed.object_type:
        value = (result.get("type") or "").lower()
        if parsed.object_type in value or value in parsed.object_type:
            bonus += 0.02
    return bonus


def image_query_search(
    db: CCTVDatabase,
    text: str,
    image_bytes: list[bytes],
    video_id: str | None = None,
    top_k: int = 30,
    anchor_ts: float | None = None,
) -> dict[str, Any]:
    """Combined image + text retrieval → embedding → DB similarity search.

    The uploaded image resolves the entity and provides appearance evidence,
    while the typed text is normalized/embedded and blended into the same query
    vector (0.7 image + 0.3 text). Explicit text attributes then filter out
    contradictory image hits, so both inputs jointly determine the results.
    The sidebar video filter is applied inside the embedding index and as a guard.
    """
    from image_query import _attach_track_info
    from search_core import SearchRequest, find_similar_image_and_text, image_search

    parsed = parse_natural_language_query(text, anchor_ts=anchor_ts) if text else None
    entity_hint = "auto"
    if parsed and parsed.target_entity in ("person", "vehicle"):
        entity_hint = parsed.target_entity

    # A pure time phrase ("pichle 2 ghante") should constrain the search window,
    # not be blended into the image embedding and dilute appearance similarity.
    semantic_text = text
    if semantic_text and detect_language(semantic_text) == "hinglish":
        # Normalize Hinglish to English before embedding so the text half of the
        # combined query is understood by the SigLIP text encoder.
        semantic_text = normalize_query(semantic_text)
    if parsed and parsed.time_range and not any(
        (
            parsed.target_entity,
            parsed.color,
            parsed.gender,
            parsed.vehicle_type,
            parsed.object_type,
        )
    ):
        semantic_text = ""

    request = SearchRequest(
        query_text=semantic_text or None,
        query_images=image_bytes,
        entity=entity_hint,
        top_k=top_k,
        video_id=video_id,
        ts_from=parsed.time_range.start_ts if parsed and parsed.time_range else None,
        ts_to=parsed.time_range.end_ts if parsed and parsed.time_range else None,
        per_track=True,
    )
    response = (
        find_similar_image_and_text(db, request)
        if semantic_text
        else image_search(db, request)
    )
    if response.error:
        return {
            "error": response.error,
            "results": [],
            "method": response.method,
            "entity": response.entity,
            "parsed": parsed,
        }

    hits = response.hits
    if response.entity:
        _attach_track_info(db, resolved_entity=response.entity, hits=hits)

    for h in hits:
        if not h.get("entity"):
            h["entity"] = response.entity or h.get("entity_type")
        h["search_method"] = (
            "Image + Text Match" if semantic_text and image_bytes else "Image Match"
        )
        h["similarity"] = float(h.get("similarity") or 0.0)
        h["peak_conf"] = float(h.get("peak_conf") or 0.0)
        h["match_score"] = h["similarity"] + _attribute_match_bonus(h, parsed)

    # Genuinely combine the two inputs: when the typed text adds explicit
    # attributes (colour/type/gender/object), discard image hits that contradict
    # them. Fall back to the unfiltered image list if nothing satisfies the text.
    if parsed and any(
        (parsed.color, parsed.vehicle_type, parsed.gender, parsed.object_type)
    ):
        constrained = [h for h in hits if _matches_query_constraints(h, parsed)]
        if constrained:
            hits = constrained

    hits.sort(
        key=lambda item: (
            -float(item.get("match_score") or item.get("similarity") or 0.0),
            -float(item.get("similarity") or 0.0),
        )
    )
    if video_id:
        hits = [h for h in hits if h.get("video_id") == video_id]

    return {
        "error": None,
        "results": hits,
        "method": response.method,
        "entity": response.entity,
        "parsed": parsed,
    }


def run_unified_search(
    db: CCTVDatabase,
    text: str,
    image_bytes: list[bytes],
    video_id: str | None = None,
    anchor_ts: float | None = None,
    top_k: int = 30,
) -> dict[str, Any]:
    """Route the unified search box to image, image+text, or text search."""
    if image_bytes:
        return image_query_search(
            db, text, image_bytes, video_id=video_id, top_k=top_k, anchor_ts=anchor_ts
        )

    results = hybrid_search(
        db, text, video_id=video_id, top_k=top_k, anchor_ts=anchor_ts
    )
    return {
        "error": None,
        "results": results,
        "method": "text + structured + vector",
        "entity": None,
        "parsed": parse_natural_language_query(text, anchor_ts=anchor_ts),
    }


def _fetch_entity_detail(
    db: CCTVDatabase,
    entity_type: str,
    entity_id: int,
    video_id: str | None = None,
    track_id: int | None = None,
) -> dict[str, Any] | None:
    """Fetch entity row from DB."""
    try:
        identity_column = "video_id = ? AND track_id = ?" if video_id is not None and track_id is not None else "id = ?"
        identity_params = (video_id, track_id) if video_id is not None and track_id is not None else (entity_id,)
        if entity_type == "person":
            rows = db.conn.execute(
                f"SELECT p.*, v.file_path, v.camera_id FROM persons p JOIN videos v ON p.video_id = v.video_id WHERE p.{identity_column}",
                identity_params,
            ).fetchall()
            if rows:
                r = dict(rows[0])
                gender_str = r.get("gender") or "unknown"
                age_str = f" (~{int(r['age_est'])}y)" if r.get("age_est") else ""
                return {
                    "entity": "person",
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "file_path": r.get("file_path", ""),
                    "track_id": r["track_id"],
                    "label": f"person #{r['track_id']} ({gender_str}{age_str})",
                    "gender": r.get("gender"),
                    "age_est": r.get("age_est"),
                    "upper_color": r.get("upper_color"),
                    "lower_color": r.get("lower_color"),
                    "first_seen_s": r.get("first_seen_s"),
                    "last_seen_s": r.get("last_seen_s"),
                    "frame_count": r.get("frame_count"),
                    "peak_conf": r.get("peak_conf"),
                    "bbox_json": r.get("bbox_json"),
                    "bbox_time_s": r.get("bbox_time_s"),
                    "bbox_frame": r.get("bbox_frame"),
                }
        elif entity_type == "vehicle":
            rows = db.conn.execute(
                f"SELECT v2.*, v.file_path, v.camera_id FROM vehicles v2 JOIN videos v ON v2.video_id = v.video_id WHERE v2.{identity_column}",
                identity_params,
            ).fetchall()
            if rows:
                r = dict(rows[0])
                return {
                    "entity": "vehicle",
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "file_path": r.get("file_path", ""),
                    "track_id": r["track_id"],
                    "label": f"{r.get('color', '?')} {r.get('vehicle_type', '?')}",
                    "type": r.get("vehicle_type"),
                    "color": r.get("color"),
                    "first_seen_s": r.get("first_seen_s"),
                    "last_seen_s": r.get("last_seen_s"),
                    "frame_count": r.get("frame_count"),
                    "peak_conf": r.get("peak_conf"),
                    "bbox_json": r.get("bbox_json"),
                    "bbox_time_s": r.get("bbox_time_s"),
                    "bbox_frame": r.get("bbox_frame"),
                }
        elif entity_type == "object":
            rows = db.conn.execute(
                f"SELECT o.*, v.file_path, v.camera_id FROM objects o JOIN videos v ON o.video_id = v.video_id WHERE o.{identity_column}",
                identity_params,
            ).fetchall()
            if rows:
                r = dict(rows[0])
                assoc = f" [Person #{r['associated_person_track']}]" if r.get("associated_person_track") else ""
                obj_type = _clean_object_type(r.get("object_type"), r.get("color"))
                return {
                    "entity": "object",
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "file_path": r.get("file_path", ""),
                    "track_id": r.get("track_id"),
                    "label": f"{r.get('color', '?')} {obj_type}{assoc}",
                    "type": obj_type,
                    "color": r.get("color"),
                    "associated_person_track": r.get("associated_person_track"),
                    "first_seen_s": r.get("first_seen_s"),
                    "last_seen_s": r.get("last_seen_s"),
                    "frame_count": r.get("frame_count"),
                    "peak_conf": r.get("peak_conf"),
                    "bbox_json": r.get("bbox_json"),
                    "bbox_time_s": r.get("bbox_time_s"),
                    "bbox_frame": r.get("bbox_frame"),
                }
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# ██  MAIN STREAMLIT APPLICATION  ██
# ═══════════════════════════════════════════════════════════════════════

def main():
    initial_upload = _upload_state()
    if "source_mode" not in st.session_state:
        st.session_state["source_mode"] = "upload" if initial_upload.get("db_path") else "demo"
    if st.session_state.get("source_mode") == "upload" and not initial_upload.get("db_path"):
        st.session_state["source_mode"] = "demo"

    db_path = get_db_path().resolve()
    database_exists = db_path.is_file()
    # The deployment bundle is already migrated. Open read-only so the app never
    # mutates the demo index and works without SQLite sidecar files.
    db = get_database(str(db_path), read_only=database_exists)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### CCTV search")
        st.caption("Search indexed person, vehicle, and object tracks.")

        upload_state = _upload_state()
        has_upload = bool(upload_state.get("db_path"))
        source_options = ["Demo footage"]
        if has_upload:
            source_options.append("Uploaded video")
        current_source = (
            "Uploaded video"
            if st.session_state.get("source_mode") == "upload" and has_upload
            else "Demo footage"
        )
        choice = st.radio(
            "Search source",
            source_options,
            index=source_options.index(current_source),
            horizontal=True,
        )
        next_mode = "upload" if choice == "Uploaded video" else "demo"
        if next_mode != st.session_state.get("source_mode"):
            st.session_state["source_mode"] = next_mode
            st.rerun()

        if has_upload:
            st.caption(
                f"Active upload: `{upload_state.get('name', 'video')}` · "
                f"{upload_state.get('persons', 0)} persons · "
                f"{upload_state.get('vehicles', 0)} vehicles"
            )

        with st.expander("Index your own video", expanded=not has_upload):
            uploaded_video = st.file_uploader(
                "Video file",
                type=UPLOAD_VIDEO_TYPES,
                accept_multiple_files=False,
                help=(
                    f"Up to {MAX_UPLOAD_MB} MB. Hosted CPU indexing works best with "
                    "short clips (under 2-3 minutes)."
                ),
            )
            upload_camera = st.text_input("Camera ID", value="upload", key="upload_camera")
            upload_start = st.text_input(
                "Footage start time (optional)",
                placeholder="YYYY-MM-DD HH:MM",
                key="upload_start",
                help="Used for time-based queries such as 'last night' or '6 pm'.",
            )
            upload_stride = st.select_slider(
                "Frame stride",
                options=[3, 5, 10, 15, 20],
                value=10,
                key="upload_stride",
                help="Higher values are faster but sample fewer frames.",
            )
            upload_objects = st.checkbox(
                "Detect clothing and objects (slower)",
                value=False,
                key="upload_objects",
                help="Skipped automatically on CPU to keep indexing responsive.",
            )
            if st.button(
                "Index video",
                type="primary",
                width="stretch",
                disabled=uploaded_video is None,
            ):
                _index_uploaded_video(
                    uploaded_video,
                    upload_camera,
                    upload_stride,
                    upload_objects,
                    upload_start,
                )
                if st.session_state.get("upload_state") and st.session_state.get("source_mode") == "upload":
                    st.rerun()

            if has_upload:
                st.caption("Uploads are kept for this browser session and cleared on app restart.")
                if st.button("Remove uploaded video", key="remove_upload"):
                    try:
                        get_database(str(Path(upload_state["db_path"]))).close()
                    except Exception:
                        pass
                    get_database.clear()
                    shutil.rmtree(Path(upload_state["db_path"]).parent, ignore_errors=True)
                    st.session_state.pop("upload_state", None)
                    st.session_state["source_mode"] = "demo"
                    st.rerun()

        if not database_exists:
            st.warning(f"No footage has been indexed yet. Database: `{db_path}`")
            st.caption("Upload a video above or run the detection pipelines locally:")
            st.code(
                "python ingest.py --source data/raw/cam1.mp4 --db outputs/cctv.db --embed",
                language="bash",
            )

        counts = db.get_total_counts()
        videos = db.get_all_videos()
        emb_count = db.count_embeddings()

        st.markdown("### Index statistics")
        c1, c2 = st.columns(2)
        c1.metric("Videos", len(videos))
        c2.metric("Persons", counts["persons"])
        c3, c4 = st.columns(2)
        c3.metric("Vehicles", counts["vehicles"])
        c4.metric("Objects", counts["objects"])

        if emb_count > 0:
            st.caption(f"SigLIP 2 embeddings: {emb_count:,} vectors")
        else:
            st.warning("No vector embeddings indexed. Run `python embed_crops.py`.")

        st.markdown("### Search settings")
        clip_pad = st.slider(
            "Footage window (± seconds)",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            help="Extracts [event - pad] to [event + pad] seconds of footage.",
        )

        auto_cut = st.checkbox(
            "Auto-extract clips for top results",
            value=True,
            help="Renders the footage clip next to the event snapshot.",
        )

        video_options = ["All videos"] + [v["video_id"] for v in videos]
        selected_vid = st.selectbox("Video", video_options)
        vid_filter = None if selected_vid == "All videos" else selected_vid

        if vid_filter:
            v_summary = db.get_video_summary(vid_filter)
            st.caption(
                f"{v_summary['persons']} persons · "
                f"{v_summary['vehicles']} vehicles · "
                f"{v_summary['objects']} objects"
            )

        if videos:
            with st.expander("Indexed videos"):
                for v in videos:
                    dur = format_time_s(v.get("duration_s")) if v.get("duration_s") else "unknown"
                    st.markdown(f"**{v['video_id']}**")
                    st.caption(
                        f"{v.get('camera_id', 'cam_default')} · "
                        f"{v.get('status', 'indexed')} · {dur}"
                    )

    # ── Main Content Area ─────────────────────────────────────────────
    st.title("CCTV footage search")
    st.caption(
        "Search by text, image, or both. The image supplies appearance evidence "
        "and the text refines the match."
    )

    # Search form: text and image are submitted together and combined.
    with st.form("search_form", border=False):
        text_value = st.text_input(
            "Query",
            placeholder="e.g. blue car, female person, white truck",
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader(
            "Reference images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            help="Optional. The image and text are combined for retrieval.",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        new_text = (text_value or "").strip()
        new_files = list(uploaded or [])
        new_images = [f.getvalue() for f in new_files][:5]
        new_names = [f.name for f in new_files][:5]
        current_names = list(st.session_state.get("active_image_names") or [])
        if (new_text or new_images) and (
            new_text != (st.session_state.get("active_text") or "")
            or new_names != current_names
        ):
            st.session_state["active_text"] = new_text
            st.session_state["active_images"] = new_images
            st.session_state["active_image_names"] = new_names
            st.session_state["active_token"] = (
                f"{new_text}|{len(new_images)}|{datetime.now().timestamp()}"
            )
            st.session_state.pop("result_cache", None)

    active_text = (st.session_state.get("active_text") or "").strip()
    active_images = list(st.session_state.get("active_images") or [])
    active_image_names = list(st.session_state.get("active_image_names") or [])
    active_token = st.session_state.get("active_token") or ""

    if not active_text and not active_images:
        st.caption("Enter a query above to search indexed footage.")
        return

    # ── Execute Search ────────────────────────────────────────────────
    # NL-time anchor: end of indexed footage, else wall clock (configurable)
    cfg = load_time_config()
    _, db_hi = db.get_indexed_time_bounds()
    anchor_ts = db_hi if (cfg.get("anchor") == "latest_indexed" and db_hi) else None

    cache_key = (active_token, vid_filter)
    cached = st.session_state.get("result_cache")
    if cached and cached.get("key") == cache_key:
        payload = cached["payload"]
    else:
        if active_images and not emb_count:
            payload = {
                "error": "No SigLIP embeddings indexed yet — run `python embed_crops.py` first.",
                "results": [],
                "method": "image",
                "entity": None,
                "parsed": None,
            }
        else:
            spinner_msg = (
                f"Embedding {len(active_images)} query image(s) & searching the database..."
                if active_images
                else f"Vectorizing query & searching database: '{active_text}'..."
            )
            with st.spinner(spinner_msg):
                try:
                    payload = run_unified_search(
                        db,
                        active_text,
                        active_images,
                        video_id=vid_filter,
                        anchor_ts=anchor_ts,
                    )
                except sqlite3.OperationalError as exc:
                    st.error(
                        "The SQLite database could not be read. "
                        "If ingestion is running, wait for it to finish and retry the search. "
                        f"Details: {exc}"
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "error": f"Search failed: {exc}",
                        "results": [],
                        "method": "error",
                        "entity": None,
                        "parsed": None,
                    }
        st.session_state["result_cache"] = {"key": cache_key, "payload": payload}

    parsed = payload.get("parsed")
    results = payload.get("results") or []

    # Show submitted query image(s) as thumbnails.
    if active_images:
        st.caption(f"Query images ({len(active_images)}), analysed in memory")
        thumb_cols = st.columns(min(5, len(active_images)))
        for i, (data, name) in enumerate(zip(active_images[:5], active_image_names[:5])):
            with thumb_cols[i]:
                st.image(data, caption=name, width=120, alt=f"Query image {i + 1}")

    if active_images and active_text:
        heading = f"Results for \u201c{active_text}\u201d (image and text)"
    elif active_images:
        heading = "Image results"
    else:
        heading = f"Results for \u201c{active_text}\u201d"

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.subheader(heading)
    with h_col2:
        st.caption(f"{len(results)} match(es)")
        if st.button("Clear", key="clear_search", width="stretch"):
            for key in (
                "active_text",
                "active_images",
                "active_image_names",
                "active_token",
                "result_cache",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    if payload.get("error"):
        st.error(payload["error"])
        return

    with st.expander("Query details", expanded=False):
        if active_text:
            lang = detect_language(active_text)
            if lang == "hinglish":
                normalized = normalize_query(active_text)
                st.markdown(f"Language: `Hinglish` -> `{normalized}`")
            else:
                st.markdown("Language: `English`")
        else:
            st.markdown("Mode: `Image only`")
        if parsed and parsed.time_range:
            tr = parsed.time_range
            start = (
                datetime.fromtimestamp(tr.start_ts, ZoneInfo(tr.tz)).strftime("%d-%m %H:%M")
                if tr.start_ts
                else "open"
            )
            end = (
                datetime.fromtimestamp(tr.end_ts, ZoneInfo(tr.tz)).strftime("%d-%m %H:%M")
                if tr.end_ts
                else "open"
            )
            st.markdown(f"Time window: `{tr.label}` ({start} to {end})")
        if parsed:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"Entity: `{parsed.target_entity or 'any'}`")
            c2.markdown(f"Color: `{parsed.color or 'any'}`")
            c3.markdown(f"Gender: `{parsed.gender or 'any'}`")
            c4.markdown(f"Type: `{parsed.vehicle_type or parsed.object_type or 'any'}`")
        st.caption(
            f"Method: {payload.get('method')} · "
            f"Resolved entity: {payload.get('entity') or 'any'} · "
            f"Video: {vid_filter or 'all videos'}"
        )

    if not results:
        st.caption("No matches found.")
        if active_images:
            st.caption(
                "Try a clearer image, add a text hint such as \"red car\", "
                "or select \"All videos\" in the sidebar."
            )
        else:
            st.caption(
                "Try broader terms such as \"car\", \"person\", \"white\", or \"blue\"."
            )
        return

    # Only surface the best few high-confidence matches first; the user can
    # expand the list on demand instead of scanning every tracked record.
    total_results = len(results)
    limit_key = f"visible_results::{active_token}"
    visible_limit = int(st.session_state.get(limit_key, DEFAULT_VISIBLE_RESULTS))
    visible_limit = min(max(DEFAULT_VISIBLE_RESULTS, visible_limit), total_results)
    st.session_state[limit_key] = visible_limit

    st.caption(
        f"Showing {visible_limit} of {total_results} matching result(s), "
        "ranked by query relevance."
    )

    # Render Result Cards
    for idx, res in enumerate(results[:visible_limit]):
        _render_side_by_side_card(
            db=db,
            result=res,
            idx=idx,
            clip_pad_s=float(clip_pad),
            auto_cut=auto_cut,
        )

    if visible_limit < total_results:
        remaining = total_results - visible_limit
        more_col, all_col = st.columns(2)
        if more_col.button(
            f"Show {min(VISIBLE_RESULTS_STEP, remaining)} more",
            key=f"show_more::{active_token}",
            width="stretch",
        ):
            st.session_state[limit_key] = min(
                visible_limit + VISIBLE_RESULTS_STEP, total_results
            )
            st.rerun()
        if all_col.button(
            f"Show all {total_results}",
            key=f"show_all::{active_token}",
            width="stretch",
        ):
            st.session_state[limit_key] = total_results
            st.rerun()


def _render_side_by_side_card(
    db: CCTVDatabase,
    result: dict[str, Any],
    idx: int,
    clip_pad_s: float,
    auto_cut: bool,
):
    """Render query result with event footage image and ±30s video footage side-by-side."""
    entity = result.get("entity", "unknown")
    track_id = result.get("track_id", "—")
    label = result.get("label", "—")
    video_id = result.get("video_id", "—")
    file_path = result.get("file_path", "")
    first_seen = result.get("first_seen_s") or 0.0
    last_seen = result.get("last_seen_s") or first_seen
    bbox_time_s = result.get("bbox_time_s")
    bbox_frame = result.get("bbox_frame")
    event_time = float(bbox_time_s) if bbox_time_s is not None else first_seen
    conf = result.get("peak_conf", 0.0)
    sim = result.get("similarity", 0.0)
    search_method = result.get("search_method", "Structured Match")
    bbox_json = result.get("bbox_json")
    source_name = Path(file_path).name if file_path else "unknown"

    time_str = f"{format_time_s(first_seen)} → {format_time_s(last_seen)}"

    # Compute clip interval (±clip_pad_s) around the detection frame.
    video_dur = get_video_duration(file_path) if file_path else 0.0
    clip_start = max(0.0, event_time - clip_pad_s)
    clip_end = min(video_dur, event_time + clip_pad_s) if video_dur > 0 else (event_time + clip_pad_s)
    clip_window_str = f"{format_time_s(clip_start)} to {format_time_s(clip_end)} (±{int(clip_pad_s)}s)"

    card_title = (
        f"Result {idx + 1}: {label} · Track #{track_id} · "
        f"{video_id} · event at {format_time_s(event_time)}"
    )
    if sim > 0:
        card_title += f" · similarity {sim:.3f}"

    with st.expander(card_title, expanded=(idx < 2)):
        # Top Meta Bar
        m1, m2, m3, m4, m5 = st.columns([2, 2, 2, 2, 2])
        m1.markdown(f"Entity: `{entity}`")
        m2.markdown(f"Track: `#{track_id}`")
        m3.markdown(f"Time: `{format_time_s(event_time)}`")
        m4.markdown(f"Confidence: `{conf:.2f}`")
        m5.markdown(f"Match: `{search_method}`")
        st.caption(f"Source: {source_name} ({video_id}) · track span {time_str}")

        if entity == "person":
            gender_txt = f"Gender: `{result.get('gender', 'unknown')}`"
            age_txt = f" · Age: `~{int(result['age_est'])}y`" if result.get("age_est") else ""
            upper_txt = f" · Upper: `{result.get('upper_color', 'unknown')}`"
            lower_txt = f" · Lower: `{result.get('lower_color', 'unknown')}`"
            st.markdown(f"{gender_txt}{age_txt}{upper_txt}{lower_txt}")
        elif entity == "vehicle":
            st.markdown(
                f"Type: `{result.get('type', '—')}` · "
                f"Color: `{result.get('color', '—')}`"
            )
        elif entity == "object":
            assoc = (
                f" · Person: `#{result['associated_person_track']}`"
                if result.get("associated_person_track")
                else ""
            )
            st.markdown(
                f"Type: `{result.get('type', '—')}` · "
                f"Color: `{result.get('color', '—')}`{assoc}"
            )

        # ── SIDE BY SIDE PRESENTATION ─────────────────────────────────
        col_image, col_video = st.columns([1, 1], gap="medium")

        # 1. LEFT SIDE: Query Footage Image (Snapshot with bbox detection)
        with col_image:
            st.markdown(f"**Snapshot** · {format_time_s(event_time)}")
            source_fps = 0.0
            if file_path and Path(file_path).is_file():
                probe = cv2.VideoCapture(str(file_path))
                if probe.isOpened():
                    source_fps = float(probe.get(cv2.CAP_PROP_FPS))
                probe.release()
            fallback_frame = (
                int(round(event_time * source_fps))
                if source_fps > 0 and bbox_json
                else None
            )
            snapshot_frame = bbox_frame if bbox_frame is not None else fallback_frame
            bbox_exact = bbox_json is not None and snapshot_frame is not None
            displayed_saved_crop = False
            if bbox_exact:
                annotated_frame = extract_annotated_frame(
                    video_path=file_path,
                    bbox_json=bbox_json,
                    timestamp_s=event_time,
                    label=f"{entity} #{track_id} ({conf:.2f})",
                    frame_index=snapshot_frame,
                )
                if annotated_frame is None:
                    annotated_frame = _load_saved_crop(entity, video_id, track_id)
                    displayed_saved_crop = annotated_frame is not None
            else:
                annotated_frame = _load_saved_crop(entity, video_id, track_id)
                displayed_saved_crop = annotated_frame is not None
                if annotated_frame is None:
                    # Legacy row: do not draw a box we cannot align to a frame.
                    annotated_frame = extract_annotated_frame(
                        video_path=file_path,
                        bbox_json=None,
                        timestamp_s=event_time,
                        label="",
                    )

            if annotated_frame is not None:
                if displayed_saved_crop:
                    caption = f"Saved detection crop for {entity} #{track_id} (re-ingest for full-frame bbox)"
                elif bbox_exact:
                    frame_note = "exact detection frame" if bbox_frame is not None else "timestamp-aligned frame"
                    caption = f"{frame_note} at {format_time_s(event_time)} with detection bounding box"
                else:
                    caption = f"Frame at {format_time_s(event_time)} (legacy record without box metadata)"
                st.image(
                    numpy_to_pil(improve_snapshot_detail(annotated_frame)),
                    caption=caption,
                    width="stretch",
                    alt="Event snapshot with detection bounding box",
                )
            else:
                crop = extract_crop_from_video(file_path, bbox_json, event_time, frame_index=snapshot_frame)
                if crop is not None:
                    st.image(
                        numpy_to_pil(upscale_crop_for_display(crop)),
                        caption=f"Crop at {format_time_s(event_time)}",
                        width="stretch",
                        alt="Detected object crop",
                    )
                else:
                    st.caption("Snapshot preview is not available (video file missing or unreachable).")

            # Inset crop if available and the main view is a full frame.
            if bbox_exact:
                crop = extract_crop_from_video(file_path, bbox_json, event_time, frame_index=snapshot_frame)
                if crop is not None and annotated_frame is not None:
                    is_distant = min(crop.shape[:2]) < 220
                    with st.expander("View zoomed crop", expanded=is_distant):
                        st.image(
                            numpy_to_pil(upscale_crop_for_display(crop)),
                            caption=f"Cropped {entity} #{track_id}",
                            width=360,
                            alt="Zoomed detection crop",
                        )

        # 2. RIGHT SIDE: Event Video Footage (±30s clip)
        with col_video:
            st.markdown(f"**Footage clip** · {clip_window_str}")
            st.caption(
                f"{int(clip_pad_s)}s before to {int(clip_pad_s)}s after the detected event."
            )

            clip_id = f"{entity}_{video_id}_trk{track_id}"
            clip_path = CLIP_DIR / f"{clip_id}_{int(clip_start)}s_to_{int(clip_end)}s.mp4"

            has_source = bool(file_path) and Path(file_path).is_file()
            saved_clip = _find_saved_clip(entity, video_id, track_id)
            if saved_clip is not None and not clip_path.is_file():
                clip_path = saved_clip

            if not has_source and not clip_path.is_file():
                st.caption("No pre-cut footage clip is available for this track.")
            else:
                if has_source and clip_path.is_file() and clip_path.stat().st_size <= 1024:
                    clip_path.unlink(missing_ok=True)

                if has_source:
                    should_extract = auto_cut and (idx < 3 or clip_path.is_file())
                    extract_btn = False

                    if not clip_path.is_file():
                        extract_btn = st.button(
                            f"Extract ±{int(clip_pad_s)}s clip", key=f"btn_cut_{idx}"
                        )

                    if should_extract or extract_btn:
                        if not clip_path.is_file():
                            with st.spinner(
                                f"Extracting ±{int(clip_pad_s)}s footage clip "
                                f"({format_time_s(clip_start)} → {format_time_s(clip_end)})..."
                            ):
                                clip_res = extract_event_clip(
                                    file_path, event_time, pad_s=clip_pad_s, clip_id=clip_id
                                )
                                if clip_res and clip_res.is_file():
                                    clip_path = clip_res

                if clip_path.is_file():
                    clip_path = ensure_browser_playable_video(clip_path, max_height=480)
                    try:
                        with open(clip_path, "rb") as f_clip:
                            clip_bytes = f_clip.read()
                        if not clip_bytes:
                            raise ValueError("Extracted clip is empty.")
                        st.video(clip_bytes, format="video/mp4")
                    except (OSError, ValueError) as exc:
                        st.error(f"Clip preview unavailable: {exc}")
                        clip_bytes = None
                    st.caption(f"Source: {source_name} (ID: {video_id})")
                    # Download button for extracted footage snippet
                    if clip_bytes:
                        try:
                            st.download_button(
                                label="Download clip",
                                data=clip_bytes,
                                file_name=clip_path.name,
                                mime="video/mp4",
                                key=f"dl_{idx}",
                                width="stretch",
                            )
                        except OSError as exc:
                            st.error(f"Clip download unavailable: {exc}")
                else:
                    st.caption("Use the extract button above to cut this footage segment.")



# ── Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
