#!/usr/bin/env python3
"""image_query.py — image-to-image "find similar" search over indexed CCTV crops.

Given one or more query images (up to 5), optionally detect person/vehicle
crops inside them, embed the crops with SigLIP 2, and return the most similar
tracked entities from the embeddings database.

Query images are processed entirely in memory — they are never written to disk.

CLI:
    python image_query.py sample.jpg --entity auto --top-k 10
    python image_query.py a.jpg b.jpg --entity vehicle --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.db import CCTVDatabase
from common.emb_index import EmbeddingIndex
from common.embed import embed_images_mean, is_available
from common.time_query import DEFAULT_TZ


_PERSON_CACHE: dict[str, Any] = {}
_VEHICLE_CACHE: dict[str, Any] = {}


def load_image_bgr(path_or_bytes) -> np.ndarray | None:
    """Decode an image from a path or bytes WITHOUT writing anything to disk."""
    import cv2

    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = np.frombuffer(path_or_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        data = Path(path_or_bytes).read_bytes()
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _get_person_detector(device: str = "cpu", model_name: str = "rfdetr-nano"):
    """Lazily construct the person detector used for QueryCrop."""
    key = f"person:{model_name}:{device}"
    if key not in _PERSON_CACHE:
        from person._detect_person import PersonDetector
        _PERSON_CACHE[key] = PersonDetector(model_name=model_name, device=device)
    return _PERSON_CACHE[key]


def _get_vehicle_detector(device: str = "cpu", model_name: str = "rfdetr-nano"):
    """Lazily construct the vehicle detector used for QueryCrop."""
    key = f"vehicle:{model_name}:{device}"
    if key not in _VEHICLE_CACHE:
        from vechical.detect_vehicle import VehicleDetector, DEFAULT_VEHICLE_CLASSES
        _VEHICLE_CACHE[key] = VehicleDetector(
            model_name=model_name, target_classes=set(DEFAULT_VEHICLE_CLASSES),
            device=device,
        )
    return _VEHICLE_CACHE[key]


def _detect_best_crop_scored(
    img_bgr: np.ndarray, kind: str, device: str
) -> tuple[np.ndarray, float] | None:
    """Best detection crop of `kind` plus an importance score.

    The score combines detector confidence with the box area, so a large clear
    subject wins over a tiny high-confidence background detection. This is what
    makes auto-resolution pick the right entity in mixed scenes.
    """
    det = _get_person_detector(device) if kind == "person" else _get_vehicle_detector(device)
    try:
        if kind == "person":
            boxes, confs, _tid = det.predict(img_bgr, conf=0.35, iou=0.5)
        else:
            boxes, det_confs, _names, _colors, _color_confs, _tid = det.predict(
                img_bgr, conf=0.35, iou=0.5
            )
            confs = det_confs
    except Exception:
        return None
    if boxes is None or len(boxes) == 0:
        return None

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    confs = np.asarray(confs, dtype=np.float32).reshape(-1)
    if confs.shape[0] != boxes.shape[0]:
        confs = np.ones(boxes.shape[0], dtype=np.float32)

    h, w = img_bgr.shape[:2]
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 1.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 1.0, None)
    area_frac = np.clip((widths * heights) / float(max(1, w * h)), 0.0, 1.0)
    scores = confs * np.sqrt(area_frac)

    best = int(np.argmax(scores))
    x1, y1, x2, y2 = (float(v) for v in boxes[best][:4])
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(max(x2, x1 + 1))), min(h, int(max(y2, y1 + 1)))
    seg = img_bgr[y1:y2, x1:x2]
    if seg.size == 0 or seg.shape[0] <= 8 or seg.shape[1] <= 8:
        return None
    return seg, float(scores[best])


def _detect_best_crop(img_bgr: np.ndarray, kind: str, device: str):
    """Best-scoring detection crop of `kind` in the image, or None."""
    result = _detect_best_crop_scored(img_bgr, kind, device)
    return result[0] if result is not None else None


def _resolve_query_crops(
    imgs: list[np.ndarray], entity: str, device: str
) -> tuple[str | None, list[np.ndarray]]:
    """Resolve the entity type and query crops for a set of uploaded images.

    Explicit person/vehicle requests keep that type. In auto mode both detectors
    compete and the most prominent subject wins instead of always preferring a
    person (which used to select a tiny bystander over a clearly framed car).
    """
    if entity in ("person", "vehicle"):
        scored = [
            r for r in (_detect_best_crop_scored(img, entity, device) for img in imgs) if r
        ]
        if scored:
            return entity, [crop for crop, _ in scored]
        # The hinted entity was not present in the uploaded image; fall back to
        # auto resolution instead of forcing a mismatched index.

    best_kind: str | None = None
    best_score = -1.0
    best_crops: list[np.ndarray] = []
    for kind in ("person", "vehicle"):
        scored = [
            r for r in (_detect_best_crop_scored(img, kind, device) for img in imgs) if r
        ]
        if not scored:
            continue
        kind_score = max(score for _, score in scored)
        if kind_score > best_score:
            best_kind = kind
            best_score = kind_score
            best_crops = [crop for crop, _ in scored]
    return best_kind, best_crops


def find_similar(
    db: CCTVDatabase,
    images,
    entity: str = "auto",
    top_k: int = 10,
    min_sim: float = 0.0,
    video_id: str | None = None,
    camera_id: str | None = None,
    ts_from: float | None = None,
    ts_to: float | None = None,
    device: str = "cpu",
    per_track: bool = True,
) -> tuple[str, list[dict]]:
    """Search the embedding DB using query images.

    Returns (resolved_entity, hits). `images` — list of BGR ndarrays or paths
    or raw bytes. `entity` — "person" | "vehicle" | "auto". In auto mode both
    detectors compete and the most prominent subject resolves the type; if no
    crop is found, a whole-image search runs across both indexes. Never persists
    the query images.
    """
    if not is_available():
        return "", [{"error": "SigLIP embedding stack unavailable (transformers not installed)"}]
    imgs: list[np.ndarray] = []
    for item in images:
        img = load_image_bgr(item)
        if img is not None:
            imgs.append(img)
    if not imgs:
        return "", [{"error": "no decodable query image"}]

    resolved, crops = _resolve_query_crops(imgs, entity, device)

    if resolved is None:
        # Whole-image fallback: try both indexes, keep the better-similarity side.
        qvec = embed_images_mean(imgs, device=device)
        if qvec is None:
            return "person", [{"error": "failed to embed query image(s)"}]
        candidates = []
        for etype in ("person", "vehicle"):
            idx = EmbeddingIndex(db, entity_type=etype)
            idx.refresh()
            if len(idx) == 0:
                continue
            fn = idx.search_tracks if per_track else idx.search
            hits = fn(qvec, top_k=top_k, min_sim=min_sim,
                      video_id=video_id, camera_id=camera_id, ts_from=ts_from, ts_to=ts_to)
            if hits:
                candidates.append((float(hits[0]["similarity"]), etype, hits))
        if not candidates:
            return "person", []
        candidates.sort(key=lambda c: -c[0])
        return candidates[0][1], candidates[0][2]

    idx = EmbeddingIndex(db, entity_type=resolved)
    idx.refresh()
    if len(idx) == 0:
        return resolved, []
    query_vec = embed_images_mean(crops if crops else imgs, device=device)
    if query_vec is None:
        return resolved, [{"error": "failed to embed query image(s)"}]
    fn = idx.search_tracks if per_track else idx.search
    hits = fn(query_vec, top_k=top_k, min_sim=min_sim, video_id=video_id,
              camera_id=camera_id, ts_from=ts_from, ts_to=ts_to)
    for h in hits:
        h["query_entity"] = resolved
    return resolved, hits


def _attach_track_info(db: CCTVDatabase, resolved_entity: str, hits: list[dict]) -> None:
    """Decorate hits with human-readable attributes from the track tables.

    Embeds are keyed by (video_id, track_id) because stale embeddings may carry an
    outdated ``entity_id`` after re-ingestion. Looking up by ``id`` alone silently
    dropped metadata for those rows, which made image search look broken (no
    snapshot, no video, no label). We now resolve by track first, then fall back
    to the numeric id for old databases.
    """
    for h in hits:
        etype = h.get("entity_type") or resolved_entity
        if etype not in ("person", "vehicle"):
            continue
        table = "persons" if etype == "person" else "vehicles"
        video_id = h.get("video_id")
        track_id = h.get("track_id")

        row = None
        if video_id is not None and track_id is not None:
            row = db.conn.execute(
                f"SELECT * FROM {table} WHERE video_id = ? AND track_id = ?",
                (video_id, track_id),
            ).fetchone()
        if row is None and h.get("entity_id") is not None:
            row = db.conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (h["entity_id"],)
            ).fetchone()
        if row is None:
            h.setdefault("entity", etype)
            continue

        r = dict(row)
        h["entity"] = etype
        h["entity_id"] = r.get("id", h.get("entity_id"))
        h["track_id"] = r.get("track_id", track_id)
        h["video_id"] = r.get("video_id", video_id)
        if etype == "person":
            h["label"] = f"person #{h['track_id']} ({r.get('gender') or 'unknown'})"
            h["gender"] = r.get("gender")
            h["age_est"] = r.get("age_est")
            h["upper_color"] = r.get("upper_color")
            h["lower_color"] = r.get("lower_color")
        else:
            h["label"] = f"{r.get('color', '?')} {r.get('vehicle_type', '?')}"
            h["color"] = r.get("color")
            h["vehicle_type"] = r.get("vehicle_type")
            h["type"] = r.get("vehicle_type")

        h["first_seen_s"] = r.get("first_seen_s")
        h["last_seen_s"] = r.get("last_seen_s")
        h["first_seen_ts"] = r.get("first_seen_ts")
        h["last_seen_ts"] = r.get("last_seen_ts")
        h["frame_count"] = r.get("frame_count")
        h["bbox_json"] = r.get("bbox_json")
        h["bbox_time_s"] = r.get("bbox_time_s")
        h["bbox_frame"] = r.get("bbox_frame")
        if r.get("peak_conf") is not None:
            h["peak_conf"] = r.get("peak_conf")

        v = db.conn.execute(
            "SELECT camera_id, file_path FROM videos WHERE video_id = ?",
            (h["video_id"],),
        ).fetchone()
        if v is not None:
            h["camera_id"], h["file_path"] = v["camera_id"], v["file_path"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Find indexed person/vehicle tracks similar to query images")
    parser.add_argument("images", nargs="+", help="Query image paths (up to 5 recommended)")
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"))
    parser.add_argument("--entity", choices=["auto", "person", "vehicle"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-sim", type=float, default=0.0)
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"[ERROR] Database not found: {args.db}", file=sys.stderr)
        return 1
    db = CCTVDatabase(str(args.db))

    resolved, hits = find_similar(
        db, args.images, entity=args.entity, top_k=args.top_k,
        min_sim=args.min_sim, video_id=args.video_id,
    )
    if hits and isinstance(hits[0], dict) and hits[0].get("error"):
        print(hits[0]["error"], file=sys.stderr)
        return 1
    _attach_track_info(db, resolved_entity=resolved, hits=hits)

    if args.json:
        print(json.dumps({"entity": resolved, "hits": hits}, indent=2, default=str))
    else:
        print(f"[IMAGE] Query type resolved: {resolved} — {len(hits)} track hit(s)")
        for i, h in enumerate(hits, start=1):
            print(f"{i:>3}. {h.get('label', '?'):<30} sim={h['similarity']:.3f} "
                  f"cam={h.get('camera_id') or h.get('video_id')} "
                  f"track#{h.get('track_id')}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
