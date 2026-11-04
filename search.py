#!/usr/bin/env python3
"""
search.py — Unified Query Search & Video Snippet Extraction Engine.

Search across all tracked persons, vehicles, and objects in the database.
Supports natural language search phrases (e.g. "red car", "female person", "white truck", "blue shirt")
as well as explicit attribute filters. Can automatically extract and save playable video snippets.

Examples:
    # Natural language search
    python search.py "red car"
    python search.py "female person"
    python search.py "white truck" --cut-clips
    python search.py "blue shirt" --video-id cam1

    # Filtered search
    python search.py --entity vehicle --color red --vehicle-type car
    python search.py --entity person --gender male --age-min 18 --age-max 35
    python search.py --entity person --upper-color blue --time-start 60 --time-end 180

    # JSON output
    python search.py "black motorcycle" --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.db import CCTVDatabase
from common.time_query import DEFAULT_TZ, TimeRange, load_time_config, parse_time
from common.video_io import cut_clip
from query_norm import normalize_query, detect_language
from search_core import (
    SearchRequest,
    find_similar_image_and_text,
    image_search,
    similar_to_track,
)


KNOWN_COLORS = {
    "black", "dark gray", "gray", "grey", "white", "red", "orange",
    "yellow", "green", "cyan", "blue", "navy", "purple", "pink", "brown", "beige"
}

VEHICLE_KEYWORDS = {
    "car": "car", "auto": "car", "sedan": "car", "suv": "car",
    "truck": "truck", "lorry": "truck", "pickup": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle", "bike": "motorcycle", "scooter": "motorcycle",
    "bicycle": "bicycle", "cycle": "bicycle",
}

PERSON_KEYWORDS = {
    "person", "people", "man", "men", "woman", "women", "male", "female",
    "boy", "girl", "guy", "lady"
}

GENDER_MAP = {
    "man": "male", "men": "male", "male": "male", "boy": "male", "guy": "male",
    "woman": "female", "women": "female", "female": "female", "girl": "female", "lady": "female"
}

CLOTHING_OBJECT_KEYWORDS = {
    "shirt", "t-shirt", "tshirt", "jacket", "coat", "sweater", "hoodie", "pants",
    "jeans", "shorts", "skirt", "dress", "shoe", "shoes", "sneakers", "hat", "cap",
    "glasses", "backpack", "bag", "handbag", "suitcase", "umbrella", "tie", "scarf",
    "laptop", "laptops", "computer", "computers", "phone", "phones",
    "mobile", "cellphone", "cellphones", "tablet", "tablets"
}


@dataclass
class ParsedQuery:
    raw_query: str
    target_entity: str | None = None  # "person" | "vehicle" | "object" | None (search all)
    color: str | None = None
    vehicle_type: str | None = None
    gender: str | None = None
    object_type: str | None = None
    event_type: str | None = None   # "entry" | "exit" | None
    time_start: float | None = None
    time_end: float | None = None
    time_range: TimeRange | None = None  # absolute epoch window from NL time phrase


def parse_natural_language_query(
    text: str,
    anchor_ts: float | None = None,
    tz: str = DEFAULT_TZ,
) -> ParsedQuery:
    """Extract entities, colors, types, genders, and a TIME window from natural language.

    Automatically normalizes Hinglish/Hindi queries to English before parsing.
    Time phrases are extracted from the RAW text (before normalization, which can
    mangle Hinglish time words). `anchor_ts` — epoch seconds the anchor (default
    wall-clock now; callers may pass the end of indexed footage instead).
    """
    if anchor_ts is None:
        anchor_ts = _time.time()

    # Stage 0a: extract the time phrase from the raw query first
    clean_text, tr = parse_time(text, now=anchor_ts, tz=tz)

    # Stage 0b: Hinglish normalization on the remaining text
    lang = detect_language(clean_text)
    normalized = normalize_query(clean_text) if lang == "hinglish" else clean_text.lower().strip()

    q = normalized
    words = re.findall(r"\b[\w-]+\b", q)

    parsed = ParsedQuery(raw_query=text, time_range=tr)

    # 1. Colors
    for color in KNOWN_COLORS:
        # Match multi-word color first
        if color in q:
            parsed.color = "gray" if color == "grey" else color
            break

    # 2. Gender
    for word in words:
        if word in GENDER_MAP:
            parsed.gender = GENDER_MAP[word]
            parsed.target_entity = "person"
            break

    # 3. Vehicle Type
    for word in words:
        if word in VEHICLE_KEYWORDS:
            parsed.vehicle_type = VEHICLE_KEYWORDS[word]
            parsed.target_entity = "vehicle"
            break

    # 4. Clothing / Objects
    for word in words:
        if word in CLOTHING_OBJECT_KEYWORDS:
            object_aliases = {
                "tshirt": "shirt",
                "t-shirt": "shirt",
                "laptops": "laptop",
                "computers": "computer",
                "phones": "phone",
                "cellphones": "cellphone",
                "tablets": "tablet",
            }
            parsed.object_type = object_aliases.get(word, word)
            if parsed.target_entity is None:
                parsed.target_entity = "object"
            break

    # 5. Directional Events (entry / exit)
    EVENT_KEYWORDS = {
        "entry": "entry", "enter": "entry", "entering": "entry", "entered": "entry",
        "exit": "exit", "exiting": "exit", "exited": "exit", "leaving": "exit",
    }
    for word in words:
        if word in EVENT_KEYWORDS:
            parsed.event_type = EVENT_KEYWORDS[word]
            break

    # 6. Generic Person
    if any(w in PERSON_KEYWORDS for w in words):
        parsed.target_entity = "person"

    return parsed


_COLOR_WORDS = {
    "black", "white", "gray", "grey", "dark", "light", "red", "blue", "green",
    "yellow", "orange", "brown", "pink", "purple", "cyan", "magenta", "navy",
    "maroon", "beige", "gold", "silver", "tan", "olive", "violet", "teal",
}


def _clean_object_type(object_type: str | None, color: str | None) -> str:
    """Remove duplicated leading colour words from legacy object labels."""
    obj = (object_type or "object").strip()
    col = (color or "").strip()

    if col and obj.lower().startswith(col.lower()):
        cleaned = obj[len(col):].strip()
        if cleaned:
            obj = cleaned

    while obj:
        first, _, rest = obj.partition(" ")
        if first.lower() in _COLOR_WORDS and rest.strip():
            obj = rest.strip()
        else:
            break
    return obj or "object"


def format_time_s(seconds: float | None) -> str:
    """Format seconds into MM:SS.s string."""
    if seconds is None:
        return "--:--"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            x1, y1, x2, y2 = (float(v) for v in parsed[:4])
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return None


def _same_physical_entity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Conservatively identify adjacent tracker IDs for the same entity."""
    if left.get("video_id") != right.get("video_id") or left.get("entity") != right.get("entity"):
        return False
    if left.get("type") != right.get("type") or left.get("color") != right.get("color"):
        return False
    if left.get("entity") == "person" and left.get("gender") != right.get("gender"):
        return False

    left_end = float(left.get("last_seen_s") or left.get("first_seen_s") or 0.0)
    right_start = float(right.get("first_seen_s") or 0.0)
    if right_start - left_end > 2.0:
        return False

    a = _parse_bbox(left.get("bbox_json"))
    b = _parse_bbox(right.get("bbox_json"))
    if a is None or b is None:
        return False
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    iou = intersection / union if union > 0 else 0.0
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    diagonal = max(1.0, ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5)
    center_distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    return iou >= 0.25 or center_distance <= diagonal * 0.75


def _deduplicate_track_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge conservative cross-track duplicates before displaying search results."""
    deduplicated: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: float(item.get("first_seen_s") or 0.0)):
        match = next(
            (
                existing
                for existing in reversed(deduplicated)
                if _same_physical_entity(existing, result)
            ),
            None,
        )
        if match is None:
            deduplicated.append(result)
            continue

        match["first_seen_s"] = min(
            float(match.get("first_seen_s") or 0.0),
            float(result.get("first_seen_s") or 0.0),
        )
        match["last_seen_s"] = max(
            float(match.get("last_seen_s") or 0.0),
            float(result.get("last_seen_s") or 0.0),
        )
        match["frame_count"] = (match.get("frame_count") or 0) + (result.get("frame_count") or 0)
        if (result.get("peak_conf") or 0.0) > (match.get("peak_conf") or 0.0):
            for key in ("id", "track_id", "label", "peak_conf", "bbox_json", "bbox_time_s", "bbox_frame"):
                if key in result:
                    match[key] = result[key]
    return deduplicated


def execute_search(
    db: CCTVDatabase,
    query: ParsedQuery,
    video_id: str | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
    upper_color: str | None = None,
    lower_color: str | None = None,
    clip_output_dir: Path | None = None,
    clip_pad_s: float = 2.0,
) -> list[dict[str, Any]]:
    """Execute search across database tables and return normalized results list."""
    results: list[dict[str, Any]] = []

    # Decide which tables to search
    search_vehicles = query.target_entity in (None, "vehicle")
    search_persons = query.target_entity in (None, "person")
    search_objects = query.target_entity in (None, "object")

    # ── Search Vehicles ──
    if search_vehicles and query.gender is None and query.object_type is None:
        v_rows = db.search_vehicles(
            video_id=video_id,
            vehicle_type=query.vehicle_type,
            color=query.color,
            time_start=query.time_start,
            time_end=query.time_end,
            ts_from=query.time_range.start_ts if query.time_range else None,
            ts_to=query.time_range.end_ts if query.time_range else None,
        )
        for r in v_rows:
            results.append({
                "entity": "vehicle",
                "id": r["id"],
                "video_id": r["video_id"],
                "file_path": r.get("file_path", ""),
                "track_id": r["track_id"],
                "label": f"{r['color']} {r['vehicle_type']}",
                "type": r["vehicle_type"],
                "color": r["color"],
                "color_conf": r.get("color_conf", 0.0),
                "gender": None,
                "first_seen_s": r["first_seen_s"],
                "last_seen_s": r["last_seen_s"],
                "frame_count": r["frame_count"],
                "peak_conf": r["peak_conf"],
                "bbox_json": r.get("bbox_json"),
                "bbox_time_s": r.get("bbox_time_s"),
                "bbox_frame": r.get("bbox_frame"),
            })

    # ── Search Persons ──
    if search_persons and query.vehicle_type is None:
        p_upper = upper_color or (query.color if query.object_type in (None, "shirt", "jacket", "coat") else None)
        p_rows = db.search_persons(
            video_id=video_id,
            gender=query.gender,
            age_min=age_min,
            age_max=age_max,
            upper_color=p_upper,
            lower_color=lower_color,
            time_start=query.time_start,
            time_end=query.time_end,
            ts_from=query.time_range.start_ts if query.time_range else None,
            ts_to=query.time_range.end_ts if query.time_range else None,
        )
        for r in p_rows:
            gender_str = r.get("gender") or "unknown"
            age_str = f" (~{int(r['age_est'])}y)" if r.get("age_est") else ""
            label_parts = [gender_str + age_str]
            if r.get("upper_color") and r["upper_color"] != "unknown":
                label_parts.append(f"top:{r['upper_color']}")
            if r.get("lower_color") and r["lower_color"] != "unknown":
                label_parts.append(f"bottom:{r['lower_color']}")
            results.append({
                "entity": "person",
                "id": r["id"],
                "video_id": r["video_id"],
                "file_path": r.get("file_path", ""),
                "track_id": r["track_id"],
                "label": f"person #{r['track_id']} ({', '.join(label_parts)})",
                "gender": r.get("gender"),
                "age_est": r.get("age_est"),
                "upper_color": r.get("upper_color"),
                "lower_color": r.get("lower_color"),
                "first_seen_s": r["first_seen_s"],
                "last_seen_s": r["last_seen_s"],
                "frame_count": r["frame_count"],
                "peak_conf": r["peak_conf"],
                "bbox_json": r.get("bbox_json"),
                "bbox_time_s": r.get("bbox_time_s"),
                "bbox_frame": r.get("bbox_frame"),
            })

    # ── Search Objects / Clothing ──
    if search_objects and query.vehicle_type is None and query.gender is None:
        o_rows = db.search_objects(
            video_id=video_id,
            object_type=query.object_type,
            color=query.color,
            time_start=query.time_start,
            time_end=query.time_end,
            ts_from=query.time_range.start_ts if query.time_range else None,
            ts_to=query.time_range.end_ts if query.time_range else None,
        )
        # The object detector writes one row per frame. Collapse those rows into
        # one result per tracked object so the UI is not flooded with duplicates.
        grouped: dict[tuple, dict[str, Any]] = {}
        for r in o_rows:
            obj_type = _clean_object_type(r.get("object_type"), r.get("color"))
            key = (r["video_id"], r.get("track_id"), obj_type, r.get("color"))
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = {
                    "entity": "object",
                    "id": r["id"],
                    "video_id": r["video_id"],
                    "file_path": r.get("file_path", ""),
                    "track_id": r.get("track_id"),
                    "label": f"{r.get('color', '?')} {obj_type}" + (
                        f" [Person #{r['associated_person_track']}]" if r.get("associated_person_track") else ""
                    ),
                    "type": obj_type,
                    "color": r.get("color"),
                    "associated_person_track": r.get("associated_person_track"),
                    "first_seen_s": r.get("first_seen_s"),
                    "last_seen_s": r.get("last_seen_s"),
                    "frame_count": r.get("frame_count", 0) or 0,
                    "peak_conf": r.get("peak_conf", 0.0) or 0.0,
                    "bbox_json": r.get("bbox_json"),
                    "bbox_time_s": r.get("bbox_time_s"),
                    "bbox_frame": r.get("bbox_frame"),
                }
            else:
                existing["frame_count"] += r.get("frame_count", 0) or 0
                if (r.get("last_seen_s") or 0) > (existing.get("last_seen_s") or 0):
                    existing["last_seen_s"] = r.get("last_seen_s")
                if (r.get("first_seen_s") or 0) < (existing.get("first_seen_s") or float("inf")):
                    existing["first_seen_s"] = r.get("first_seen_s")
                has_better_box = (
                    r.get("bbox_json")
                    and not existing.get("bbox_json")
                )
                has_better_conf = (
                    (r.get("peak_conf") or 0) > (existing.get("peak_conf") or 0)
                )
                if has_better_conf:
                    existing["peak_conf"] = r.get("peak_conf")
                if has_better_box or has_better_conf:
                    existing["bbox_json"] = r.get("bbox_json")
                    existing["bbox_time_s"] = r.get("bbox_time_s")
                    existing["bbox_frame"] = r.get("bbox_frame")
        results.extend(grouped.values())

    # ── Search Gate Events directly if event query ──
    if query.event_type:
        e_rows = db.search_events(
            video_id=video_id,
            event_type=query.event_type,
            time_start=query.time_start,
            time_end=query.time_end,
            ts_from=query.time_range.start_ts if query.time_range else None,
            ts_to=query.time_range.end_ts if query.time_range else None,
        )
        for er in e_rows:
            results.append({
                "entity": er["entity_type"],
                "id": er["id"],
                "video_id": er["video_id"],
                "file_path": er.get("file_path", ""),
                "track_id": er["track_id"],
                "label": f"{er['entity_type']} #{er['track_id']} ({er['event_type']} @ {er['line_name']})",
                "event_type": er["event_type"],
                "first_seen_s": er["timestamp_s"],
                "last_seen_s": er["timestamp_s"],
                "peak_conf": er.get("confidence", 0.0),
                "bbox_json": er.get("bbox_json"),
                "bbox_time_s": er["timestamp_s"],
                "bbox_frame": None,
            })

    results = _deduplicate_track_results(results)

    # Sort results by video and timestamp
    results.sort(key=lambda x: (x.get("video_id", ""), x.get("first_seen_s") or 0.0))

    # Cut clips if requested
    if clip_output_dir is not None:
        clip_output_dir.mkdir(parents=True, exist_ok=True)
        for item in results:
            src_file = item.get("file_path")
            t_start = item.get("first_seen_s")
            t_end = item.get("last_seen_s")
            if src_file and Path(src_file).is_file() and t_start is not None and t_end is not None:
                start_cut = max(0.0, float(t_start) - clip_pad_s)
                end_cut = float(t_end) + clip_pad_s
                clip_name = f"{item['entity']}_{item.get('track_id', item['id'])}_{int(start_cut)}s_{int(end_cut)}s.mp4"
                target_path = clip_output_dir / clip_name
                try:
                    final_clip = cut_clip(src_file, start_cut, end_cut, target_path)
                    item["clip_path"] = str(final_clip.resolve())
                except Exception as exc:
                    item["clip_path"] = None
                    item["clip_error"] = str(exc)

    return results


def print_results_table(results: list[dict[str, Any]], cut_clips: bool) -> None:
    """Print results nicely formatted to terminal."""
    if not results:
        print("\n[INFO] No matching tracks found.")
        return

    print(f"\nFound {len(results)} matching track(s):\n")
    print(f"{'#':<4} {'Entity':<10} {'Video':<18} {'Track':<7} {'Time Interval':<17} {'Description':<32} {'Conf':<6}")
    print("-" * 100)

    for i, r in enumerate(results, start=1):
        t_str = f"{format_time_s(r.get('first_seen_s'))} - {format_time_s(r.get('last_seen_s'))}"
        conf_str = f"{r.get('peak_conf', 0.0):.2f}"
        desc = r.get("label", "")[:31]
        v_name = Path(r.get("video_id", "")).stem[:17]
        tid_str = str(r.get("track_id", "-"))
        print(f"{i:<4} {r['entity']:<10} {v_name:<18} #{tid_str:<6} {t_str:<17} {desc:<32} {conf_str:<6}")
        if cut_clips and r.get("clip_path"):
            print(f"     └─ Saved Clip: {r['clip_path']}")

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search CCTV tracked database for persons, vehicles, and objects by text query or attributes."
    )
    parser.add_argument("query", nargs="?", default=None, help="Natural language query string (e.g. 'red car', 'female person')")
    parser.add_argument("--db", type=Path, default=Path("outputs/cctv.db"), help="Path to SQLite database")
    parser.add_argument("--video-id", type=str, default=None, help="Filter by video ID")
    parser.add_argument("--entity", choices=["person", "vehicle", "object"], default=None, help="Target entity filter")
    parser.add_argument("--color", type=str, default=None, help="Target color (e.g. red, blue, white, black)")
    parser.add_argument("--vehicle-type", type=str, default=None, help="Vehicle type (car, motorcycle, bus, truck, bicycle)")
    parser.add_argument("--gender", choices=["male", "female", "unknown"], default=None, help="Person gender")
    parser.add_argument("--age-min", type=float, default=None, help="Minimum estimated age")
    parser.add_argument("--age-max", type=float, default=None, help="Maximum estimated age")
    parser.add_argument("--upper-color", type=str, default=None, help="Upper clothing color")
    parser.add_argument("--lower-color", type=str, default=None, help="Lower clothing color")
    parser.add_argument("--object-type", type=str, default=None, help="Object/clothing type (e.g. shirt, bag, shoe)")
    parser.add_argument("--event-type", choices=["entry", "exit"], default=None, help="Gate line event filter (entry or exit)")
    parser.add_argument("--time-start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--time-end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--cut-clips", action="store_true", help="Extract and save video clips for matching tracks")
    parser.add_argument("--clip-dir", type=Path, default=Path("outputs/clips"), help="Directory to save extracted clips")
    parser.add_argument("--clip-pad", type=float, default=30.0, help="Padding in seconds around detected event (default: 30.0s for ±30s window)")
    parser.add_argument("--vector", action="store_true", help="Use SigLIP 2 contextual vector similarity search")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--list-videos", action="store_true", help="List all indexed videos and summary stats")

    # ── Image / similar-to / absolute-time search (Phase 5) ──
    parser.add_argument("--image", nargs="+", default=None, metavar="IMG",
                        help="Query image paths (1–5); finds similar indexed tracks")
    parser.add_argument("--similar-to", type=str, default=None, metavar="ID",
                        help="entity@video:track_id (e.g. person@cam1_10s:4) — find similar indexed tracks")
    parser.add_argument("--mode", choices=["query", "image", "similar", "hybrid"], default="query",
                        help="Search mode: query (text/SQL), image (--image), similar (--similar-to), hybrid (text+SQL+vector)")
    parser.add_argument("--min-sim", type=float, default=0.0, help="Minimum similarity for image/similar search (0–1; calibrate with tools/sim_calibrate.py)")
    parser.add_argument("--top-k", type=int, default=10, help="Max results for image/similar/vector modes")
    parser.add_argument("--camera", type=str, default=None, help="Filter by camera_id (absolute-time search)")
    parser.add_argument("--anchor", choices=["latest_indexed", "wall"], default=None,
                        help="What 'now/pichle' resolve against: end of indexed footage or wall clock (default: config)")
    parser.add_argument("--since", type=str, default=None, metavar="PHRASE",
                        help="Natural-language start bound, e.g. \"2 ghante\", \"aaj subah\"")
    parser.add_argument("--from", dest="time_from", type=str, default=None, metavar="DATETIME",
                        help="Absolute start ISO \"YYYY-MM-DD HH:MM\"")
    parser.add_argument("--to", dest="time_to", type=str, default=None, metavar="DATETIME",
                        help="Absolute end ISO \"YYYY-MM-DD HH:MM\"")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.db.is_file():
        print(f"[ERROR] Database file not found: {args.db}", file=sys.stderr)
        print("Run detection pipelines first to index videos.", file=sys.stderr)
        return 1

    db = CCTVDatabase(args.db)

    try:
        # Check if user just wants video summary
        if args.list_videos:
            videos = db.get_all_videos()
            if not videos:
                print("No indexed videos found.")
                return 0
            print(f"\nIndexed Videos ({len(videos)}):\n")
            print(f"{'Video ID':<25} {'Camera':<12} {'Duration':<10} {'Status':<12} {'Counts (P/V/O)'}")
            print("-" * 80)
            for v in videos:
                summary = db.get_video_summary(v["video_id"])
                p_c = summary["persons"]
                v_c = summary["vehicles"]
                o_c = summary["objects"]
                dur = f"{v.get('duration_s', 0):.1f}s" if v.get("duration_s") else "--"
                print(f"{v['video_id']:<25} {v.get('camera_id', ''):<12} {dur:<10} {v.get('status', ''):<12} P:{p_c} | V:{v_c} | O:{o_c}")
            print()
            return 0

        # Parse query (resolve the NL-time anchor from indexed footage first)
        cfg = load_time_config()
        anchor_choice = args.anchor or cfg.get("anchor", "latest_indexed")
        lo_ts, hi_ts = db.get_indexed_time_bounds()
        anchor_ts = hi_ts if (anchor_choice == "latest_indexed" and hi_ts) else _time.time()
        query = parse_natural_language_query(args.query or "", anchor_ts=anchor_ts)
        if args.query and query.time_range:
            lo = query.time_range.start_ts
            hi = query.time_range.end_ts
            print(f"[TIME] {query.time_range.label} "
                  f"({lo if lo else 'open'} .. {hi if hi else 'open'}, tz={query.time_range.tz})"
                  + (f" | assumptions: {query.time_range.assumptions}" if query.time_range.assumptions else ""))
        else:
            query = ParsedQuery(raw_query="")

        # CLI overrides
        if args.entity:
            query.target_entity = args.entity
        if args.color:
            query.color = args.color
        if args.vehicle_type:
            query.vehicle_type = args.vehicle_type
        if args.gender:
            query.gender = args.gender
        if args.object_type:
            query.object_type = args.object_type
        if args.event_type:
            query.event_type = args.event_type
        if args.time_start is not None:
            query.time_start = args.time_start
        if args.time_end is not None:
            query.time_end = args.time_end

        # ── Absolute-time CLI window (--since / --from / --to) ──
        def _iso_local_ts(s: str) -> float:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ))
            return dt.timestamp()

        ts_from, ts_to = None, None
        if args.since:
            _c, tr_since = parse_time(args.since, now=anchor_ts, tz=DEFAULT_TZ)
            if tr_since and tr_since.start_ts is not None:
                ts_from = tr_since.start_ts
                print(f"[TIME] --since → {tr_since.label}")
        if args.time_from:
            ts_from = _iso_local_ts(args.time_from)
        if args.time_to:
            ts_to = _iso_local_ts(args.time_to)
        if ts_from is not None or ts_to is not None:
            query.time_range = TimeRange(
                start_ts=ts_from if ts_from is not None else (
                    query.time_range.start_ts if query.time_range else None),
                end_ts=ts_to if ts_to is not None else (
                    query.time_range.end_ts if query.time_range else None),
                label="CLI time window", tz=DEFAULT_TZ,
                source_phrase="cli", assumptions=[],
            )

        # ── Image / similar-to vector modes ──
        if args.image or args.similar_to:
            request = SearchRequest(
                query_text=args.query,
                query_images=args.image or [],
                entity=args.entity or "auto",
                top_k=args.top_k,
                min_sim=args.min_sim,
                video_id=args.video_id,
                camera_id=args.camera,
                ts_from=query.time_range.start_ts if query.time_range else None,
                ts_to=query.time_range.end_ts if query.time_range else None,
            )
            if args.similar_to:
                m = re.match(r"^(\w+)@([\w\-\.]+):(\d+)$", args.similar_to)
                if not m:
                    print("[ERROR] --similar-to expects entity@video_id:track_id", file=sys.stderr)
                    return 1
                resp = similar_to_track(db, m.group(1), m.group(2), int(m.group(3)), request)
            elif args.query:
                resp = find_similar_image_and_text(db, request)
            else:
                resp = image_search(db, request)
            if resp.error:
                print(f"[ERROR] {resp.error}", file=sys.stderr)
                return 1
            from image_query import _attach_track_info
            if resp.entity:
                _attach_track_info(db, resolved_entity=resp.entity, hits=resp.hits)
            if args.camera:
                cam_map = {r["video_id"]: r["camera_id"] for r in db.get_all_videos()}
                resp.hits = [h for h in resp.hits if cam_map.get(h.get("video_id")) == args.camera]
            if args.json:
                print(json.dumps({"method": resp.method, "entity": resp.entity, "hits": resp.hits},
                                 indent=2, default=str))
            else:
                print(f"[{resp.method.upper()}] entity={resp.entity or 'any'} hits={len(resp.hits)}")
                for i, h in enumerate(resp.hits, start=1):
                    print(f"{i:>3}. {h.get('label', '?'):<34} sim={h.get('similarity', 0.0):.3f} "
                          f"cam={h.get('camera_id') or '-'} track#{h.get('track_id')} "
                          f"video={h.get('video_id')}")
            return 0

        clip_dir = args.clip_dir if args.cut_clips else None

        if args.vector and query.raw_query:
            try:
                from common.embed import batch_cosine_similarity, embed_text
                import numpy as np

                emb_rows = db.get_all_embeddings(video_id=args.video_id)
                if emb_rows:
                    q_vec = embed_text(query.raw_query)
                    if q_vec is not None:
                        valid_rows = []
                        vecs = []
                        for r in emb_rows:
                            v = np.frombuffer(r["embedding"], dtype=np.float32)
                            if v.shape[0] > 0:
                                vecs.append(v)
                                valid_rows.append(r)
                        if vecs:
                            scores = batch_cosine_similarity(q_vec, np.stack(vecs))
                            top_indices = np.argsort(-scores)[:50]
                            print(f"[VECTOR] SigLIP 2 query vector computed. Scored {len(vecs)} embeddings.")
            except Exception as e:
                print(f"[WARN] Vector search fallback: {e}", file=sys.stderr)

        results = execute_search(
            db=db,
            query=query,
            video_id=args.video_id,
            age_min=args.age_min,
            age_max=args.age_max,
            upper_color=args.upper_color,
            lower_color=args.lower_color,
            clip_output_dir=clip_dir,
            clip_pad_s=args.clip_pad,
        )

        if args.camera:
            cam_map = {r["video_id"]: r["camera_id"] for r in db.get_all_videos()}
            results = [r for r in results if cam_map.get(r.get("video_id")) == args.camera]

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"[QUERY] Text: '{query.raw_query}' | Entity: {query.target_entity or 'ALL'} | "
                  f"Color: {query.color or 'ANY'} | Gender: {query.gender or 'ANY'} | "
                  f"Type: {query.vehicle_type or query.object_type or 'ANY'}")
            print_results_table(results, cut_clips=args.cut_clips)

        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
