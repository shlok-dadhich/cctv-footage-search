"""
common/search_images.py - small helper to mimic a "search images" path using DB + crops.

Main idea:
- video compare / search can happen DB-first (query_norm + SQL + hybrid)
- batch-across-video work ka ek chhota entry point is module ho sakta hai
- ye sirf ek organizing helper hai, core logic search.py / ingest.py se alag hai
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def summarize_image_findings(db_path: str | Path, video_id: str | None = None) -> dict[str, Any]:
    """
    Return a compact summary over detection + embedding presence for use in
    batch / cross-video review flows.
    """
    from common.db import CCTVDatabase

    db = CCTVDatabase(db_path)
    try:
        counts = db.get_total_counts()
        summary: dict[str, Any] = {
            "counts": counts,
            "video_id": video_id,
        }
        if video_id:
            summary["video_summary"] = db.get_video_summary(video_id)
            summary["crops"] = db.get_crops_for_video(video_id)
        return summary
    finally:
        db.close()


def dataset_video_list(db_path: str | Path) -> list[dict[str, Any]]:
    """List all indexed videos with summary stats."""

    from common.db import CCTVDatabase

    db = CCTVDatabase(db_path)
    try:
        rows = db.get_all_videos()
        out: list[dict[str, Any]] = []
        for r in rows:
            summary = db.get_video_summary(r["video_id"])
            out.append({**r, "summary": summary})
        return out
    finally:
        db.close()
