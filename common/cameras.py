"""Camera timezone configuration loading.

Cameras are configured in config/cameras.json as {"camera_id": {"tz": "Asia/Kolkata"}}.
Internal timestamps are UTC epoch seconds; the camera tz is used only when parsing
local wall-clock input (e.g. --start-time) and when displaying results.
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT_DIR = Path(__file__).resolve().parent.parent
CAMERAS_CONFIG = ROOT_DIR / "config" / "cameras.json"
DEFAULT_TZ_NAME = "Asia/Kolkata"


def load_camera_tz_map() -> dict[str, str]:
    """Load camera_id -> IANA tz name map. Missing file or syntax error -> {}."""
    try:
        raw = json.loads(Path(CAMERAS_CONFIG).read_text(encoding="utf-8"))
        return {cid: str(cfg["tz"]) for cid, cfg in raw.items() if isinstance(cfg, dict) and "tz" in cfg}
    except (OSError, ValueError, TypeError):
        return {}


def camera_tz_name(camera_id: str | None, tz_map: dict[str, str] | None = None) -> str:
    """Resolve the IANA tz name for a camera, defaulting to Asia/Kolkata."""
    if tz_map is None:
        tz_map = load_camera_tz_map()
    return tz_map.get(camera_id or "", DEFAULT_TZ_NAME)


def get_camera_zone(camera_id: str | None, tz_map: dict[str, str] | None = None) -> ZoneInfo:
    """Resolve the ZoneInfo for a camera with a sane fallback."""
    name = camera_tz_name(camera_id, tz_map)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return ZoneInfo("UTC")
