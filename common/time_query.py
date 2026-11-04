#!/usr/bin/env python3
"""Natural-language TIME extraction for CCTV searches (Hinglish + English).

Parses time phrases like "pichle 2 ghante", "aaj subah", "kal shaam 6 baje",
"3 se 5 baje", "last 30 minutes", "sunday dopahar" from the raw query and
returns an absolute epoch-second window plus the remaining (clean) query text.

Design rules:
- Pure: no clock or I/O calls. `now` (anchor epoch seconds) and `tz` are injected.
- Windows are inclusive-overlap ready: start_ts <= seen_end and end_ts >= seen_start.
- Future guard: any day-phrases whose end lands after the anchor are shifted one
  day into the past (recorded in `assumptions`) because CCTV footage is historical.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Kolkata"
DEFAULT_POINT_WINDOW_MIN = 15

_CONFIG_CANDIDATES = ("config/time.json", "config/cctv.json")

NUMBER_WORDS = {
    "a": 1, "an": 1, "ek": 1, "one": 1,
    "do": 2, "two": 2, "teen": 3, "three": 3,
    "chaar": 4, "char": 4, "four": 4,
    "paanch": 5, "panch": 5, "five": 5,
    "chhe": 6, "six": 6, "saat": 7, "seven": 7,
    "aath": 8, "eight": 8, "nau": 9, "nine": 9,
    "das": 10, "ten": 10, "gyarah": 11, "eleven": 11,
    "barah": 12, "twelve": 12,
    "bees": 20, "twenty": 20, "tees": 30, "thirty": 30,
    "chalis": 40, "forty": 40, "pachas": 50, "fifty": 50,
}

UNIT_SECONDS = {
    "minute": 60.0, "min": 60.0, "mins": 60.0, "minutes": 60.0,
    "hour": 3600.0, "hrs": 3600.0, "hr": 3600.0, "hours": 3600.0,
    "ghanta": 3600.0, "ghante": 3600.0,
    "day": 86400.0, "days": 86400.0, "din": 86400.0,
    "week": 604800.0, "weeks": 604800.0, "hafte": 604800.0, "hafte_y": 604800.0,
}

# Weekday index, Monday = 0
WEEKDAYS = {
    "monday": 0, "somvaar": 0, "somwar": 0, "somvar": 0,
    "tuesday": 1, "mangalvaar": 1, "mangalwar": 1, "mangal": 1,
    "wednesday": 2, "budhvaar": 2, "budhwar": 2, "budh": 2,
    "thursday": 3, "guruvaar": 3, "guruwar": 3, "guru": 3,
    "friday": 4, "shukravaar": 4, "shukrawar": 4, "shukra": 4,
    "saturday": 5, "shanivaar": 5, "shaniwar": 5, "shani": 5,
    "sunday": 6, "ravivaar": 6, "raviwar": 6, "ravi": 6, "itwaar": 6, "itwar": 6,
}

# Part of day -> (start_hour, end_hour_exclusive)
PARTS_OF_DAY = {
    "subah": (5, 12), "morning": (5, 12),
    "dopahar": (12, 17), "afternoon": (12, 17),
    "shaam": (17, 21), "sham": (17, 21), "evening": (17, 21),
    "raat": (21, 24), "night": (21, 24),
}

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class TimeRange:
    """Absolute UTC-epoch window resolved from a natural-language time phrase."""
    start_ts: Optional[float]
    end_ts: Optional[float]
    label: str
    tz: str
    source_phrase: str
    assumptions: list[str] = field(default_factory=list)


def load_time_config() -> dict:
    """Load time config with defaults (point window, anchor strategy)."""
    cfg = {"point_window_min": DEFAULT_POINT_WINDOW_MIN, "anchor": "latest_indexed"}
    root = Path(__file__).resolve().parent.parent
    for candidate in _CONFIG_CANDIDATES:
        path = root / candidate
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                scope = data.get("time", data)
                if isinstance(scope, dict):
                    if "point_window_min" in scope:
                        cfg["point_window_min"] = int(scope["point_window_min"])
                    if "anchor" in scope:
                        cfg["anchor"] = str(scope["anchor"])
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                pass
            break
    return cfg


def _localize(epoch_or_dt, tz: str) -> datetime:
    zone = ZoneInfo(tz)
    if isinstance(epoch_or_dt, datetime):
        dt = epoch_or_dt
    else:
        dt = datetime.fromtimestamp(float(epoch_or_dt), tz=zone)
    return dt.astimezone(zone)


def _epoch(dt: datetime, tz: str) -> float:
    return dt.astimezone(ZoneInfo(tz)).timestamp()


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _most_recent_weekday(anchor: datetime, target_wd: int) -> datetime:
    delta = (anchor.weekday() - target_wd) % 7
    return anchor - timedelta(days=delta)


def _num_or_int(tok: str) -> int:
    if tok.isdigit():
        return int(tok)
    return NUMBER_WORDS.get(tok.lower(), 0)


def _guard_future(anchor: datetime, start: datetime, end: datetime,
                  assumptions: list[str]) -> tuple[float, float]:
    """Keep a day-based window in the past relative to the anchor.

    - Window entirely after the anchor: shifted one day into the past.
    - Window straddling the anchor: end clipped to the anchor.
    Returns `(start_ts, end_ts)` epoch seconds after applying the guard.
    """
    if start > anchor:
        assumptions.append("time phrase referenced a future moment; shifted one day into the past")
        return (start - timedelta(days=1)).timestamp(), (end - timedelta(days=1)).timestamp()
    if end > anchor:
        assumptions.append("window end clipped to the anchor time")
        return start.timestamp(), anchor.timestamp()
    return start.timestamp(), end.timestamp()


_PART_MERIDIAN = {"subah": "am", "morning": "am",
                  "dopahar": "pm", "afternoon": "pm",
                  "shaam": "pm", "sham": "pm", "evening": "pm",
                  "raat": "pm", "night": "pm"}


def _match_relative_duration(text: str) -> Optional[tuple[str, re.Match, tuple[str, str]]]:
    """... returns (kind, match, (num_tok, unit_tok))"""
    m = re.search(
        r"(?:\bpich(?:h)?le\b|\bpehle\b|\blast\b|\bprevious\b|\bpast\b)\s+"
        r"(\d+|\b" + "|".join(NUMBER_WORDS) + r"\b)\s+"
        r"(" + "|".join(UNIT_SECONDS) + r")\b",
        text, flags=re.IGNORECASE,
    )
    if m:
        return "pre", m, (m.group(1), m.group(2))
    m = re.search(
        r"(\d+|\b" + "|".join(NUMBER_WORDS) + r"\b)\s+"
        r"(" + "|".join(UNIT_SECONDS) + r")\b\s*(?:\bpehle\b|\bago\b)",
        text, flags=re.IGNORECASE,
    )
    if m:
        return "post", m, (m.group(1), m.group(2))
    # Bare "pichle hafte/din/hafte" (singular, no explicit count) -> 1 unit
    m = re.search(
        r"(?:\bpich(?:h)?le\b|\blast\b|\bprevious\b|\bpast\b)\s+"
        r"(" + "|".join(UNIT_SECONDS) + r")\b",
        text, flags=re.IGNORECASE,
    )
    if m:
        return "pre", m, ("1", m.group(1))
    return None


def _infer_meridian(hour: float, anchor: datetime, assumptions: list[str]) -> float:
    if hour >= 13:
        return hour if hour < 24 else 23.999
    if hour == 12:
        return 12.0
    a_h = anchor.hour + anchor.minute / 60.0
    pm = hour + 12
    if abs(pm - a_h) < abs(hour - a_h):
        assumptions.append(f"assumed PM for bare clock hour {int(hour)}")
        return pm
    return hour


def parse_time(text: str, now: float, tz: str = DEFAULT_TZ) -> tuple[str, Optional[TimeRange]]:
    """Extract a time window from `text`.

    Returns `(clean_text, TimeRange | None)`. `now` is the anchor epoch seconds
    resolved by the caller (e.g. end of indexed footage or wall-clock now).
    """
    zone_dt = _localize(now, tz)
    assumptions: list[str] = []
    clean = text

    result: Optional[TimeRange] = None

    # ── 1. Relative durations: "pichle 2 ghante", "last 30 minutes", "2 din pehle"
    rel = _match_relative_duration(clean)
    if rel:
        _kind, m, (num_tok, unit_tok) = rel
        n = _num_or_int(num_tok)
        span = n * UNIT_SECONDS.get(unit_tok.lower(), UNIT_SECONDS[unit_tok.lower()])
        result = TimeRange(
            start_ts=now - span, end_ts=now,
            label=f"last {n} {unit_tok}",
            tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
        )
        clean = clean.replace(m.group(0), " ", 1)

    # ── 2. Open-ended: "<clock> pehle/before" and "<clock> ke baad/after"

    if result is None:
        m = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:baje|bajay)?\s*"
            r".*?\b(?:pehle|before|pahle)\b",
            clean, flags=re.IGNORECASE,
        )
        hour = int(m.group(1)) if m and m.group(1) else None
        if m and hour is not None and hour <= 24:
            if not m.group(3) and 0 <= hour <= 11:
                hour = int(_infer_meridian(float(hour), zone_dt, assumptions))
            elif m.group(3) and m.group(3).lower() == "pm" and hour < 12:
                hour += 12
            elif m.group(3) and m.group(3).lower() == "am" and hour == 12:
                hour = 0
            ref = _day_start(zone_dt) + timedelta(hours=hour, minutes=int(m.group(2) or 0))
            if ref > zone_dt:
                assumptions.append("before-time clamped to end of day before anchor")
            result = TimeRange(
                start_ts=None, end_ts=min(_epoch(ref, tz), now),
                label=f"before {hour:02d}:{m.group(2) or '00'}",
                tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
            )
            clean = clean.replace(m.group(0), " ", 1)

    if result is None:
        m = re.search(
            r"\b(?:baad|after)\b.*?\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            clean, flags=re.IGNORECASE,
        )
        hour = int(m.group(1)) if m and m.group(1) else None
        if m and hour is not None and hour <= 24:
            if m.group(3) and m.group(3).lower() == "pm" and hour < 12:
                hour += 12
            elif not m.group(3) and 0 <= hour <= 11:
                hour = int(_infer_meridian(float(hour), zone_dt, assumptions))
            ref = _day_start(zone_dt) + timedelta(hours=hour, minutes=int(m.group(2) or 0))
            result = TimeRange(
                start_ts=_epoch(ref, tz), end_ts=now,
                label=f"after {hour:02d}:{m.group(2) or '00'}",
                tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
            )
            clean = clean.replace(m.group(0), " ", 1)

    if result is None:
        m = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje|bajay)?\s*.*?\b(?:baad|after|se\s+lekar)\b",
            clean, flags=re.IGNORECASE,
        )
        hour = int(m.group(1)) if m and m.group(1) else None
        if m and hour is not None and hour <= 24 and (m.group(3) or hour >= 13):
            if m.group(3) and m.group(3).lower() == "pm" and hour < 12:
                hour += 12
            elif m.group(3) and m.group(3).lower() == "am" and hour == 12:
                hour = 0
            ref = _day_start(zone_dt) + timedelta(hours=hour, minutes=int(m.group(2) or 0))
            if ref > zone_dt:
                assumptions.append("after-time shifted to previous day")
                ref -= timedelta(days=1)
            result = TimeRange(
                start_ts=_epoch(ref, tz), end_ts=now,
                label=f"after {hour:02d}:{m.group(2) or '00'}",
                tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
            )
            clean = clean.replace(m.group(0), " ", 1)

    # ── 3. Instant "abhi / just now / recently" → point window ending at anchor
    if result is None:
        m = re.search(r"\b(abhi|abhi\s+abhi|just\s+now|recently|abhi\s+thi)\b", clean, flags=re.IGNORECASE)
        if m:
            cfg = load_time_config()
            span = cfg["point_window_min"] * 60.0
            result = TimeRange(
                start_ts=now - span, end_ts=now,
                label=f"last {cfg['point_window_min']} minutes (just now)",
                tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
            )
            clean = clean.replace(m.group(0), " ", 1)

    # ── 4. Clock ranges: "3 se 5 baje", "5pm se 7pm", "between 3 and 5"
    if result is None:
        m = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:baje|bajay)?\s*"
            r"(?:se|se\s+lekar|to|until|till|tak|and|–|—|-\s?)\s+"
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:baje|bajay)?",
            clean, flags=re.IGNORECASE,
        )
        if m and re.search(r"(?:baje|bajay|[ap]m)\b", m.group(0), flags=re.IGNORECASE):
            h1 = int(m.group(1)); min1 = int(m.group(2) or 0)
            h2 = int(m.group(4)); min2 = int(m.group(5) or 0)
            mer1 = (m.group(3) or m.group(6) or "").lower()
            mer2 = (m.group(6) or mer1).lower()
            bare_hours = not (m.group(3) or m.group(6) or
                              re.search(r"\b(?:baje|bajay)\b", m.group(0)))
            if not mer1 and not bare_hours and h1 < 12:
                # "3 se 5 baje" — infer meridian from the anchor hour
                if zone_dt.hour >= 13:
                    h1 += 12
                    h2 = h2 + 12 if h2 <= 12 else h2
                    assumptions.append("assumed PM for bare clock range based on anchor hour")
            if mer1 == "pm" and h1 < 12:
                h1 += 12
            elif mer1 == "am" and h1 == 12:
                h1 = 0
            if mer2 == "pm" and h2 < 12:
                h2 += 12
            elif mer2 == "am" and h2 == 12:
                h2 = 0
            start_h = float(h1) + min1 / 60.0
            end_h = float(h2) + min2 / 60.0
            if end_h <= start_h:
                # Range appears to cross midnight — assume PM start first.
                if h1 < 12 and mer1 != "am":
                    start_h += 12.0
                    assumptions.append("assumed PM for range start to keep end after start")
                if end_h <= start_h:
                    end_h += 24.0
                    assumptions.append("range crossed midnight; end pushed to next day")
            start = _day_start(zone_dt) + timedelta(hours=start_h)
            end = _day_start(zone_dt) + timedelta(hours=end_h)
            start_ts, end_ts = _epoch(start, tz), _epoch(end, tz)
            if end_ts > now:
                if start_ts <= now:
                    # Range partially covered by indexed footage: clip the end.
                    assumptions.append("range end clipped to the anchor time")
                    end_ts = now
                else:
                    # Entire range is in the future: shift one day back.
                    assumptions.append("clock range in future relative to anchor; shifted one day back")
                    start_ts, end_ts = start_ts - _SECONDS_PER_DAY, end_ts - _SECONDS_PER_DAY
                    if end_ts > now:
                        assumptions.append("range end clipped to the anchor time")
                        end_ts = now
            result = TimeRange(
                start_ts=start_ts, end_ts=end_ts,
                label=f"{m.group(1)}:{m.group(2) or '00'}–{m.group(4)}:{m.group(5) or '00'}",
                tz=tz, source_phrase=m.group(0).strip(), assumptions=assumptions,
            )
            clean = clean.replace(m.group(0), " ", 1)

    # ── 5. Day references (aaj/kal/parso/weekday) + optional part of day + clock hour
    if result is None:
        day_m = re.search(
            r"\b(aaj|today|kal|yesterday|parso|parson|"
            + "|".join(WEEKDAYS) + r")\b",
            clean, flags=re.IGNORECASE,
        )
        part_m = re.search(r"\b(" + "|".join(PARTS_OF_DAY) + r")\b", clean, flags=re.IGNORECASE)
        hour_m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|baje|bajay)?", clean, flags=re.IGNORECASE)

        day_tok = day_m.group(1).lower() if day_m else None
        part_key = part_m.group(1).lower() if part_m else None

        if day_tok or part_key or (hour_m and (hour_m.group(3) in ("am", "pm", "baje", "bajay"))):
            # Resolve the reference day
            if day_tok in ("aaj", "today") or day_tok is None:
                day = _day_start(zone_dt)
            elif day_tok in ("kal", "yesterday"):
                day = _day_start(zone_dt) - timedelta(days=1)
                if day_tok == "kal":
                    assumptions.append("kal interpreted as yesterday (past) by default")
            elif day_tok in ("parso", "parson"):
                day = _day_start(zone_dt) - timedelta(days=2)
                assumptions.append("parso interpreted as day before yesterday")
            else:
                day = _day_start(_most_recent_weekday(zone_dt, WEEKDAYS[day_tok]))

            hour_is_clock = hour_m is not None and hour_m.group(3) in ("am", "pm", "baje", "bajay")
            if hour_is_clock and hour_m:
                hh = int(hour_m.group(1))
                mm = int(hour_m.group(2) or 0)
                mer = (hour_m.group(3) or "").lower()
                if mer == "pm" and hh < 12:
                    hh += 12
                elif mer == "am" and hh == 12:
                    hh = 0
                elif mer in ("baje", "bajay") and hh < 12:
                    if part_key:
                        # Part of day provides the meridian (e.g. "subah 6 baje" → 6am)
                        hint = _PART_MERIDIAN[part_key]
                        if hint == "pm" and hh < 12:
                            hh += 12
                    else:
                        hh = int(_infer_meridian(float(hh), zone_dt, assumptions))
                center = _day_start(day) + timedelta(hours=hh, minutes=mm)
                start_ts, _end = _guard_future(zone_dt, center, center, assumptions)
                win = load_time_config()["point_window_min"] * 60.0
                end_ts = _end + win
                label = f"{day_tok or 'today'}" + (f" {part_key}" if part_key else "") + \
                    f" {hh:02d}:{mm:02d} (+{int(win // 60)} min window)"
            elif part_key:
                p_start, p_end = PARTS_OF_DAY[part_key]
                start = day + timedelta(hours=p_start)
                end = day + timedelta(days=1 if p_end == 24 else 0) + timedelta(hours=p_end % 24)
                start_ts, end_ts = _guard_future(zone_dt, start, end, assumptions)
                label = f"{day_tok or 'this'} {part_key} ({p_start:02d}:00–{p_end % 24 if p_end != 24 else 24:02d}:00)"
            else:
                start, end = day, day + timedelta(days=1)
                start_ts, end_ts = _guard_future(zone_dt, start, end, assumptions)
                label = day_tok or "today"

            if result is None:
                result = TimeRange(
                    start_ts=start_ts, end_ts=end_ts,
                    label=label, tz=tz, source_phrase=" ".join(
                        g for g in (day_m.group(0) if day_m else "",
                                    part_m.group(0) if part_m else "",
                                    hour_m.group(0) if hour_m and hour_is_clock else "")
                        if g),
                    assumptions=assumptions,
                )
                if day_m:
                    clean = clean.replace(day_m.group(0), " ", 1)
                if part_m:
                    clean = clean.replace(part_m.group(0), " ", 1)
                if hour_m and hour_is_clock:
                    clean = clean.replace(hour_m.group(0), " ", 1)

    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,.")
    return clean, result
