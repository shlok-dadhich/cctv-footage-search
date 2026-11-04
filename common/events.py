"""
common/events.py — Virtual Gate Line Crossing & Directional Event Engine.

Computes directional vector cross-products between tracked object trajectories
and user-defined virtual gate lines (e.g. entry, exit).
Includes a refractory cooldown buffer to prevent false multiple crossings caused
by bounding box jitter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class CrossingEvent:
    track_id: int
    entity_type: str  # "person" | "vehicle" | "object"
    event_type: str   # "entry" | "exit" | "crossing"
    line_name: str
    timestamp_s: float
    confidence: float
    bbox: list[float]
    point: Tuple[float, float]


class VirtualGateLine:
    """A directional tripwire defined by a 2D line segment (p1 -> p2).

    Convention:
        - Line is directed from p1 to p2.
        - Moving from LEFT of vector (p1 -> p2) to RIGHT: "entry" (or "in").
        - Moving from RIGHT to LEFT: "exit" (or "out").
    """

    def __init__(
        self,
        name: str,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        refractory_sec: float = 2.0,
    ) -> None:
        self.name = name
        self.p1 = Point(float(p1[0]), float(p1[1]))
        self.p2 = Point(float(p2[0]), float(p2[1]))
        self.refractory_sec = refractory_sec

        # Track previous side for each track: track_id -> side (+1 or -1)
        self._previous_side: dict[int, int] = {}
        # Track last crossing timestamp: track_id -> timestamp_s
        self._last_crossing_time: dict[int, float] = {}
        # Track previous point: track_id -> Point
        self._previous_point: dict[int, Point] = {}

    def _cross_product_sign(self, pt: Point) -> int:
        """Compute the signed side of pt relative to directed line p1 -> p2.

        val = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        Returns:
            +1 if pt is on left of line
            -1 if pt is on right of line
             0 if collinear
        """
        val = (self.p2.x - self.p1.x) * (pt.y - self.p1.y) - (self.p2.y - self.p1.y) * (pt.x - self.p1.x)
        if abs(val) < 1e-4:
            return 0
        return 1 if val > 0 else -1

    @staticmethod
    def get_anchor_point(bbox_xyxy: list[float] | np.ndarray, anchor: str = "bottom_center") -> Point:
        """Compute representative anchor point for bounding box."""
        x1, y1, x2, y2 = bbox_xyxy[:4]
        if anchor == "bottom_center":
            return Point(float((x1 + x2) / 2.0), float(y2))
        elif anchor == "center":
            return Point(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
        return Point(float((x1 + x2) / 2.0), float(y2))

    def update(
        self,
        track_id: int,
        entity_type: str,
        bbox_xyxy: list[float] | np.ndarray,
        timestamp_s: float,
        confidence: float = 0.0,
        anchor: str = "bottom_center",
    ) -> Optional[CrossingEvent]:
        """Update tracker position and return a CrossingEvent if a crossing occurred.

        Applies refractory debouncing so a single physical crossing generates
        at most one event within refractory_sec.
        """
        curr_pt = self.get_anchor_point(bbox_xyxy, anchor)
        curr_side = self._cross_product_sign(curr_pt)

        if track_id not in self._previous_side:
            if curr_side != 0:
                self._previous_side[track_id] = curr_side
                self._previous_point[track_id] = curr_pt
            return None

        prev_side = self._previous_side[track_id]

        # Check if side changed across the line
        if curr_side != 0 and prev_side != 0 and curr_side != prev_side:
            # Check refractory buffer
            last_time = self._last_crossing_time.get(track_id, -1e9)
            if (timestamp_s - last_time) >= self.refractory_sec:
                # Left -> Right is entry, Right -> Left is exit
                direction = "entry" if (prev_side > 0 and curr_side < 0) else "exit"

                event = CrossingEvent(
                    track_id=track_id,
                    entity_type=entity_type,
                    event_type=direction,
                    line_name=self.name,
                    timestamp_s=timestamp_s,
                    confidence=confidence,
                    bbox=[float(v) for v in bbox_xyxy[:4]],
                    point=curr_pt.to_tuple(),
                )

                self._last_crossing_time[track_id] = timestamp_s
                self._previous_side[track_id] = curr_side
                self._previous_point[track_id] = curr_pt
                return event

        if curr_side != 0:
            self._previous_side[track_id] = curr_side
            self._previous_point[track_id] = curr_pt
        return None

    def cleanup_inactive(self, active_track_ids: set[int]) -> None:
        """Remove state for tracks that have ended."""
        for tid in list(self._previous_side.keys()):
            if tid not in active_track_ids:
                self._previous_side.pop(tid, None)
                self._previous_point.pop(tid, None)
                self._last_crossing_time.pop(tid, None)
