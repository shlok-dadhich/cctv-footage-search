"""
Shared color estimation utilities.
Extracts and consolidates color estimation code used by vehicle and object pipelines.
"""
from __future__ import annotations

import cv2
import numpy as np

# Reference BGR colours for perceptual CIE LAB lookup
NAMED_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "black": (22, 22, 22),
    "dark gray": (75, 75, 75),
    "gray": (140, 140, 140),
    "white": (242, 242, 242),
    "red": (35, 35, 215),
    "orange": (25, 130, 240),
    "yellow": (35, 220, 240),
    "green": (45, 165, 45),
    "cyan": (205, 205, 35),
    "blue": (210, 85, 30),
    "navy": (115, 40, 20),
    "purple": (160, 45, 135),
    "pink": (180, 135, 245),
    "brown": (40, 75, 130),
    "beige": (175, 210, 230),
}

_CHROMATIC_NAMES = [
    k for k in NAMED_COLORS_BGR if k not in ("black", "dark gray", "gray", "white")
]
_CHROMATIC_BGR = np.array(
    [NAMED_COLORS_BGR[k] for k in _CHROMATIC_NAMES], dtype=np.uint8
).reshape(-1, 1, 3)
_CHROMATIC_LAB = (
    cv2.cvtColor(_CHROMATIC_BGR, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
)


def estimate_color_lab(
    frame_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    inner_ratio: float = 0.22,
    reject_skin: bool = True,
) -> str:
    """
    Estimate dominant color of a region using CIE LAB space.
    Works for clothing, vehicles, and general objects.

    Args:
        frame_bgr: Full frame in BGR
        box_xyxy: Bounding box [x1, y1, x2, y2]
        inner_ratio: Margin ratio to crop inward (avoids background)
        reject_skin: Whether to filter out skin-tone pixels

    Returns:
        Color name string (e.g. "red", "blue", "white", "unknown")
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return "unknown"

    # Central crop to strip background edges
    px, py = int(bw * inner_ratio), int(bh * inner_ratio)
    cx1, cy1 = x1 + px, y1 + py
    cx2, cy2 = x2 - px, y2 - py
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    roi_bgr = frame_bgr[cy1:cy2, cx1:cx2]
    if roi_bgr.size == 0:
        return "unknown"
    if roi_bgr.shape[0] > 48 or roi_bgr.shape[1] > 48:
        roi_bgr = cv2.resize(roi_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    pixels_bgr = roi_bgr.reshape(-1, 3)

    if len(pixels_bgr) == 0:
        return "unknown"

    # YCrCb skin-pixel filter
    if reject_skin and len(pixels_bgr) >= 16:
        ycrcb = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        y_val, cr, cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]
        skin_mask = (
            (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (y_val >= 45) & (y_val <= 235)
        )
        non_skin = pixels_bgr[~skin_mask]
        if len(non_skin) >= max(8, int(0.25 * len(pixels_bgr))):
            pixels_bgr = non_skin

    lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    l_chan = lab[:, 0]
    a_shift = lab[:, 1] - 128.0
    b_shift = lab[:, 2] - 128.0
    chroma = np.hypot(a_shift, b_shift)

    chromatic_mask = (chroma >= 14.0) & (l_chan >= 35.0) & (l_chan <= 245.0)
    chromatic_ratio = float(np.count_nonzero(chromatic_mask)) / float(len(lab))

    if chromatic_ratio < 0.22:
        med_l = float(np.median(l_chan))
        if med_l < 58.0:
            return "black"
        if med_l < 110.0:
            return "dark gray"
        if med_l > 205.0:
            return "white"
        return "gray"

    chroma_lab = lab[chromatic_mask]
    if len(chroma_lab) >= 12:
        q25 = np.percentile(chroma_lab, 25, axis=0)
        q75 = np.percentile(chroma_lab, 75, axis=0)
        c0, c1 = q25.copy(), q75.copy()
        mask0 = np.ones(len(chroma_lab), dtype=bool)
        for _ in range(5):
            d0 = np.sum((chroma_lab - c0) ** 2, axis=1)
            d1 = np.sum((chroma_lab - c1) ** 2, axis=1)
            mask0 = d0 <= d1
            if np.any(mask0) and np.any(~mask0):
                c0 = chroma_lab[mask0].mean(axis=0)
                c1 = chroma_lab[~mask0].mean(axis=0)
            else:
                break
        dominant_lab = c0 if np.count_nonzero(mask0) >= np.count_nonzero(~mask0) else c1
    else:
        dominant_lab = np.median(chroma_lab, axis=0)

    diff = _CHROMATIC_LAB - dominant_lab[None, :]
    dists = np.sqrt((0.75 * diff[:, 0]) ** 2 + (1.35 * diff[:, 1]) ** 2 + (1.35 * diff[:, 2]) ** 2)
    return _CHROMATIC_NAMES[int(np.argmin(dists))]


def estimate_person_colors(
    frame_bgr: np.ndarray,
    box_xyxy: np.ndarray,
) -> tuple[str, str]:
    """
    Estimate upper body and lower body colors for a person.

    Splits the person bounding box into upper 45% and lower 45%
    (skipping the middle 10% which is often belt/waist area).

    Returns:
        (upper_color, lower_color) tuple
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in box_xyxy]
    h = y2 - y1
    if h < 20:
        return "unknown", "unknown"

    # Upper body: top 45% (typically torso/shirt area, skip head ~15%)
    head_offset = int(h * 0.15)
    upper_y2 = y1 + int(h * 0.50)
    upper_box = np.array([x1, y1 + head_offset, x2, upper_y2], dtype=np.float32)
    upper_color = estimate_color_lab(frame_bgr, upper_box, inner_ratio=0.15, reject_skin=True)

    # Lower body: bottom 45% (typically pants/legs area)
    lower_y1 = y1 + int(h * 0.55)
    lower_box = np.array([x1, lower_y1, x2, y2], dtype=np.float32)
    lower_color = estimate_color_lab(frame_bgr, lower_box, inner_ratio=0.15, reject_skin=False)

    return upper_color, lower_color
