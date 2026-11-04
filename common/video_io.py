"""Shared video I/O utilities — VideoWriter with codec fallback and clip cutter."""
from __future__ import annotations

import os
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

_BROWSER_CODECS = {"h264", "vp8", "vp9", "av1", "theora"}


def create_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> tuple[cv2.VideoWriter, Path]:
    """Open cv2.VideoWriter with safe codec fallback (mp4v -> avc1 -> XVID)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        ("mp4v", output_path.with_suffix(".mp4")),
        ("avc1", output_path.with_suffix(".mp4")),
        ("XVID", output_path.with_suffix(".avi")),
    ]
    for fourcc_str, candidate_path in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(candidate_path), fourcc, float(fps), (width, height))
        if writer.isOpened():
            if candidate_path != output_path:
                print(f"[WARN] Codec fallback switched output file to: {candidate_path}", file=sys.stderr)
            return writer, candidate_path
        writer.release()
    raise RuntimeError(f"Failed to initialize VideoWriter for {output_path} with codecs mp4v, avc1, or XVID.")


def _av_available() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def probe_creation_time(path: str | Path) -> float | None:
    """Return the container creation time as UTC epoch seconds, or None.

    Only container-level metadata is considered — absent metadata (very common
    for plain .avi camera exports) yields None so callers fall through to
    mtime-based estimation. Returns None when PyAV is unavailable too.
    """
    if not _av_available():
        return None
    try:
        import av
        with av.open(str(path)) as container:
            creation = container.creation_time
            if creation:
                return float(creation)
    except Exception:
        return None
    return None


def video_codec_name(path: str | Path) -> str | None:
    """Return the video codec short name (e.g. 'h264', 'mpeg4') or None."""
    try:
        import av
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            return (stream.codec_context.name or "").lower() if stream else None
    except Exception:
        return None


def is_browser_playable_video(path: str | Path) -> bool:
    """Return True when the clip uses a codec browsers can play in <video>."""
    return (video_codec_name(path) or "") in _BROWSER_CODECS


def _transcode_with_pyav(
    source_path: Path,
    output_path: Path,
    fps: float,
    out_w: int,
    out_h: int,
) -> Path:
    """Transcode a video to H.264/yuv420p so Streamlit/browsers can play it."""
    import av

    out_w -= out_w % 2
    out_h -= out_h % 2
    rate = Fraction(fps).limit_denominator(1000000)

    with av.open(str(source_path)) as input_container:
        in_stream = next((s for s in input_container.streams if s.type == "video"), None)
        if in_stream is None:
            raise RuntimeError(f"No video stream in {source_path}")
        with av.open(str(output_path), mode="w", format="mp4", options={"movflags": "+faststart"}) as output_container:
            out_stream = output_container.add_stream("libx264", rate=rate)
            out_stream.width = out_w
            out_stream.height = out_h
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "23", "preset": "veryfast"}
            for frame in input_container.decode(in_stream):
                # Rebuild the frame from raw pixels; some legacy MPEG-4 files carry
                # time-base metadata that makes libx264 refuse to mux.
                pixels = frame.to_ndarray(format="bgr24")
                video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="bgr24")
                if (out_w, out_h) != (frame.width, frame.height):
                    video_frame = video_frame.reformat(width=out_w, height=out_h, format="yuv420p")
                else:
                    video_frame = video_frame.reformat(format="yuv420p")
                # Dropping stale pts avoids muxer EINVAL on Windows.
                video_frame.pts = None
                for packet in out_stream.encode(video_frame):
                    output_container.mux(packet)
            for packet in out_stream.encode():
                output_container.mux(packet)
    return output_path


def ensure_browser_playable_video(path: str | Path, max_height: int | None = None) -> Path:
    """Transcode an existing clip in place if its codec cannot play in browsers."""
    path = Path(path)
    if not path.is_file() or is_browser_playable_video(path):
        return path
    if not _av_available():
        return path

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return path
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        return path
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    scale = min(1.0, max_height / height) if (max_height and height > max_height) else 1.0
    out_w, out_h = int(width * scale), int(height * scale)

    temp_path = path.with_name(f"{path.stem}.h264tmp{path.suffix}")
    try:
        _transcode_with_pyav(path, temp_path, fps, out_w, out_h)
        os.replace(temp_path, path)
    except Exception as exc:
        print(f"[WARN] Could not make clip browser-playable ({path}): {exc}", file=sys.stderr)
        temp_path.unlink(missing_ok=True)
    return path


def cut_clip(
    source_path: str | Path,
    start_s: float,
    end_s: float,
    output_path: str | Path,
    max_height: int = 720,
    browser_compatible: bool = True,
) -> Path:
    """Cut a video clip from start_s to end_s using OpenCV (no ffmpeg dependency).

    Handles seeking accurately via frame numbers. When PyAV + libx264 are
    available the clip is encoded directly as browser-playable H.264.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale down if needed
    scale = min(1.0, max_height / height) if height > max_height else 1.0
    out_w = int(width * scale)
    out_h = int(height * scale)

    start_frame = max(0, int(start_s * fps))
    end_frame = int(end_s * fps)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    use_pyav = browser_compatible and _av_available()
    temp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if use_pyav:
        try:
            import av

            out_w -= out_w % 2
            out_h -= out_h % 2
            with av.open(str(temp_path), mode="w", format="mp4", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000000))
                stream.width = out_w
                stream.height = out_h
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "veryfast"}
                try:
                    current_frame = start_frame
                    while current_frame <= end_frame:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        if (out_w, out_h) != (width, height):
                            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                        video_frame = video_frame.reformat(format="yuv420p")
                        video_frame.pts = None
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                        current_frame += 1
                finally:
                    cap.release()
                    # Flush the encoder before closing the container.
                    for packet in stream.encode():
                        container.mux(packet)
            os.replace(temp_path, output_path)
            return output_path
        except Exception as exc:
            print(f"[WARN] H.264 clip encode failed, falling back to OpenCV: {exc}", file=sys.stderr)
            temp_path.unlink(missing_ok=True)
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {source_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer, final_path = create_video_writer(output_path, fps, out_w, out_h)

    try:
        current_frame = start_frame
        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            current_frame += 1
    finally:
        cap.release()
        writer.release()

    return final_path


def extract_frame(
    source_path: str | Path,
    timestamp_s: float,
    bbox: list[float] | None = None,
    label: str | None = None,
    frame_index: int | None = None,
) -> np.ndarray | None:
    """Extract a frame at timestamp_s (or an exact frame index) with optional bbox."""
    source_path = Path(source_path)
    if not source_path.is_file():
        return None

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    if frame_index is not None and frame_index >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    else:
        target_frame = max(0, int(timestamp_s * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback to msec seek
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s * 1000.0))
            ok, frame = cap.read()
            if not ok or frame is None:
                return None

        # Draw bbox if provided
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if label:
                    text = str(label)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.55
                    thick = 1
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x1 + 3, y1 - 4), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        return frame
    finally:
        cap.release()
