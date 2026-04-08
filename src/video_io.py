"""
Video I/O utilities: reading frames, writing output, and detecting I-frames.
"""
import cv2
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def read_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    gray: bool = False,
) -> Tuple[List, float]:
    """
    Read all frames (up to max_frames) from a video file.

    Returns
    -------
    frames : list of np.ndarray  (BGR uint8)
    fps    : float
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return frames, fps


def write_video(
    frames: List,
    output_path: str,
    fps: float = 30.0,
) -> None:
    """Write a list of BGR frames to an mp4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    out.release()


def get_iframe_positions(video_path: str) -> List[int]:
    """
    Use ffprobe to extract the coded picture numbers of I-frames.

    Returns an empty list if ffprobe is unavailable or the file has no I-frames.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=pict_type,coded_picture_number",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        data = json.loads(result.stdout)
        return sorted(
            int(f["coded_picture_number"])
            for f in data.get("frames", [])
            if f.get("pict_type") == "I"
        )
    except Exception:
        return []


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info
