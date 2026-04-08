#!/usr/bin/env python3
"""
Generate synthetic test videos that do not require downloading from YouTube.

Produces a small set of video clips covering different motion / colour
characteristics similar to the YouTube categories in the original paper:
  static_gradient   ??? slow colour gradient (low motion, like a lecture slide)
  moving_shapes     ??? coloured shapes moving across the frame  (medium motion)
  fast_noise        ??? high-frequency noise (simulates fast action / sports)
  scene_cuts        ??? abrupt brightness/colour changes (tests scene detection)

All clips are written to  data/videos/  by default.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is on the path so we can import config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config   # noqa: E402
from src.video_io import write_video   # noqa: E402


# ?????? Generators ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def make_static_gradient(n_frames: int = 120, h: int = 240, w: int = 320) -> list:
    """Slow hue-shifting colour gradient ??? minimal motion, like lecture slides."""
    frames = []
    for i in range(n_frames):
        hue = int(180 * i / n_frames)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for col in range(w):
            img[:, col, 0] = (hue + col * 180 // w) % 180
            img[:, col, 1] = 200
            img[:, col, 2] = 200
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        frames.append(bgr)
    return frames


def make_moving_shapes(n_frames: int = 120, h: int = 240, w: int = 320) -> list:
    """Solid-colour circles/rectangles moving across a gradient background."""
    frames = []
    bg_base = np.zeros((h, w, 3), dtype=np.uint8)
    Y, X = np.mgrid[:h, :w]
    bg_base[:, :, 0] = (X * 128 // w).astype(np.uint8)
    bg_base[:, :, 1] = (Y * 128 // h).astype(np.uint8)
    bg_base[:, :, 2] = 80

    for i in range(n_frames):
        img = bg_base.copy()
        t   = i / n_frames
        # Moving red circle
        cx  = int(w * t)
        cy  = h // 3
        cv2.circle(img, (cx % w, cy), 30, (0, 0, 220), -1)
        # Bouncing blue rectangle
        rx  = int(w * 0.3)
        ry  = int(h * abs(np.sin(np.pi * t * 4)))
        cv2.rectangle(img, (rx, ry), (rx + 60, ry + 40), (200, 50, 0), -1)
        frames.append(img)
    return frames


def make_fast_noise(n_frames: int = 120, h: int = 240, w: int = 320) -> list:
    """High-frequency random noise ??? simulates fast sports content."""
    rng    = np.random.default_rng(42)
    frames = []
    for _ in range(n_frames):
        noise = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # Add a slowly shifting coloured wash so adjacent frames are related
        wash       = np.zeros((h, w, 3), dtype=np.uint8)
        wash[:, :] = rng.integers(0, 80, (3,), dtype=np.uint8)
        frame      = np.clip(noise.astype(np.int32) // 2 + wash.astype(np.int32), 0, 255)
        frames.append(frame.astype(np.uint8))
    return frames


def make_scene_cuts(n_frames: int = 120, h: int = 240, w: int = 320) -> list:
    """
    Alternating uniform-colour 'scenes' to stress-test scene-change detection.
    Each scene lasts ~20 frames; there are 6 scene changes total.
    """
    scenes = [
        (220, 30,  30),   # blue-ish
        (30,  220, 30),   # green-ish
        (30,  30,  220),  # red-ish
        (220, 220, 30),   # cyan-ish
        (220, 30,  220),  # magenta-ish
        (30,  220, 220),  # yellow-ish
    ]
    frames = []
    for i in range(n_frames):
        scene_idx = (i * len(scenes)) // n_frames
        b, g, r   = scenes[scene_idx % len(scenes)]
        img       = np.full((h, w, 3), [b, g, r], dtype=np.uint8)
        # Overlay a small moving blob so it is not perfectly static
        cx = int(w * (i % (n_frames // len(scenes))) / (n_frames // len(scenes)))
        cv2.circle(img, (cx, h // 2), 20, (255, 255, 255), -1)
        frames.append(img)
    return frames


# ?????? Main ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

_CLIP_GENERATORS = {
    "static_gradient": make_static_gradient,
    "moving_shapes":   make_moving_shapes,
    "fast_noise":      make_fast_noise,
    "scene_cuts":      make_scene_cuts,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test videos for VideoChef evaluation"
    )
    parser.add_argument("--output-dir", default=str(config.VIDEOS_DIR),
                        help="Directory to write videos (default: data/videos/)")
    parser.add_argument("--n-frames", type=int, default=120,
                        help="Frames per clip (default: 120)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width",  type=int, default=320)
    parser.add_argument("--clips",  nargs="+",
                        default=list(_CLIP_GENERATORS),
                        choices=list(_CLIP_GENERATORS),
                        help="Which clip types to generate (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.clips:
        gen    = _CLIP_GENERATORS[name]
        frames = gen(args.n_frames, args.height, args.width)
        path   = str(out_dir / f"{name}.mp4")
        from src.video_io import write_video
        write_video(frames, path, args.fps)
        print(f"  Written {len(frames)} frames -> {path}")

    print(f"\nDone. {len(args.clips)} clip(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
