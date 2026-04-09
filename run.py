#!/usr/bin/env python3
"""
Run VideoChef on a single video and optionally compare against exact output.

Usage
-----
  python run.py input.mp4
  python run.py input.mp4 --output out.mp4 --pipeline UIV --model CAD
  python run.py input.mp4 --compare          # also show PSNR/SSIM vs exact
  python run.py input.mp4 --keyframe iframe  # use H.264 I-frames
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import config
from src.quality import compute_psnr, compute_ssim
from src.video_io import get_iframe_positions, read_frames, write_video
from src.videochef import ExactProcessor, IRAProcessor, VideoChef


def format_time(seconds: float) -> str:
    return f"{seconds:.3f}s"


def print_results(label: str, result: dict, exact_time: float) -> None:
    tt   = result["total_time"]
    so   = result.get("search_overhead", 0.0)
    kf   = len(result.get("keyframes", []))
    spd  = exact_time / tt if tt > 0 else 1.0
    pct  = 100.0 * so / tt if tt > 0 else 0.0
    print(f"\n{'-'*55}")
    print(f"  {label}")
    print(f"{'-'*55}")
    print(f"  Total time      : {format_time(tt)}")
    print(f"  Search overhead : {format_time(so)} ({pct:.1f}%)")
    print(f"  Speedup vs exact: {spd:.2f}x")
    print(f"  Key frames      : {kf}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VideoChef approximate video processing"
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: <input>_videochef.mp4)")
    parser.add_argument("--pipeline", default="BVD",
                        choices=list(config.PIPELINES),
                        help="Filter pipeline to use")
    parser.add_argument("--model", default="CAD",
                        choices=["C", "CA", "CAD", "IRA"],
                        help="Error mapping model")
    parser.add_argument("--keyframe", default="scene_change",
                        choices=["fixed", "scene_change", "iframe"],
                        help="Key-frame detection strategy")
    parser.add_argument("--metric", default="psnr",
                        choices=["psnr", "ssim"],
                        help="Quality metric for the approximation constraint")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Quality threshold (default: 30 dB PSNR / 0.85 SSIM)")
    parser.add_argument("--max-frames", type=int, default=config.MAX_FRAMES_EVAL,
                        help="Maximum frames to process")
    parser.add_argument("--compare", action="store_true",
                        help="Compute quality vs exact output and IRA baseline")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame search details")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: '{args.input}' not found.")
        sys.exit(1)

    output_path = args.output or str(
        config.RESULTS_DIR / f"{input_path.stem}_videochef.mp4"
    )

    threshold = args.threshold or (
        config.DEFAULT_PSNR_THRESHOLD if args.metric == "psnr"
        else config.SSIM_THRESHOLD
    )

    # load video 
    print(f"Loading: {input_path.name}  (max {args.max_frames} frames)")
    frames, fps = read_frames(str(input_path), max_frames=args.max_frames)
    if not frames:
        print("Error: could not read any frames.")
        sys.exit(1)
    print(f"Loaded {len(frames)} frames @ {fps:.1f} fps  "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")

    iframe_positions = (
        get_iframe_positions(str(input_path))
        if args.keyframe == "iframe" else None
    )

    # exact baseline 
    print("\nRunning exact baseline...")
    exact_proc   = ExactProcessor(args.pipeline)
    exact_result = exact_proc.process_video(frames)
    ref_frames   = exact_result["output_frames"]
    exact_time   = exact_result["total_time"]
    print_results("Exact (baseline)", exact_result, exact_time)

    # VideoChef 
    print("\nRunning VideoChef...")
    print(f"  pipeline={args.pipeline}  model={args.model}  "
          f"keyframe={args.keyframe}  metric={args.metric}>={threshold}")

    vc = VideoChef(
        pipeline_name     = args.pipeline,
        error_model_type  = args.model,
        keyframe_strategy = args.keyframe,
        quality_metric    = args.metric,
        quality_threshold = threshold,
    )
    vc_result    = vc.process_video(frames, fps=fps,
                                     iframe_positions=iframe_positions,
                                     verbose=args.verbose)
    vc_frames    = vc_result["output_frames"]
    print_results("VideoChef", vc_result, exact_time)

    if args.compare:
        # IRA baseline 
        print("\nRunning IRA baseline...")
        ira     = IRAProcessor(args.pipeline, args.metric, threshold)
        ira_res = ira.process_video(frames)
        print_results("IRA (baseline)", ira_res, exact_time)

        # quality comparison 
        print("\n Quality vs exact output ")
        for label, out_frames in [("VideoChef", vc_frames), ("IRA", ira_res["output_frames"])]:
            psnr_vals, ssim_vals = [], []
            for ref, out in zip(ref_frames, out_frames):
                p = compute_psnr(ref, out)
                s = compute_ssim(ref, out)
                if np.isfinite(p):
                    psnr_vals.append(p)
                ssim_vals.append(s)
            viol = np.mean([p < threshold for p in psnr_vals]) if psnr_vals else 0.0
            print(f"  {label:12s}  mean PSNR={np.mean(psnr_vals):.2f} dB  "
                  f"mean SSIM={np.mean(ssim_vals):.4f}  "
                  f"violation={100*viol:.1f}%")

    # save output 
    write_video(vc_frames, output_path, fps)
    print(f"\nSaved VideoChef output -> {output_path}")


if __name__ == "__main__":
    main()

