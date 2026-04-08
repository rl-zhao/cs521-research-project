#!/usr/bin/env python3
"""
Comprehensive evaluation of VideoChef vs baselines across test videos.

Produces a CSV summary (results/combined_results.csv) and per-video CSVs.

What is compared
----------------
  Exact             ??? no approximation (ground-truth runtime / quality)
  IRA               ??? fixed 10-frame recalibration, no error model
  VideoChef-C-*     ??? quadratic model C, various keyframe / metric combos
  VideoChef-CA-*    ??? model CA
  VideoChef-CAD-*   ??? model CAD (paper's best)

Metrics reported
----------------
  total_time       ??? wall-clock seconds
  speedup          ??? exact_time / total_time
  mean_psnr        ??? mean PSNR vs exact output
  mean_ssim        ??? mean SSIM vs exact output
  violation_rate   ??? fraction of frames with PSNR < threshold
  n_keyframes      ??? number of search invocations
  search_pct       ??? search overhead as % of total_time

Usage
-----
  python evaluate.py                           # all pipelines, all test videos
  python evaluate.py --pipeline BVD            # one pipeline
  python evaluate.py --max-frames 60           # faster evaluation
  python evaluate.py --threshold 30 --metric psnr
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.quality import compute_psnr, compute_ssim
from src.video_io import get_iframe_positions, read_frames
from src.videochef import ExactProcessor, IRAProcessor, VideoChef


# ?????? Single-configuration runner ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def run_config(
    frames: list,
    ref_frames: list,
    exact_time: float,
    pipeline_name: str,
    model_type: str,
    kf_strategy: str,
    quality_metric: str,
    threshold: float,
    iframe_positions: list,
) -> dict:
    """
    Process *frames* with one VideoChef configuration and return a metrics dict.
    """
    vc = VideoChef(
        pipeline_name     = pipeline_name,
        error_model_type  = model_type,
        keyframe_strategy = kf_strategy,
        quality_metric    = quality_metric,
        quality_threshold = threshold,
    )
    result = vc.process_video(frames, iframe_positions=iframe_positions)
    out_frames = result["output_frames"]

    psnr_vals, ssim_vals = [], []
    for ref, out in zip(ref_frames, out_frames):
        p = compute_psnr(ref, out)
        s = compute_ssim(ref, out)
        if np.isfinite(p):
            psnr_vals.append(p)
        ssim_vals.append(s)

    tt      = result["total_time"]
    so      = result["search_overhead"]
    speedup = exact_time / tt if tt > 0 else 1.0
    viol    = (np.mean([p < threshold for p in psnr_vals])
               if psnr_vals else 0.0)

    return {
        "total_time":     tt,
        "speedup":        speedup,
        "search_pct":     100.0 * so / tt if tt > 0 else 0.0,
        "mean_psnr":      np.mean(psnr_vals) if psnr_vals else float("nan"),
        "mean_ssim":      np.mean(ssim_vals),
        "violation_rate": viol,
        "n_keyframes":    len(result["keyframes"]),
    }


# ?????? Per-video evaluation ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def evaluate_video(
    video_path: str,
    pipeline_name: str,
    threshold_psnr: float,
    threshold_ssim: float,
    max_frames: int,
    output_dir: Path,
) -> pd.DataFrame:
    frames, fps = read_frames(video_path, max_frames=max_frames)
    if not frames:
        print(f"  Could not read frames from {Path(video_path).name}")
        return pd.DataFrame()

    iframe_positions = get_iframe_positions(video_path)

    # Ground-truth exact output
    exact_proc   = ExactProcessor(pipeline_name)
    exact_result = exact_proc.process_video(frames)
    ref_frames   = exact_result["output_frames"]
    exact_time   = exact_result["total_time"]

    rows = [{
        "config":         "Exact",
        "pipeline":       pipeline_name,
        "model":          "???",
        "keyframe":       "???",
        "metric":         "???",
        "total_time":     exact_time,
        "speedup":        1.0,
        "search_pct":     0.0,
        "mean_psnr":      float("inf"),
        "mean_ssim":      1.0,
        "violation_rate": 0.0,
        "n_keyframes":    0,
    }]

    # IRA baseline
    for metric, thr in [("psnr", threshold_psnr), ("ssim", threshold_ssim)]:
        ira = IRAProcessor(pipeline_name, metric, thr)
        r   = ira.process_video(frames)
        psnr_vals, ssim_vals = [], []
        for ref, out in zip(ref_frames, r["output_frames"]):
            p = compute_psnr(ref, out)
            s = compute_ssim(ref, out)
            if np.isfinite(p):
                psnr_vals.append(p)
            ssim_vals.append(s)
        tt      = r["total_time"]
        speedup = exact_time / tt if tt > 0 else 1.0
        viol    = (np.mean([p < threshold_psnr for p in psnr_vals])
                   if psnr_vals else 0.0)
        rows.append({
            "config":         f"IRA-{metric}",
            "pipeline":       pipeline_name,
            "model":          "IRA",
            "keyframe":       "fixed-10",
            "metric":         metric,
            "total_time":     tt,
            "speedup":        speedup,
            "search_pct":     100.0 * r["search_overhead"] / tt if tt > 0 else 0.0,
            "mean_psnr":      np.mean(psnr_vals) if psnr_vals else float("nan"),
            "mean_ssim":      np.mean(ssim_vals),
            "violation_rate": viol,
            "n_keyframes":    len(r["keyframes"]),
        })

    # VideoChef configurations
    configs = [
        (model, kf, metric)
        for model  in ["C", "CA", "CAD"]
        for kf     in ["fixed", "scene_change"]
        for metric in ["psnr", "ssim"]
    ]

    for model, kf, metric in tqdm(configs, desc=f"  {Path(video_path).stem} / {pipeline_name}", leave=False):
        thr = threshold_psnr if metric == "psnr" else threshold_ssim
        try:
            m = run_config(
                frames, ref_frames, exact_time,
                pipeline_name, model, kf, metric, thr, iframe_positions,
            )
        except Exception as exc:
            print(f"    WARN: {model}-{kf}-{metric}: {exc}")
            traceback.print_exc()
            continue

        rows.append({
            "config":  f"VC-{model}-{kf}-{metric}",
            "pipeline": pipeline_name,
            "model":    model,
            "keyframe": kf,
            "metric":   metric,
            **m,
        })

    df = pd.DataFrame(rows)

    # Per-video save
    video_name = Path(video_path).stem
    out_path   = output_dir / f"{video_name}_{pipeline_name}.csv"
    df.to_csv(str(out_path), index=False)
    return df


# ?????? CLI entry point ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate VideoChef configurations on test videos"
    )
    parser.add_argument("--videos-dir", default=str(config.VIDEOS_DIR))
    parser.add_argument("--pipeline", default="all",
                        choices=list(config.PIPELINES) + ["all"])
    parser.add_argument("--threshold-psnr", type=float,
                        default=config.DEFAULT_PSNR_THRESHOLD)
    parser.add_argument("--threshold-ssim", type=float,
                        default=config.SSIM_THRESHOLD)
    parser.add_argument("--max-frames", type=int, default=config.MAX_FRAMES_EVAL)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()

    videos_dir  = Path(args.videos_dir)
    video_files = sorted(
        list(videos_dir.rglob("*.mp4")) +
        list(videos_dir.rglob("*.avi")) +
        list(videos_dir.rglob("*.mkv"))
    )

    if not video_files:
        print(f"No video files found in {videos_dir}.")
        print("Add videos or run  python scripts/generate_test_videos.py  first.")
        sys.exit(0)

    # Use the test split
    n_test_start = int(len(video_files) * (config.TRAIN_RATIO + config.VAL_RATIO))
    test_videos  = video_files[n_test_start:] or video_files
    if args.max_videos:
        test_videos = test_videos[: args.max_videos]

    print(f"Evaluating {len(test_videos)} test video(s)")

    pipelines  = (list(config.PIPELINES) if args.pipeline == "all"
                  else [args.pipeline])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for pipeline in pipelines:
        for vp in test_videos:
            print(f"\n{Path(str(vp)).name}  [{pipeline}]")
            df = evaluate_video(
                str(vp), pipeline,
                args.threshold_psnr, args.threshold_ssim,
                args.max_frames, output_dir,
            )
            if not df.empty:
                df["video"] = Path(str(vp)).stem
                all_dfs.append(df)

    if not all_dfs:
        print("No results generated.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_csv  = output_dir / "combined_results.csv"
    combined.to_csv(str(out_csv), index=False)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY  (mean across all test videos and pipelines)")
    print("=" * 70)
    num_cols = ["speedup", "mean_psnr", "mean_ssim", "violation_rate"]
    summary  = (combined.groupby("config")[num_cols]
                .mean()
                .sort_values("speedup", ascending=False))
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", 50)
    print(summary.to_string())
    print(f"\nFull results saved to: {out_csv}")


if __name__ == "__main__":
    main()
