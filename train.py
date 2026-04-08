#!/usr/bin/env python3
"""
Offline training script for VideoChef error-mapping models.

For each (pipeline, quality-metric) pair:
  1. Reads the training split of videos from data/videos/.
  2. For every sampled frame ?? sampled AL combination, computes:
       C  = quality(canary_approx, canary_exact)
       F  = quality(full_approx,   full_exact)
       D  = row-difference features
  3. Trains Model-C, Model-CA and Model-CAD.
  4. Saves the fitted models to models/.

Usage
-----
  python train.py                          # all pipelines, PSNR
  python train.py --pipeline BVD           # one pipeline
  python train.py --metric ssim            # use SSIM instead of PSNR
  python train.py --max-frames 20 --max-videos 4   # quick smoke-test
"""

import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from src.canary import generate_canary, compute_row_difference_features
from src.error_model import ModelC, ModelCA, ModelCAD, save_model, evaluate_model_f1
from src.filters import apply_pipeline
from src.quality import compute_quality
from src.video_io import read_frames


# ?????? Data collection ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def collect_training_data(
    video_path: str,
    pipeline_name: str,
    quality_metric: str = "psnr",
    max_frames: int = 50,
    al_sample_rate: float = 0.10,
) -> dict:
    """
    Return arrays C, AL, D, F from one video.

    C  : (N,)        canary quality  (metric applied to canary_approx vs canary_exact)
    AL : (N, n_filt) approximation level per filter
    D  : (N, 3)      row-difference features (Y, U, V)
    F  : (N,)        full-frame quality (metric applied to full_approx vs full_exact)
    """
    pipeline  = config.PIPELINES[pipeline_name]
    n_filters = len(pipeline)
    exact_al  = [1] * n_filters

    frames, _ = read_frames(str(video_path), max_frames=max_frames)
    if not frames:
        return {}

    # Sample AL combinations (always include exact for reference)
    all_combos = list(itertools.product(range(1, config.MAX_AL + 1), repeat=n_filters))
    n_sample   = max(1, int(len(all_combos) * al_sample_rate))
    sampled    = random.sample(all_combos, min(n_sample, len(all_combos)))
    sampled_set = set(sampled)
    exact_key   = tuple(exact_al)
    # Remove exact from training pairs (quality is ???)
    sampled_set.discard(exact_key)
    sampled = list(sampled_set)

    C_list, AL_list, D_list, F_list = [], [], [], []

    for frame in frames:
        canary = generate_canary(frame, config.CANARY_SCALE)
        D      = compute_row_difference_features(frame)

        canary_exact = apply_pipeline(canary, pipeline, exact_al)
        full_exact   = apply_pipeline(frame,  pipeline, exact_al)

        for combo in sampled:
            al = list(combo)

            canary_approx = apply_pipeline(canary, pipeline, al)
            full_approx   = apply_pipeline(frame,  pipeline, al)

            C = compute_quality(canary_exact, canary_approx, quality_metric)
            F = compute_quality(full_exact,   full_approx,   quality_metric)

            if not (np.isfinite(C) and np.isfinite(F)):
                continue

            C_list.append(C)
            AL_list.append(al)
            D_list.append(D.tolist())
            F_list.append(F)

    if not C_list:
        return {}
    return {
        "C":  np.array(C_list,  dtype=np.float32),
        "AL": np.array(AL_list, dtype=np.float32),
        "D":  np.array(D_list,  dtype=np.float32),
        "F":  np.array(F_list,  dtype=np.float32),
    }


# ?????? Training ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def train_and_save(
    data: dict,
    n_filters: int,
    pipeline_name: str,
    quality_metric: str,
    models_dir: Path,
    quality_threshold: float,
) -> None:
    C  = data["C"]
    AL = data["AL"]
    D  = data["D"]
    F  = data["F"]

    if len(C) < 20:
        print(f"  ???  Only {len(C)} samples ??? skipping (need ??? 20).")
        return

    print(f"  Training on {len(C)} samples ???")

    for name, model, fit_kwargs in [
        ("C",   ModelC(),             {"C": C, "F": F}),
        ("CA",  ModelCA(n_filters),   {"C": C, "AL": AL, "F": F}),
        ("CAD", ModelCAD(n_filters),  {"C": C, "AL": AL, "D": D, "F": F}),
    ]:
        model.fit(**fit_kwargs)

        path = models_dir / f"model_{pipeline_name}_{name}_{quality_metric}.pkl"
        save_model(model, str(path))

        f1 = evaluate_model_f1(model, C, AL, D, F, quality_threshold)
        print(f"    Model-{name:3s}  saved -> {path.name}   F-1={f1:.4f}")


# ?????? CLI entry point ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train VideoChef error-mapping models"
    )
    parser.add_argument(
        "--videos-dir", default=str(config.VIDEOS_DIR),
        help="Directory containing training videos",
    )
    parser.add_argument(
        "--pipeline", default="all",
        choices=list(config.PIPELINES) + ["all"],
        help="Pipeline to train (default: all)",
    )
    parser.add_argument(
        "--metric", default="psnr",
        choices=["psnr", "ssim"],
        help="Quality metric (default: psnr)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=config.MAX_FRAMES_TRAIN,
        help="Max frames sampled per video",
    )
    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max training videos to use",
    )
    parser.add_argument(
        "--al-sample-rate", type=float, default=config.AL_SAMPLE_RATE,
        help="Fraction of AL combinations to sample (default 0.10)",
    )
    args = parser.parse_args()

    videos_dir  = Path(args.videos_dir)
    video_files = sorted(
        list(videos_dir.rglob("*.mp4")) +
        list(videos_dir.rglob("*.avi")) +
        list(videos_dir.rglob("*.mkv"))
    )

    if not video_files:
        print(f"No video files found in {videos_dir}.")
        print("Add .mp4 / .avi / .mkv files to data/videos/ and re-run.")
        print("(Use  python scripts/generate_test_videos.py  for synthetic clips.)")
        sys.exit(0)

    n_train     = max(1, int(len(video_files) * config.TRAIN_RATIO))
    train_set   = video_files[:n_train]
    if args.max_videos:
        train_set = train_set[: args.max_videos]

    print(f"Training on {len(train_set)} / {len(video_files)} videos")

    pipelines = (list(config.PIPELINES) if args.pipeline == "all"
                 else [args.pipeline])

    threshold = (config.DEFAULT_PSNR_THRESHOLD if args.metric == "psnr"
                 else config.SSIM_THRESHOLD)

    for pipeline_name in pipelines:
        print(f"\n{'='*60}")
        print(f"Pipeline: {pipeline_name}   Metric: {args.metric}")
        print('='*60)
        n_filters = len(config.PIPELINES[pipeline_name])

        all_C, all_AL, all_D, all_F = [], [], [], []

        for vp in tqdm(train_set, desc=f"  {pipeline_name}"):
            d = collect_training_data(
                str(vp), pipeline_name,
                quality_metric=args.metric,
                max_frames=args.max_frames,
                al_sample_rate=args.al_sample_rate,
            )
            if d:
                all_C.extend(d["C"].tolist())
                all_AL.extend(d["AL"].tolist())
                all_D.extend(d["D"].tolist())
                all_F.extend(d["F"].tolist())

        if not all_C:
            print("  No training data collected ??? check video format / path.")
            continue

        train_and_save(
            {"C":  np.array(all_C),
             "AL": np.array(all_AL),
             "D":  np.array(all_D),
             "F":  np.array(all_F)},
            n_filters, pipeline_name, args.metric,
            config.MODELS_DIR, threshold,
        )


if __name__ == "__main__":
    main()
