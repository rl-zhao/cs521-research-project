# CS521 Research Project — Extending VideoChef

**Raymond Zhao & Ray Lin · University of Illinois Urbana-Champaign**

A Python reimplementation and extension of the
[VideoChef (USENIX ATC '18)](https://www.usenix.org/conference/atc18/presentation/xu-ran)
approximate video-processing system, as described in our project proposal.

---

## Overview

VideoChef achieves significant video-processing speedup by using small
*canary* frames to cheaply find the best approximation level for each
filter, then applying that setting to full-resolution frames.  This project:

- **Re-implements** VideoChef in Python + OpenCV (4 filter pipelines, 6 ALs).
- **Extends** it with SSIM as an alternative quality metric (vs. PSNR only).
- **Evaluates** three error-mapping models (C, CA, CAD) and their
  generalisation across video categories.
- **Compares** three key-frame detection strategies (fixed interval, scene
  change, H.264 I-frames).

---

## Project Structure

```
cs521-research-project/
├── config.py                  # All global parameters
├── requirements.txt
├── train.py                   # Offline model training
├── run.py                     # Run VideoChef on one video
├── evaluate.py                # Full benchmark vs baselines
│
├── src/
│   ├── video_io.py            # Frame I/O, ffprobe I-frame detection
│   ├── filters.py             # Filter implementations + approx levels
│   ├── canary.py              # Canary generation, SMM/SMSD, row-diff features
│   ├── keyframe.py            # Fixed / scene-change / I-frame detection
│   ├── quality.py             # PSNR, SSIM, optional VMAF
│   ├── error_model.py         # Model-C, Model-CA, Model-CAD, IRA baseline
│   ├── search.py              # Greedy search + payoff estimation
│   └── videochef.py           # Main controller, ExactProcessor, IRAProcessor
│
├── scripts/
│   └── generate_test_videos.py   # Create synthetic benchmark clips
│
├── notebooks/
│   └── analysis.ipynb            # Figures & tables for the report
│
├── data/videos/               # Place .mp4/.avi/.mkv clips here
├── models/                    # Trained error models (.pkl)
└── results/                   # CSV outputs and PDF figures
```

---

## Setup

### 1. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 3. Install FFmpeg (optional, needed for I-frame detection and VMAF)

Download from <https://ffmpeg.org/download.html> and add `ffmpeg.exe` /
`ffprobe.exe` to your `PATH`.

---

## Quick Start

### Generate synthetic test videos

```powershell
python scripts/generate_test_videos.py
```

This writes four short clips to `data/videos/`:
`static_gradient.mp4`, `moving_shapes.mp4`, `fast_noise.mp4`, `scene_cuts.mp4`.

### (Optional) Add real YouTube clips

Download 2 clips per category (Lectures, Ads, Car Races, Entertainment,
Movie Trailers, Nature, News, Sports) and place them under `data/videos/`.
Any tool that outputs `.mp4` files works (e.g. `yt-dlp`).

### Train error-mapping models

```powershell
python train.py                         # all pipelines, PSNR metric
python train.py --metric ssim           # SSIM metric
python train.py --pipeline BVD          # one pipeline only
python train.py --max-frames 20 --max-videos 2   # fast smoke-test
```

Trained models are saved to `models/`.

### Run VideoChef on a single video

```powershell
python run.py data/videos/moving_shapes.mp4
python run.py data/videos/moving_shapes.mp4 --compare   # vs exact + IRA
python run.py data/videos/fast_noise.mp4 --pipeline UIV --model CAD --keyframe scene_change
```

### Full evaluation (all configurations × all test videos)

```powershell
python evaluate.py
python evaluate.py --pipeline BVD --max-frames 60   # faster
```

Results go to `results/combined_results.csv`.

### Analysis notebook

```powershell
jupyter notebook notebooks/analysis.ipynb
```

---

## Pipelines

| Name | Filters (→ approx. type) |
|------|--------------------------|
| BVD  | Boxblur(memo) → Vignette(perf) → Dilation(perf) |
| BVI  | Boxblur(memo) → Vignette(perf) → Inflate(perf) |
| UIV  | Unsharp(perf) → Inflate(perf) → Vignette(perf) |
| DVE  | Deflate(perf) → Vignette(perf) → Emboss(perf) |

**Approximation types:**
- `perf` (loop perforation) — apply filter to 1/AL fraction of rows; interpolate rest.
- `memo` (memoization) — apply filter to 1/AL²-sized frame; upsample result.

Both are simulated by downsampling before the filter and upsampling after.

---

## Error Models

| Model | Features | Paper result (F-1 @ 30 dB) |
|-------|----------|-----------------------------|
| IRA (baseline) | F ≡ C (no mapping) | 0.865 |
| Model-C  | C only (quadratic) | 0.958 |
| Model-CA | C + AL vector | 0.959 |
| Model-CAD | C + AL + row-diff D | **0.969** |

---

## Key-frame Strategies

| Strategy | Description | Overhead |
|----------|-------------|---------|
| `fixed` | Every 30 frames | Predictable |
| `scene_change` | Histogram diff on canary Y-channel | Content-dependent |
| `iframe` | H.264 I-frame positions (requires ffprobe) | Low |

---

## Research Questions Addressed

1. **Does SSIM provide a better approximation constraint than PSNR?**
   Compare `VC-*-psnr` vs `VC-*-ssim` in the evaluation.

2. **How well does the error model generalise across video categories?**
   Train on lectures/ads, test on sports/nature (separate category splits).

3. **Which key-frame strategy gives the best speedup/quality trade-off?**
   Compare `fixed` vs `scene_change` columns in the results.

---

## Citation

> Xu, R., Koo, J., Kumar, R., Bai, P., Mitra, S., Misailovic, S., & Bagchi, S.
> (2018). *VideoChef: Efficient Approximation for Streaming Video Processing
> Pipelines.* USENIX ATC '18.
