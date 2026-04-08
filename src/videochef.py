"""
VideoChef main controller.

Orchestrates the end-to-end approximate video processing pipeline:
  1. Build a timing lookup table from a sample frame.
  2. Detect key frames (fixed interval / scene change / I-frame).
  3. For each key frame: generate canary, run greedy search, cache the
     optimal AL vector.
  4. For non-key frames: reuse the cached AL vector.
  5. Apply the pipeline and return per-frame output and timing stats.

Also provides ExactProcessor (no approximation) and IRAProcessor baselines.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from src.canary import generate_canary
from src.error_model import create_model, load_model, IRABaseline
from src.filters import apply_pipeline, build_timing_table
from src.keyframe import (
    detect_keyframes_fixed,
    detect_keyframes_scene_change,
    detect_keyframes_iframe,
)
from src.quality import get_quality_threshold, get_error_margin
from src.search import greedy_search, ira_search


# ?????? VideoChef ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

class VideoChef:
    """
    Approximate video processor.

    Parameters
    ----------
    pipeline_name      : one of config.PIPELINES
    error_model_type   : 'C', 'CA', 'CAD', or 'IRA'
    keyframe_strategy  : 'fixed', 'scene_change', or 'iframe'
    quality_metric     : 'psnr' or 'ssim'
    quality_threshold  : minimum acceptable quality score
    models_dir         : directory from which trained models are loaded
    canary_scale       : downsampling factor for canary generation
    """

    def __init__(
        self,
        pipeline_name: str,
        error_model_type: str = "CAD",
        keyframe_strategy: str = "scene_change",
        quality_metric: str = "psnr",
        quality_threshold: Optional[float] = None,
        models_dir: Optional[str] = None,
        canary_scale: int = None,
    ):
        if pipeline_name not in config.PIPELINES:
            raise ValueError(f"Unknown pipeline '{pipeline_name}'. "
                             f"Available: {list(config.PIPELINES)}")

        self.pipeline_name    = pipeline_name
        self.pipeline         = config.PIPELINES[pipeline_name]
        self.n_filters        = len(self.pipeline)
        self.error_model_type = error_model_type
        self.keyframe_strategy = keyframe_strategy
        self.quality_metric   = quality_metric
        self.quality_threshold = (
            quality_threshold if quality_threshold is not None
            else get_quality_threshold(quality_metric)
        )
        self.error_margin     = get_error_margin(quality_metric)
        self.canary_scale     = canary_scale or config.CANARY_SCALE
        self.models_dir       = Path(models_dir or config.MODELS_DIR)
        self.error_model      = self._load_or_create_model()
        self.timing_table: Dict[tuple, float] = {}

    # ?????? model management ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    def _load_or_create_model(self):
        path = (self.models_dir /
                f"model_{self.pipeline_name}_{self.error_model_type}"
                f"_{self.quality_metric}.pkl")
        if path.exists():
            m = load_model(str(path))
            print(f"[VideoChef] Loaded model from {path.name}")
            return m
        m = create_model(self.error_model_type, self.n_filters)
        print(f"[VideoChef] No trained model found at {path.name}; "
              "using untrained (IRA-like) fallback.")
        return m

    # ?????? key-frame detection ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    def _detect_keyframes(
        self,
        frames: List,
        iframe_positions: Optional[List[int]],
    ) -> List[int]:
        n = len(frames)
        if self.keyframe_strategy == "fixed":
            return detect_keyframes_fixed(n)
        if self.keyframe_strategy == "scene_change":
            return detect_keyframes_scene_change(frames)
        if self.keyframe_strategy == "iframe":
            if iframe_positions:
                return detect_keyframes_iframe(iframe_positions, n)
            print("[VideoChef] No I-frame positions supplied; "
                  "falling back to fixed-interval detection.")
            return detect_keyframes_fixed(n)
        raise ValueError(f"Unknown keyframe strategy '{self.keyframe_strategy}'")

    # ?????? main processing entry point ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    def process_video(
        self,
        frames: List,
        fps: float = 30.0,
        iframe_positions: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Process a list of frames with adaptive approximate computation.

        Returns a dict with:
          output_frames   ??? list of processed np.ndarray
          al_history      ??? AL vector applied to each frame
          timing          ??? per-frame processing time (seconds)
          keyframes       ??? list of key-frame indices
          search_overhead ??? total search time (seconds)
          total_time      ??? processing + search (seconds)
        """
        if not frames:
            return {}

        # Build timing table from a representative frame
        sample = frames[min(10, len(frames) - 1)]
        self.timing_table = build_timing_table(sample, self.pipeline)

        keyframe_set = set(self._detect_keyframes(frames, iframe_positions))
        if verbose:
            print(f"[VideoChef] {len(keyframe_set)} keyframes detected "
                  f"out of {len(frames)} frames")

        output_frames: List = []
        al_history:    List = []
        timing:        List[float] = []
        search_overhead = 0.0
        current_al      = [1] * self.n_filters   # start exact

        for idx, frame in enumerate(frames):
            is_kf = idx in keyframe_set

            if is_kf:
                t0     = time.perf_counter()
                canary = generate_canary(frame, self.canary_scale)
                optimal_al, pred_q = greedy_search(
                    canary_frame      = canary,
                    full_frame        = frame,
                    pipeline          = self.pipeline,
                    error_model       = self.error_model,
                    timing_table      = self.timing_table,
                    quality_threshold = self.quality_threshold,
                    quality_metric    = self.quality_metric,
                    error_margin      = self.error_margin,
                    verbose           = verbose,
                )
                search_overhead += time.perf_counter() - t0
                current_al       = optimal_al
                if verbose:
                    print(f"  frame {idx:4d} [KEY] AL={current_al}  "
                          f"F??={pred_q:.2f}")

            t0 = time.perf_counter()
            out = apply_pipeline(frame, self.pipeline, current_al)
            timing.append(time.perf_counter() - t0)

            output_frames.append(out)
            al_history.append(current_al[:])

        return {
            "output_frames":   output_frames,
            "al_history":      al_history,
            "timing":          timing,
            "keyframes":       sorted(keyframe_set),
            "search_overhead": search_overhead,
            "total_time":      sum(timing) + search_overhead,
        }


# ?????? Exact (no-approximation) baseline ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

class ExactProcessor:
    """Apply the pipeline at AL=1 for every frame (ground-truth baseline)."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.pipeline      = config.PIPELINES[pipeline_name]
        self.n_filters     = len(self.pipeline)

    def process_video(self, frames: List) -> Dict:
        exact_al = [1] * self.n_filters
        output_frames: List  = []
        timing: List[float]  = []

        for frame in frames:
            t0 = time.perf_counter()
            output_frames.append(apply_pipeline(frame, self.pipeline, exact_al))
            timing.append(time.perf_counter() - t0)

        return {
            "output_frames": output_frames,
            "timing":        timing,
            "total_time":    sum(timing),
        }


# ?????? IRA processor baseline ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

class IRAProcessor:
    """
    IRA baseline: same greedy search as VideoChef but with no error model
    (assumes F == C) and a fixed recalibration interval of every 10 frames.
    """

    def __init__(
        self,
        pipeline_name: str,
        quality_metric: str = "psnr",
        quality_threshold: Optional[float] = None,
        interval: int = 10,
        canary_scale: int = None,
    ):
        self.pipeline_name    = pipeline_name
        self.pipeline         = config.PIPELINES[pipeline_name]
        self.n_filters        = len(self.pipeline)
        self.quality_metric   = quality_metric
        self.quality_threshold = (
            quality_threshold if quality_threshold is not None
            else get_quality_threshold(quality_metric)
        )
        self.interval    = interval
        self.canary_scale = canary_scale or config.CANARY_SCALE

    def process_video(self, frames: List) -> Dict:
        keyframe_set   = set(range(0, len(frames), self.interval))
        output_frames: List = []
        al_history:    List = []
        timing:        List[float] = []
        search_overhead = 0.0
        current_al      = [1] * self.n_filters

        for idx, frame in enumerate(frames):
            if idx in keyframe_set:
                t0     = time.perf_counter()
                canary = generate_canary(frame, self.canary_scale)
                current_al = ira_search(
                    canary, self.pipeline,
                    self.quality_threshold, self.quality_metric,
                )
                search_overhead += time.perf_counter() - t0

            t0 = time.perf_counter()
            output_frames.append(apply_pipeline(frame, self.pipeline, current_al))
            timing.append(time.perf_counter() - t0)
            al_history.append(current_al[:])

        return {
            "output_frames":   output_frames,
            "al_history":      al_history,
            "timing":          timing,
            "keyframes":       sorted(keyframe_set),
            "search_overhead": search_overhead,
            "total_time":      sum(timing) + search_overhead,
        }
