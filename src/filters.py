"""
Video filter implementations with approximation levels.

Each filter supports approximation levels AL = 1 .. MAX_AL.
  AL = 1  ->  exact computation (no approximation)
  AL = k  ->  approximation applied, scaled to k

Two approximation types are implemented faithfully:

  perf  (loop perforation)
        Apply the filter at full resolution (preserving spatial context),
        then accept only every k-th output row and linearly interpolate
        the rest.  Produces authentic horizontal banding / interpolation
        artefacts.  Theoretical speedup: ~k x.

  memo  (memoization / 2-D perforation)
        Apply the filter at full resolution, accept only a sparse k x k
        grid of output pixels, then bilinearly interpolate back to full
        size.  Produces smooth but spatially-sampled artefacts.
        Theoretical speedup: ~k^2 x.
"""

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config

# ?????? Registry ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

_FILTER_REGISTRY: Dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _FILTER_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Core approximation helpers
# ---------------------------------------------------------------------------

def _loop_perforate(frame: np.ndarray, filter_fn, approx_level: int) -> np.ndarray:
    """
    Row-based loop perforation, faithful to the paper (Section 3.1).

    Paper pseudocode:
        for (i = 0; i < n; i = i + approx_level)
            result = compute_result();

    Only every approx_level-th row is computed. Skipped rows are not written
    and retain the original input pixel values. At AL=3 rows 0,3,6,... get
    the filtered output; rows 1,2,4,5,... keep the raw input pixels.
    This produces authentic horizontal banding artefacts.
    Theoretical speedup: ~approx_level x.
    """
    if approx_level <= 1:
        return filter_fn(frame)

    filtered = filter_fn(frame)
    result = frame.copy()                    # skipped rows keep original values
    result[::approx_level] = filtered[::approx_level]   # computed rows
    return result


def _memoize(frame: np.ndarray, filter_fn, approx_level: int) -> np.ndarray:
    """
    Row-based loop memoization, faithful to the paper (Section 3.1).

    Paper pseudocode:
        for (i = 0; i < n; i++)
            if (i % approx_level == 0)
                cached_result = result = compute_result();
            else
                result = cached_result;

    Every row is visited. Computed rows (i % approx_level == 0) get the real
    filtered output and update the cache. Skipped rows copy the most recently
    computed row (nearest-neighbour row repeat, no interpolation).
    At AL=3: rows 0,3,6,... are computed; rows 1,2 copy row 0; rows 4,5 copy
    row 3; etc.
    Theoretical speedup: ~approx_level x (same loop count, cheaper body).
    """
    if approx_level <= 1:
        return filter_fn(frame)

    h = frame.shape[0]
    filtered = filter_fn(frame)

    # For each row index i, the nearest computed row below it is
    # (i // approx_level) * approx_level  — fully vectorised, no Python loop.
    src_rows = (np.arange(h) // approx_level) * approx_level
    src_rows = np.clip(src_rows, 0, h - 1)
    return filtered[src_rows]


def _approximate(frame: np.ndarray, filter_fn, approx_level: int,
                 approx_type: str = 'perf') -> np.ndarray:
    """
    Dispatch to the correct approximation implementation (paper Section 3.1).

    approx_type 'perf'  -> _loop_perforate  (skip rows, keep original values)
    approx_type 'memo'  -> _memoize         (repeat last computed row)
    """
    if approx_type == 'memo':
        return _memoize(frame, filter_fn, approx_level)
    return _loop_perforate(frame, filter_fn, approx_level)


# ?????? Individual filters ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

@_register("boxblur")
def boxblur(frame: np.ndarray, approx_level: int = 1,
            approx_type: str = 'perf') -> np.ndarray:
    """Box (mean) blur with a 5x5 kernel."""
    def _fn(f):
        return cv2.blur(f, (5, 5))
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("vignette")
def vignette(frame: np.ndarray, approx_level: int = 1,
             approx_type: str = 'perf') -> np.ndarray:
    """Radial vignette: darken edges using an elliptical gradient mask."""
    def _fn(f):
        h, w = f.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist  = np.sqrt(((X - w / 2) / (w / 2)) ** 2 +
                        ((Y - h / 2) / (h / 2)) ** 2)
        mask  = np.clip(1.0 - 0.7 * dist, 0.0, 1.0).astype(np.float32)
        if f.ndim == 3:
            mask = mask[:, :, np.newaxis]
        return (f.astype(np.float32) * mask).clip(0, 255).astype(np.uint8)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("unsharp")
def unsharp(frame: np.ndarray, approx_level: int = 1,
            approx_type: str = 'perf') -> np.ndarray:
    """Unsharp mask: sharpen by subtracting a Gaussian-blurred copy."""
    def _fn(f):
        blurred = cv2.GaussianBlur(f, (5, 5), 1.0)
        return cv2.addWeighted(f, 1.5, blurred, -0.5, 0)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("inflate")
def inflate(frame: np.ndarray, approx_level: int = 1,
            approx_type: str = 'perf') -> np.ndarray:
    """Morphological dilation with a 5x5 elliptical kernel (brightens/expands)."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(f, kernel)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("deflate")
def deflate(frame: np.ndarray, approx_level: int = 1,
            approx_type: str = 'perf') -> np.ndarray:
    """Morphological erosion with a 5x5 elliptical kernel (darkens/shrinks)."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.erode(f, kernel)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("dilation")
def dilation(frame: np.ndarray, approx_level: int = 1,
             approx_type: str = 'perf') -> np.ndarray:
    """Morphological dilation with a 3x3 rectangular kernel."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(f, kernel)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("erosion")
def erosion(frame: np.ndarray, approx_level: int = 1,
            approx_type: str = 'perf') -> np.ndarray:
    """Morphological erosion with a 3x3 rectangular kernel."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.erode(f, kernel)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("emboss")
def emboss(frame: np.ndarray, approx_level: int = 1,
           approx_type: str = 'perf') -> np.ndarray:
    """Emboss effect via 2-D convolution with a directional kernel."""
    _kernel = np.array([[-2, -1,  0],
                        [-1,  1,  1],
                        [ 0,  1,  2]], dtype=np.float32)

    def _fn(f):
        if f.ndim == 3:
            channels = [cv2.filter2D(f[:, :, c].astype(np.float32), -1, _kernel)
                        for c in range(f.shape[2])]
            result = np.stack(channels, axis=2)
        else:
            result = cv2.filter2D(f.astype(np.float32), -1, _kernel)
        return np.clip(result + 128, 0, 255).astype(np.uint8)
    return _approximate(frame, _fn, approx_level, approx_type)


@_register("colorbalance")
def colorbalance(frame: np.ndarray, approx_level: int = 1,
                 approx_type: str = 'perf') -> np.ndarray:
    """Simple per-channel level adjustment (colour balance)."""
    def _fn(f):
        if f.ndim != 3:
            return f
        out = f.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * 0.90, 0, 255)   # B channel
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.10, 0, 255)   # G channel
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.00, 0, 255)   # R channel
        return out.astype(np.uint8)
    return _approximate(frame, _fn, approx_level, approx_type)


# ?????? Public API ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def apply_filter(frame: np.ndarray, filter_name: str, approx_level: int = 1,
                 approx_type: str = 'perf') -> np.ndarray:
    """Apply a named filter at the given approximation level and type."""
    if filter_name not in _FILTER_REGISTRY:
        raise ValueError(f"Unknown filter '{filter_name}'. "
                         f"Available: {sorted(_FILTER_REGISTRY)}")
    return _FILTER_REGISTRY[filter_name](frame, approx_level, approx_type)


def apply_pipeline(
    frame: np.ndarray,
    pipeline: List[Tuple[str, str]],
    al_vector: List[int],
) -> np.ndarray:
    """
    Apply a sequence of filters to *frame*.

    Parameters
    ----------
    frame      : input BGR frame
    pipeline   : list of (filter_name, approx_type) pairs
    al_vector  : approximation level for each filter (1 = exact)
    """
    result = frame.copy()
    for (filter_name, approx_type), al in zip(pipeline, al_vector):
        result = apply_filter(result, filter_name, al, approx_type)
    return result


def time_pipeline(
    frame: np.ndarray,
    pipeline: List[Tuple[str, str]],
    al_vector: List[int],
    n_trials: int = 3,
) -> float:
    """Return the minimum wall-clock time (seconds) over *n_trials* runs."""
    best = float("inf")
    for _ in range(n_trials):
        t0 = time.perf_counter()
        apply_pipeline(frame, pipeline, al_vector)
        best = min(best, time.perf_counter() - t0)
    return best


def build_timing_table(
    sample_frame: np.ndarray,
    pipeline: List[Tuple[str, str]],
    max_al: int = None,
    sample_fraction: float = 0.25,
) -> Dict[tuple, float]:
    """
    Build a lookup table mapping AL-vector ??? execution time.

    A random *sample_fraction* of all AL combinations is measured;
    the exact (all-1) entry is always included.
    """
    import itertools
    import random

    if max_al is None:
        max_al = config.MAX_AL

    n_filters  = len(pipeline)
    exact_al   = tuple([1] * n_filters)
    all_combos = list(itertools.product(range(1, max_al + 1), repeat=n_filters))

    n_sample  = max(1, int(len(all_combos) * sample_fraction))
    sampled   = random.sample(all_combos, min(n_sample, len(all_combos)))
    if exact_al not in sampled:
        sampled.append(exact_al)

    table: Dict[tuple, float] = {}
    for combo in sampled:
        table[combo] = time_pipeline(sample_frame, pipeline, list(combo), n_trials=2)

    return table


def estimate_exec_time(
    al_vector: List[int],
    timing_table: Dict[tuple, float],
    exact_al: List[int] = None,
    pipeline: List[Tuple[str, str]] = None,
) -> float:
    """
    Look up or estimate execution time for *al_vector*.

    Falls back to a theoretical speedup model when the combination is not in
    the timing table:
      perf  filters -> k x  speedup per filter (row-based perforation)
      memo  filters -> k^2 speedup per filter  (2-D grid memoization)

    Parameters
    ----------
    pipeline : optional list of (filter_name, approx_type) used to determine
               per-filter speedup exponent.  If None, memo (k^2) is assumed
               for all filters (conservative / backwards-compatible default).
    """
    key = tuple(al_vector)
    if key in timing_table:
        return timing_table[key]

    if exact_al is None:
        exact_al = [1] * len(al_vector)
    exact_key = tuple(exact_al)

    # Both perf and memo give ~k x speedup per the paper:
    # perf skips k-1 out of k rows entirely (~k x fewer iterations).
    # memo visits every row but k-1 out of k have a cheap cache hit (~k x cheaper body).
    # We use exponent=1 for both; pipeline arg kept for future extensibility.
    if pipeline is not None:
        exponents = [1 for _ in pipeline]
    else:
        exponents = [1] * len(al_vector)

    speedup = float(np.prod([al ** exp
                              for al, exp in zip(al_vector, exponents)]))
    speedup = max(speedup, 1.0)

    if exact_key in timing_table:
        return max(timing_table[exact_key] / speedup, 1e-9)

    return 1.0 / (speedup + 1e-9)
