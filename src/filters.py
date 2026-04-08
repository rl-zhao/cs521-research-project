"""
Video filter implementations with approximation levels.

Each filter supports approximation levels AL = 1 ??? MAX_AL.
  AL = 1  ???  exact computation (no downsampling)
  AL = k  ???  downsample each dimension by k before applying the filter,
              then upsample back.  This simulates loop-perforation / memoization:
              only 1/k?? of the pixels are processed, yielding an ~k?? speedup
              at the cost of resampling artefacts.

The "approx_type" field in a pipeline entry ('perf' / 'memo') distinguishes
the two approximation families described in the paper but both are realised
with the same resize-based simulation in Python.
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


# ?????? Core approximation helper ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _approximate(frame: np.ndarray, filter_fn, approx_level: int) -> np.ndarray:
    """
    Apply *filter_fn* with approximation via symmetric resize.

    AL=1 ??? filter_fn applied to original dimensions (exact).
    AL=k ??? downsample by k per dimension, apply filter, upsample.
    """
    if approx_level <= 1:
        return filter_fn(frame)

    h, w = frame.shape[:2]
    small_h = max(1, h // approx_level)
    small_w = max(1, w // approx_level)

    small  = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = filter_fn(small)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)


# ?????? Individual filters ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

@_register("boxblur")
def boxblur(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Box (mean) blur with a 5??5 kernel."""
    def _fn(f):
        return cv2.blur(f, (5, 5))
    return _approximate(frame, _fn, approx_level)


@_register("vignette")
def vignette(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
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
    return _approximate(frame, _fn, approx_level)


@_register("unsharp")
def unsharp(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Unsharp mask: sharpen by subtracting a Gaussian-blurred copy."""
    def _fn(f):
        blurred = cv2.GaussianBlur(f, (5, 5), 1.0)
        return cv2.addWeighted(f, 1.5, blurred, -0.5, 0)
    return _approximate(frame, _fn, approx_level)


@_register("inflate")
def inflate(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Morphological dilation with a 5??5 elliptical kernel (brightens/expands)."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(f, kernel)
    return _approximate(frame, _fn, approx_level)


@_register("deflate")
def deflate(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Morphological erosion with a 5??5 elliptical kernel (darkens/shrinks)."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.erode(f, kernel)
    return _approximate(frame, _fn, approx_level)


@_register("dilation")
def dilation(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Morphological dilation with a 3??3 rectangular kernel."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(f, kernel)
    return _approximate(frame, _fn, approx_level)


@_register("erosion")
def erosion(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Morphological erosion with a 3??3 rectangular kernel."""
    def _fn(f):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.erode(f, kernel)
    return _approximate(frame, _fn, approx_level)


@_register("emboss")
def emboss(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
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
    return _approximate(frame, _fn, approx_level)


@_register("colorbalance")
def colorbalance(frame: np.ndarray, approx_level: int = 1) -> np.ndarray:
    """Simple per-channel level adjustment (colour balance)."""
    def _fn(f):
        if f.ndim != 3:
            return f
        out = f.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * 0.90, 0, 255)   # B channel
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.10, 0, 255)   # G channel
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.00, 0, 255)   # R channel
        return out.astype(np.uint8)
    return _approximate(frame, _fn, approx_level)


# ?????? Public API ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def apply_filter(frame: np.ndarray, filter_name: str, approx_level: int = 1) -> np.ndarray:
    """Apply a named filter at the given approximation level."""
    if filter_name not in _FILTER_REGISTRY:
        raise ValueError(f"Unknown filter '{filter_name}'. "
                         f"Available: {sorted(_FILTER_REGISTRY)}")
    return _FILTER_REGISTRY[filter_name](frame, approx_level)


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
    for (filter_name, _), al in zip(pipeline, al_vector):
        result = apply_filter(result, filter_name, al)
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
) -> float:
    """
    Look up or estimate execution time for *al_vector*.

    Falls back to a theoretical model (speedup ??? ??? al_i??) if the exact
    entry is not in the table.
    """
    key = tuple(al_vector)
    if key in timing_table:
        return timing_table[key]

    if exact_al is None:
        exact_al = [1] * len(al_vector)
    exact_key = tuple(exact_al)

    if exact_key in timing_table:
        exact_time = timing_table[exact_key]
        speedup = float(np.prod([al ** 2 for al in al_vector]))
        return max(exact_time / speedup, 1e-9)

    return 1.0 / float(np.prod([al ** 2 for al in al_vector]) + 1e-9)
