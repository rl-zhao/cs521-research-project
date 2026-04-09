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
    Real row-based loop perforation.

    The filter is applied at full resolution so that each accepted row has
    correct spatial context (neighbouring pixels are real, not interpolated).
    Only every approx_level-th row of the filtered output is kept; the
    remaining rows are filled by linear interpolation between the accepted
    rows.  This produces authentic horizontal banding artefacts distinct
    from the blur introduced by naive resize-based simulation.
    """
    if approx_level <= 1:
        return filter_fn(frame)

    h, w = frame.shape[:2]
    filtered = filter_fn(frame).astype(np.float32)

    selected = np.arange(0, h, approx_level)          # accepted row indices
    flat_sel = filtered.reshape(h, -1)[selected]       # (n_sel, w*c)

    all_rows = np.arange(h)
    i_low  = np.clip(np.searchsorted(selected, all_rows, side='right') - 1,
                     0, len(selected) - 1)
    i_high = np.clip(i_low + 1, 0, len(selected) - 1)

    r_low  = selected[i_low].astype(np.float32)
    r_high = selected[i_high].astype(np.float32)
    denom  = np.where(r_high > r_low, r_high - r_low, 1.0)
    t      = np.clip((all_rows - r_low) / denom, 0.0, 1.0)

    result_flat = ((1.0 - t)[:, None] * flat_sel[i_low] +
                   t[:, None]         * flat_sel[i_high])
    return result_flat.reshape(frame.shape).clip(0, 255).astype(np.uint8)


def _memoize(frame: np.ndarray, filter_fn, approx_level: int) -> np.ndarray:
    """
    2-D block memoization (2-D loop perforation).

    The filter is applied at full resolution for correct context, but only
    a sparse approx_level x approx_level grid of output pixels is accepted.
    The gaps are filled by bilinear interpolation (cv2.resize), producing
    smooth but spatially-subsampled artefacts distinct from loop perforation.
    Theoretical speedup is approx_level^2 x.
    """
    if approx_level <= 1:
        return filter_fn(frame)

    h, w = frame.shape[:2]
    filtered = filter_fn(frame).astype(np.float32)

    row_sel = np.arange(0, h, approx_level)
    col_sel = np.arange(0, w, approx_level)
    sparse  = filtered[np.ix_(row_sel, col_sel)]      # (n_rows, n_cols [, c])

    result = cv2.resize(sparse, (w, h), interpolation=cv2.INTER_LINEAR)
    return result.clip(0, 255).astype(np.uint8)


def _approximate(frame: np.ndarray, filter_fn, approx_level: int,
                 approx_type: str = 'perf') -> np.ndarray:
    """
    Dispatch to the correct approximation implementation.

    approx_type 'perf'  -> _loop_perforate  (row-based, ~k x speedup)
    approx_type 'memo'  -> _memoize         (2-D grid,  ~k^2 x speedup)
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

    # Determine exponent per filter: perf -> 1, memo -> 2
    if pipeline is not None:
        exponents = [1 if atype == 'perf' else 2
                     for _, atype in pipeline]
    else:
        exponents = [2] * len(al_vector)

    speedup = float(np.prod([al ** exp
                              for al, exp in zip(al_vector, exponents)]))
    speedup = max(speedup, 1.0)

    if exact_key in timing_table:
        return max(timing_table[exact_key] / speedup, 1e-9)

    return 1.0 / (speedup + 1e-9)
