"""
Online approximation-level search (Section 3.4 of the VideoChef paper).

The greedy search starts at exact computation (all ALs = 1) and
incrementally increases one filter's AL at a time.  At each step it uses
the Approximation Payoff Estimation Algorithm to decide whether the
expected speedup gain is worth the search overhead, then checks the
predicted full-frame quality via the error model.

Also implements:
  - exhaustive_search  ??? oracle upper-bound (no search overhead counted)
  - ira_search         ??? IRA baseline (no error model, F assumed == C)
"""

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from src.canary import compute_row_difference_features
from src.filters import apply_pipeline, estimate_exec_time
from src.quality import compute_quality, quality_acceptable


# ?????? Main VideoChef greedy search ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def greedy_search(
    canary_frame,
    full_frame,
    pipeline: List[Tuple[str, str]],
    error_model,
    timing_table: Dict[tuple, float],
    quality_threshold: float,
    quality_metric: str = "psnr",
    error_margin: float = None,
    max_al: int = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Greedy search for the most aggressive approximation level vector that
    still meets the quality constraint.

    Parameters
    ----------
    canary_frame      : small (downsampled) version of the current key frame
    full_frame        : the full-resolution key frame
    pipeline          : list of (filter_name, approx_type)
    error_model       : fitted error model (ModelC / ModelCA / ModelCAD / IRA)
    timing_table      : dict mapping AL-vector tuple ??? measured exec time
    quality_threshold : minimum acceptable quality (e.g. 30 dB PSNR)
    quality_metric    : 'psnr' or 'ssim'
    error_margin      : safety slack subtracted from threshold during search
    max_al            : maximum approximation level per filter
    verbose           : print per-step debug info

    Returns
    -------
    best_al_vector    : list of ints, one per filter
    predicted_quality : float, model's quality estimate for best_al_vector
    """
    if error_margin is None:
        error_margin = (config.ERROR_MARGIN_PSNR
                        if quality_metric == "psnr"
                        else config.ERROR_MARGIN_SSIM)
    if max_al is None:
        max_al = config.MAX_AL

    n_filters  = len(pipeline)
    exact_al   = [1] * n_filters
    current_al = exact_al[:]

    # Precompute reference canary output (AL=1) and row-diff features
    canary_exact = apply_pipeline(canary_frame, pipeline, exact_al)
    D_features   = compute_row_difference_features(full_frame)

    # Search against a relaxed threshold so we explore more of the space
    search_threshold = quality_threshold - error_margin

    best_al               = current_al[:]
    # Exact AL produces identical output -> quality is infinite.
    # Use a large finite sentinel so the model is never called with inf.
    best_predicted_quality = 999.0

    for _iteration in range(max_al * n_filters):
        # ?????? generate candidates: increment each filter AL by 1 ?????????????????????????????????????????????
        candidates = []
        for i in range(n_filters):
            if current_al[i] < max_al:
                cand = current_al[:]
                cand[i] += 1
                candidates.append((i, cand))

        if not candidates:
            break

        # ?????? payoff estimation (Section 3.4.1) ????????????????????????????????????????????????????????????????????????????????????????????????
        t_current = estimate_exec_time(current_al, timing_table, exact_al)

        # Maximum benefit B: best possible time saving with any candidate
        B = max(
            t_current - estimate_exec_time(cand, timing_table, exact_al)
            for _, cand in candidates
        )
        # Overhead O: cost of running all candidates on the canary
        # Canary is 1 / CANARY_SCALE?? ??? 1/64 the pixel count
        canary_ratio = 1.0 / (config.CANARY_SCALE ** 2)
        O = sum(
            estimate_exec_time(cand, timing_table, exact_al) * canary_ratio
            for _, cand in candidates
        )

        if B <= O:
            if verbose:
                print(f"    [search] Stopping: B={B:.4f} ??? O={O:.4f}")
            break

        # ?????? try each candidate on the canary ???????????????????????????????????????????????????????????????????????????????????????????????????
        found = False
        for filter_idx, cand_al in candidates:
            canary_approx = apply_pipeline(canary_frame, pipeline, cand_al)
            C = compute_quality(canary_exact, canary_approx, quality_metric)
            # Guard against exact match (shouldn't happen here, but be safe)
            if not np.isfinite(C):
                C = 60.0 if quality_metric == "psnr" else 1.0
            F_hat = error_model.predict(C, cand_al, D_features)

            if verbose:
                print(f"    [search] filter {filter_idx} AL???{cand_al[filter_idx]}: "
                      f"C={C:.3f}  F??={F_hat:.3f}  thr={search_threshold:.3f}")

            if quality_acceptable(F_hat, search_threshold):
                current_al             = cand_al
                best_al                = cand_al[:]
                best_predicted_quality = F_hat
                found = True
                break   # greedy: take first acceptable move

        if not found:
            break

    return best_al, best_predicted_quality


def _canary_quality_to_full(
    metric, canary_exact, canary_approx, al_vector, D_features, error_model
) -> float:
    C = compute_quality(canary_exact, canary_approx, metric)
    # Clamp infinity (exact-computation result) to a large finite value
    if not np.isfinite(C):
        C = 60.0 if metric == "psnr" else 1.0
    return error_model.predict(C, al_vector, D_features)


# ?????? Exhaustive / Oracle search ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def exhaustive_search(
    canary_frame,
    pipeline: List[Tuple[str, str]],
    error_model,
    quality_threshold: float,
    quality_metric: str = "psnr",
    error_margin: float = 0.0,
    max_al: int = None,
) -> List[int]:
    """
    Oracle: try every possible AL combination on the canary and return the
    most aggressive setting that still predicts acceptable quality.

    Does not count search overhead (upper-bound baseline).
    """
    if max_al is None:
        max_al = config.MAX_AL

    n_filters   = len(pipeline)
    exact_al    = [1] * n_filters
    canary_exact = apply_pipeline(canary_frame, pipeline, exact_al)
    D_zeros      = np.zeros(3)
    threshold    = quality_threshold - error_margin

    best_al  = exact_al[:]
    best_sum = sum(exact_al)

    for combo in itertools.product(range(1, max_al + 1), repeat=n_filters):
        al = list(combo)
        if al == exact_al:
            continue
        canary_approx = apply_pipeline(canary_frame, pipeline, al)
        C     = compute_quality(canary_exact, canary_approx, quality_metric)
        F_hat = error_model.predict(C, al, D_zeros)
        if quality_acceptable(F_hat, threshold):
            s = sum(combo)
            if s > best_sum:
                best_sum = s
                best_al  = al

    return best_al


# ?????? IRA baseline search ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def ira_search(
    canary_frame,
    pipeline: List[Tuple[str, str]],
    quality_threshold: float,
    quality_metric: str = "psnr",
    max_al: int = None,
) -> List[int]:
    """
    IRA-style search: greedy increment with F assumed == C (no error model).
    Used as a weaker baseline to isolate the benefit of the error mapping.
    """
    if max_al is None:
        max_al = config.MAX_AL

    n_filters    = len(pipeline)
    exact_al     = [1] * n_filters
    canary_exact = apply_pipeline(canary_frame, pipeline, exact_al)
    current_al   = exact_al[:]

    for _ in range(max_al * n_filters):
        improved = False
        for i in range(n_filters):
            if current_al[i] < max_al:
                cand = current_al[:]
                cand[i] += 1
                canary_approx = apply_pipeline(canary_frame, pipeline, cand)
                C = compute_quality(canary_exact, canary_approx, quality_metric)
                if quality_acceptable(C, quality_threshold):
                    current_al = cand
                    improved   = True
                    break
        if not improved:
            break

    return current_al
