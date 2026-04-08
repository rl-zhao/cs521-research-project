"""
Canary-input utilities.

A canary frame is a small, downsampled version of the original frame used to
cheaply estimate the quality impact of approximation settings before applying
them to the full frame.

Implements the dissimilarity metrics SMM and SMSD from Section 3.2 of the
VideoChef paper and the row-difference features used by Model-CAD (Section 3.5.3).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


# ?????? Canary generation ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def generate_canary(frame: np.ndarray, scale: int = 8) -> np.ndarray:
    """
    Downsample *frame* by taking every *scale*-th pixel in each dimension.

    Default scale=8 gives 1/64 of the original pixel count (1/8 per axis),
    matching the canary size used in the VideoChef paper.
    """
    return frame[::scale, ::scale].copy()


# ?????? Dissimilarity metrics (SMM / SMSD) ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def _yuv_means_stds(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-channel mean and std for Y, U, V."""
    if frame.ndim == 3:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
    else:
        yuv = np.stack([frame.astype(np.float32)] * 3, axis=2)
    means = yuv.mean(axis=(0, 1))
    stds  = yuv.std(axis=(0, 1))
    return means, stds


# YUV channel weights: Y carries more perceptual weight
_YUV_WEIGHTS = np.array([0.6, 0.2, 0.2], dtype=np.float32)


def compute_smm(full_frame: np.ndarray, canary: np.ndarray) -> float:
    """
    Dissimilarity Metric for Mean (SMM).

    SMM = weighted sum over channels of |mean_small - mean_full| / mean_full
    """
    means_full,  _ = _yuv_means_stds(full_frame)
    means_small, _ = _yuv_means_stds(canary)
    safe_denom = np.where(means_full > 1e-6, means_full, 1e-6)
    per_channel = np.abs(means_small - means_full) / safe_denom
    return float((_YUV_WEIGHTS * per_channel).sum())


def compute_smsd(full_frame: np.ndarray, canary: np.ndarray) -> float:
    """
    Dissimilarity Metric for Standard Deviation (SMSD).

    SMSD = weighted sum over channels of |std_small - std_full| / std_full
    """
    _, stds_full  = _yuv_means_stds(full_frame)
    _, stds_small = _yuv_means_stds(canary)
    safe_denom = np.where(stds_full > 1e-6, stds_full, 1e-6)
    per_channel = np.abs(stds_small - stds_full) / safe_denom
    return float((_YUV_WEIGHTS * per_channel).sum())


def select_canary(
    frame: np.ndarray,
    threshold: float = 0.10,
    candidate_scales: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """
    Choose the smallest canary that satisfies both SMM ??? threshold and
    SMSD ??? threshold.

    Returns (canary_frame, scale_used).
    """
    if candidate_scales is None:
        candidate_scales = [2, 4, 8, 16]  # 1/4, 1/16, 1/64, 1/256 of pixels

    best_canary: Optional[np.ndarray] = None
    best_scale  = candidate_scales[0]

    for scale in candidate_scales:
        canary = generate_canary(frame, scale)
        if compute_smm(frame, canary) <= threshold and \
           compute_smsd(frame, canary) <= threshold:
            # Update: prefer larger scale (= smaller canary)
            best_canary = canary
            best_scale  = scale

    if best_canary is None:
        # Nothing passed; fall back to coarsest valid canary
        best_canary = generate_canary(frame, candidate_scales[0])
        best_scale  = candidate_scales[0]

    return best_canary, best_scale


# ?????? Row-difference features (Model-CAD) ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

def compute_row_difference_features(frame: np.ndarray) -> np.ndarray:
    """
    Compute per-channel row-difference features D = [d_Y, d_U, d_V].

    Each d_c is the mean absolute difference between consecutive rows in
    channel c of the YUV representation. These features capture how much
    the image content varies between rows, which is correlated with how
    much quality loop perforation will lose.
    """
    if frame.ndim == 3:
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
        features = np.array([
            np.abs(np.diff(yuv[:, :, c], axis=0)).mean()
            for c in range(3)
        ])
    else:
        channel  = frame.astype(np.float32)
        row_diff = np.abs(np.diff(channel, axis=0)).mean()
        features = np.array([row_diff, 0.0, 0.0])

    return features
