"""Evaluation metrics and the case-level statistics used in the paper."""
import itertools

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(pred, ref, roi=None):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    if roi is not None:
        pred, ref = pred & roi, ref & roi
    s = pred.sum() + ref.sum()
    return 1.0 if s == 0 else 2.0 * (pred & ref).sum() / s


def iou(pred, ref):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    u = (pred | ref).sum()
    return 1.0 if u == 0 else (pred & ref).sum() / u


def false_positive_pixels(pred, ref):
    return int((np.asarray(pred, bool) & ~np.asarray(ref, bool)).sum())


def area_ratio(pred, ref):
    return np.asarray(pred, bool).sum() / max(1, np.asarray(ref, bool).sum())


def hd95(pred, ref):
    """95th-percentile symmetric Hausdorff distance in pixels."""
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    if not pred.any() or not ref.any():
        return np.nan
    bp = pred & ~binary_erosion(pred)
    br = ref & ~binary_erosion(ref)
    d1 = distance_transform_edt(~br)[bp]
    d2 = distance_transform_edt(~bp)[br]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def case_bootstrap_ci(case_deltas, n=10000, seed=0, alpha=0.05):
    """Mean of per-case paired differences and percentile bootstrap CI (case = unit)."""
    v = np.asarray(case_deltas, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


def exact_sign_flip_p(case_deltas):
    """Two-sided exact sign-flip test over all 2^n sign patterns (n <= ~22 is practical)."""
    v = np.asarray(case_deltas, float)
    obs = abs(v.mean())
    count = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(v)):
        count += abs((v * np.asarray(signs)).mean()) >= obs - 1e-15
        total += 1
    return count / total
