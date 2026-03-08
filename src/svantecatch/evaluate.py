"""
Threshold tuning and classification-metric helpers.

The trained pipeline outputs continuous probabilities for the positive
class (Svante).  These utilities convert probabilities into binary
decisions by sweeping over candidate thresholds and selecting the one
that maximises a configurable metric (default: F1) on a held-out
validation fold.

The 200-point grid in ``[0.01, 0.99]`` provides ~0.005 resolution,
which is sufficient for this task.  Finer grids were tested but yielded
negligible improvement while increasing evaluation time.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Evaluate precision, recall, F1, and confusion counts at *threshold*.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (``1`` = Svante, ``0`` = not Svante).
    y_prob : array-like of shape (n_samples,)
        Model-predicted probability of the positive class.
    threshold : float
        Decision boundary: predict positive when ``y_prob >= threshold``.

    Returns
    -------
    dict[str, float]
        Keys: ``threshold``, ``f1``, ``precision``, ``recall``,
        ``tn``, ``fp``, ``fn``, ``tp``.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def add_auc_metrics(
    stats: Dict[str, float], y_true: np.ndarray, y_prob: np.ndarray
) -> None:
    """Augment *stats* in-place with ``roc_auc`` and ``pr_auc``.

    Both metrics are threshold-independent and characterise the full
    ranking quality of the model.  They are wrapped in try/except
    because AUC is undefined when only one class is present in
    *y_true* (e.g. in very small validation folds).

    Parameters
    ----------
    stats : dict
        Existing metrics dict to augment (modified in-place).
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    """
    try:
        stats["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        stats["roc_auc"] = float("nan")
    try:
        stats["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        stats["pr_auc"] = float("nan")


def pick_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> Tuple[float, Dict[str, float]]:
    """Sweep 200 thresholds in ``[0.01, 0.99]`` and return the best.

    The grid avoids the extremes ``0`` and ``1`` to prevent degenerate
    all-positive or all-negative predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    metric : str, default ``"f1"``
        Name of the metric to maximise.  Must be a key in the dict
        returned by :func:`compute_threshold_metrics` (one of
        ``"f1"``, ``"precision"``, or ``"recall"``).

    Returns
    -------
    best_threshold : float
        Threshold that achieved the highest *metric* value.
    best_stats : dict
        Full metrics dict at that threshold, augmented with
        ``roc_auc`` and ``pr_auc``.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t = 0.5
    best = -1.0
    best_stats: Dict[str, float] = {}

    for t in thresholds:
        stats = compute_threshold_metrics(y_true, y_prob, float(t))
        score = stats.get(metric, stats["f1"])
        if score > best:
            best = score
            best_t = float(t)
            best_stats = stats

    add_auc_metrics(best_stats, y_true, y_prob)

    return best_t, best_stats
