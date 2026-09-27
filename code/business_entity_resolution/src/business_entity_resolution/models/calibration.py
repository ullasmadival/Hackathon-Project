from __future__ import annotations

import numpy as np


def select_threshold(y_true, scores):
    thresholds = np.linspace(0.1, 0.99, 90)
    best = {"threshold": 0.5, "f05": -1.0}
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        if f05 > best["f05"]:
            best = {"threshold": float(threshold), "f05": float(f05)}
    return best["threshold"]
