from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score


class EntityResolutionModel:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(random_state=42, max_depth=8)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def evaluate_model(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    precision = precision_score(y_val, preds, zero_division=0)
    recall = recall_score(y_val, preds, zero_division=0)
    f05 = fbeta_score(y_val, preds, beta=0.5, zero_division=0)
    return {"precision": precision, "recall": recall, "f05": f05}
