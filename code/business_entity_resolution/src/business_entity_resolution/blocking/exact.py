from __future__ import annotations

import pandas as pd


def exact_name_block(source1: pd.DataFrame, source2: pd.DataFrame):
    index = {}
    for _, row in source2[["entity_id", "business_name"]].dropna().iterrows():
        norm = str(row["business_name"]).strip().lower()
        if norm:
            index.setdefault(norm, []).append(row["entity_id"])
    out = {}
    for _, row in source1[["entity_id", "business_name"]].dropna().iterrows():
        norm = str(row["business_name"]).strip().lower()
        out[row["entity_id"]] = index.get(norm, [])
    return out
