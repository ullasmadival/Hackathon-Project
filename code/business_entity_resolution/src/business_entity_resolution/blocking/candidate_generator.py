from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from business_entity_resolution.normalization.addresses import address_tokens
from business_entity_resolution.normalization.names import normalize_business_name


class CandidateGenerator:
    def __init__(self, max_candidates_per_entity: int = 100):
        self.max_candidates_per_entity = max_candidates_per_entity

    def generate_candidates(self, source1: pd.DataFrame, source2_and_3: pd.DataFrame, max_candidates_per_entity: int | None = None):
        source1 = source1.copy()
        source2_and_3 = source2_and_3.copy()
        if source1.empty:
            return {}
        if max_candidates_per_entity is None:
            max_candidates_per_entity = self.max_candidates_per_entity

        source1["_norm_name"] = source1["business_name"].map(normalize_business_name)
        source2_and_3["_norm_name"] = source2_and_3["business_name"].map(normalize_business_name)

        source1["_key"] = source1["entity_id"]
        source2_and_3["_key"] = source2_and_3["entity_id"]

        candidate_map: dict[str, set[str]] = defaultdict(set)
        exact = defaultdict(set)
        for _, row in source2_and_3[["entity_id", "_norm_name"]].dropna().iterrows():
            if row["_norm_name"]:
                exact[row["_norm_name"]].add(row["entity_id"])

        for _, row in source1[["entity_id", "_norm_name"]].dropna().iterrows():
            norm = row["_norm_name"]
            if norm and norm in exact:
                candidate_map[row["entity_id"]].update(exact[norm])

        addr_index = defaultdict(set)
        for _, row in source2_and_3[["entity_id", "business_address"]].dropna().iterrows():
            for token in address_tokens(row["business_address"]):
                if token:
                    addr_index[token].add(row["entity_id"])

        for _, row in source1[["entity_id", "business_address"]].dropna().iterrows():
            s1_id = row["entity_id"]
            for token in address_tokens(row["business_address"]):
                candidate_map[s1_id].update(addr_index.get(token, set()))

        # Aggregate and keep only S2-/S3- candidate IDs
        result: dict[str, list[str]] = {}
        source1_ids = set(source1["entity_id"].tolist())
        for s1_id in sorted(source1_ids):
            cands = sorted({
                cand
                for cand in candidate_map.get(s1_id, set())
                if (cand.startswith("S2-") or cand.startswith("S3-")) and cand not in source1_ids
            })
            if len(cands) > max_candidates_per_entity:
                cands = cands[:max_candidates_per_entity]
            result[s1_id] = cands
        return result
