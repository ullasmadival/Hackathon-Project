from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_submission(matching_path: str | Path, candidate_path: str | Path, test_dir: str | Path) -> None:
    matching = pd.read_csv(matching_path, sep="\t") if Path(matching_path).exists() else pd.DataFrame()
    candidate = pd.read_csv(candidate_path, sep="\t") if Path(candidate_path).exists() else pd.DataFrame()

    if matching.empty and candidate.empty:
        raise ValueError("Submission files are empty, nothing to validate.")

    if not {"source1_entity_id", "matched_entity_ids"}.issubset(matching.columns):
        raise ValueError("matching_results.tsv must include source1_entity_id and matched_entity_ids")
    if not {"source1_entity_id", "candidate_entity_ids"}.issubset(candidate.columns):
        raise ValueError("candidate_pairs.tsv must include source1_entity_id and candidate_entity_ids")

    if matching["source1_entity_id"].duplicated().any():
        raise ValueError("Duplicate source1_entity_id rows found in matching_results.tsv")

    for _, row in matching.iterrows():
        ids = [item.strip() for item in str(row["matched_entity_ids"]).split(",") if item.strip()]
        assert all(item.startswith("S2-") or item.startswith("S3-") for item in ids), ids

    test_df = pd.read_csv(Path(test_dir) / "test_source1.tsv", sep="\t")
    required = set(test_df["entity_id"].tolist())
    present = set(matching["source1_entity_id"].tolist())
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Missing Source 1 entities in matching output: {missing[:10]}")

    # Fallback: candidate subset check
    for _, row in matching.iterrows():
        s1 = row["source1_entity_id"]
        ids = [item.strip() for item in str(row["matched_entity_ids"]).split(",") if item.strip()]
        cand_row = candidate[candidate["source1_entity_id"] == s1]
        if cand_row.empty:
            continue
        cand_ids = [item.strip() for item in str(cand_row.iloc[0, 1]).split(",") if item.strip()]
        bad = [item for item in ids if item not in cand_ids]
        if bad:
            raise ValueError(f"Matched IDs are not in the candidate set for {s1}: {bad}")

    return None
