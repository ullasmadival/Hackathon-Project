#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Validate submission output files.")
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    matching = pd.read_csv(args.matching, sep="\t") if Path(args.matching).exists() else pd.DataFrame()
    candidate = pd.read_csv(args.candidate, sep="\t") if Path(args.candidate).exists() else pd.DataFrame()

    if matching.empty or candidate.empty:
        raise ValueError("Matching or candidate files are missing or empty.")
    if "source1_entity_id" not in matching.columns or "matched_entity_ids" not in matching.columns:
        raise ValueError("Matching must contain source1_entity_id and matched_entity_ids")
    if "source1_entity_id" not in candidate.columns or "candidate_entity_ids" not in candidate.columns:
        raise ValueError("Candidate must contain source1_entity_id and candidate_entity_ids")

    test_df = pd.read_csv(Path(args.test_dir) / "test_source1.tsv", sep="\t")
    required_ids = set(test_df["entity_id"].astype(str))
    present_ids = set(matching["source1_entity_id"].astype(str))
    missing = sorted(required_ids - present_ids)
    if missing:
        raise ValueError(f"Missing source1 entities in output: {missing[:10]}")

    for _, row in matching.iterrows():
        match_ids = [sid.strip() for sid in str(row["matched_entity_ids"]).split(",") if sid.strip()]
        if any(not sid.startswith("S2-") and not sid.startswith("S3-") for sid in match_ids):
            raise ValueError(f"Invalid matched id in row: {row}")

    print("Validation passed.")


if __name__ == "__main__":
    main()
