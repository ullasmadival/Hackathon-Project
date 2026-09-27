from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from business_entity_resolution.config import load_config
from business_entity_resolution.features.pair_features import compute_pair_features
from business_entity_resolution.io import read_tsv, validate_dataframe, write_candidate_pairs, write_matching_results
from business_entity_resolution.models.calibration import select_threshold
from business_entity_resolution.models.train import EntityResolutionModel, evaluate_model
from business_entity_resolution.normalization.addresses import normalize_address
from business_entity_resolution.normalization.names import normalize_business_name
from business_entity_resolution.validation import validate_submission


LOGGER = logging.getLogger(__name__)


def _ensure_source_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    for required in ["entity_id", "business_name", "business_address", "country"]:
        if required not in df.columns:
            raise ValueError(f"{label} is missing required column: {required}")
    df["entity_id"] = df["entity_id"].astype(str)
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["normalized_name"] = df["business_name"].map(normalize_business_name)
    df["normalized_address"] = df["business_address"].map(normalize_address)
    return df


def _build_training_pairs(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame, ground_truth: pd.DataFrame):
    all_secondaries = pd.concat([source2, source3], ignore_index=True)
    gen = CandidateGenerator(max_candidates_per_entity=100)
    candidates = gen.generate_candidates(source1, all_secondaries)

    pair_rows = []
    match_ids = ground_truth.set_index("source1_entity_id")["matched_entity_ids"].fillna("")
    for s1_id, candidate_ids in candidates.items():
        truth_ids = set(str(match_ids.get(s1_id, "")).split(",")) if str(match_ids.get(s1_id, "")) else set()
        truth_ids = {x.strip() for x in truth_ids if x.strip()}
        for cand_id in candidate_ids:
            label = 1 if cand_id in truth_ids else 0
            s1_row = source1[source1["entity_id"] == s1_id].iloc[0]
            cand_row = all_secondaries[all_secondaries["entity_id"] == cand_id]
            if cand_row.empty:
                continue
            cand_row = cand_row.iloc[0]
            feat = compute_pair_features(
                s1_row["business_name"],
                cand_row["business_name"],
                s1_row["business_address"],
                cand_row["business_address"],
                s1_row["country"],
                cand_row["country"],
                "",
            )
            row = {"source1_entity_id": s1_id, "candidate_entity_id": cand_id, "label": label}
            row.update(feat)
            pair_rows.append(row)

    return pd.DataFrame(pair_rows)


def _generate_test_candidates(source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame):
    all_secondaries = pd.concat([source2, source3], ignore_index=True)
    gen = CandidateGenerator(max_candidates_per_entity=100)
    return gen.generate_candidates(source1, all_secondaries)


def run_pipeline():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    cfg = load_config()
    root = cfg.root_dir

    train_dir = root / "student_resource" / "dataset" / "train"
    test_dir = root / "student_resource" / "dataset" / "test"
    output_dir = root / "student_resource" / "output"

    logging.info("Loading data...")
    source1_train = validate_dataframe(read_tsv(train_dir / "train_source1.tsv"), source_name="train_source1")
    source2_train = validate_dataframe(read_tsv(train_dir / "train_source2.tsv"), source_name="train_source2")
    source3_train = validate_dataframe(read_tsv(train_dir / "train_source3.tsv"), source_name="train_source3")
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")
    source1_test = validate_dataframe(read_tsv(test_dir / "test_source1.tsv"), source_name="test_source1")
    source2_test = validate_dataframe(read_tsv(test_dir / "test_source2.tsv"), source_name="test_source2")
    source3_test = validate_dataframe(read_tsv(test_dir / "test_source3.tsv"), source_name="test_source3")

    source1_train = _ensure_source_columns(source1_train, "train_source1")
    source2_train = _ensure_source_columns(source2_train, "train_source2")
    source3_train = _ensure_source_columns(source3_train, "train_source3")
    source1_test = _ensure_source_columns(source1_test, "test_source1")
    source2_test = _ensure_source_columns(source2_test, "test_source2")
    source3_test = _ensure_source_columns(source3_test, "test_source3")

    logging.info("Building training candidate pairs...")
    training_pairs = _build_training_pairs(source1_train, source2_train, source3_train, gt)
    numeric_cols = [c for c in training_pairs.columns if c not in {"source1_entity_id", "candidate_entity_id", "label"}]
    X = training_pairs[numeric_cols].fillna(0.0)
    y = training_pairs["label"].astype(int)

    model = EntityResolutionModel()
    model.fit(X, y)
    val_scores = model.predict_proba(X)[:, 1]
    best_threshold = select_threshold(y.to_numpy(), val_scores)

    logging.info("Generating test candidates...")
    test_candidates = _generate_test_candidates(source1_test, source2_test, source3_test)
    write_candidate_pairs(test_candidates, output_dir / "candidate_pairs.tsv")

    final_matches = {}
    for source1_id in source1_test["entity_id"].tolist():
        candidates = test_candidates.get(source1_id, [])
        selected = []
        for cand_id in candidates:
            cand_row = pd.concat([source2_test, source3_test], ignore_index=True)
            cand_row = cand_row[cand_row["entity_id"] == cand_id]
            if cand_row.empty:
                continue
            cand_row = cand_row.iloc[0]
            s1_row = source1_test[source1_test["entity_id"] == source1_id].iloc[0]
            feat = compute_pair_features(
                s1_row["business_name"],
                cand_row["business_name"],
                s1_row["business_address"],
                cand_row["business_address"],
                s1_row["country"],
                cand_row["country"],
                "",
            )
            row = pd.DataFrame([feat])
            prob = model.predict_proba(row[numeric_cols].fillna(0.0))[0, 1]
            if prob >= best_threshold:
                selected.append(cand_id)
        final_matches[source1_id] = selected

    write_matching_results(final_matches, output_dir / "matching_results.tsv")
    validate_submission(output_dir / "matching_results.tsv", output_dir / "candidate_pairs.tsv", test_dir)

    logging.info("SUCCESS")
    return {"candidate_pairs": test_candidates, "matches": final_matches, "threshold": best_threshold}
