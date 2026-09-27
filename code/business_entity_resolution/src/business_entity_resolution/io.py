from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"entity_id", "business_name", "business_address", "country"}


def read_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required TSV file: {path}")
    return pd.read_csv(path, sep="\t")


def validate_dataframe(df: pd.DataFrame, required: Iterable[str] | None = None, source_name: str = "dataset") -> pd.DataFrame:
    required = set(required or REQUIRED_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} missing required columns: {sorted(missing)}")
    return df


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_matching_results(matches: dict[str, list[str]], path: str | Path) -> None:
    out_path = Path(path)
    ensure_directory(out_path.parent)
    rows = []
    for source1_id in sorted(matches.keys()):
        values = matches[source1_id]
        rows.append({"source1_entity_id": source1_id, "matched_entity_ids": ",".join(values)})
    pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(out_path, sep="\t", index=False)


def write_candidate_pairs(candidates: dict[str, list[str]], path: str | Path) -> None:
    out_path = Path(path)
    ensure_directory(out_path.parent)
    rows = []
    for source1_id in sorted(candidates.keys()):
        values = candidates[source1_id]
        rows.append({"source1_entity_id": source1_id, "candidate_entity_ids": ",".join(values)})
    pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(out_path, sep="\t", index=False)
