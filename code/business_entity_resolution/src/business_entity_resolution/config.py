from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    root_dir: Path = Path(__file__).resolve().parents[4]
    dataset_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    blocking: Dict[str, Any] = field(default_factory=lambda: {
        "exact_name": True,
        "name_ngrams": True,
        "address_tokens": True,
        "postal_code": True,
        "tfidf": True,
    })
    candidate_limits: Dict[str, Any] = field(default_factory=lambda: {
        "tfidf_top_k": 20,
        "max_candidates_per_entity": 100,
    })
    model: Dict[str, Any] = field(default_factory=lambda: {
        "type": "hist_gradient_boosting",
        "threshold": "auto",
    })

    def __post_init__(self):
        self.dataset_dir = self.root_dir / "student_resource" / "dataset"
        self.train_dir = self.dataset_dir / "train"
        self.test_dir = self.dataset_dir / "test"
        self.output_dir = self.root_dir / "student_resource" / "output"


def load_config(path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is None:
        return cfg
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg
