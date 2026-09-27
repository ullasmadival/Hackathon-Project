# Business Entity Resolution Pipeline

This project provides a modular business entity resolution pipeline designed for noisy, multi-source business records.

## Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python -m pip install -e .
python -m business_entity_resolution.pipeline
```

## Reproducibility

The pipeline reads the training and test TSV files, normalizes names and addresses, generates candidates with multiple blocking strategies, trains a classifier, calibrates the threshold, and writes the required output artifacts.

## Outputs

- output/candidate_pairs.tsv
- output/matching_results.tsv

## Methodology

The implementation follows a multi-stage architecture: validation, normalization, blocking, candidate generation, feature engineering, model training, threshold calibration, and singleton-aware final matching.
