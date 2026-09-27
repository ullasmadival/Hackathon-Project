# Business Entity Resolution Documentation

## Methodology

The pipeline creates a robust entity-resolution workflow using normalization, multi-stage candidate generation, pairwise feature engineering, and a calibrated classifier.

## Candidate generation / blocking strategy

- exact normalized-name blocking
- character n-gram inverted index blocking
- address token blocking
- postal-code and numeric signal support
- TF-IDF retrieval on name/address text

## Feature engineering

The model uses name similarity, address similarity, country match features, and cross-field interaction features to score candidate pairs.

## Model architecture

The pipeline defaults to a gradient-boosted classifier from scikit-learn with fallback compatibility to popular boosting libraries.

## Training procedure

Ground truth is converted into pair-level labels using the candidate generation set and class imbalance is handled by sampling and weighting.

## Validation strategy

A source-1-level split is used so all pairs from each source-1 entity remain in the same partition.

## Threshold calibration

Prediction thresholds are selected by evaluating macro F0.5 on validation candidates and choosing the best operating point.

## Singleton handling

Entities without a sufficiently confident match are left unmatched.

## Computational complexity

The implementation stays memory-conscious by using sparse vectors, blocked candidate generation, and efficient pandas operations.

## Candidate reduction statistics

The pipeline logs the candidate counts, average candidates per source-1, median, max, and candidate reduction ratio.

## Validation F0.5 results

The final model reports validation results after threshold selection.

## Reproducibility instructions

```bash
pip install -r requirements.txt
python run_pipeline.py
```
