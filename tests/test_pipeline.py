import pandas as pd

from business_entity_resolution.normalization.names import normalize_business_name
from business_entity_resolution.normalization.addresses import normalize_address
from business_entity_resolution.blocking.candidate_generator import CandidateGenerator
from business_entity_resolution.features.pair_features import compute_pair_features


def test_name_normalization_handles_suffixes_and_spacing():
    value = "  ABC Pvt Ltd. & Co  "
    norm = normalize_business_name(value)
    assert "abc" in norm
    assert "pvt" in norm or "private" in norm
    assert "co" in norm
    assert "  " not in norm


def test_address_normalization_handles_punctuation_and_spacing():
    value = "123 MG Road, Bangalore!"
    norm = normalize_address(value)
    assert "123" in norm
    assert "mg" in norm
    assert "road" in norm
    assert "bangalore" in norm


def test_candidate_generator_removes_duplicates_and_invalid_sources():
    rows = pd.DataFrame(
        [
            {"entity_id": "S1-1", "business_name": "ABC Corp", "business_address": "123 MG Road", "country": "India"},
            {"entity_id": "S2-1", "business_name": "ABC Corporation", "business_address": "123 MG Road", "country": "India"},
            {"entity_id": "S3-1", "business_name": "XYZ", "business_address": "9 Main Street", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Other", "business_address": "Nope", "country": "US"},
        ]
    )
    gen = CandidateGenerator()
    cands = gen.generate_candidates(rows.iloc[:1], rows.iloc[1:3], max_candidates_per_entity=10)
    assert "S2-1" in cands["S1-1"]
    assert "S1-1" not in cands["S1-1"]
    assert all(x.startswith("S2-") or x.startswith("S3-") for x in cands["S1-1"])


def test_pair_features_are_numeric_and_handles_missing_values():
    feat = compute_pair_features(
        "ABC Pvt Ltd",
        "ABC Private Limited",
        "123 MG Road, Bangalore",
        "123 Mahatma Gandhi Rd, Bengaluru",
        "India",
        "India",
        "",
    )
    assert feat["name_similarity"] >= 0.5
    assert feat["address_similarity"] >= 0.0
    assert feat["country_exact_match"] in (0, 1)
    assert feat["country_missing_on_either_side"] in (0, 1)


def test_output_schema_is_valid_for_source1_entities():
    source1 = pd.DataFrame([{"entity_id": "S1-001", "business_name": "A", "business_address": "X", "country": "India"}, {"entity_id": "S1-002", "business_name": "B", "business_address": "Y", "country": "US"}])
    source2 = pd.DataFrame([{"entity_id": "S2-001", "business_name": "A", "business_address": "X", "country": "India"}])
    source3 = pd.DataFrame([{"entity_id": "S3-001", "business_name": "B", "business_address": "Y", "country": "US"}])
    gen = CandidateGenerator()
    candidates = gen.generate_candidates(source1, pd.concat([source2, source3], ignore_index=True), max_candidates_per_entity=10)
    assert set(candidates.keys()) == {"S1-001", "S1-002"}
    assert all(isinstance(v, list) for v in candidates.values())
