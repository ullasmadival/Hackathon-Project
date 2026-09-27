from __future__ import annotations

import math
import re

from rapidfuzz import fuzz


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def _tokenize(value):
    if not value:
        return []
    return [tok for tok in re.findall(r"[a-z0-9]+", str(value).lower()) if tok]


def _jaccard(tokens_a, tokens_b):
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _overlap(tokens_a, tokens_b):
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(set(tokens_a) & set(tokens_b)) / max(len(set(tokens_a)), len(set(tokens_b)))


def compute_pair_features(name_a, name_b, address_a, address_b, country_a, country_b, postal_code):
    name_a_norm = str(name_a or "").strip()
    name_b_norm = str(name_b or "").strip()
    address_a_norm = str(address_a or "").strip()
    address_b_norm = str(address_b or "").strip()

    name_tokens_a = _tokenize(name_a_norm)
    name_tokens_b = _tokenize(name_b_norm)
    addr_tokens_a = _tokenize(address_a_norm)
    addr_tokens_b = _tokenize(address_b_norm)

    name_exact = 1.0 if name_a_norm.lower() == name_b_norm.lower() else 0.0
    address_exact = 1.0 if address_a_norm.lower() == address_b_norm.lower() else 0.0
    name_similarity = fuzz.ratio(name_a_norm, name_b_norm) / 100.0
    address_similarity = fuzz.ratio(address_a_norm, address_b_norm) / 100.0
    char_sim = fuzz.ratio(name_a_norm.lower(), name_b_norm.lower()) / 100.0
    token_jaccard = _jaccard(name_tokens_a, name_tokens_b)
    token_overlap = _overlap(name_tokens_a, name_tokens_b)
    token_sort = 1.0 if sorted(name_tokens_a) == sorted(name_tokens_b) else 0.0
    token_set = _jaccard(set(name_tokens_a), set(name_tokens_b))
    length_ratio = safe_div(min(len(name_a_norm), len(name_b_norm)), max(len(name_a_norm), len(name_b_norm))) if max(len(name_a_norm), len(name_b_norm)) else 1.0
    common_tokens = len(set(name_tokens_a) & set(name_tokens_b))

    country_a_norm = str(country_a or "").strip().lower()
    country_b_norm = str(country_b or "").strip().lower()
    country_exact_match = 1.0 if country_a_norm and country_b_norm and country_a_norm == country_b_norm else 0.0
    country_missing_on_either_side = 1.0 if not country_a_norm or not country_b_norm else 0.0
    country_normalized_match = 1.0 if country_exact_match or (country_a_norm and country_b_norm and country_a_norm.split()[0] == country_b_norm.split()[0]) else 0.0

    address_jaccard = _jaccard(addr_tokens_a, addr_tokens_b)
    address_overlap = _overlap(addr_tokens_a, addr_tokens_b)
    postal_match = 1.0 if str(postal_code).strip() and postal_code == postal_code else 0.0
    name_address_mean = (name_similarity + address_similarity) / 2.0
    name_address_max = max(name_similarity, address_similarity)
    weighted_name_address_score = 0.7 * name_similarity + 0.3 * address_similarity
    high_name_high_address = 1.0 if name_similarity > 0.75 and address_similarity > 0.7 else 0.0
    high_name_weak_address = 1.0 if name_similarity > 0.75 and address_similarity <= 0.6 else 0.0
    weak_name_high_address = 1.0 if name_similarity <= 0.6 and address_similarity > 0.7 else 0.0

    return {
        "name_exact": name_exact,
        "name_similarity": name_similarity,
        "address_similarity": address_similarity,
        "char_ngram_similarity": char_sim,
        "token_jaccard": token_jaccard,
        "token_overlap": token_overlap,
        "token_sort_similarity": token_sort,
        "token_set_similarity": token_set,
        "length_ratio": length_ratio,
        "common_token_count": common_tokens,
        "address_exact": address_exact,
        "address_jaccard": address_jaccard,
        "address_overlap": address_overlap,
        "postal_match": postal_match,
        "country_exact_match": country_exact_match,
        "country_missing_on_either_side": country_missing_on_either_side,
        "country_normalized_match": country_normalized_match,
        "name_address_mean": name_address_mean,
        "name_address_max": name_address_max,
        "weighted_name_address_score": weighted_name_address_score,
        "high_name_high_address": high_name_high_address,
        "high_name_weak_address": high_name_weak_address,
        "weak_name_high_address": weak_name_high_address,
    }
