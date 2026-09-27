from __future__ import annotations

from collections import defaultdict


def build_inverted_index(records, field_name):
    index = defaultdict(set)
    for record in records:
        tokens = record.get(field_name, "").lower().split()
        for token in tokens:
            index[token].add(record["entity_id"])
    return index
