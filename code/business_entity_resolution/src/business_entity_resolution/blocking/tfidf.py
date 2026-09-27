from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_candidates(source1_texts: Iterable[str], source2_texts: Iterable[str], top_k: int = 20):
    texts1 = list(source1_texts)
    texts2 = list(source2_texts)
    if not texts1 or not texts2:
        return {}
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts1 + texts2)
    n1 = len(texts1)
    sims = cosine_similarity(matrix[:n1], matrix[n1:])
    result = {}
    for idx, row in enumerate(sims):
        top = np.argsort(row)[::-1][:top_k]
        result[idx] = top.tolist()
    return result
