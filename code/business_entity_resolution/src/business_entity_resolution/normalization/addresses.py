import re

from .text import normalize_text


def normalize_address(value):
    text = normalize_text(value, lowercase=True, remove_punctuation=False)
    text = text.replace("road", " road ")
    text = text.replace("street", " street ")
    text = text.replace("st.", " street ")
    text = text.replace("rd.", " road ")
    text = text.replace("ave", " avenue ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def address_tokens(value):
    text = normalize_address(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t and t not in {"near", "road", "street", "main", "india", "usa"}]
