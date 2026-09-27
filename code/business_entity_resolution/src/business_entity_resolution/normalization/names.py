import re

from .text import normalize_text

LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co",
    "company", "pvt", "private", "plc", "group", "industries", "services", "systems",
    "holdings", "partners", "associates", "enterprise", "enterprises"
}


def normalize_business_name(value):
    text = normalize_text(value, lowercase=True, remove_punctuation=False)
    text = re.sub(r"\s+", " ", text)
    tokens = text.split()
    normalized = []
    for token in tokens:
        token = token.strip(".,;:-/\\")
        token = token.replace("&", " and ")
        if token:
            normalized.append(token)
    text = " ".join(normalized)
    text = re.sub(r"\b(and|& )\b", " and ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Keep meaningful legal suffix tokens but remove obvious noise-only tokens.
    clean_tokens = []
    for token in text.split():
        if token in {"the", "of"}:
            continue
        clean_tokens.append(token)
    return " ".join(clean_tokens)


def name_tokens(value):
    return [t for t in normalize_business_name(value).split() if t]
