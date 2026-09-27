import re
import unicodedata


def normalize_text(value, lowercase=True, remove_punctuation=False):
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"[\u00A0\t\r]+", " ", text)
    text = text.replace("\n", " ")
    if remove_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)
    else:
        text = text.replace(".", " ")
        text = text.replace(",", " ")
        text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text
