from __future__ import annotations

import re
import unicodedata


def normalize_text_for_match(text: str) -> str:
    """Normalize full-width digits/punctuation and compatibility forms for comparison."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_numbers(text: str) -> set[str]:
    normalized = normalize_text_for_match(text)
    return set(re.findall(r"\d+(?:\.\d+)?", normalized))


GARBLED_TEXT_PATTERN = re.compile(r"犌犅|犜[０-９0-9]|犐犮犛|犲犮犺狀犻犮")


def bad_text_layer_score(text: str) -> float:
    """Estimate how corrupted an embedded PDF text layer is (0=normal, 1=bad)."""
    if not text:
        return 0.0

    length = len(text)
    fullwidth = sum(
        1 for char in text if "\uff00" <= char <= "\uffef" or "\u3000" <= char <= "\u303f"
    )
    rare_cjk = sum(1 for char in text if "\u72c0" <= char <= "\u737f")
    garbled_hits = len(GARBLED_TEXT_PATTERN.findall(text))

    score = (
        (fullwidth / length) * 0.65
        + min(rare_cjk / max(length, 1) * 4.0, 0.25)
        + min(garbled_hits * 0.08, 0.2)
    )
    return min(score, 1.0)


def fuzzy_contains(haystack: str, needle: str, threshold: float = 0.6) -> bool:
    haystack_norm = normalize_text_for_match(haystack)
    needle_norm = normalize_text_for_match(needle)
    if not needle_norm:
        return False
    if needle_norm in haystack_norm:
        return True
    needle_chars = set(needle_norm)
    haystack_chars = set(haystack_norm)
    overlap = len(needle_chars & haystack_chars) / max(len(needle_chars), 1)
    return overlap >= threshold
