from __future__ import annotations

import re
from dataclasses import dataclass

from src.text_utils import normalize_text_for_match

OCR_CORRECTIONS = {
    "裂縫": "裂纹",
    "抗拉强庋": "抗拉强度",
    "抗拉强渡": "抗拉强度",
    "590MPa": "590 MPa",
    "590M Pa": "590 MPa",
    "氧化皮 ": "氧化皮",
    "浮锈 ": "浮锈",
    "毛刺 ": "毛刺",
    "A QL": "AQL",
    "A.Q.L": "AQL",
    "G B/T": "GB/T",
    "G8/T": "GB/T",
    "GB /T": "GB/T",
    "GB/T ": "GB/T",
    "l": "1",
}

CLAUSE_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s*[\.\、]?\s*(.*)$")
TABLE_HINT_PATTERN = re.compile(r"表\s*(\d+)")


@dataclass
class OcrLine:
    text: str
    bbox: list[float]
    confidence: float
    page: int


def normalize_ocr_text(text: str) -> str:
    result = normalize_text_for_match(text)
    for wrong, correct in OCR_CORRECTIONS.items():
        if wrong == "l":
            continue
        result = result.replace(wrong, correct)
    result = re.sub(r"\s+", " ", result)
    return result.strip()


def extract_clause_id(text: str) -> tuple[str | None, str]:
    match = CLAUSE_PATTERN.match(text.strip())
    if not match:
        return None, text
    clause_id, rest = match.groups()
    if not rest and len(clause_id.split(".")) == 1 and clause_id.isdigit():
        return None, text
    if len(clause_id) > 6:
        return None, text
    return clause_id, rest.strip() or text


def is_table_hint(text: str) -> bool:
    lowered = text.lower()
    return bool(TABLE_HINT_PATTERN.search(text)) or "aql" in lowered or "合格质量水平" in text
