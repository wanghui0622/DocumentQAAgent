from __future__ import annotations

import re

from src.models import QueryType


class QueryRouter:
    TABLE_KEYWORDS = ("表", "AQL", "合格质量水平", "尺寸公差", "验收")
    OUT_OF_SCOPE_KEYWORDS = ("花键", "螺栓", "螺母", "螺钉", "拧紧力矩")
    CLAUSE_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")

    def classify(self, question: str) -> QueryType:
        lowered = question.lower()

        if any(keyword in question for keyword in self.OUT_OF_SCOPE_KEYWORDS):
            return "out_of_scope_hint"

        if any(keyword.lower() in lowered for keyword in [k.lower() for k in self.TABLE_KEYWORDS]):
            return "table_query"

        if self.CLAUSE_PATTERN.search(question):
            return "clause_lookup"

        return "general"
