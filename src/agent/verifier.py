from __future__ import annotations

import json

from openai import OpenAI

from src.text_utils import extract_numbers
from src.models import Citation, GroundingLevel, RetrievalHit, SelfCheckResult
from src.text_utils import extract_numbers, fuzzy_contains, normalize_text_for_match


class AnswerVerifier:
    GROUNDING_PROMPT = """判断答案是否被证据完全支持。
输出 JSON: {"grounding_level": "supported|partial|unsupported", "grounding_score": 0.0-1.0, "reason": "..."}
"""

    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_api_base,
        )

    def verify(
        self,
        question: str,
        answer: str | None,
        citations: list[Citation],
        evidence: list[RetrievalHit],
        confidence: float,
    ) -> SelfCheckResult:
        evidence_text = "\n".join(hit.text for hit in evidence)
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        citation_valid = True
        for citation in citations:
            if not fuzzy_contains(evidence_text, citation.quote):
                citation_valid = False
                checks_failed.append(f"citation_not_in_evidence: {citation.quote[:30]}")
            else:
                checks_passed.append("citation_match")

        if not citations:
            citation_valid = False
            checks_failed.append("missing_citations")

        numeric_valid = True
        if answer:
            answer_numbers = extract_numbers(answer)
            evidence_numbers = extract_numbers(evidence_text)
            extra_numbers = answer_numbers - evidence_numbers
            if extra_numbers:
                normalized_evidence = normalize_text_for_match(evidence_text)
                still_missing = {num for num in extra_numbers if num not in normalized_evidence}
                if still_missing:
                    numeric_valid = False
                    checks_failed.append(f"unsupported_numbers: {sorted(still_missing)}")
                else:
                    checks_passed.append("numeric_check")
            else:
                checks_passed.append("numeric_check")

        grounding_level, grounding_score = self._llm_grounding(question, answer, evidence_text)
        if grounding_level == "supported":
            checks_passed.append("llm_grounding_supported")
        elif grounding_level == "partial":
            checks_failed.append("llm_grounding_partial")
        else:
            checks_failed.append("llm_grounding_unsupported")

        if confidence < self.settings.verifier_min_grounding:
            checks_failed.append("low_generator_confidence")

        return SelfCheckResult(
            grounding_level=grounding_level,
            grounding_score=grounding_score,
            citation_valid=citation_valid,
            numeric_valid=numeric_valid,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def should_refuse(self, self_check: SelfCheckResult, confidence: float) -> tuple[bool, str | None]:
        if self_check.grounding_level == "unsupported":
            return True, "REFUSE_HALLUCINATION: 答案无法被检索证据支持"
        if not self_check.citation_valid:
            return True, "REFUSE_HALLUCINATION: 引用片段无法在证据中找到"
        if not self_check.numeric_valid:
            return True, "REFUSE_HALLUCINATION: 答案包含证据中不存在的数值"
        if self_check.grounding_level == "partial" and confidence < self.settings.verifier_min_grounding:
            return True, "REFUSE_INSUFFICIENT: 证据不足，无法给出可靠答案"
        if self_check.grounding_score < self.settings.verifier_min_grounding:
            return True, "REFUSE_INSUFFICIENT: 自检置信度过低"
        return False, None

    def _llm_grounding(self, question: str, answer: str | None, evidence_text: str) -> tuple[GroundingLevel, float]:
        if not answer:
            return "unsupported", 0.0

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": self.GROUNDING_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence_text[:6000]}",
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            level = data.get("grounding_level", "partial")
            if level not in ("supported", "partial", "unsupported"):
                level = "partial"
            score = float(data.get("grounding_score", 0.5) or 0.5)
            return level, score
        except Exception:
            if fuzzy_contains(evidence_text, answer[: min(20, len(answer))]):
                return "partial", 0.6
            return "unsupported", 0.2
