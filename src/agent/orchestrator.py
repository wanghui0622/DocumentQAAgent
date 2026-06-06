from __future__ import annotations

import re

from loguru import logger
from openai import APIConnectionError, APIStatusError

from src.agent.generator import AnswerGenerator
from src.agent.router import QueryRouter
from src.agent.verifier import AnswerVerifier
from src.config.settings import Settings
from src.indexing.retriever import HybridRetriever
from src.models import QAResponse, RetrievalHit, SelfCheckResult


class DocumentQAAgent:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.router = QueryRouter()
        self.retriever = HybridRetriever(self.settings)
        self.generator = AnswerGenerator(self.settings)
        self.verifier = AnswerVerifier(self.settings)

    def ask(self, question: str, doc_ids: list[str] | None = None) -> QAResponse:
        query_type = self.router.classify(question)
        clause_filter = HybridRetriever.extract_clause_filter(question) if query_type == "clause_lookup" else None

        hits = self.retriever.retrieve(
            question,
            query_type=query_type,
            clause_filter=clause_filter,
            doc_ids=doc_ids,
        )

        if not hits:
            return self._refuse(question, query_type, hits, "REFUSE_NO_EVIDENCE: 未检索到相关文档片段", doc_ids)

        if query_type == "out_of_scope_hint":
            out_of_scope_reason = self._check_out_of_scope_refusal(question, hits)
            if out_of_scope_reason:
                return self._refuse(question, query_type, hits, out_of_scope_reason, doc_ids)

        max_score = max(hit.score for hit in hits)
        if max_score < self.settings.retrieval_score_threshold:
            reason = (
                f"REFUSE_NO_EVIDENCE: 最高检索分数 {max_score:.3f} "
                f"低于阈值 {self.settings.retrieval_score_threshold}"
            )
            if query_type == "out_of_scope_hint":
                reason += "；该问题可能超出本文档范围"
            return self._refuse(question, query_type, hits, reason, doc_ids)

        if query_type == "out_of_scope_hint":
            return self._refuse(
                question,
                query_type,
                hits,
                "REFUSE_NO_EVIDENCE: 问题涉及文档明确排除或未覆盖的内容",
                doc_ids,
            )

        try:
            answer, citations, confidence = self.generator.generate(question, hits)
        except (APIStatusError, APIConnectionError, RuntimeError) as exc:
            return self._refuse(question, query_type, hits, f"API_ERROR: {exc}", doc_ids)
        if not answer:
            return self._refuse(
                question, query_type, hits, "REFUSE_NO_EVIDENCE: 模型判断证据不足以生成答案", doc_ids
            )

        try:
            self_check = self.verifier.verify(question, answer, citations, hits, confidence)
        except (APIStatusError, APIConnectionError, RuntimeError) as exc:
            return self._refuse(question, query_type, hits, f"API_ERROR: {exc}", doc_ids)
        should_refuse, refuse_reason = self.verifier.should_refuse(self_check, confidence)
        if should_refuse:
            return self._refuse(question, query_type, hits, refuse_reason, doc_ids, self_check)

        return QAResponse(
            question=question,
            answer=answer,
            status="answered",
            refuse_reason=None,
            citations=citations,
            retrieved_chunks=hits,
            self_check=self_check,
            query_type=query_type,
            confidence=confidence,
            doc_ids=doc_ids or sorted({hit.metadata.get("doc_id", "") for hit in hits if hit.metadata.get("doc_id")}),
        )

    @staticmethod
    def _check_out_of_scope_refusal(question: str, hits: list[RetrievalHit]) -> str | None:
        evidence_text = "\n".join(hit.text for hit in hits)
        for keyword in QueryRouter.OUT_OF_SCOPE_KEYWORDS:
            if keyword not in question:
                continue
            if re.search(rf"除\s*{re.escape(keyword)}\s*外", evidence_text):
                return (
                    f"REFUSE_NO_EVIDENCE: 文档明确说明适用范围为「除{keyword}外」，"
                    f"未包含{keyword}的技术要求"
                )
        return None

    def _refuse(
        self,
        question: str,
        query_type,
        hits,
        reason: str,
        doc_ids: list[str] | None = None,
        self_check: SelfCheckResult | None = None,
    ) -> QAResponse:
        logger.info(f"Refused question: {reason}")
        return QAResponse(
            question=question,
            answer=None,
            status="refused",
            refuse_reason=reason,
            citations=[],
            retrieved_chunks=hits,
            self_check=self_check,
            query_type=query_type,
            confidence=0.0,
            doc_ids=doc_ids or [],
        )
