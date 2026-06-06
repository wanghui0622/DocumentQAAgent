from __future__ import annotations

import json
import re

from openai import APIConnectionError, APIStatusError, OpenAI

from src.config.settings import Settings
from src.models import Citation, RetrievalHit
from src.text_utils import extract_numbers


class AnswerGenerator:
    SYSTEM_PROMPT = """你是文档问答助手。必须严格依据提供的 Evidence 回答，不得使用外部知识。
输出 JSON 格式：
{
  "answer": "简洁准确的答案",
  "citations": [{"page": 1, "clause_id": "3.1", "quote": "证据原文片段", "doc_id": "文档ID"}],
  "confidence": 0.0-1.0
}
规则：
1. 每个结论都必须有 citations 支撑。
2. quote 必须来自 Evidence 原文，尽量短。
3. 若 Evidence 不足以回答，answer 设为 null，confidence 设为 0。
4. 表格问题必须引用表格中的具体数值。
"""

    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_api_base,
        )

    def generate(self, question: str, evidence: list[RetrievalHit]) -> tuple[str | None, list[Citation], float]:
        evidence_text = self._format_evidence(evidence)
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence_text}"

        try:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except APIStatusError as exc:
            detail = exc.message if hasattr(exc, "message") else str(exc)
            raise RuntimeError(f"LLM API 错误 ({exc.status_code}): {detail}") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"LLM API 连接失败: {exc}") from exc

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        answer = data.get("answer")
        confidence = float(data.get("confidence", 0.0) or 0.0)
        citations = []
        for item in data.get("citations", []):
            if not item.get("quote"):
                continue
            doc_id = item.get("doc_id")
            source_file = None
            page = int(item.get("page", 0) or 0)
            for hit in evidence:
                if doc_id and hit.metadata.get("doc_id") != doc_id:
                    continue
                if page and int(hit.metadata.get("page", 0)) != page:
                    continue
                source_file = hit.metadata.get("source_file")
                doc_id = doc_id or hit.metadata.get("doc_id")
                break
            if not doc_id and evidence:
                doc_id = evidence[0].metadata.get("doc_id")
                source_file = evidence[0].metadata.get("source_file")
            citations.append(
                Citation(
                    page=page or int(evidence[0].metadata.get("page", 0) if evidence else 0),
                    clause_id=item.get("clause_id") or None,
                    quote=str(item.get("quote", "")).strip(),
                    doc_id=doc_id,
                    source_file=source_file,
                )
            )
        return answer, citations, confidence

    @staticmethod
    def _format_evidence(evidence: list[RetrievalHit]) -> str:
        parts = []
        for idx, hit in enumerate(evidence, start=1):
            meta = hit.metadata
            parts.append(
                f"[{idx}] page={meta.get('page')} clause_id={meta.get('clause_id')} "
                f"type={meta.get('block_type')} score={hit.score:.3f}\n{hit.text}"
            )
        return "\n\n".join(parts)

    extract_numbers = staticmethod(extract_numbers)
