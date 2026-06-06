from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


BlockType = Literal["clause", "table", "header", "footnote", "paragraph"]
PdfType = Literal["scanned", "text_native"]
QueryType = Literal["clause_lookup", "table_query", "general", "out_of_scope_hint"]
AnswerStatus = Literal["answered", "refused"]
GroundingLevel = Literal["supported", "partial", "unsupported"]


@dataclass
class PdfProfile:
    pdf_type: PdfType
    page_count: int
    has_tables_hint: bool
    source_file: str


@dataclass
class DocumentBlock:
    block_id: str
    block_type: BlockType
    clause_id: str | None
    page: int
    content: str
    bbox: list[float] | None = None
    table_id: str | None = None
    doc_id: str = "default"


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0


@dataclass
class Citation:
    page: int
    clause_id: str | None
    quote: str
    doc_id: str | None = None
    source_file: str | None = None


@dataclass
class SelfCheckResult:
    grounding_level: GroundingLevel
    grounding_score: float
    citation_valid: bool
    numeric_valid: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class QAResponse:
    question: str
    answer: str | None
    status: AnswerStatus
    refuse_reason: str | None
    citations: list[Citation]
    retrieved_chunks: list[RetrievalHit]
    self_check: SelfCheckResult | None
    query_type: QueryType
    confidence: float = 0.0
    doc_ids: list[str] = field(default_factory=list)
