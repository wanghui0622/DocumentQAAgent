from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from src.config.settings import Settings
from src.indexing.embedder import EmbeddingClient
from src.indexing.indexer import DocumentIndexer
from src.models import QueryType, RetrievalHit


class VectorRetriever:
    """Pure vector retrieval using configured embedding model (e.g. text-embedding-v4)."""

    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.indexer = DocumentIndexer(self.settings)
        self._embedder: Optional[EmbeddingClient] = None

    @property
    def embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(self.settings)
        return self._embedder

    def retrieve(
        self,
        query: str,
        query_type: QueryType = "general",
        top_k: int | None = None,
        final_k: int | None = None,
        clause_filter: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievalHit]:
        top_k = top_k or self.settings.retrieval_top_k
        final_k = final_k or self.settings.retrieval_final_k

        chunks = self.indexer.load_chunks()
        if not chunks:
            logger.warning("No chunks available for retrieval")
            return []

        if doc_ids:
            chunks = [chunk for chunk in chunks if chunk.metadata.get("doc_id") in doc_ids]
        if not chunks:
            logger.warning("No chunks matched doc_ids filter")
            return []

        try:
            collection = self.indexer.get_collection()
            query_embedding = self.embedder.embed_query(query)
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, len(chunks)),
                "include": ["documents", "metadatas", "distances"],
            }
            if doc_ids:
                query_kwargs["where"] = {"doc_id": {"$in": doc_ids}}
            vector_result = collection.query(**query_kwargs)
        except Exception as exc:
            logger.error(f"Vector retrieval failed: {exc}")
            return []

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        hits: list[RetrievalHit] = []

        for chunk_id, distance in zip(vector_result["ids"][0], vector_result["distances"][0]):
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue

            score = max(0.0, 1.0 - float(distance))

            if query_type == "table_query" and chunk.metadata.get("block_type") == "table":
                score *= 1.2
            if clause_filter and chunk.metadata.get("clause_id") == clause_filter:
                score *= 1.5

            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                    bm25_score=0.0,
                    vector_score=score,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:final_k]

    @staticmethod
    def extract_clause_filter(query: str) -> str | None:
        match = re.search(r"\b(\d+(?:\.\d+)+|\d+)\b", query)
        if match:
            return match.group(1)
        return None


# Backward-compatible alias
HybridRetriever = VectorRetriever
