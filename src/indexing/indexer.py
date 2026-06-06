from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config.settings import Settings
from src.indexing.chunker import SemanticChunker
from src.indexing.embedder import EmbeddingClient
from src.models import DocumentBlock, DocumentChunk


class DocumentIndexer:
    COLLECTION_NAME = "document_qa"

    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.chunker = SemanticChunker()
        self._embedder: Optional[EmbeddingClient] = None
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @property
    def embedder(self) -> EmbeddingClient:
        if self._embedder is None:
            self._embedder = EmbeddingClient(self.settings)
        return self._embedder

    def build_from_blocks(
        self,
        blocks: list[DocumentBlock],
        source_file: str = "",
        source_files: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        chunks = self.chunker.chunk_blocks(blocks, source_file=source_file, source_files=source_files)
        self._save_chunks(chunks)
        self._build_vector_store(chunks)
        return chunks

    def _save_chunks(self, chunks: list[DocumentChunk]) -> None:
        payload = [
            {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
            for chunk in chunks
        ]
        self.settings.processed_dir.mkdir(parents=True, exist_ok=True)
        self.settings.chunks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _ensure_embedding_available(self) -> None:
        if not self.settings.openai_api_key or self.settings.openai_api_key.startswith("sk-your"):
            raise RuntimeError("OPENAI_API_KEY 未配置，无法构建 text-embedding-v4 向量索引")

    def _build_vector_store(self, chunks: list[DocumentChunk]) -> None:
        self._ensure_embedding_available()
        logger.info(f"Building vector index with embedding model: {self.settings.embedding_model}")

        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        collection = self.client.create_collection(name=self.COLLECTION_NAME)
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.embed_texts([chunk.text for chunk in batch])
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[chunk.metadata for chunk in batch],
            )
        logger.info(f"Built Chroma collection with {len(chunks)} chunks")

    @classmethod
    def load_chunks(cls, path: Path | None = None) -> list[DocumentChunk]:
        from src.config.settings import get_settings

        settings = get_settings()
        path = path or settings.chunks_path
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [DocumentChunk(**item) for item in data]

    def get_collection(self):
        return self.client.get_or_create_collection(name=self.COLLECTION_NAME)
