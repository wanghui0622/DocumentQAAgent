from __future__ import annotations

from src.models import DocumentBlock, DocumentChunk


class SemanticChunker:
    def chunk_blocks(
        self,
        blocks: list[DocumentBlock],
        source_file: str = "",
        source_files: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        seen_block_ids: dict[str, int] = {}

        for block in blocks:
            block_source = (source_files or {}).get(block.doc_id, source_file)
            occurrence = seen_block_ids.get(block.block_id, 0)
            seen_block_ids[block.block_id] = occurrence + 1
            chunk_id = block.block_id if occurrence == 0 else f"{block.block_id}_{occurrence}"

            metadata = {
                "doc_id": block.doc_id,
                "page": block.page,
                "clause_id": block.clause_id or "",
                "block_type": block.block_type,
                "source_file": block_source,
                "block_id": block.block_id,
                "table_id": block.table_id or "",
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=block.content,
                    metadata=metadata,
                )
            )

            if block.block_type == "clause" and len(block.content) > 300:
                overlap_chunks = self._split_with_overlap(block, chunk_id, block_source)
                chunks.extend(overlap_chunks)

        return chunks

    def _split_with_overlap(
        self, block: DocumentBlock, base_chunk_id: str, source_file: str, size: int = 200, overlap: int = 50
    ) -> list[DocumentChunk]:
        text = block.content
        parts: list[DocumentChunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(len(text), start + size)
            segment = text[start:end]
            idx += 1
            parts.append(
                DocumentChunk(
                    chunk_id=f"{base_chunk_id}_part_{idx}",
                    text=segment,
                    metadata={
                        "doc_id": block.doc_id,
                        "page": block.page,
                        "clause_id": block.clause_id or "",
                        "block_type": block.block_type,
                        "source_file": source_file,
                        "block_id": block.block_id,
                        "table_id": block.table_id or "",
                        "is_partial": True,
                    },
                )
            )
            if end >= len(text):
                break
            start = end - overlap
        return parts
