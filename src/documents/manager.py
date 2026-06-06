from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config.settings import Settings
from src.documents.registry import DocumentRegistry
from src.indexing.indexer import DocumentIndexer
from src.models import DocumentBlock, PdfProfile
from src.pdf.pipeline import PdfPipeline


class DocumentManager:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.registry = DocumentRegistry(self.settings)
        self.pipeline = PdfPipeline(self.settings)
        self.indexer = DocumentIndexer(self.settings)

    def parse_pdf(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
        force_ocr: bool | None = None,
    ) -> tuple[str, PdfProfile, list[DocumentBlock]]:
        pdf_path = Path(pdf_path).resolve()
        doc_id = doc_id or self.registry.make_doc_id(pdf_path)
        profile, blocks = self.pipeline.parse(str(pdf_path), doc_id=doc_id, force_ocr=force_ocr)
        self.registry.save_document(doc_id, profile, blocks)
        return doc_id, profile, blocks

    def parse_all_raw(self, force_ocr: bool | None = None) -> list[tuple[str, PdfProfile, list[DocumentBlock]]]:
        results: list[tuple[str, PdfProfile, list[DocumentBlock]]] = []
        for pdf_path in self.registry.list_raw_pdfs():
            logger.info(f"Parsing {pdf_path.name}")
            results.append(self.parse_pdf(pdf_path, force_ocr=force_ocr))
        return results

    def index_documents(self, doc_ids: list[str] | None = None) -> int:
        loaded = self.registry.load_all_blocks(doc_ids=None)
        if not loaded:
            raise FileNotFoundError("No parsed documents found. Run parse first.")

        all_blocks: list[DocumentBlock] = []
        source_map: dict[str, str] = {}
        for profile, blocks in loaded:
            source_file = profile.get("source_file", "")
            doc_id = profile.get("doc_id", "")
            source_map[doc_id] = source_file
            for block in blocks:
                block.doc_id = doc_id or block.doc_id
            all_blocks.extend(blocks)

        chunks = self.indexer.build_from_blocks(all_blocks, source_files=source_map)
        indexed_ids = doc_ids or [profile.get("doc_id") for profile, _ in loaded]
        self.registry.mark_indexed([doc_id for doc_id in indexed_ids if doc_id])
        logger.info(f"Unified index rebuilt from {len(loaded)} document(s), {len(chunks)} chunks")
        return len(chunks)

    def list_status(self) -> list[dict]:
        records = {record.doc_id: record for record in self.registry.load_records()}
        status: list[dict] = []
        for pdf_path in self.registry.list_raw_pdfs():
            doc_id = self.registry.make_doc_id(pdf_path)
            record = records.get(doc_id)
            status.append(
                {
                    "doc_id": doc_id,
                    "filename": pdf_path.name,
                    "parsed": self.registry.doc_blocks_path(doc_id).exists(),
                    "indexed": bool(record and record.indexed_at),
                    "block_count": record.block_count if record else 0,
                    "page_count": record.page_count if record else 0,
                    "pdf_type": record.pdf_type if record else None,
                }
            )
        return status
