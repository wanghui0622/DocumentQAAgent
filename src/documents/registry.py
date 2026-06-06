from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config.settings import Settings
from src.models import DocumentBlock, PdfProfile


@dataclass
class DocumentRecord:
    doc_id: str
    source_file: str
    filename: str
    pdf_type: str
    page_count: int
    block_count: int
    indexed_at: str | None = None


class DocumentRegistry:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.settings.processed_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.settings.processed_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    @property
    def registry_path(self) -> Path:
        return self.settings.processed_dir / "registry.json"

    @staticmethod
    def make_doc_id(pdf_path: str | Path) -> str:
        stem = Path(pdf_path).stem
        doc_id = re.sub(r"[^\w\-+.\u4e00-\u9fff]+", "_", stem)
        return doc_id.strip("_") or "document"

    def doc_blocks_path(self, doc_id: str) -> Path:
        return self.documents_dir / doc_id / "blocks.json"

    def list_raw_pdfs(self) -> list[Path]:
        self.settings.raw_dir.mkdir(parents=True, exist_ok=True)
        return sorted(self.settings.raw_dir.glob("*.pdf"))

    def load_records(self) -> list[DocumentRecord]:
        self._migrate_legacy_blocks()
        if not self.registry_path.exists():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return [DocumentRecord(**item) for item in data.get("documents", [])]

    def get_record(self, doc_id: str) -> DocumentRecord | None:
        return next((record for record in self.load_records() if record.doc_id == doc_id), None)

    def save_document(
        self,
        doc_id: str,
        profile: PdfProfile,
        blocks: list[DocumentBlock],
    ) -> DocumentRecord:
        doc_dir = self.documents_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "doc_id": doc_id,
            "profile": {
                "pdf_type": profile.pdf_type,
                "page_count": profile.page_count,
                "has_tables_hint": profile.has_tables_hint,
                "source_file": profile.source_file,
            },
            "blocks": [asdict(block) for block in blocks],
        }
        blocks_path = self.doc_blocks_path(doc_id)
        blocks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        record = DocumentRecord(
            doc_id=doc_id,
            source_file=profile.source_file,
            filename=Path(profile.source_file).name,
            pdf_type=profile.pdf_type,
            page_count=profile.page_count,
            block_count=len(blocks),
            indexed_at=None,
        )
        self._upsert_record(record)
        logger.info(f"Saved document blocks: {doc_id} ({len(blocks)} blocks)")
        return record

    def load_document_blocks(self, doc_id: str) -> tuple[dict, list[DocumentBlock]]:
        path = self.doc_blocks_path(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"Document blocks not found for doc_id={doc_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        blocks = [DocumentBlock(**item) for item in data["blocks"]]
        profile = data.get("profile", {})
        profile["doc_id"] = doc_id
        return profile, blocks

    def load_all_blocks(self, doc_ids: list[str] | None = None) -> list[tuple[dict, list[DocumentBlock]]]:
        records = self.load_records()
        if doc_ids:
            records = [record for record in records if record.doc_id in doc_ids]
        loaded: list[tuple[dict, list[DocumentBlock]]] = []
        for record in records:
            if self.doc_blocks_path(record.doc_id).exists():
                loaded.append(self.load_document_blocks(record.doc_id))
        return loaded

    def mark_indexed(self, doc_ids: list[str] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        records = self.load_records()
        for record in records:
            if doc_ids is None or record.doc_id in doc_ids:
                record.indexed_at = now
        self._save_records(records)

    def remove_document(self, doc_id: str) -> None:
        records = [record for record in self.load_records() if record.doc_id != doc_id]
        self._save_records(records)
        doc_dir = self.documents_dir / doc_id
        if doc_dir.exists():
            for path in doc_dir.glob("*"):
                path.unlink()
            doc_dir.rmdir()

    def _upsert_record(self, record: DocumentRecord) -> None:
        records = [item for item in self.load_records() if item.doc_id != record.doc_id]
        records.append(record)
        records.sort(key=lambda item: item.filename)
        self._save_records(records)

    def _save_records(self, records: list[DocumentRecord]) -> None:
        payload = {"documents": [asdict(record) for record in records]}
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _migrate_legacy_blocks(self) -> None:
        if self.registry_path.exists():
            return
        legacy_path = self.settings.blocks_path
        if not legacy_path.exists():
            return

        data = json.loads(legacy_path.read_text(encoding="utf-8"))
        source_file = data.get("profile", {}).get("source_file", "legacy.pdf")
        doc_id = self.make_doc_id(source_file)
        doc_dir = self.documents_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        migrated = {"doc_id": doc_id, **data}
        self.doc_blocks_path(doc_id).write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")

        record = DocumentRecord(
            doc_id=doc_id,
            source_file=source_file,
            filename=Path(source_file).name,
            pdf_type=data.get("profile", {}).get("pdf_type", "unknown"),
            page_count=data.get("profile", {}).get("page_count", 0),
            block_count=len(data.get("blocks", [])),
            indexed_at=datetime.now(timezone.utc).isoformat() if self.settings.chunks_path.exists() else None,
        )
        self._save_records([record])
        logger.info(f"Migrated legacy blocks.json to document registry: {doc_id}")
