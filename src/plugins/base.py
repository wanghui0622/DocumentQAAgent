from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.models import DocumentBlock, PdfProfile


class DocumentParser(Protocol):
    def supports(self, pdf_profile: PdfProfile) -> bool: ...

    def parse(self, pdf_path: Path, doc_id: str) -> list[DocumentBlock]: ...
