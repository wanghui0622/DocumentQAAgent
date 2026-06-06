from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.config.settings import get_settings
from src.indexing.indexer import DocumentIndexer
from src.models import DocumentBlock
from src.pdf.pipeline import PdfPipeline


def seed_sample_index() -> None:
    """Build index from bundled sample blocks for smoke testing without OCR."""
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    fixture = Path(__file__).parent / "sample_blocks.json"
    shutil.copy(fixture, settings.blocks_path)

    data = json.loads(fixture.read_text(encoding="utf-8"))
    blocks = [DocumentBlock(**item) for item in data["blocks"]]
    indexer = DocumentIndexer(settings)
    chunks = indexer.build_from_blocks(blocks, source_file=data["profile"]["source_file"])
    print(f"Seeded {len(chunks)} chunks from sample blocks")


if __name__ == "__main__":
    seed_sample_index()
