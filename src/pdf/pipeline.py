from __future__ import annotations

import json
from pathlib import Path

import fitz
from loguru import logger

from src.config.settings import Settings
from src.models import DocumentBlock, PdfProfile
from src.pdf.detector import PdfTypeDetector
from src.pdf.ocr import normalize_ocr_text
from src.pdf.ocr_engine import OcrEngine
from src.pdf.structurer import ClauseParser, LayoutParser, TableParser


class TextExtractor:
    def extract(self, pdf_path: str, doc_id: str = "default") -> list[DocumentBlock]:
        doc = fitz.open(pdf_path)
        blocks: list[DocumentBlock] = []
        counter = 0
        for page_num, page in enumerate(doc, start=1):
            text = normalize_ocr_text(page.get_text("text"))
            if not text:
                continue
            counter += 1
            blocks.append(
                DocumentBlock(
                    block_id=f"{doc_id}_text_{counter}",
                    block_type="paragraph",
                    clause_id=None,
                    page=page_num,
                    content=text,
                    doc_id=doc_id,
                )
            )
        doc.close()
        return blocks


class DocumentStructurer:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.layout_parser = LayoutParser()
        self.table_parser = TableParser()
        self.clause_parser = ClauseParser()

    def structure_ocr_lines(self, lines, doc_id: str = "default") -> list[DocumentBlock]:
        page_map: dict[int, list] = {}
        for line in lines:
            page_map.setdefault(line.page, []).append(line)

        all_blocks: list[DocumentBlock] = []
        for page in sorted(page_map):
            page_lines = page_map[page]
            text_lines, table_lines = self.layout_parser.split_page_lines(page_lines)
            all_blocks.extend(self.clause_parser.parse_clause_blocks(text_lines, doc_id=doc_id))
            all_blocks.extend(self.table_parser.parse_table_blocks(table_lines, doc_id=doc_id))
        return all_blocks


class PdfPipeline:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.detector = PdfTypeDetector(self.settings)
        self.ocr_engine = OcrEngine(self.settings)
        self.text_extractor = TextExtractor()
        self.structurer = DocumentStructurer(self.settings)

    def parse(
        self,
        pdf_path: str,
        doc_id: str | None = None,
        force_ocr: bool | None = None,
    ) -> tuple[PdfProfile, list[DocumentBlock]]:
        pdf_path = str(Path(pdf_path).resolve())
        doc_id = doc_id or Path(pdf_path).stem.replace(" ", "_")
        profile = self.detector.detect(pdf_path, force_ocr=force_ocr)

        logger.info(f"PDF type detected: {profile.pdf_type}, pages={profile.page_count}")

        if profile.pdf_type == "text_native":
            blocks = self.text_extractor.extract(pdf_path, doc_id=doc_id)
        else:
            lines = self.ocr_engine.recognize_pdf(pdf_path)
            blocks = self.structurer.structure_ocr_lines(lines, doc_id=doc_id)

        logger.info(f"Parsed {len(blocks)} document blocks")
        return profile, blocks

    def parse_and_save(self, pdf_path: str, doc_id: str | None = None) -> tuple[PdfProfile, list[DocumentBlock]]:
        profile, blocks = self.parse(pdf_path, doc_id=doc_id)
        self.save_blocks(blocks, profile)
        return profile, blocks

    def save_blocks(self, blocks: list[DocumentBlock], profile: PdfProfile) -> Path:
        self.settings.processed_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile": {
                "pdf_type": profile.pdf_type,
                "page_count": profile.page_count,
                "has_tables_hint": profile.has_tables_hint,
                "source_file": profile.source_file,
            },
            "blocks": [
                {
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "clause_id": block.clause_id,
                    "page": block.page,
                    "content": block.content,
                    "bbox": block.bbox,
                    "table_id": block.table_id,
                    "doc_id": block.doc_id,
                }
                for block in blocks
            ],
        }
        output_path = self.settings.blocks_path
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved blocks to {output_path}")
        return output_path

    @staticmethod
    def load_blocks(path: Path | None = None) -> tuple[dict, list[DocumentBlock]]:
        from src.config.settings import get_settings

        settings = get_settings()
        path = path or settings.blocks_path
        data = json.loads(path.read_text(encoding="utf-8"))
        blocks = [DocumentBlock(**item) for item in data["blocks"]]
        return data["profile"], blocks
