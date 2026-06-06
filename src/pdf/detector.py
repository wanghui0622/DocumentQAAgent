from __future__ import annotations

import fitz
from loguru import logger

from src.config.settings import Settings
from src.models import PdfProfile, PdfType
from src.text_utils import bad_text_layer_score


class PdfTypeDetector:
    def __init__(self, settings: Settings | None = None, text_threshold: int | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.text_threshold = text_threshold or self.settings.pdf_text_threshold

    def detect(self, pdf_path: str, force_ocr: bool | None = None) -> PdfProfile:
        use_ocr = force_ocr if force_ocr is not None else self.settings.pdf_force_ocr
        if use_ocr:
            profile = self._build_profile(pdf_path, pdf_type="scanned")
            logger.info("PDF type forced to scanned (--force-ocr / PDF_FORCE_OCR)")
            return profile

        doc = fitz.open(pdf_path)
        low_text_pages = 0
        has_tables_hint = False
        bad_scores: list[float] = []
        image_pages = 0

        for page in doc:
            text = page.get_text("text").strip()
            if len(text) < self.text_threshold:
                low_text_pages += 1
            lowered = text.lower()
            if "表" in text or "table" in lowered or "aql" in lowered:
                has_tables_hint = True
            if page.get_images():
                image_pages += 1
            bad_scores.append(bad_text_layer_score(text))

        page_count = len(doc)
        doc.close()

        low_text_ratio = low_text_pages / page_count if page_count else 1.0
        max_bad_score = max(bad_scores) if bad_scores else 0.0
        avg_bad_score = sum(bad_scores) / len(bad_scores) if bad_scores else 0.0
        image_ratio = image_pages / page_count if page_count else 0.0

        reasons: list[str] = []
        if low_text_ratio >= 0.8:
            reasons.append(f"low_text_pages={low_text_ratio:.0%}")

        page_threshold = self.settings.pdf_bad_text_page_threshold
        avg_threshold = self.settings.pdf_bad_text_avg_threshold
        if max_bad_score >= page_threshold:
            reasons.append(f"bad_text_layer_page={max_bad_score:.2f}>={page_threshold}")
        if avg_bad_score >= avg_threshold:
            reasons.append(f"bad_text_layer_avg={avg_bad_score:.2f}>={avg_threshold}")
        if image_ratio >= 0.5 and max_bad_score >= page_threshold * 0.6:
            reasons.append(f"image_overlay(image_ratio={image_ratio:.0%})")

        pdf_type: PdfType = "scanned" if reasons else "text_native"
        if reasons:
            logger.info(f"PDF classified as scanned: {', '.join(reasons)}")

        return PdfProfile(
            pdf_type=pdf_type,
            page_count=page_count,
            has_tables_hint=has_tables_hint,
            source_file=pdf_path,
        )

    def _build_profile(self, pdf_path: str, pdf_type: PdfType) -> PdfProfile:
        doc = fitz.open(pdf_path)
        has_tables_hint = False
        for page in doc:
            text = page.get_text("text").strip().lower()
            if "表" in text or "table" in text or "aql" in text:
                has_tables_hint = True
                break
        page_count = len(doc)
        doc.close()
        return PdfProfile(
            pdf_type=pdf_type,
            page_count=page_count,
            has_tables_hint=has_tables_hint,
            source_file=pdf_path,
        )
