from __future__ import annotations

import fitz
import numpy as np
from loguru import logger

from src.config.settings import Settings
from src.pdf.ocr import OcrLine, normalize_ocr_text


class OcrEngine:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR is required for scanned PDF parsing. "
                    "Install with: pip install -e '.[ocr]'"
                ) from exc

            logger.info("Initializing PaddleOCR (first run may download models)...")
            self._ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
        return self._ocr

    @staticmethod
    def _parse_ocr_result(result) -> list[tuple[list, str, float]]:
        """Normalize PaddleOCR 2.x / 3.x result shapes into (bbox, text, confidence)."""
        if not result:
            return []

        parsed: list[tuple[list, str, float]] = []

        for page_result in result:
            if hasattr(page_result, "get") and page_result.get("rec_texts"):
                texts = page_result.get("rec_texts") or []
                scores = page_result.get("rec_scores") or [1.0] * len(texts)
                polys = page_result.get("rec_polys") or page_result.get("dt_polys") or []
                for idx, text in enumerate(texts):
                    bbox = polys[idx] if idx < len(polys) else []
                    score = float(scores[idx]) if idx < len(scores) else 1.0
                    parsed.append((bbox, str(text), score))
                continue

            items = page_result if isinstance(page_result, list) else [page_result]
            for item in items:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                bbox, text_payload = item[0], item[1]
                if isinstance(text_payload, (list, tuple)) and text_payload:
                    text, confidence = str(text_payload[0]), float(text_payload[1])
                elif isinstance(text_payload, str):
                    text, confidence = text_payload, 1.0
                else:
                    continue
                parsed.append((bbox, text, confidence))

        return parsed

    def render_page(self, page: fitz.Page, dpi: int | None = None) -> np.ndarray:
        dpi = dpi or self.settings.ocr_dpi
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = image[:, :, :3]
        return image

    def recognize_page(self, page: fitz.Page, page_num: int) -> list[OcrLine]:
        image = self.render_page(page)
        try:
            result = self.ocr.predict(image)
        except AttributeError:
            result = self.ocr.ocr(image, cls=True)
        lines: list[OcrLine] = []

        for bbox, text, confidence in self._parse_ocr_result(result):
            cleaned = normalize_ocr_text(text)
            if not cleaned:
                continue
            flat_bbox = [coord for point in bbox for coord in point]
            lines.append(
                OcrLine(
                    text=cleaned,
                    bbox=flat_bbox,
                    confidence=float(confidence),
                    page=page_num,
                )
            )
        return lines

    def recognize_pdf(self, pdf_path: str) -> list[OcrLine]:
        doc = fitz.open(pdf_path)
        all_lines: list[OcrLine] = []
        for idx, page in enumerate(doc, start=1):
            logger.info(f"OCR page {idx}/{len(doc)}")
            all_lines.extend(self.recognize_page(page, idx))
        doc.close()
        return all_lines
