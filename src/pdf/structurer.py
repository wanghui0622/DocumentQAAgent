from __future__ import annotations

import re
from collections import defaultdict

from src.models import DocumentBlock
from src.pdf.ocr import OcrLine, extract_clause_id, is_table_hint


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    xs = bbox[0::2]
    ys = bbox[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def group_lines_by_row(lines: list[OcrLine], y_tolerance: float = 12.0) -> list[list[OcrLine]]:
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda line: (_bbox_center(line.bbox)[1], _bbox_center(line.bbox)[0]))
    rows: list[list[OcrLine]] = []
    current_row: list[OcrLine] = []
    current_y: float | None = None

    for line in sorted_lines:
        _, y = _bbox_center(line.bbox)
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current_row.append(line)
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            rows.append(sorted(current_row, key=lambda item: _bbox_center(item.bbox)[0]))
            current_row = [line]
            current_y = y
    if current_row:
        rows.append(sorted(current_row, key=lambda item: _bbox_center(item.bbox)[0]))
    return rows


def rows_to_markdown(rows: list[list[OcrLine]]) -> str:
    if not rows:
        return ""
    table_rows = [[cell.text for cell in row] for row in rows]
    max_cols = max(len(row) for row in table_rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in table_rows]
    header = normalized[0]
    separator = ["---"] * max_cols
    body = normalized[1:] if len(normalized) > 1 else []
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class LayoutParser:
    def split_page_lines(self, lines: list[OcrLine]) -> tuple[list[OcrLine], list[OcrLine]]:
        table_lines: list[OcrLine] = []
        text_lines: list[OcrLine] = []
        in_table = False

        for line in lines:
            if is_table_hint(line.text) or "AQL" in line.text.upper() or "尺寸公差" in line.text:
                in_table = True
            if in_table and ("表" in line.text and "续" not in line.text and len(line.text) < 8):
                in_table = True
            if in_table:
                table_lines.append(line)
            else:
                text_lines.append(line)
        return text_lines, table_lines


class TableParser:
    TABLE1_KEYWORDS = ("AQL", "合格质量水平", "普通", "半圆键", "楔键", "薄型", "导向")

    def parse_table_blocks(self, lines: list[OcrLine], doc_id: str = "default") -> list[DocumentBlock]:
        if not lines:
            return []

        page_groups: dict[int, list[OcrLine]] = defaultdict(list)
        for line in lines:
            page_groups[line.page].append(line)

        blocks: list[DocumentBlock] = []
        table_counter = 0

        for page, page_lines in sorted(page_groups.items()):
            rows = group_lines_by_row(page_lines)
            if not rows:
                continue
            markdown = rows_to_markdown(rows)
            if not any(keyword in markdown for keyword in self.TABLE1_KEYWORDS):
                continue

            table_counter += 1
            table_id = f"table_{table_counter}"
            blocks.append(
                DocumentBlock(
                    block_id=f"{doc_id}_table_{table_counter}_full",
                    block_type="table",
                    clause_id=None,
                    page=page,
                    content=f"表{table_counter}\n{markdown}",
                    table_id=table_id,
                    doc_id=doc_id,
                )
            )

            for row_idx, row in enumerate(rows[1:], start=1):
                row_text = " | ".join(cell.text for cell in row)
                if len(row_text.strip()) < 4:
                    continue
                blocks.append(
                    DocumentBlock(
                        block_id=f"{doc_id}_table_{table_counter}_row_{row_idx}",
                        block_type="table",
                        clause_id=None,
                        page=page,
                        content=f"表{table_counter} 第{row_idx}行: {row_text}",
                        table_id=table_id,
                        doc_id=doc_id,
                    )
                )
        return blocks


class ClauseParser:
    def parse_clause_blocks(self, lines: list[OcrLine], doc_id: str = "default") -> list[DocumentBlock]:
        page_groups: dict[int, list[OcrLine]] = defaultdict(list)
        for line in lines:
            page_groups[line.page].append(line)

        blocks: list[DocumentBlock] = []
        block_counter = 0

        for page, page_lines in sorted(page_groups.items()):
            merged_rows = [" ".join(cell.text for cell in row) for row in group_lines_by_row(page_lines)]
            current_clause: str | None = None
            current_parts: list[str] = []
            current_page = page

            def flush():
                nonlocal block_counter, current_clause, current_parts, current_page
                if not current_parts:
                    return
                content = " ".join(current_parts).strip()
                if not content:
                    current_parts = []
                    return
                block_counter += 1
                blocks.append(
                    DocumentBlock(
                        block_id=f"{doc_id}_clause_{block_counter}",
                        block_type="clause" if current_clause else "paragraph",
                        clause_id=current_clause,
                        page=current_page,
                        content=f"{current_clause} {content}".strip() if current_clause else content,
                        doc_id=doc_id,
                    )
                )
                current_parts = []

            for row_text in merged_rows:
                if is_table_hint(row_text) and "AQL" in row_text.upper():
                    flush()
                    current_clause = None
                    continue

                clause_id, rest = extract_clause_id(row_text)
                if clause_id:
                    flush()
                    current_clause = clause_id
                    current_page = page
                    if rest and rest != row_text:
                        current_parts.append(rest)
                    continue

                if re.match(r"^GB/T\s*\d+", row_text):
                    flush()
                    current_clause = None
                    current_parts = [row_text]
                    continue

                current_parts.append(row_text)

            flush()

        return blocks
