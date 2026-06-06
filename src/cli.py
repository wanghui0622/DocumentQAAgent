from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent.orchestrator import DocumentQAAgent
from src.config.settings import get_settings
from src.documents.manager import DocumentManager
from src.documents.registry import DocumentRegistry
from src.models import QAResponse

app = typer.Typer(help="Document QA Agent CLI")
console = Console()


def _response_to_dict(response: QAResponse) -> dict:
    return asdict(response)


@app.command("docs")
def docs_command() -> None:
    """List documents in data/raw and their parse/index status."""
    manager = DocumentManager()
    rows = manager.list_status()
    if not rows:
        console.print("[yellow]No PDFs found in data/raw/[/yellow]")
        return

    table = Table(title="Documents")
    table.add_column("Doc ID")
    table.add_column("Filename")
    table.add_column("Parsed")
    table.add_column("Indexed")
    table.add_column("Blocks")
    for row in rows:
        table.add_row(
            row["doc_id"],
            row["filename"],
            "yes" if row["parsed"] else "no",
            "yes" if row["indexed"] else "no",
            str(row["block_count"]),
        )
    console.print(table)


@app.command("index")
def index_command(
    pdf: Optional[Path] = typer.Option(None, "--pdf", help="Path to a single PDF file"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Document ID"),
    all_docs: bool = typer.Option(False, "--all", help="Parse and index all PDFs in data/raw"),
    skip_parse: bool = typer.Option(False, "--skip-parse", help="Only rebuild index from parsed blocks"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Force PaddleOCR instead of text extraction"),
) -> None:
    """Parse PDF(s) and build unified search index."""
    manager = DocumentManager()

    if all_docs:
        if not skip_parse:
            results = manager.parse_all_raw(force_ocr=force_ocr or None)
            console.print(f"[green]Parsed {len(results)} documents[/green]")
        chunk_count = manager.index_documents()
        console.print(Panel.fit(f"Unified index built with {chunk_count} chunks", title="Indexing Complete"))
        return

    if pdf is None:
        pdfs = DocumentRegistry().list_raw_pdfs()
        if not pdfs:
            raise typer.BadParameter("No PDF found. Place PDFs under data/raw or pass --pdf / --all")
        pdf = pdfs[0]

    if not pdf.exists():
        raise typer.BadParameter(f"PDF not found: {pdf}")

    if not skip_parse:
        doc_id, profile, blocks = manager.parse_pdf(pdf, doc_id=doc_id, force_ocr=force_ocr or None)
        console.print(f"Parsed {doc_id}: {len(blocks)} blocks, type={profile.pdf_type}")
    else:
        doc_id = doc_id or DocumentRegistry.make_doc_id(pdf)

    chunk_count = manager.index_documents()
    console.print(Panel.fit(f"Indexed doc_id={doc_id}, unified chunks={chunk_count}", title="Indexing Complete"))


@app.command("parse")
def parse_command(
    pdf: Optional[Path] = typer.Option(None, "--pdf", help="Path to a single PDF file"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Document ID"),
    all_docs: bool = typer.Option(False, "--all", help="Parse all PDFs in data/raw"),
    force_ocr: bool = typer.Option(False, "--force-ocr", help="Force PaddleOCR instead of text extraction"),
) -> None:
    """Parse PDF(s) and save per-document blocks."""
    manager = DocumentManager()

    if all_docs:
        results = manager.parse_all_raw(force_ocr=force_ocr or None)
        for doc_id, profile, blocks in results:
            console.print(f"{doc_id}: {len(blocks)} blocks, pages={profile.page_count}, type={profile.pdf_type}")
        console.print(f"[green]Parsed {len(results)} documents[/green]")
        return

    if pdf is None:
        pdfs = DocumentRegistry().list_raw_pdfs()
        if not pdfs:
            raise typer.BadParameter(f"No PDF found under {get_settings().raw_dir}")
        pdf = pdfs[0]

    doc_id, profile, blocks = manager.parse_pdf(pdf, doc_id=doc_id, force_ocr=force_ocr or None)
    console.print(
        Panel.fit(
            f"doc_id: {doc_id}\nBlocks: {len(blocks)}\nPages: {profile.page_count}\nType: {profile.pdf_type}",
            title="Parse Complete",
        )
    )


@app.command("ask")
def ask_command(
    question: str = typer.Argument(..., help="Question to ask"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Limit search to one document"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON response"),
) -> None:
    """Ask a question against indexed document(s)."""
    doc_ids = [doc_id] if doc_id else None
    agent = DocumentQAAgent()
    response = agent.ask(question, doc_ids=doc_ids)

    if json_output:
        console.print_json(json.dumps(_response_to_dict(response), ensure_ascii=False, indent=2))
        return

    if response.status == "refused":
        console.print(Panel.fit(response.refuse_reason or "拒答", title="Refused", border_style="red"))
    else:
        console.print(Panel.fit(response.answer or "", title="Answer", border_style="green"))

    if response.citations:
        table = Table(title="Citations")
        table.add_column("Doc")
        table.add_column("Page")
        table.add_column("Clause")
        table.add_column("Quote")
        for citation in response.citations:
            table.add_row(
                citation.doc_id or "-",
                str(citation.page),
                citation.clause_id or "-",
                citation.quote[:120],
            )
        console.print(table)

    if response.self_check:
        console.print(
            Panel.fit(
                f"grounding={response.self_check.grounding_level} "
                f"score={response.self_check.grounding_score:.2f}\n"
                f"passed={response.self_check.checks_passed}\n"
                f"failed={response.self_check.checks_failed}",
                title="Self Check",
            )
        )


@app.command("seed")
def seed_command() -> None:
    """Seed index from bundled sample blocks (smoke test without PDF/OCR)."""
    from tests.fixtures.seed_sample_index import seed_sample_index

    seed_sample_index()
    console.print("[green]Sample index seeded. Try: python -m src.cli ask \"键的抗拉强度最低要求是多少?\"[/green]")


@app.command("eval")
def eval_command(
    suite: str = typer.Option("all", "--suite", help="gbt | agent | combined | all"),
) -> None:
    """Run golden QA evaluation."""
    import tests.eval.run_eval as run_eval_module

    raise typer.Exit(run_eval_module.run_eval(suite))


@app.command("blocks")
def blocks_command(
    doc_id: Optional[str] = typer.Option(None, "--doc-id", help="Preview blocks for one document"),
    limit: int = typer.Option(10, help="Number of blocks to preview"),
) -> None:
    """Preview parsed blocks."""
    registry = DocumentRegistry()
    if doc_id:
        _, blocks = registry.load_document_blocks(doc_id)
        doc_ids = [doc_id]
    else:
        loaded = registry.load_all_blocks()
        if not loaded:
            raise typer.BadParameter("No parsed documents found. Run `parse --all` first.")
        blocks = []
        doc_ids = []
        for profile, doc_blocks in loaded:
            doc_ids.append(profile.get("doc_id", ""))
            blocks.extend(doc_blocks)

    for block in blocks[:limit]:
        console.print(
            Panel.fit(
                f"doc={block.doc_id} type={block.block_type} page={block.page} clause={block.clause_id}\n"
                f"{block.content[:500]}",
                title=block.block_id,
            )
        )
    console.print(f"Showing {min(limit, len(blocks))}/{len(blocks)} blocks from {len(set(doc_ids))} document(s)")


def main() -> None:
    logger.remove()
    logger.add(lambda msg: console.print(msg, end=""), level="INFO")
    app()


if __name__ == "__main__":
    main()
