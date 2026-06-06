from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_base: str = Field(default="https://api.openai.com/v1", alias="OPENAI_API_BASE")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")

    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    retrieval_final_k: int = Field(default=5, alias="RETRIEVAL_FINAL_K")
    retrieval_score_threshold: float = Field(default=0.35, alias="RETRIEVAL_SCORE_THRESHOLD")

    verifier_min_grounding: float = Field(default=0.7, alias="VERIFIER_MIN_GROUNDING")

    ocr_dpi: int = Field(default=300, alias="OCR_DPI")
    pdf_text_threshold: int = Field(default=50, alias="PDF_TEXT_THRESHOLD")
    pdf_force_ocr: bool = Field(default=False, alias="PDF_FORCE_OCR")
    pdf_bad_text_page_threshold: float = Field(default=0.15, alias="PDF_BAD_TEXT_PAGE_THRESHOLD")
    pdf_bad_text_avg_threshold: float = Field(default=0.10, alias="PDF_BAD_TEXT_AVG_THRESHOLD")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_dir: Path = Field(default=Path("data/raw"), alias="RAW_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DIR")
    chroma_dir: Path = Field(default=Path("data/processed/chroma"), alias="CHROMA_DIR")

    @property
    def blocks_path(self) -> Path:
        return self.processed_dir / "blocks.json"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
