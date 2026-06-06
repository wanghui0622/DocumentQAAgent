from __future__ import annotations

from openai import APIStatusError, OpenAI

from src.config.settings import Settings

DASHSCOPE_EMBEDDING_MODELS = {"text-embedding-v3", "text-embedding-v4"}


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None):
        from src.config.settings import get_settings

        self.settings = settings or get_settings()
        self.model = self._normalize_model_name(self.settings.embedding_model)
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_api_base,
        )

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        if "/" in model or model.startswith("Qwen"):
            raise ValueError(
                f"Embedding 模型名 `{model}` 格式不正确。"
                "DashScope 兼容接口请使用 text-embedding-v4 或 text-embedding-v3，"
                "不要使用 HuggingFace 模型 ID（如 Qwen/Qwen3-Embedding-8B）。"
            )
        return model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
        except APIStatusError as exc:
            if exc.status_code == 404:
                raise RuntimeError(
                    f"Embedding 模型 `{self.model}` 不可用。"
                    "若使用阿里云 DashScope，请设置 EMBEDDING_MODEL=text-embedding-v4"
                ) from exc
            raise
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
