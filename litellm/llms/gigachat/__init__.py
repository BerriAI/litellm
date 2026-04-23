"""
GigaChat Provider for LiteLLM

GigaChat is Sber AI's large language model (Russia's leading LLM).
Supports:
- Chat completions (sync/async)
- Streaming (sync/async)
- Function calling / Tools
- Structured output via JSON schema (emulated through function calls)
- Image input (base64 and URL)
- Embeddings

API Documentation: https://developers.sber.ru/docs/ru/gigachat/api/overview
"""

from .chat.transformation import GigaChatConfig, GigaChatError
from .embedding.transformation import GigaChatEmbeddingConfig

__all__ = [
    "GigaChatConfig",
    "GigaChatEmbeddingConfig",
    "GigaChatError",
]
