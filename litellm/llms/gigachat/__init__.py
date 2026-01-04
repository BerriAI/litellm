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

from .chat.transformation import GigaChatConfig, gigachat_chat_config
from .embedding.transformation import GigaChatEmbeddingConfig, gigachat_embedding_config
from .common_utils import GigaChatError

__all__ = [
    "GigaChatConfig",
    "gigachat_chat_config",
    "GigaChatEmbeddingConfig",
    "gigachat_embedding_config",
    "GigaChatError",
]
