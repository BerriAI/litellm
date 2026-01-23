"""
GigaChat Provider for LiteLLM

GigaChat is Sber AI's large language model (Russia's leading LLM).

Supports:
- Chat completions (sync/async)
- Streaming (sync/async) with SSE parsing
- Function calling / Tools with automatic format conversion
- Structured output via JSON schema (emulated through function calls)
- Image input (base64 and URL) with automatic file upload
- Embeddings
- OAuth 2.0 and basic auth (user/password)

GigaChat-specific features:
- repetition_penalty: Control repetition in generated text
- profanity_check: Enable content filtering
- flags: Feature flags for the API
- reasoning_effort: Reasoning effort level for reasoning models

API Documentation: https://developers.sber.ru/docs/ru/gigachat/api/overview
SDK: https://github.com/ai-forever/gigachat
"""

from .authenticator import (
    get_access_token,
    get_access_token_async,
)
from .chat.transformation import GigaChatConfig
from .common_utils import (
    GIGACHAT_AUTH_URL,
    GIGACHAT_BASE_URL,
    GIGACHAT_SCOPE,
    TOKEN_EXPIRY_BUFFER_MS,
    USER_AGENT,
    GigaChatAuthError,
    GigaChatError,
    build_url,
)
from .embedding.transformation import GigaChatEmbeddingConfig

__all__ = [
    # Config classes
    "GigaChatConfig",
    "GigaChatEmbeddingConfig",
    # Constants
    "GIGACHAT_BASE_URL",
    "GIGACHAT_AUTH_URL",
    "GIGACHAT_SCOPE",
    "TOKEN_EXPIRY_BUFFER_MS",
    "USER_AGENT",
    # Exception classes
    "GigaChatError",
    "GigaChatAuthError",
    # Auth functions
    "get_access_token",
    "get_access_token_async",
    # Utils
    "build_url",
]
