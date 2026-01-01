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

from litellm.llms.gigachat.chat.handler import GigaChatChatHandler, gigachat_chat_handler

__all__ = ["GigaChatChatHandler", "gigachat_chat_handler"]
