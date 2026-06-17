"""
TokenHub LLM Provider (Tencent Cloud)
Support for TokenHub chat models via OpenAI-compatible API.
Reference: https://cloud.tencent.com/document/product/1823/130051
"""

from .chat.transformation import TokenHubChatConfig
from .common_utils import (
    TokenHubError,
    get_tokenhub_base_url,
    get_tokenhub_headers,
)

# For backward compatibility
TokenHubConfig = TokenHubChatConfig

__all__ = [
    "TokenHubChatConfig",
    "TokenHubConfig",
    "TokenHubError",
    "get_tokenhub_base_url",
    "get_tokenhub_headers",
]
