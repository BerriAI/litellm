"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat, embedding, and responses models.
"""

from typing import Final

from .chat.transformation import VolcEngineChatConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)
from .embedding import VolcEngineEmbeddingConfig
from .responses.transformation import VolcEngineResponsesAPIConfig

# For backward compatibility, keep the old class name
VolcEngineConfig: Final = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingConfig",
    "VolcEngineError",
    "VolcEngineResponsesAPIConfig",
    "get_volcengine_base_url",
    "get_volcengine_headers",
]
