"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat, embedding, and responses models.
"""

from .chat.transformation import VolcEngineChatConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)
from .embedding import VolcEngineEmbeddingConfig
from .responses.transformation import VolcEngineResponsesAPIConfig

# For backward compatibility, keep the old class name
VolcEngineConfig = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingConfig",
    "VolcEngineResponsesAPIConfig",
    "VolcEngineError",
    "get_volcengine_base_url",
    "get_volcengine_headers",
]
