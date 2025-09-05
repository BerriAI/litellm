"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat and embedding models
"""

from .chat.transformation import VolcEngineChatConfig
from .embedding import VolcEngineEmbeddingHandler, VolcEngineEmbeddingConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)

# For backward compatibility, keep the old class name
VolcEngineConfig = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingHandler",
    "VolcEngineEmbeddingConfig",
    "VolcEngineError",
    "get_volcengine_base_url",
    "get_volcengine_headers",
]
