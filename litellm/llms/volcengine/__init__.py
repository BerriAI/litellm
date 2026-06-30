"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat, embedding, and responses models.
"""

from .chat.transformation import VolcEngineChatConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
    get_volcengine_speech_api_key,
)
from .embedding import VolcEngineEmbeddingConfig
from .realtime.transformation import VolcEngineRealtimeConfig
from .responses.transformation import VolcEngineResponsesAPIConfig

# For backward compatibility, keep the old class name
VolcEngineConfig = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingConfig",
    "VolcEngineRealtimeConfig",
    "VolcEngineResponsesAPIConfig",
    "VolcEngineError",
    "get_volcengine_base_url",
    "get_volcengine_headers",
    "get_volcengine_speech_api_key",
]
