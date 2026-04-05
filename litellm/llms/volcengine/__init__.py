"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat, embedding, image generation, and responses models.
"""

from .chat.transformation import VolcEngineChatConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)
from .embedding import VolcEngineEmbeddingConfig
from .image_generation import VolcEngineImageGenerationConfig
from .responses.transformation import VolcEngineResponsesAPIConfig

# For backward compatibility, keep the old class name
VolcEngineConfig = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingConfig",
    "VolcEngineImageGenerationConfig",
    "VolcEngineResponsesAPIConfig",
    "VolcEngineError",
    "get_volcengine_base_url",
    "get_volcengine_headers",
]
