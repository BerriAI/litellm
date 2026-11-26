"""
Volcengine LLM Provider
Support for Volcengine (ByteDance) chat, embedding, image generation, image editing, and responses models.
"""

from typing import Final

from .chat.transformation import VolcEngineChatConfig
from .common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)
from .embedding import VolcEngineEmbeddingConfig
from .image_edit import VolcEngineImageEditConfig
from .image_generation import VolcEngineImageGenerationConfig
from .responses.transformation import VolcEngineResponsesAPIConfig

# For backward compatibility, keep the old class name
VolcEngineConfig: Final = VolcEngineChatConfig

__all__ = [
    "VolcEngineChatConfig",
    "VolcEngineConfig",  # backward compatibility
    "VolcEngineEmbeddingConfig",
    "VolcEngineError",
    "VolcEngineImageEditConfig",
    "VolcEngineImageGenerationConfig",
    "VolcEngineResponsesAPIConfig",
    "get_volcengine_base_url",
    "get_volcengine_headers",
]
