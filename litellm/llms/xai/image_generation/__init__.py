from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import XAIImageGenerationConfig

__all__ = ["XAIImageGenerationConfig", "get_xai_image_generation_config"]


def get_xai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return XAIImageGenerationConfig()
