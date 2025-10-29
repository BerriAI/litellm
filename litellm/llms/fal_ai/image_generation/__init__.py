from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import FalAIImageGenerationConfig

__all__ = [
    "FalAIImageGenerationConfig",
]


def get_fal_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return FalAIImageGenerationConfig()

