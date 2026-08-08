from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import TogetherAIImageGenerationConfig

__all__ = [
    "TogetherAIImageGenerationConfig",
]


def get_together_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return TogetherAIImageGenerationConfig()
