from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import OpenRouterImageGenerationConfig

__all__ = [
    "OpenRouterImageGenerationConfig",
]


def get_openrouter_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return OpenRouterImageGenerationConfig()