from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import AimlImageGenerationConfig

__all__ = [
    "AimlImageGenerationConfig",
]


def get_aiml_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return AimlImageGenerationConfig()
