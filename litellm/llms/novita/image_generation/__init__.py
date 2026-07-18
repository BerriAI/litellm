from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import NovitaImageGenerationConfig

__all__ = [
    "NovitaImageGenerationConfig",
]


def get_novita_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return NovitaImageGenerationConfig()
