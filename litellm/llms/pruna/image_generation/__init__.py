from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import PrunaImageGenerationConfig

__all__ = [
    "PrunaImageGenerationConfig",
]


def get_pruna_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return PrunaImageGenerationConfig()
