from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import RecraftImageGenerationConfig

__all__ = [
    "RecraftImageGenerationConfig",
]


def get_recraft_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return RecraftImageGenerationConfig()
