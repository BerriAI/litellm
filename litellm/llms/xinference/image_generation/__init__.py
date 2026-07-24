from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import XInferenceImageGenerationConfig

__all__ = [
    "XInferenceImageGenerationConfig",
]


def get_xinference_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return XInferenceImageGenerationConfig()
