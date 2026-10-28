from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import GoogleImageGenConfig

__all__ = [
    "GoogleImageGenConfig",
]


def get_gemini_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return GoogleImageGenConfig()
