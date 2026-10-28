from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import CometAPIImageGenerationConfig

__all__ = [
    "CometAPIImageGenerationConfig",
]


def get_cometapi_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return CometAPIImageGenerationConfig()
