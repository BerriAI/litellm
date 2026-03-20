from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import RunwareImageGenerationConfig

__all__ = [
    "RunwareImageGenerationConfig",
]


def get_runware_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the Runware image generation configuration.

    All Runware image generation models use the same config since
    the API has a single endpoint with taskType-based routing.
    """
    return RunwareImageGenerationConfig()
