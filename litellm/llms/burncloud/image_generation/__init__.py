from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import BurnCloudImageGenerationConfig

__all__ = [
    "BurnCloudImageGenerationConfig",
]


def get_burncloud_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return BurnCloudImageGenerationConfig()
