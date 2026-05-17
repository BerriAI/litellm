from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .handler import HunyuanImageGeneration, hunyuan_image_generation
from .transformation import HunyuanImageGenerationConfig

__all__ = [
    "HunyuanImageGeneration",
    "HunyuanImageGenerationConfig",
    "hunyuan_image_generation",
]


def get_hunyuan_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return HunyuanImageGenerationConfig()
