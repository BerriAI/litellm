from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import HunyuanImageGenerationConfig

__all__ = [
    "HunyuanImageGenerationConfig",
]


def get_hunyuan_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return HunyuanImageGenerationConfig()
