from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import VolcEngineImageGenerationConfig

__all__ = [
    "VolcEngineImageGenerationConfig",
]


def get_volcengine_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return VolcEngineImageGenerationConfig()
