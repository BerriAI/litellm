from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import SiliconFlowImageGenerationConfig

__all__ = ["SiliconFlowImageGenerationConfig"]


def get_siliconflow_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return SiliconFlowImageGenerationConfig()
