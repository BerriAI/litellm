from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import XAIImageGenerationConfig

__all__ = ["XAIImageGenerationConfig", "get_xai_image_generation_config"]  # mutable-ok: provider JSON body and base-class dict signature


def get_xai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return XAIImageGenerationConfig()
