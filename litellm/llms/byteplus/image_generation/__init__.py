from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig

from .transformation import BytePlusImageGenerationConfig

__all__ = ["BytePlusImageGenerationConfig", "get_byteplus_image_generation_config"]


def get_byteplus_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return BytePlusImageGenerationConfig()
