from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import RunwayMLImageGenerationConfig

__all__ = [
    "RunwayMLImageGenerationConfig",
]


def get_runwayml_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return RunwayMLImageGenerationConfig()
