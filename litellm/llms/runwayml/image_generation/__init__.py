from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import RunwayMLBaseConfig, RunwayMLImageGenerationConfig

__all__ = [
    "RunwayMLBaseConfig",
    "RunwayMLImageGenerationConfig",
]


def get_runwayml_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return RunwayMLImageGenerationConfig()

