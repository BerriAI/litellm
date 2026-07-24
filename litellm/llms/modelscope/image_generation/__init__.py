"""
ModelScope Image Generation Module

Factory function for getting the appropriate config class.
"""

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import ModelScopeImageGenerationConfig

__all__ = [
    "ModelScopeImageGenerationConfig",
    "get_modelscope_image_generation_config",
]


def get_modelscope_image_generation_config(
    model: str,
) -> BaseImageGenerationConfig:
    """
    Get the ModelScope config for image generation.

    Args:
        model: The model name (e.g., "modelscope/Qwen/Qwen-Image-Edit")

    Returns:
        BaseImageGenerationConfig instance for ModelScope
    """
    return ModelScopeImageGenerationConfig()
