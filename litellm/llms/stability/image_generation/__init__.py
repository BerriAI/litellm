"""
Stability AI Image Generation Module

Factory function for getting the appropriate config class.
"""

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import StabilityImageGenerationConfig

__all__ = [
    "StabilityImageGenerationConfig",
    "get_stability_image_generation_config",
]


def get_stability_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the appropriate Stability AI config for the given model.

    Currently all models use the same config class, but this factory
    allows for model-specific configs in the future.

    Args:
        model: The model name (e.g., "stability/sd3", "stability/stable-image-ultra")

    Returns:
        BaseImageGenerationConfig instance for Stability AI
    """
    # For now, all models use the same config
    # In the future, we could have model-specific configs:
    # - StabilitySD3Config for SD3 models
    # - StabilityUltraConfig for Ultra models
    # - etc.
    return StabilityImageGenerationConfig()
