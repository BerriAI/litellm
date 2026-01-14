"""
Stability AI Image Edit Module

Factory function for getting the appropriate config class.
"""

from litellm.llms.base_llm.image_edit.transformation import (
    BaseImageEditConfig,
)

from .transformations import StabilityImageEditConfig

__all__ = [
    "StabilityImageEditConfig",
    "get_stability_image_edit_config",
]


def get_stability_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate Stability AI config for the given model.

    Currently all models use the same config class, but this factory
    allows for model-specific configs in the future.

    Args:
        model: The model name (e.g., "stability/inpaint", "stability/outpaint")

    Returns:
        BaseImageEditConfig instance for Stability AI
    """
    # For now, all models use the same config
    # In the future, we could have model-specific configs:
    # - StabilityInpaintConfig for Inpaint models
    # - StabilityOutpaintConfig for Outpaint models
    # - etc.
    return StabilityImageEditConfig()
