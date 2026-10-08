from typing import Final

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .flux_lora_depth_transformation import FalAIFluxLoraDepthEditConfig
from .transformation import FalAIImageEditConfig

__all__ = ("FalAIFluxLoraDepthEditConfig", "FalAIImageEditConfig")


def get_fal_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate Fal AI image edit configuration based on the model.

    Args:
        model: The Fal AI model name (e.g., "openai/gpt-image-2.5/flare/edit", "fal-ai/flux-lora-depth")

    Returns:
        The appropriate configuration class for the specified model
    """
    model_lower: Final = model.lower()
    if "flux-lora-depth" in model_lower:
        return FalAIFluxLoraDepthEditConfig()
    return FalAIImageEditConfig()
