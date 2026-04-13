from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import FalAIImageEditConfig

__all__ = [
    "FalAIImageEditConfig",
    "get_fal_ai_image_edit_config",
]


def get_fal_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate Fal AI image edit configuration based on the model.

    Currently all Fal AI edit models use the same generic config since they
    share the same JSON request/response format. Model-specific configs can
    be added here in the future if needed.
    """
    return FalAIImageEditConfig()
