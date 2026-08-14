from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import OpenRouterImageEditConfig

__all__ = [
    "OpenRouterImageEditConfig",
]


def get_openrouter_image_edit_config(model: str) -> BaseImageEditConfig:
    return OpenRouterImageEditConfig()
