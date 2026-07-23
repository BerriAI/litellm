from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import TogetherAIImageEditConfig

__all__ = [
    "TogetherAIImageEditConfig",
]


def get_together_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    return TogetherAIImageEditConfig()
