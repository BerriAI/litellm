from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import XInferenceImageEditConfig

__all__ = [
    "XInferenceImageEditConfig",
]


def get_xinference_image_edit_config(model: str) -> BaseImageEditConfig:
    return XInferenceImageEditConfig()
