from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import VolcEngineImageEditConfig

__all__ = [
    "VolcEngineImageEditConfig",
]


def get_volcengine_image_edit_config(model: str) -> BaseImageEditConfig:
    return VolcEngineImageEditConfig()
