from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import DashScopeImageEditConfig

__all__ = ["DashScopeImageEditConfig"]


def get_dashscope_image_edit_config(model: str) -> BaseImageEditConfig:
    return DashScopeImageEditConfig()
