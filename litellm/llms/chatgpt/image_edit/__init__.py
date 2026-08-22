from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .transformation import ChatGPTImageEditConfig

__all__ = ["ChatGPTImageEditConfig", "get_chatgpt_image_edit_config"]


def get_chatgpt_image_edit_config(model: str) -> BaseImageEditConfig:
    return ChatGPTImageEditConfig()
