from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .handler import HunyuanImageEdit, hunyuan_image_edit
from .transformation import HunyuanImageEditConfig

__all__ = [
    "HunyuanImageEdit",
    "HunyuanImageEditConfig",
    "hunyuan_image_edit",
]


def get_hunyuan_image_edit_config(model: str) -> BaseImageEditConfig:
    return HunyuanImageEditConfig()
