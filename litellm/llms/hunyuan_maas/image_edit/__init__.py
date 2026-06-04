from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .handler import HunyuanMaasImageEdit, hunyuan_maas_image_edit
from .transformation import HunyuanMaasImageEditConfig

__all__ = [
    "HunyuanMaasImageEdit",
    "HunyuanMaasImageEditConfig",
    "hunyuan_maas_image_edit",
]


def get_hunyuan_maas_image_edit_config(model: str) -> BaseImageEditConfig:
    return HunyuanMaasImageEditConfig()
