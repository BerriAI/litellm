from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .handler import HunyuanGptMaasImageEdit, hunyuan_gpt_maas_image_edit
from .transformation import HunyuanGptMaasImageEditConfig

__all__ = [
    "HunyuanGptMaasImageEdit",
    "HunyuanGptMaasImageEditConfig",
    "hunyuan_gpt_maas_image_edit",
]


def get_hunyuan_gpt_maas_image_edit_config(model: str) -> BaseImageEditConfig:
    return HunyuanGptMaasImageEditConfig()
