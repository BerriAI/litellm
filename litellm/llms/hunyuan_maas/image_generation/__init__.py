from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .handler import HunyuanMaasImageGeneration, hunyuan_maas_image_generation
from .transformation import HunyuanMaasImageGenerationConfig

__all__ = [
    "HunyuanMaasImageGeneration",
    "HunyuanMaasImageGenerationConfig",
    "hunyuan_maas_image_generation",
]


def get_hunyuan_maas_image_generation_config(
    model: str,
) -> BaseImageGenerationConfig:
    return HunyuanMaasImageGenerationConfig()
