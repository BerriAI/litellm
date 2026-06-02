from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .handler import HunyuanGptMaasImageGeneration, hunyuan_gpt_maas_image_generation
from .transformation import HunyuanGptMaasImageGenerationConfig

__all__ = [
    "HunyuanGptMaasImageGeneration",
    "HunyuanGptMaasImageGenerationConfig",
    "hunyuan_gpt_maas_image_generation",
]


def get_hunyuan_gpt_maas_image_generation_config(
    model: str,
) -> BaseImageGenerationConfig:
    return HunyuanGptMaasImageGenerationConfig()
