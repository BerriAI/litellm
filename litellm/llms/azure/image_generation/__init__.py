from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .dall_e_2_transformation import AzureDallE2ImageGenerationConfig
from .dall_e_3_transformation import AzureDallE3ImageGenerationConfig
from .gpt_transformation import AzureGPTImageGenerationConfig

__all__ = [
    "AzureDallE2ImageGenerationConfig",
    "AzureDallE3ImageGenerationConfig",
    "AzureGPTImageGenerationConfig",
]


def get_azure_image_generation_config(model: str) -> BaseImageGenerationConfig:
    if model.startswith("dall-e-2") or model == "":  # empty model is dall-e-2
        return AzureDallE2ImageGenerationConfig()
    elif model.startswith("dall-e-3"):
        return AzureDallE3ImageGenerationConfig()
    else:
        return AzureGPTImageGenerationConfig()
