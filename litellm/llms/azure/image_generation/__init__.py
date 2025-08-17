from litellm._logging import verbose_logger
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
    model = model.lower()
    model = model.replace("-", "")
    model = model.replace("_", "")
    if model == "" or "dalle2" in model:  # empty model is dall-e-2
        return AzureDallE2ImageGenerationConfig()
    elif "dalle3" in model:
        return AzureDallE3ImageGenerationConfig()
    else:
        verbose_logger.debug(
            f"Using AzureGPTImageGenerationConfig for model: {model}. This follows the gpt-image-1 model format."
        )
        return AzureGPTImageGenerationConfig()
