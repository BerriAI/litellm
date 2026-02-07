from litellm._logging import verbose_logger
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .dall_e_2_transformation import AzureFoundryDallE2ImageGenerationConfig
from .dall_e_3_transformation import AzureFoundryDallE3ImageGenerationConfig
from .flux_transformation import AzureFoundryFluxImageGenerationConfig
from .gpt_transformation import AzureFoundryGPTImageGenerationConfig

__all__ = [
    "AzureFoundryFluxImageGenerationConfig",
    "AzureFoundryGPTImageGenerationConfig",
    "AzureFoundryDallE2ImageGenerationConfig",
    "AzureFoundryDallE3ImageGenerationConfig",
]


def get_azure_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    model = model.lower()
    model = model.replace("-", "")
    model = model.replace("_", "")
    if model == "" or "dalle2" in model:  # empty model is dall-e-2
        return AzureFoundryDallE2ImageGenerationConfig()
    elif "dalle3" in model:
        return AzureFoundryDallE3ImageGenerationConfig()
    elif "flux" in model:
        return AzureFoundryFluxImageGenerationConfig()
    else:
        verbose_logger.debug(
            f"Using AzureGPTImageGenerationConfig for model: {model}. This follows the gpt-image-1 model format."
        )
        return AzureFoundryGPTImageGenerationConfig()
