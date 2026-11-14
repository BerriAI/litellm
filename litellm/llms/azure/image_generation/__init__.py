from litellm._logging import verbose_logger
from litellm.llms.azure_ai.image_generation import AzureFoundryMAIImageGenerationConfig
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .gpt_transformation import AzureGPTImageGenerationConfig
from .http_utils import azure_deployment_image_generation_json_body

__all__ = [
    "AzureGPTImageGenerationConfig",
    "azure_deployment_image_generation_json_body",
]


def get_azure_image_generation_config(model: str) -> BaseImageGenerationConfig:
    model = model.lower()
    model = model.replace("-", "")
    model = model.replace("_", "")
    if AzureFoundryMAIImageGenerationConfig.is_mai_model(model):
        return AzureFoundryMAIImageGenerationConfig()
    else:
        verbose_logger.debug(
            "Using AzureGPTImageGenerationConfig for model: %s. This follows the gpt-image model format.", model
        )
        return AzureGPTImageGenerationConfig()
