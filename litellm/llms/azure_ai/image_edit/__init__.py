from typing import Final

from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.llms.azure_ai.image_generation.mai_transformation import (
    AzureFoundryMAIImageGenerationConfig,
)
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .flux2_transformation import AzureFoundryFlux2ImageEditConfig
from .gpt_transformation import AzureFoundryGPTImageEditConfig
from .mai_transformation import AzureFoundryMAIImageEditConfig
from .transformation import AzureFoundryFluxImageEditConfig

__all__ = [
    "AzureFoundryFlux2ImageEditConfig",
    "AzureFoundryFluxImageEditConfig",
    "AzureFoundryGPTImageEditConfig",
    "AzureFoundryMAIImageEditConfig",
]


def get_azure_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate image edit config for an Azure AI model.

    - gpt-image models use /openai/deployments/{deployment}/images/edits with multipart form data
    - MAI models use /mai/v1/images/edits with multipart form data
    - FLUX 2 models use JSON with base64 image
    - FLUX 1 models use multipart/form-data
    """
    if AzureFoundryMAIImageGenerationConfig.is_mai_model(model):
        return AzureFoundryMAIImageEditConfig()

    model_normalized: Final = model.lower().replace("-", "").replace("_", "")

    if "gptimage" in model_normalized:
        return AzureFoundryGPTImageEditConfig()

    # Check if it's a FLUX 2 model
    if AzureFoundryFluxImageGenerationConfig.is_flux2_model(model):
        return AzureFoundryFlux2ImageEditConfig()

    # Default to FLUX 1 config for other FLUX models
    if model_normalized == "" or "flux" in model_normalized:
        return AzureFoundryFluxImageEditConfig()

    raise ValueError(f"Model {model} is not supported for Azure AI image editing.")
