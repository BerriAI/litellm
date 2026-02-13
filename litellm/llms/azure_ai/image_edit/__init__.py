from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .flux2_transformation import AzureFoundryFlux2ImageEditConfig
from .transformation import AzureFoundryFluxImageEditConfig

__all__ = ["AzureFoundryFluxImageEditConfig", "AzureFoundryFlux2ImageEditConfig"]


def get_azure_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate image edit config for an Azure AI model.

    - FLUX 2 models use JSON with base64 image
    - FLUX 1 models use multipart/form-data
    """
    # Check if it's a FLUX 2 model
    if AzureFoundryFluxImageGenerationConfig.is_flux2_model(model):
        return AzureFoundryFlux2ImageEditConfig()

    # Default to FLUX 1 config for other FLUX models
    model_normalized = model.lower().replace("-", "").replace("_", "")
    if model_normalized == "" or "flux" in model_normalized:
        return AzureFoundryFluxImageEditConfig()

    raise ValueError(f"Model {model} is not supported for Azure AI image editing.")
