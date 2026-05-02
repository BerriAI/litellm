import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .dall_e_2_transformation import AzureFoundryDallE2ImageGenerationConfig
from .dall_e_3_transformation import AzureFoundryDallE3ImageGenerationConfig
from .flux_transformation import AzureFoundryFluxImageGenerationConfig
from .gpt_transformation import AzureFoundryGPTImageGenerationConfig
from .mai_transformation import AzureFoundryMAIImageGenerationConfig

__all__ = [
    "AzureFoundryFluxImageGenerationConfig",
    "AzureFoundryGPTImageGenerationConfig",
    "AzureFoundryDallE2ImageGenerationConfig",
    "AzureFoundryDallE3ImageGenerationConfig",
    "AzureFoundryMAIImageGenerationConfig",
]


def _normalize_model_name(model: str) -> str:
    return model.lower().replace("-", "").replace("_", "")


def _supports_mai_endpoint(model: str) -> bool:
    model_key = model if model.startswith("azure_ai/") else f"azure_ai/{model}"
    try:
        resolved_model_info = litellm.get_model_info(model=model_key)
    except Exception:
        verbose_logger.debug(
            "Azure AI model info not found for image model: %s", model_key
        )
        return False
    resolved_key = resolved_model_info.get("key", model_key)
    model_info = litellm.model_cost.get(resolved_key, {})
    return bool(model_info.get("supports_mai_endpoint"))


def get_azure_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    if _supports_mai_endpoint(model):
        return AzureFoundryMAIImageGenerationConfig()

    normalized_model = _normalize_model_name(model)
    if (
        normalized_model == "" or "dalle2" in normalized_model
    ):  # empty model is dall-e-2
        return AzureFoundryDallE2ImageGenerationConfig()
    elif "dalle3" in normalized_model:
        return AzureFoundryDallE3ImageGenerationConfig()
    elif "flux" in normalized_model:
        return AzureFoundryFluxImageGenerationConfig()
    else:
        verbose_logger.debug(
            "Using AzureGPTImageGenerationConfig for model: %s. This follows the "
            "gpt-image-1 model format.",
            normalized_model,
        )
        return AzureFoundryGPTImageGenerationConfig()
