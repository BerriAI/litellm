from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.openai.image_generation import GPTImageGenerationConfig
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class AzureFoundryFluxImageGenerationConfig(GPTImageGenerationConfig):
    """Azure Foundry BFL API configuration for FLUX image generation."""

    @staticmethod
    def get_flux2_image_generation_url(
        api_base: str | None,
        model: str,
        api_version: str | None,
    ) -> str:
        """
        Constructs the complete URL for Azure AI FLUX 2 image generation.

        FLUX 2 models on Azure AI use a different URL pattern than standard Azure OpenAI:
        - Standard: /openai/deployments/{model}/images/generations
        - FLUX 2: /providers/blackforestlabs/v1/{model-path}

        Args:
            api_base: Base URL (e.g., https://litellm-ci-cd-prod.services.ai.azure.com)
            model: Model name (e.g., FLUX.2-flex or FLUX.2-pro)
            api_version: API version (e.g., preview)

        Returns:
            Complete URL for the FLUX 2 image generation endpoint
        """
        if api_base is None:
            raise ValueError("api_base is required for Azure AI FLUX 2 image generation")

        api_base = api_base.rstrip("/")
        api_version = api_version or "preview"

        # If the api_base already contains /providers/, it's already a complete path
        if "/providers/" in api_base:
            if "?" in api_base:
                return api_base
            return f"{api_base}?api-version={api_version}"

        provider_model_path: Final = AzureFoundryFluxImageGenerationConfig.get_flux2_provider_model_path(model)
        return f"{api_base}/providers/blackforestlabs/v1/{provider_model_path}?api-version={api_version}"

    @staticmethod
    def is_flux2_model(model: str) -> bool:
        """
        Check if the model is an Azure AI FLUX 2 model.

        Args:
            model: Model name (e.g., flux.2-pro, azure_ai/flux.2-pro)

        Returns:
            True if the model is a FLUX 2 model
        """
        model_lower: Final = model.lower().replace(".", "-").replace("_", "-")
        return "flux-2" in model_lower or "flux2" in model_lower

    @staticmethod
    def get_flux2_provider_model_path(model: str) -> str:
        normalized_model: Final = model.lower().replace(".", "-").replace("_", "-")
        return "flux-2-flex" if "flux-2-flex" in normalized_model else "flux-2-pro"

    def get_supported_openai_params(  # mutable-ok: inherited config contract returns a list
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:
        return [  # mutable-ok: BaseImageGenerationConfig requires a list
            "n",
            "size",
            "output_format",
            "seed",
            "safety_tolerance",
            "aspect_ratio",
            "width",
            "height",
            "num_images",
            "guidance",
            "steps",
        ]

    @staticmethod
    def _map_parameter(name: str, value: object) -> tuple[tuple[str, object], ...]:
        if name == "n":
            return (("num_images", value),)
        if name != "size":
            return ((name, value),)

        try:
            width, height = (int(dimension) for dimension in str(value).lower().split("x"))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid size format '{value}'. Expected 'WxH', for example '1024x1024'.")
        return (("width", width), ("height", height))

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: inherited config contract returns a dict
        supported_params: Final = self.get_supported_openai_params(model)
        unsupported_params: Final = tuple(name for name in non_default_params if name not in supported_params)
        if unsupported_params and not drop_params:
            raise ValueError(
                f"Parameters {unsupported_params} are not supported for model {model}. "
                f"Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
            )

        mapped_params: Final[Mapping[str, object]] = MappingProxyType(
            {
                mapped_name: mapped_value
                for name, value in non_default_params.items()
                if name in supported_params
                for mapped_name, mapped_value in self._map_parameter(name, value)
            }
        )
        return {**optional_params, **mapped_params}  # mutable-ok: inherited config contract returns a dict
