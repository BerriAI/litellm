import base64
from collections.abc import Mapping, Sequence
from io import BufferedReader
from types import MappingProxyType
from typing import Any, Final

from httpx._types import RequestFiles

import litellm
from litellm.llms.azure_ai.common_utils import (
    AzureFoundryModelInfo,
    get_azure_ai_auth_headers,
)
from litellm.llms.azure_ai.image_generation.flux_transformation import (
    AzureFoundryFluxImageGenerationConfig,
)
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams


class AzureFoundryFlux2ImageEditConfig(OpenAIImageEditConfig):
    """
    Azure AI Foundry FLUX 2 image edit config

    Supports FLUX 2 models (e.g., flux.2-pro) for image editing.
    Uses the model-specific /providers/blackforestlabs/v1/flux-2-* endpoint as image generation,
    with the image passed as base64 in JSON body.
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        FLUX 2 supports a subset of OpenAI image edit params
        """
        return [
            "n",
            "size",
            "width",
            "height",
            "num_images",
            "seed",
            "safety_tolerance",
            "output_format",
            "aspect_ratio",
            "guidance",
            "steps",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to FLUX 2 params.
        FLUX 2 uses the same param names as OpenAI for supported params.
        """
        return AzureFoundryFluxImageGenerationConfig().map_openai_params(
            non_default_params=MappingProxyType(
                {key: value for key, value in image_edit_optional_params.items() if value is not None}
            ),
            optional_params=MappingProxyType({}),
            model=model,
            drop_params=drop_params,
        )

    def use_multipart_form_data(self) -> bool:
        """FLUX 2 uses JSON requests, not multipart/form-data."""
        return False

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate Azure AI Foundry environment and set up authentication
        """
        headers.update(
            {
                **get_azure_ai_auth_headers(
                    api_key=AzureFoundryModelInfo.get_api_key(api_key),
                    litellm_params=litellm_params,
                    api_key_header="Api-Key",
                ),
                "Content-Type": "application/json",
            }
        )
        return headers

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes] | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        """
        Transform image edit request for FLUX 2.

        FLUX 2 uses the same endpoint for generation and editing,
        with the image passed as base64 in the JSON body.
        """
        if prompt is None:
            raise ValueError("FLUX 2 image edit requires a prompt.")

        if image is None:
            raise ValueError("FLUX 2 image edit requires an image.")

        images: Final = tuple(image) if isinstance(image, list) else (image,)
        if not images:
            raise ValueError("FLUX 2 image edit requires at least one image.")
        max_reference_images: Final = 10 if "flex" in model.lower() else 8
        if len(images) > max_reference_images:
            raise ValueError(f"{model} supports at most {max_reference_images} reference images.")

        reference_images: Final[Mapping[str, str]] = MappingProxyType(
            {
                "input_image" if index == 1 else f"input_image_{index}": self._convert_image_to_base64(reference_image)
                for index, reference_image in enumerate(images, start=1)
            }
        )
        request_body: Final[dict[str, Any]] = {
            "prompt": prompt,
            "model": model,
            **reference_images,
            **image_edit_optional_request_params,
        }
        return request_body, []

    def _convert_image_to_base64(self, image: Any) -> str:
        """Convert image file to base64 string"""
        if isinstance(image, BufferedReader):
            image_bytes = image.read()
            image.seek(0)  # Reset file pointer for potential reuse
        elif isinstance(image, bytes):
            image_bytes = image
        elif hasattr(image, "read"):
            image_bytes = image.read()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for Azure AI Foundry FLUX 2 image edits.

        Uses the same model-specific BFL provider endpoint as image generation.
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base)

        if api_base is None:
            raise ValueError(
                "Azure AI API base is required. Set AZURE_AI_API_BASE environment variable or pass api_base parameter."
            )

        api_version: Final = (
            litellm_params.get("api_version")
            or litellm.api_version
            or get_secret_str("AZURE_AI_API_VERSION")
            or "preview"
        )

        return AzureFoundryFluxImageGenerationConfig.get_flux2_image_generation_url(
            api_base=api_base,
            model=model,
            api_version=api_version,
        )
