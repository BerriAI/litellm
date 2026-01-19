import base64
from io import BufferedReader
from typing import Any, Dict, Optional, Tuple

from httpx._types import RequestFiles

import litellm
from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo
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
    Uses the same /providers/blackforestlabs/v1/flux-2-pro endpoint as image generation,
    with the image passed as base64 in JSON body.
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        FLUX 2 supports a subset of OpenAI image edit params
        """
        return [
            "prompt",
            "image",
            "model",
            "n",
            "size",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI params to FLUX 2 params.
        FLUX 2 uses the same param names as OpenAI for supported params.
        """
        mapped_params: Dict[str, Any] = {}
        supported_params = self.get_supported_openai_params(model)

        for key, value in dict(image_edit_optional_params).items():
            if key in supported_params and value is not None:
                mapped_params[key] = value

        return mapped_params

    def use_multipart_form_data(self) -> bool:
        """FLUX 2 uses JSON requests, not multipart/form-data."""
        return False

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate Azure AI Foundry environment and set up authentication
        """
        api_key = AzureFoundryModelInfo.get_api_key(api_key)

        if not api_key:
            raise ValueError(
                f"Azure AI API key is required for model {model}. Set AZURE_AI_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Api-Key": api_key,
                "Content-Type": "application/json",
            }
        )
        return headers

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform image edit request for FLUX 2.

        FLUX 2 uses the same endpoint for generation and editing,
        with the image passed as base64 in the JSON body.
        """
        if prompt is None:
            raise ValueError("FLUX 2 image edit requires a prompt.")
        
        if image is None:
            raise ValueError("FLUX 2 image edit requires an image.")
        
        image_b64 = self._convert_image_to_base64(image)

        # Build request body with required params
        request_body: Dict[str, Any] = {
            "prompt": prompt,
            "image": image_b64,
            "model": model,
        }

        # Add mapped optional params (already filtered by map_openai_params)
        request_body.update(image_edit_optional_request_params)

        # Return JSON body and empty files list (FLUX 2 doesn't use multipart)
        return request_body, []

    def _convert_image_to_base64(self, image: Any) -> str:
        """Convert image file to base64 string"""
        # Handle list of images (take first one)
        if isinstance(image, list):
            if len(image) == 0:
                raise ValueError("Empty image list provided")
            image = image[0]

        if isinstance(image, BufferedReader):
            image_bytes = image.read()
            image.seek(0)  # Reset file pointer for potential reuse
        elif isinstance(image, bytes):
            image_bytes = image
        elif hasattr(image, "read"):
            image_bytes = image.read()  # type: ignore
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for Azure AI Foundry FLUX 2 image edits.

        Uses the same /providers/blackforestlabs/v1/flux-2-pro endpoint as image generation.
        """
        api_base = AzureFoundryModelInfo.get_api_base(api_base)

        if api_base is None:
            raise ValueError(
                "Azure AI API base is required. Set AZURE_AI_API_BASE environment variable or pass api_base parameter."
            )

        api_version = (
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

