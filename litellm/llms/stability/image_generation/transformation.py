"""
Stability AI Image Generation Config

Handles transformation between OpenAI-compatible format and Stability AI API format.

API Reference: https://platform.stability.ai/docs/api-reference
"""

from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.llms.stability import (
    OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO,
    STABILITY_GENERATION_MODELS,
    StabilityImageGenerationRequest,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class StabilityImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Stability AI image generation.

    Supports:
    - Stable Diffusion 3 (SD3, SD3.5)
    - Stable Image Ultra
    - Stable Image Core
    """

    DEFAULT_BASE_URL: str = "https://api.stability.ai"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Return list of OpenAI params supported by Stability AI.

        https://platform.stability.ai/docs/api-reference
        """
        return [
            "n",  # Number of images (Stability always returns 1, we can loop)
            "size",  # Maps to aspect_ratio
            "response_format",  # b64_json or url (Stability only returns b64)
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Stability AI parameters.

        OpenAI -> Stability mappings:
        - size -> aspect_ratio
        - n -> (handled separately, Stability returns 1 image per request)
        """
        supported_params = self.get_supported_openai_params(model)

        for k, v in non_default_params.items():
            if k not in optional_params:
                if k in supported_params:
                    # Map size to aspect_ratio
                    if k == "size" and v in OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO:
                        optional_params["aspect_ratio"] = (
                            OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO[v]
                        )
                    elif k == "n":
                        # Store n for later, but don't pass to Stability
                        optional_params["_n"] = v
                    elif k == "response_format":
                        # Stability only returns base64, store for response handling
                        optional_params["_response_format"] = v
                    else:
                        optional_params[k] = v
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        f"Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def _get_model_endpoint(self, model: str) -> str:
        """
        Get the API endpoint for a given model.
        """
        # Remove "stability/" prefix if present
        model_name = model.lower()
        if model_name.startswith("stability/"):
            model_name = model_name[10:]  # Remove "stability/" prefix

        # Check if model is in our mapping
        for key, endpoint in STABILITY_GENERATION_MODELS.items():
            if key in model_name:
                return endpoint

        # Default to SD3 endpoint
        return "/v2beta/stable-image/generate/sd3"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the Stability AI API request.
        """
        base_url: str = (
            api_base
            or get_secret_str("STABILITY_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        base_url = base_url.rstrip("/")

        endpoint = self._get_model_endpoint(model)
        return f"{base_url}{endpoint}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up headers for Stability AI.
        """
        final_api_key: Optional[str] = api_key or get_secret_str("STABILITY_API_KEY")

        if not final_api_key:
            raise ValueError(
                "STABILITY_API_KEY is not set. "
                "Please set it via environment variable or pass api_key parameter."
            )

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Accept"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style request to Stability AI request format.

        Note: Stability AI uses multipart/form-data, but the HTTP handler
        will handle the conversion from dict to form data.
        """
        # Build Stability request
        stability_request: StabilityImageGenerationRequest = {
            "prompt": prompt,
            "output_format": "png",  # Default to PNG
        }

        # Add optional params (already mapped in map_openai_params)
        for key, value in optional_params.items():
            # Skip internal params (prefixed with _)
            if key.startswith("_"):
                continue
            # Add supported Stability params
            if key in [
                "negative_prompt",
                "aspect_ratio",
                "seed",
                "output_format",
                "model",
                "mode",
                "strength",
                "style_preset",
            ]:
                stability_request[key] = value  # type: ignore

        return dict(stability_request)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform Stability AI response to OpenAI-compatible ImageResponse.

        Stability returns: {"image": "base64...", "finish_reason": "SUCCESS", "seed": 123}
        OpenAI expects: {"data": [{"b64_json": "base64..."}], "created": timestamp}
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Stability AI response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Check for errors in response
        if "errors" in response_data:
            raise self.get_error_class(
                error_message=f"Stability AI error: {response_data['errors']}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Check finish_reason
        finish_reason = response_data.get("finish_reason", "")
        if finish_reason == "CONTENT_FILTERED":
            raise self.get_error_class(
                error_message="Content was filtered by Stability AI safety systems",
                status_code=400,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        # Extract image from response
        image_b64 = response_data.get("image")
        if image_b64:
            model_response.data.append(
                ImageObject(
                    b64_json=image_b64,
                    url=None,
                    revised_prompt=None,
                )
            )

        return model_response

    def use_multipart_form_data(self) -> bool:
        """
        Stability AI requires multipart/form-data for image generation.
        """
        return True
