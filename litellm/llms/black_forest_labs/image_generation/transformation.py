"""
Black Forest Labs Image Generation Configuration

Handles transformation between OpenAI-compatible format and Black Forest Labs API format
for image generation endpoints (flux-pro-1.1, flux-pro-1.1-ultra, flux-dev, flux-pro).

API Reference: https://docs.bfl.ai/
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    DEFAULT_API_BASE,
    IMAGE_GENERATION_MODELS,
    BlackForestLabsError,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BlackForestLabsImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Black Forest Labs image generation (text-to-image).

    Supports:
    - flux-pro-1.1: Fast & reliable standard generation
    - flux-pro-1.1-ultra: Ultra high-resolution (up to 4MP)
    - flux-dev: Development/open-source variant
    - flux-pro: Original pro model

    Note: HTTP requests and polling are handled by the handler (handler.py).
    This class only handles data transformation.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Return list of OpenAI params supported by Black Forest Labs.

        Note: BFL uses different parameter names, these are mapped in map_openai_params.
        """
        return [
            "n",  # Number of images (BFL returns 1 per request, but ultra supports up to 4)
            "size",  # Maps to width/height or aspect_ratio
            "response_format",  # b64_json or url
            "quality",  # Maps to raw mode for ultra
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Black Forest Labs parameters.

        BFL-specific params are passed through directly.
        """
        supported_params = self.get_supported_openai_params(model)

        for k, v in non_default_params.items():
            if k in optional_params:
                continue

            if k in supported_params:
                # Map OpenAI 'size' to BFL width/height
                if k == "size" and v:
                    self._map_size_param(v, optional_params)
                elif k == "n":
                    # BFL uses num_images for ultra model
                    if "ultra" in model.lower():
                        optional_params["num_images"] = v
                elif k == "quality" and v == "hd":
                    # Map 'hd' quality to raw mode for more natural look
                    if "ultra" in model.lower():
                        optional_params["raw"] = True
                else:
                    optional_params[k] = v
            elif not drop_params:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    f"Set drop_params=True to drop unsupported parameters."
                )

        return optional_params

    def _map_size_param(self, size: str, optional_params: dict) -> None:
        """Map OpenAI size parameter to BFL width/height."""
        # Common size mappings
        size_mapping = {
            "1024x1024": (1024, 1024),
            "1792x1024": (1792, 1024),
            "1024x1792": (1024, 1792),
            "512x512": (512, 512),
            "256x256": (256, 256),
        }

        if size in size_mapping:
            width, height = size_mapping[size]
            optional_params["width"] = width
            optional_params["height"] = height
        elif "x" in size:
            # Parse custom size
            try:
                width, height = map(int, size.lower().split("x"))
                optional_params["width"] = width
                optional_params["height"] = height
            except ValueError:
                pass  # Ignore invalid size format

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
        Validate environment and set up headers for Black Forest Labs.

        BFL uses x-key header for authentication.
        """
        final_api_key: Optional[str] = (
            api_key
            or get_secret_str("BFL_API_KEY")
            or get_secret_str("BLACK_FOREST_LABS_API_KEY")
        )

        if not final_api_key:
            raise BlackForestLabsError(
                status_code=401,
                message="BFL_API_KEY is not set. Please set it via environment variable or pass api_key parameter.",
            )

        headers["x-key"] = final_api_key
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"

        return headers

    def _get_model_endpoint(self, model: str) -> str:
        """
        Get the API endpoint for a given model.
        """
        # Remove provider prefix if present (e.g., "black_forest_labs/flux-pro-1.1")
        model_name = model.lower()
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        # Check if model is in our mapping
        if model_name in IMAGE_GENERATION_MODELS:
            return IMAGE_GENERATION_MODELS[model_name]

        # Default to flux-pro-1.1
        return IMAGE_GENERATION_MODELS["flux-pro-1.1"]

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
        Get the complete URL for the Black Forest Labs API request.
        """
        base_url: str = (
            api_base or get_secret_str("BFL_API_BASE") or DEFAULT_API_BASE
        )
        base_url = base_url.rstrip("/")

        endpoint = self._get_model_endpoint(model)
        return f"{base_url}{endpoint}"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style request to Black Forest Labs request format.

        https://docs.bfl.ai/flux_models/flux_1_1_pro
        """
        # Build request body with prompt
        request_body: Dict[str, Any] = {
            "prompt": prompt,
        }

        # BFL-specific params that can be passed through
        bfl_params = [
            "width",
            "height",
            "aspect_ratio",
            "seed",
            "output_format",
            "safety_tolerance",
            "prompt_upsampling",
            # Ultra-specific
            "raw",
            "num_images",
            "image_url",
            "image_prompt_strength",
        ]

        for param in bfl_params:
            if param in optional_params and optional_params[param] is not None:
                request_body[param] = optional_params[param]

        # Set default output format if not specified
        if "output_format" not in request_body:
            request_body["output_format"] = "png"

        return request_body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        """
        Transform Black Forest Labs response to OpenAI-compatible ImageResponse.

        This is called with the FINAL polled response (after handler does polling).
        The response contains: {"status": "Ready", "result": {"sample": "https://..."}}
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise BlackForestLabsError(
                status_code=raw_response.status_code,
                message=f"Error parsing BFL response: {e}",
            )

        result = response_data.get("result", {})

        if not model_response.data:
            model_response.data = []

        # Handle single image (sample) or multiple images
        if isinstance(result, dict) and "sample" in result:
            model_response.data.append(ImageObject(url=result["sample"]))
        elif isinstance(result, list):
            # Multiple images returned
            for img in result:
                if isinstance(img, str):
                    model_response.data.append(ImageObject(url=img))
                elif isinstance(img, dict) and "url" in img:
                    model_response.data.append(ImageObject(url=img["url"]))

        if not model_response.data:
            raise BlackForestLabsError(
                status_code=500,
                message="No image URL in BFL result",
            )

        model_response.created = int(time.time())
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BlackForestLabsError:
        """Return the appropriate error class for Black Forest Labs."""
        return BlackForestLabsError(
            status_code=status_code,
            message=error_message,
        )


def get_black_forest_labs_image_generation_config(
    model: str,
) -> BlackForestLabsImageGenerationConfig:
    """
    Get the appropriate image generation config for a Black Forest Labs model.

    Currently returns a single config class, but can be extended
    for model-specific configurations if needed.
    """
    return BlackForestLabsImageGenerationConfig()
