"""
Black Forest Labs Image Edit Configuration

Handles transformation between OpenAI-compatible format and Black Forest Labs API format
for image editing endpoints (flux-kontext-pro, flux-kontext-max, etc.).

API Reference: https://docs.bfl.ai/
"""

import base64
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

from ..common_utils import (
    DEFAULT_API_BASE,
    IMAGE_EDIT_MODELS,
    BlackForestLabsError,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BlackForestLabsImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Black Forest Labs image editing.

    Supports:
    - flux-kontext-pro: General image editing with prompts
    - flux-kontext-max: Premium quality editing
    - flux-pro-1.0-fill: Inpainting with mask
    - flux-pro-1.0-expand: Outpainting (expand image borders)

    Note: HTTP requests and polling are handled by the handler (handler.py).
    This class only handles data transformation.
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Return list of OpenAI params supported by Black Forest Labs.

        Note: BFL uses different parameter names, these are mapped in map_openai_params.
        """
        return [
            "n",  # Number of images (BFL returns 1 per request)
            "size",  # Maps to aspect_ratio
            "response_format",  # b64_json or url
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Black Forest Labs parameters.

        BFL-specific params are passed through directly.
        """
        optional_params: Dict[str, Any] = {}

        # Pass through BFL-specific params
        bfl_params = [
            "seed",
            "output_format",
            "safety_tolerance",
            "prompt_upsampling",
            # Kontext-specific
            "aspect_ratio",
            # Fill/Inpaint-specific
            "steps",
            "guidance",
            "grow_mask",
            # Expand-specific
            "top",
            "bottom",
            "left",
            "right",
        ]

        # Convert TypedDict to regular dict for access
        params_dict = dict(image_edit_optional_params)

        for param in bfl_params:
            if param in params_dict:
                value = params_dict[param]
                if value is not None:
                    optional_params[param] = value

        # Set default output format
        if "output_format" not in optional_params:
            optional_params["output_format"] = "png"

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
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

    def use_multipart_form_data(self) -> bool:
        """
        BFL uses JSON requests, not multipart/form-data.
        """
        return False

    def _get_model_endpoint(self, model: str) -> str:
        """
        Get the API endpoint for a given model.
        """
        # Remove provider prefix if present (e.g., "black_forest_labs/flux-kontext-pro")
        model_name = model.lower()
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        # Check if model is in our mapping
        if model_name in IMAGE_EDIT_MODELS:
            return IMAGE_EDIT_MODELS[model_name]

        # Default to kontext-pro
        return IMAGE_EDIT_MODELS["flux-kontext-pro"]

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for the Black Forest Labs API request.
        """
        base_url: str = (
            api_base
            or get_secret_str("BFL_API_BASE")
            or DEFAULT_API_BASE
        )
        base_url = base_url.rstrip("/")

        endpoint = self._get_model_endpoint(model)
        return f"{base_url}{endpoint}"

    def _read_image_bytes(self, image: Any) -> bytes:
        """Read image bytes from various input types."""
        if isinstance(image, bytes):
            return image
        elif isinstance(image, list):
            # If it's a list, take the first image
            return self._read_image_bytes(image[0])
        elif hasattr(image, "read"):
            # File-like object
            pos = getattr(image, "tell", lambda: 0)()
            if hasattr(image, "seek"):
                image.seek(0)
            data = image.read()
            if hasattr(image, "seek"):
                image.seek(pos)
            return data
        else:
            return image

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform OpenAI-style request to Black Forest Labs request format.

        BFL uses JSON body with base64-encoded images, not multipart/form-data.
        """
        # Read and encode image
        image_bytes = self._read_image_bytes(image)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Build request body
        request_body: Dict[str, Any] = {
            "prompt": prompt,
            "input_image": b64_image,
        }

        # Add optional params
        for key, value in image_edit_optional_request_params.items():
            if key not in ["extra_headers", "extra_body"] and value is not None:
                request_body[key] = value

        # Handle mask if provided (for inpainting)
        if "mask" in image_edit_optional_request_params:
            mask = image_edit_optional_request_params["mask"]
            mask_bytes = self._read_image_bytes(mask)
            request_body["mask"] = base64.b64encode(mask_bytes).decode("utf-8")

        # BFL uses JSON, not multipart - return empty files
        return request_body, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
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

        # Get image URL from result
        image_url = response_data.get("result", {}).get("sample")
        if not image_url:
            raise BlackForestLabsError(
                status_code=500,
                message="No image URL in BFL result",
            )

        # Build ImageResponse
        return ImageResponse(
            created=int(time.time()),
            data=[ImageObject(url=image_url)],
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BlackForestLabsError:
        """Return the appropriate error class for Black Forest Labs."""
        return BlackForestLabsError(
            status_code=status_code,
            message=error_message,
        )
