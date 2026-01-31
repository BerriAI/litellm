"""
Stability AI Image Edit Config

Handles transformation between OpenAI-compatible format and Stability AI API format.

API Reference: https://platform.stability.ai/docs/api-reference
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.stability import (
    OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO,
    STABILITY_EDIT_ENDPOINTS,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class StabilityImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Stability AI image edit.

    Supports:
    - Stable Diffusion 3 (SD3, SD3.5) Image Edit
    """

    DEFAULT_BASE_URL: str = "https://api.stability.ai"

    def get_supported_openai_params(
        self, model: str
    ) -> List[str]:
        """
        Return list of OpenAI params supported by Stability AI.

        https://platform.stability.ai/docs/api-reference
        """
        return [
            "n",  # Number of images (Stability always returns 1, we can loop)
            "size",  # Maps to aspect_ratio
            "response_format",  # b64_json or url (Stability only returns b64)
            "mask"  
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Stability AI parameters.

        OpenAI -> Stability mappings:
        - size -> aspect_ratio
        - n -> (handled separately, Stability returns 1 image per request)
        """
        supported_params = self.get_supported_openai_params(model)
        # Define mapping from OpenAI params to Stability params
        param_mapping = {
            "size": "aspect_ratio",
            # "n" and "response_format" are handled separately
        }

        # Create a copy to not mutate original - convert TypedDict to regular dict
        mapped_params: Dict[str, Any] = dict(image_edit_optional_params)

        for k, v in image_edit_optional_params.items():
            if k in param_mapping:
                # Map param if mapping exists and value is valid
                if k == "size" and v in OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO:
                    mapped_params[param_mapping[k]] = OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO[v]  # type: ignore
                # Don't copy "size" itself to final dict
            elif k == "n":
                # Store for logic but do not add to outgoing params
                mapped_params["_n"] = v
            elif k == "response_format":
                # Only b64 supported at Stability; store for postprocessing
                mapped_params["_response_format"] = v
            elif k not in supported_params:
                if not drop_params:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        f"Set drop_params=True to drop unsupported parameters."
                    )
                # Otherwise, param will simply be dropped
            else:
                # param is supported and not mapped, keep as-is
                continue

        # Remove OpenAI params that have been mapped unless they're in stability
        for mapped in ["size", "n", "response_format"]:
            if mapped in mapped_params:
                del mapped_params[mapped]

        return mapped_params

    def _get_model_endpoint(self, model: str) -> str:
        """
        Get the API endpoint for a given model.
        """
        # Remove "stability/" prefix if present
        model_name = model.lower()
        if model_name.startswith("stability/"):
            model_name = model_name[10:]  # Remove "stability/" prefix

        # Check if model is in our mapping
        for key, endpoint in STABILITY_EDIT_ENDPOINTS.items():
            if key in model_name:
                return endpoint

        # Default to SD3 endpoint
        return "/v2beta/stable-image/edit/inpaint"

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for the Stability AI API request.
        """
        base_url: str = (
            api_base
            or get_secret_str("STABILITY_API_BASE")
            or litellm_params.get("api_base", None)
            or self.DEFAULT_BASE_URL
        )
        base_url = base_url.rstrip("/")

        endpoint = self._get_model_endpoint(model)
        return f"{base_url}{endpoint}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
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
        Transform OpenAI-style request to Stability AI request format.

        Note: Stability AI uses multipart/form-data, but the HTTP handler
        will handle the conversion from dict to form data.
        """
        # Build Stability request
        # Populate multipart form-data as separate text fields (data) and files.
        # Stability expects prompt/output_format/etc. as normal form fields, not file parts.
        data: Dict[str, Any] = {
            "output_format": "png",  # Default to PNG
        }
        
        # Add prompt only if provided (some Stability endpoints don't require it)
        if prompt is not None and prompt != "":
            data["prompt"] = prompt
        # Handle image parameter - could be a single file or list
        image_file = image[0] if isinstance(image, list) else image  # type: ignore
        files: Dict[str, Any] = {}
        if image is not None:
            image_file = image[0] if isinstance(image, list) else image  # type: ignore
            files["image"] = image_file

        # Add optional params (already mapped in map_openai_params)
        for key, value in image_edit_optional_request_params.items():  # type: ignore
            # Skip internal params (prefixed with _)
            if key.startswith("_") or value is None:
                continue

            # File-like optional param
            if key == "mask":
                # Handle case where mask might be in a list
                mask_value = value
                if isinstance(value, list) and len(value) > 0:
                    mask_value = value[0]
                files["mask"] = mask_value  # type: ignore
                continue

            # File-like optional params (init_image, style_image, etc.)
            if key in ["init_image", "style_image"]:
                # Handle case where value might be in a list
                file_value = value
                if isinstance(value, list) and len(value) > 0:
                    file_value = value[0]
                files[key] = file_value  # type: ignore
                continue

            # Supported text fields
            if key in [
                "negative_prompt",
                "aspect_ratio",
                "seed",
                "mode",
                "strength",
                "style_preset",
                "left",
                "bottom",
                "right",
                "top",
                "creativity",
                "search_prompt",
                "grow_mask",
                "select_prompt",
                "control_strength",
                "composition_fidelity",
                "change_strength"
            ]:
                data[key] = value  # type: ignore

        return data, files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
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

        model_response = ImageResponse()
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

        if not hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = {}
        if "additional_headers" not in model_response._hidden_params:
            model_response._hidden_params["additional_headers"] = {}
        # Override: fetch model-cost from model_cost map based on the provided model name
        model_info = get_model_info(model, custom_llm_provider="stability")
        cost_per_image = model_info.get("output_cost_per_image", 0)
        if cost_per_image is not None:
            model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(cost_per_image)
        return model_response

    def use_multipart_form_data(self) -> bool:
        """
        Stability AI requires multipart/form-data for image generation.
        """
        return True
