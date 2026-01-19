"""
Bedrock Stability AI Image Edit Transformation

Handles transformation between OpenAI-compatible format and Bedrock Stability AI Image Edit API format.

Supported models:
- stability.stable-conservative-upscale-v1:0
- stability.stable-creative-upscale-v1:0
- stability.stable-fast-upscale-v1:0
- stability.stable-outpaint-v1:0
- stability.stable-image-control-sketch-v1:0
- stability.stable-image-control-structure-v1:0
- stability.stable-image-erase-object-v1:0
- stability.stable-image-inpaint-v1:0
- stability.stable-image-remove-background-v1:0
- stability.stable-image-search-recolor-v1:0
- stability.stable-image-search-replace-v1:0
- stability.stable-image-style-guide-v1:0
- stability.stable-style-transfer-v1:0

API Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
"""

import base64
import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx

from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.stability import (
    OPENAI_SIZE_TO_STABILITY_ASPECT_RATIO,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BedrockStabilityImageEditConfig(BaseImageEditConfig):
    """
    Configuration for Bedrock Stability AI image edit.

    Supports all Stability image edit operations through Bedrock.
    """

    @classmethod
    def _is_stability_edit_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a Bedrock Stability edit model.
        
        Bedrock Stability edit models follow this pattern:
            stability.stable-conservative-upscale-v1:0
            stability.stable-creative-upscale-v1:0
            stability.stable-fast-upscale-v1:0
            stability.stable-outpaint-v1:0
            stability.stable-image-inpaint-v1:0
            stability.stable-image-erase-object-v1:0
            etc.
        """
        if model:
            model_lower = model.lower()
            if "stability." in model_lower and any([
                "upscale" in model_lower,
                "outpaint" in model_lower,
                "inpaint" in model_lower,
                "erase" in model_lower,
                "remove-background" in model_lower,
                "search-recolor" in model_lower,
                "search-replace" in model_lower,
                "control-sketch" in model_lower,
                "control-structure" in model_lower,
                "style-guide" in model_lower,
                "style-transfer" in model_lower,
            ]):
                return True
        return False

    def get_supported_openai_params(
        self, model: str
    ) -> list:
        """
        Return list of OpenAI params supported by Bedrock Stability.
        """
        return [
            "n",  # Number of images (Stability always returns 1, we can loop)
            "size",  # Maps to aspect_ratio
            "response_format",  # b64_json or url (Stability only returns b64)
            "mask",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Bedrock Stability parameters.

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

    def transform_image_edit_request(  #noqa: PLR0915
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, Any]:
        """
        Transform OpenAI-style request to Bedrock Stability request format.

        Returns the request body dict that will be JSON-encoded by the handler.
        """
        # Build Bedrock Stability request
        data: Dict[str, Any] = {
            "output_format": "png",  # Default to PNG
        }
        
        # Add prompt only if provided (some models don't require it)
        if prompt is not None and prompt != "":
            data["prompt"] = prompt
        
        # Convert image to base64 if provided
        if image is not None:
            image_b64: str
            if hasattr(image, 'read') and callable(getattr(image, 'read', None)):
                # File-like object (e.g., BufferedReader from open())
                image_bytes = image.read()  # type: ignore
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')  # type: ignore
            elif isinstance(image, bytes):
                # Raw bytes
                image_b64 = base64.b64encode(image).decode('utf-8')
            elif isinstance(image, str):
                # Already a base64 string
                image_b64 = image
            else:
                # Try to handle as bytes
                image_b64 = base64.b64encode(bytes(image)).decode('utf-8')  # type: ignore

            # For style-transfer models, map image to init_image
            model_lower = model.lower()
            if "style-transfer" in model_lower:
                data["init_image"] = image_b64
            else:
                data["image"] = image_b64

        # Add optional params (already mapped in map_openai_params)
        for key, value in image_edit_optional_request_params.items():  # type: ignore
            # Skip internal params (prefixed with _)
            if key.startswith("_") or value is None:
                continue

            # File-like optional params (mask, init_image, style_image, etc.)
            if key in ["mask", "init_image", "style_image"]:
                # Handle case where value might be in a list
                file_value = value
                if isinstance(value, list) and len(value) > 0:
                    file_value = value[0]
                
                if hasattr(file_value, 'read') and callable(getattr(file_value, 'read', None)):
                    file_bytes = file_value.read()  # type: ignore
                elif isinstance(file_value, bytes):
                    file_bytes = file_value
                elif isinstance(file_value, str):
                    # Already a base64 string
                    data[key] = file_value
                    continue
                else:
                    file_bytes = file_value  # type: ignore
                
                if isinstance(file_bytes, bytes):
                    file_b64 = base64.b64encode(file_bytes).decode('utf-8')
                else:
                    file_b64 = str(file_bytes)
                data[key] = file_b64
                continue
            
            # Numeric fields that need to be converted to int/float
            numeric_int_fields = ["left", "right", "up", "down", "seed"]
            numeric_float_fields = [
                "strength",
                "creativity",
                "control_strength",
                "grow_mask",
                "fidelity",
                "composition_fidelity",
                "style_strength",
                "change_strength",
            ]
            
            if key in numeric_int_fields:
                # Convert to int (these are pixel values for outpaint)
                try:
                    data[key] = int(value)  # type: ignore
                except (ValueError, TypeError):
                    data[key] = value  # type: ignore
            elif key in numeric_float_fields:
                # Convert to float
                try:
                    data[key] = float(value)  # type: ignore
                except (ValueError, TypeError):
                    data[key] = value  # type: ignore

            # Supported text fields
            elif key in [
                "negative_prompt",
                "aspect_ratio",
                "output_format",
                "model",
                "mode",
                "style_preset",
                "select_prompt",
                "search_prompt",
            ]:
                data[key] = value  # type: ignore

        return data, {}

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform Bedrock Stability response to OpenAI-compatible ImageResponse.

        Bedrock returns: {"images": ["base64..."], "finish_reasons": [null], "seeds": [123]}
        OpenAI expects: {"data": [{"b64_json": "base64..."}], "created": timestamp}
        """
        try:
            response_data = raw_response.json()
            with open("response_data.json", "w") as f:
                json.dump(response_data, f)
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Bedrock Stability response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Check for errors in response
        if "errors" in response_data:
            raise self.get_error_class(
                error_message=f"Bedrock Stability error: {response_data['errors']}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Check finish_reasons
        finish_reasons = response_data.get("finish_reasons", [])
        if finish_reasons and finish_reasons[0]:
            raise self.get_error_class(
                error_message=f"Bedrock Stability error: {finish_reasons[0]}",
                status_code=400,
                headers=raw_response.headers,
            )

        model_response = ImageResponse()
        if not model_response.data:
            model_response.data = []

        # Extract images from response
        images = response_data.get("images", [])
        if images:
            for image_b64 in images:
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
        
        # Set cost based on model
        model_info = get_model_info(model, custom_llm_provider="bedrock")
        cost_per_image = model_info.get("output_cost_per_image", 0)
        if cost_per_image is not None:
            model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(cost_per_image)
        
        return model_response

    def use_multipart_form_data(self) -> bool:
        """
        Bedrock Stability uses JSON format, not multipart/form-data.
        """
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for the Bedrock Image Edit API.
        
        For Bedrock, this is handled by the handler which constructs the endpoint URL
        based on the model ID and AWS region. This method is required by the base class
        but the actual URL construction happens in BedrockImageEdit.image_edit().
        
        Returns a placeholder - the real endpoint is constructed in the handler.
        """
        # Bedrock URLs are constructed in the handler using boto3
        # This is a placeholder for the abstract method requirement
        return "bedrock://image-edit"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate environment for Bedrock Stability image edit.
        
        For Bedrock, AWS credentials are managed by the BaseAWSLLM class.
        This method validates that headers are properly set up.
        
        Args:
            headers: The request headers to validate/update
            model: The model name being used
            api_key: Optional API key (not used for Bedrock, which uses AWS credentials)
        
        Returns:
            Updated headers dict
        """
        if headers is None:
            headers = {}
        
        # Bedrock uses AWS credentials, not API keys
        # Headers are set up by the handler's get_request_headers() method
        # This just ensures basic headers are present
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        return headers

