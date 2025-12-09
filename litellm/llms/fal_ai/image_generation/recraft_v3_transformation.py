from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIRecraftV3Config(FalAIBaseConfig):
    """
    Configuration for Fal AI Recraft v3 Text-to-Image model.
    
    Recraft v3 is a text-to-image model with multiple style options including
    realistic images, digital illustrations, and vector illustrations.
    
    Model endpoint: fal-ai/recraft/v3/text-to-image
    Documentation: https://fal.ai/models/fal-ai/recraft/v3/text-to-image
    """
    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/recraft/v3/text-to-image"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for Recraft v3.
        """
        return [
            "n",
            "response_format",
            "size",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Recraft v3 parameters.
        
        Mappings:
        - size -> image_size (can be preset or custom width/height)
        - response_format -> ignored (Recraft returns URLs)
        - n -> ignored (Recraft doesn't support multiple images)
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Recraft v3 params
        param_mapping = {
            "size": "image_size",
        }
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    # Use mapped parameter name if exists
                    mapped_key = param_mapping.get(k, k)
                    mapped_value = non_default_params[k]
                    
                    # Transform specific parameters
                    if k == "response_format":
                        # Recraft always returns URLs, so we can ignore this
                        continue
                    elif k == "n":
                        # Recraft doesn't support multiple images, ignore
                        continue
                    elif k == "size":
                        # Map OpenAI size format to Recraft image_size
                        mapped_value = self._map_image_size(mapped_value)
                    
                    optional_params[mapped_key] = mapped_value
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def _map_image_size(self, size: str) -> Any:
        """
        Map OpenAI size format to Recraft v3 image_size format.
        
        OpenAI format: "1024x1024", "1792x1024", etc.
        Recraft format: Can be preset strings or {"width": int, "height": int}
        
        Available presets:
        - square_hd (default)
        - square
        - portrait_4_3
        - portrait_16_9
        - landscape_4_3
        - landscape_16_9
        """
        # Map common OpenAI sizes to Recraft presets
        size_mapping = {
            "1024x1024": "square_hd",
            "512x512": "square",
            "768x1024": "portrait_4_3",
            "576x1024": "portrait_16_9",
            "1024x768": "landscape_4_3",
            "1024x576": "landscape_16_9",
        }
        
        if size in size_mapping:
            return size_mapping[size]
        
        # Parse custom size format "WIDTHxHEIGHT"
        if "x" in size:
            try:
                width, height = size.split("x")
                return {
                    "width": int(width),
                    "height": int(height),
                }
            except (ValueError, AttributeError):
                pass
        
        # Default to square_hd
        return "square_hd"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Recraft v3 request body.
        
        Required parameters:
        - prompt: Text prompt (max 1000 characters)
        
        Optional parameters:
        - image_size: Preset or {"width": int, "height": int} (default: "square_hd")
        - style: Style preset (default: "realistic_image")
          Options: "any", "realistic_image", "digital_illustration", "vector_illustration", etc.
        - colors: Array of RGB color objects [{"r": 0-255, "g": 0-255, "b": 0-255}]
        - enable_safety_checker: Enable safety checker (default: false)
        - style_id: UUID for custom style reference
        
        Note: Vector illustrations cost 2X as much.
        """
        recraft_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        
        return recraft_request_body

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
        Transform the Recraft v3 response to litellm ImageResponse format.
        
        Expected response format:
        {
            "images": [
                {
                    "url": "https://...",
                    "content_type": "image/webp",
                    "file_name": "...",
                    "file_size": 123456
                }
            ]
        }
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        if not model_response.data:
            model_response.data = []
        
        # Handle Recraft v3 response format
        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=None,  # Recraft returns URLs only
                        )
                    )
                elif isinstance(image_data, str):
                    # If images is just a list of URLs
                    model_response.data.append(
                        ImageObject(
                            url=image_data,
                            b64_json=None,
                        )
                    )
        
        return model_response

