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


class FalAIBriaConfig(FalAIBaseConfig):
    """
    Configuration for Bria Text-to-Image 3.2 model.
    
    Bria 3.2 is a commercial-grade text-to-image model with prompt enhancement
    and multiple aspect ratio options.
    
    Model endpoint: bria/text-to-image/3.2
    Documentation: https://fal.ai/models/bria/text-to-image/3.2
    """
    IMAGE_GENERATION_ENDPOINT: str = "bria/text-to-image/3.2"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for Bria 3.2.
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
        Map OpenAI parameters to Bria 3.2 parameters.
        
        Mappings:
        - size -> aspect_ratio (1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9)
        - response_format -> ignored (Bria returns URLs)
        - n -> ignored (Bria doesn't support multiple images in one call)
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Bria params
        param_mapping = {
            "size": "aspect_ratio",
        }
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    # Use mapped parameter name if exists
                    mapped_key = param_mapping.get(k, k)
                    mapped_value = non_default_params[k]
                    
                    # Transform specific parameters
                    if k == "response_format":
                        # Bria always returns URLs, so we can ignore this
                        continue
                    elif k == "n":
                        # Bria doesn't support multiple images, ignore
                        continue
                    elif k == "size":
                        # Map OpenAI size format to Bria aspect ratio
                        mapped_value = self._map_aspect_ratio(mapped_value)
                    
                    optional_params[mapped_key] = mapped_value
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def _map_aspect_ratio(self, size: str) -> str:
        """
        Map OpenAI size format to Bria aspect ratio format.
        
        OpenAI format: "1024x1024", "1792x1024", etc.
        Bria format: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"
        """
        # Map common OpenAI sizes to Bria aspect ratios
        size_to_aspect_ratio = {
            "1024x1024": "1:1",
            "512x512": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1024x768": "4:3",
            "768x1024": "3:4",
            "1280x960": "4:3",
            "960x1280": "3:4",
        }
        
        if size in size_to_aspect_ratio:
            return size_to_aspect_ratio[size]
        
        # Parse custom size format "WIDTHxHEIGHT" and calculate aspect ratio
        if "x" in size:
            try:
                width_str, height_str = size.split("x")
                width = int(width_str)
                height = int(height_str)
                
                # Calculate aspect ratio and find closest match
                ratio = width / height
                
                # Map to closest supported aspect ratio
                if 0.95 <= ratio <= 1.05:  # Close to 1:1
                    return "1:1"
                elif ratio >= 1.7:  # Close to 16:9
                    return "16:9"
                elif ratio <= 0.6:  # Close to 9:16
                    return "9:16"
                elif 1.3 <= ratio <= 1.4:  # Close to 4:3
                    return "4:3"
                elif 0.7 <= ratio <= 0.8:  # Close to 3:4
                    return "3:4"
                elif 1.45 <= ratio <= 1.55:  # Close to 3:2
                    return "3:2"
                elif 0.65 <= ratio <= 0.7:  # Close to 2:3
                    return "2:3"
                elif 1.2 <= ratio <= 1.3:  # Close to 5:4
                    return "5:4"
                elif 0.75 <= ratio <= 0.85:  # Close to 4:5
                    return "4:5"
            except (ValueError, AttributeError, ZeroDivisionError):
                pass
        
        # Default to 1:1
        return "1:1"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Bria 3.2 request body.
        
        Required parameters:
        - prompt: Prompt for image generation
        
        Optional parameters:
        - aspect_ratio: "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9" (default: "1:1")
        - prompt_enhancer: Improve the prompt (default: true)
        - sync_mode: Return image directly in response (default: false)
        - truncate_prompt: Truncate the prompt (default: true)
        - guidance_scale: Guidance scale 1-10 (default: 5)
        - num_inference_steps: Inference steps 20-50 (default: 30)
        - seed: Random seed for reproducibility (default: 5555)
        - negative_prompt: Negative prompt string
        """
        bria_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        
        return bria_request_body

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
        Transform the Bria 3.2 response to litellm ImageResponse format.
        
        Expected response format:
        {
            "image": {
                "url": "https://...",
                "content_type": "image/png",
                "file_name": "...",
                "file_size": 123456,
                "width": 1024,
                "height": 1024
            }
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
        
        # Handle Bria response format - uses "image" (singular) not "images"
        image_data = response_data.get("image")
        if image_data and isinstance(image_data, dict):
            model_response.data.append(
                ImageObject(
                    url=image_data.get("url", None),
                    b64_json=None,  # Bria returns URLs only
                )
            )
        
        return model_response

