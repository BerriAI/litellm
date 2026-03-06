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


class FalAIFluxProV11UltraConfig(FalAIBaseConfig):
    """
    Configuration for Fal AI Flux Pro v1.1-ultra model.
    
    FLUX Pro v1.1-ultra is a high-quality text-to-image model with enhanced detail
    and support for image prompts.
    
    Model endpoint: fal-ai/flux-pro/v1.1-ultra
    Documentation: https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra
    """
    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/flux-pro/v1.1-ultra"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for Flux Pro v1.1-ultra.
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
        Map OpenAI parameters to Flux Pro v1.1-ultra parameters.
        
        Mappings:
        - n -> num_images (1-4, default 1)
        - response_format -> output_format (jpeg or png)
        - size -> aspect_ratio (21:9, 16:9, 4:3, 3:2, 1:1, 2:3, 3:4, 9:16, 9:21)
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Flux Pro v1.1-ultra params
        param_mapping = {
            "n": "num_images",
            "response_format": "output_format",
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
                        # Map OpenAI response formats to image formats
                        if mapped_value in ["b64_json", "url"]:
                            mapped_value = "jpeg"
                    elif k == "size":
                        # Map OpenAI size format to Flux aspect ratio
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
        Map OpenAI size format to Flux Pro aspect ratio format.
        
        OpenAI format: "1024x1024", "1792x1024", etc.
        Flux format: "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"
        
        Default: "16:9"
        """
        # Map common OpenAI sizes to Flux aspect ratios
        size_to_aspect_ratio = {
            "1024x1024": "1:1",
            "512x512": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1024x768": "4:3",
            "768x1024": "3:4",
            "1536x1024": "3:2",
            "1024x1536": "2:3",
            "2048x876": "21:9",
            "876x2048": "9:21",
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
                elif ratio >= 2.3:  # Close to 21:9
                    return "21:9"
                elif 1.7 <= ratio < 2.3:  # Close to 16:9
                    return "16:9"
                elif 1.3 <= ratio < 1.7:  # Close to 4:3
                    return "4:3"
                elif 1.4 <= ratio < 1.6:  # Close to 3:2
                    return "3:2"
                elif 0.6 <= ratio < 0.7:  # Close to 3:4
                    return "3:4"
                elif 0.65 <= ratio < 0.75:  # Close to 2:3
                    return "2:3"
                elif 0.5 <= ratio < 0.6:  # Close to 9:16
                    return "9:16"
                elif ratio < 0.5:  # Close to 9:21
                    return "9:21"
            except (ValueError, AttributeError, ZeroDivisionError):
                pass
        
        # Default to 16:9
        return "16:9"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Flux Pro v1.1-ultra request body.
        
        Required parameters:
        - prompt: The prompt to generate an image from
        
        Optional parameters:
        - num_images: Number of images (1-4, default: 1)
        - aspect_ratio: Aspect ratio (default: "16:9")
        - raw: Generate less processed images (default: false)
        - output_format: "jpeg" or "png" (default: "jpeg")
        - image_url: Image URL for image-to-image generation
        - sync_mode: Return data URI (default: false)
        - safety_tolerance: Safety level "1"-"6" (default: "2")
        - enable_safety_checker: Enable safety checker (default: true)
        - seed: Random seed for reproducibility
        - image_prompt_strength: Strength of image prompt 0-1 (default: 0.1)
        - enhance_prompt: Enhance prompt for better results (default: false)
        """
        flux_pro_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        
        return flux_pro_request_body

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
        Transform the Flux Pro v1.1-ultra response to litellm ImageResponse format.
        
        Expected response format:
        {
            "images": [
                {
                    "url": "https://...",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg"
                }
            ],
            "timings": {"inference": 2.5, ...},
            "seed": 42,
            "has_nsfw_concepts": [false],
            "prompt": "original prompt"
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
        
        # Handle Flux Pro v1.1-ultra response format
        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=None,  # Flux Pro returns URLs only
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
        
        # Add additional metadata from Flux Pro response
        if hasattr(model_response, "_hidden_params"):
            if "seed" in response_data:
                model_response._hidden_params["seed"] = response_data["seed"]
            if "timings" in response_data:
                model_response._hidden_params["timings"] = response_data["timings"]
            if "has_nsfw_concepts" in response_data:
                model_response._hidden_params["has_nsfw_concepts"] = response_data[
                    "has_nsfw_concepts"
                ]
        
        return model_response

