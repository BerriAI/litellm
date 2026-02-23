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


class FalAIStableDiffusionConfig(FalAIBaseConfig):
    """
    Configuration for Fal AI Stable Diffusion models.
    
    Supports Stable Diffusion v3.5 variants and other Stable Diffusion models on Fal AI.
    
    Example models:
    - fal-ai/stable-diffusion-v35-medium
    - fal-ai/stable-diffusion-v35-large
    
    Documentation: https://fal.ai/models/fal-ai/stable-diffusion-v35-medium
    """
    IMAGE_GENERATION_ENDPOINT: str = ""  # Will be set from model name
    
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
        Get the complete url for the request.
        
        For Stable Diffusion models, extract the endpoint from the model name.
        """
        from litellm.secret_managers.main import get_secret_str
        
        complete_url: str = (
            api_base 
            or get_secret_str("FAL_AI_API_BASE") 
            or self.DEFAULT_BASE_URL
        )
        
        complete_url = complete_url.rstrip("/")
        
        # Extract endpoint from model name
        # e.g., "fal-ai/stable-diffusion-v35-medium" or "stable-diffusion-v35-medium"
        endpoint = model
        if "/" in model and not model.startswith("fal-ai/"):
            # If model is like "custom/stable-diffusion-v35-medium", use full path
            endpoint = model
        elif not model.startswith("fal-ai/"):
            # If model is just "stable-diffusion-v35-medium", prepend fal-ai
            endpoint = f"fal-ai/{model}"
        
        complete_url = f"{complete_url}/{endpoint}"
        return complete_url
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for Stable Diffusion models.
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
        Map OpenAI parameters to Stable Diffusion parameters.
        
        Mappings:
        - n -> num_images (1-4, default 1)
        - response_format -> output_format (jpeg or png)
        - size -> image_size (can be preset or custom width/height)
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Stable Diffusion params
        param_mapping = {
            "n": "num_images",
            "response_format": "output_format",
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
                        # Map OpenAI response formats to image formats
                        if mapped_value in ["b64_json", "url"]:
                            mapped_value = "jpeg"
                    elif k == "size":
                        # Map OpenAI size format to Stable Diffusion image_size
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
        Map OpenAI size format to Stable Diffusion image_size format.
        
        OpenAI format: "1024x1024", "1792x1024", etc.
        Stable Diffusion format: Can be preset strings or {"width": int, "height": int}
        
        Available presets:
        - square_hd
        - square
        - portrait_4_3
        - portrait_16_9
        - landscape_4_3 (default)
        - landscape_16_9
        """
        # Map common OpenAI sizes to Stable Diffusion presets
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
        
        # Default to landscape_4_3
        return "landscape_4_3"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Stable Diffusion request body.
        
        Required parameters:
        - prompt: The prompt to generate an image from
        
        Optional parameters:
        - num_images: Number of images (1-4, default: 1)
        - image_size: Size preset or {"width": int, "height": int} (default: landscape_4_3)
        - output_format: "jpeg" or "png" (default: jpeg)
        - sync_mode: Wait for image upload before returning (default: false)
        - guidance_scale: CFG scale 0-20 (default: 4.5)
        - num_inference_steps: Inference steps 1-50 (default: 40)
        - seed: Random seed for reproducibility
        - negative_prompt: Negative prompt string (default: "")
        - enable_safety_checker: Enable safety checker (default: true)
        """
        stable_diffusion_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        
        return stable_diffusion_request_body

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
        Transform the Stable Diffusion response to litellm ImageResponse format.
        
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
        
        # Handle Stable Diffusion response format
        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=None,  # Stable Diffusion returns URLs only
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
        
        # Add additional metadata from Stable Diffusion response
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

