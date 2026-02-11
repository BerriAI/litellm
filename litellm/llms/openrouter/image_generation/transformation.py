"""
OpenRouter Image Generation Support

OpenRouter provides image generation through chat completion endpoints.
Models like google/gemini-2.5-flash-image return images in the message content.

Response format:
{
    "choices": [{
        "message": {
            "content": "Here is a beautiful sunset for you! ",
            "role": "assistant",
            "images": [{
                "image_url": {"url": "data:image/png;base64,..."},
                "index": 0,
                "type": "image_url"
            }]
        }
    }],
    "usage": {
        "completion_tokens": 1299,
        "prompt_tokens": 6,
        "total_tokens": 1305,
        "completion_tokens_details": {"image_tokens": 1290},
        "cost": 0.0387243
    }
}
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams, AllMessageValues
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails
from litellm.llms.openrouter.common_utils import OpenRouterException


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for OpenRouter image generation via chat completions.
    
    OpenRouter uses chat completion endpoints for image generation,
    so we need to transform image generation requests to chat format
    and extract images from chat responses.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for OpenRouter image generation.
        
        Since OpenRouter uses chat completions for image generation,
        we support standard image generation params.
        """
        return [
            "size",
            "quality",
            "n",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map image generation params to OpenRouter chat completion format.
        
        Maps OpenAI parameters to OpenRouter's image_config format:
        - size -> image_config.aspect_ratio
        - quality -> image_config.image_size
        """
        supported_params = self.get_supported_openai_params(model)
        
        for key, value in non_default_params.items():
            if key in supported_params:
                if key == "size":
                    # Map OpenAI size to OpenRouter aspect_ratio
                    aspect_ratio = self._map_size_to_aspect_ratio(value)
                    if "image_config" not in optional_params:
                        optional_params["image_config"] = {}
                    optional_params["image_config"]["aspect_ratio"] = aspect_ratio
                elif key == "quality":
                    # Map OpenAI quality to OpenRouter image_size
                    image_size = self._map_quality_to_image_size(value)
                    if image_size:
                        if "image_config" not in optional_params:
                            optional_params["image_config"] = {}
                        optional_params["image_config"]["image_size"] = image_size
                else:
                    # Pass through other supported params (like n)
                    optional_params[key] = value
            elif not drop_params:
                # If not supported and drop_params is False, pass through
                optional_params[key] = value
        
        return optional_params
    
    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """
        Map OpenAI size format to OpenRouter aspect_ratio format.
        
        OpenAI sizes:
        - 1024x1024 (square)
        - 1536x1024 (landscape)
        - 1024x1536 (portrait)
        - 1792x1024 (wide landscape, dall-e-3)
        - 1024x1792 (tall portrait, dall-e-3)
        - 256x256, 512x512 (dall-e-2)
        - auto (default)
        
        OpenRouter aspect_ratios:
        - 1:1 → 1024×1024 (default)
        - 2:3 → 832×1248
        - 3:2 → 1248×832
        - 3:4 → 864×1184
        - 4:3 → 1184×864
        - 4:5 → 896×1152
        - 5:4 → 1152×896
        - 9:16 → 768×1344
        - 16:9 → 1344×768
        - 21:9 → 1536×672
        """
        size_to_aspect_ratio = {
            # Square formats
            "256x256": "1:1",
            "512x512": "1:1",
            "1024x1024": "1:1",
            # Landscape formats
            "1536x1024": "3:2",  # 1.5:1 ratio, closest to 3:2
            "1792x1024": "16:9",  # 1.75:1 ratio, closest to 16:9
            # Portrait formats
            "1024x1536": "2:3",  # 0.67:1 ratio, closest to 2:3
            "1024x1792": "9:16",  # 0.57:1 ratio, closest to 9:16
            # Default
            "auto": "1:1",
        }
        return size_to_aspect_ratio.get(size, "1:1")
    
    def _map_quality_to_image_size(self, quality: str) -> Optional[str]:
        """
        Map OpenAI quality to OpenRouter image_size format.
        
        OpenAI quality values:
        - auto (default) - automatically select best quality
        - high, medium, low - for GPT image models
        - hd, standard - for dall-e-3
        
        OpenRouter image_size values (Gemini only):
        - 1K → Standard resolution (default)
        - 2K → Higher resolution
        - 4K → Highest resolution
        """
        quality_to_image_size = {
            # OpenAI quality mappings
            "low": "1K",
            "standard": "1K",
            "medium": "2K",
            "high": "4K",
            "hd": "4K",
            # Auto defaults to standard
            "auto": "1K",
        }
        return quality_to_image_size.get(quality)
    
    def _set_usage_and_cost(
        self,
        model_response: ImageResponse,
        response_json: dict,
        model: str,
    ) -> None:
        """
        Extract and set usage and cost information from OpenRouter response.
        
        Args:
            model_response: ImageResponse object to populate
            response_json: Parsed JSON response from OpenRouter
            model: The model name
        """
        usage_data = response_json.get("usage", {})
        if usage_data:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)
            
            completion_tokens_details = usage_data.get("completion_tokens_details", {})
            image_tokens = completion_tokens_details.get("image_tokens", 0)
            
            model_response.usage = ImageUsage(
                input_tokens=prompt_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0,  # Input doesn't contain images for generation
                    text_tokens=prompt_tokens,
                ),
                output_tokens=image_tokens,
                total_tokens=total_tokens,
            )
            
            cost = usage_data.get("cost")
            if cost is not None:
                if not hasattr(model_response, "_hidden_params"):
                    model_response._hidden_params = {}
                if "additional_headers" not in model_response._hidden_params:
                    model_response._hidden_params["additional_headers"] = {}
                model_response._hidden_params["additional_headers"][
                    "llm_provider-x-litellm-response-cost"
                ] = float(cost)
            
            cost_details = usage_data.get("cost_details", {})
            if cost_details:
                if "response_cost_details" not in model_response._hidden_params:
                    model_response._hidden_params["response_cost_details"] = {}
                model_response._hidden_params["response_cost_details"].update(cost_details)
        
        model_response._hidden_params["model"] = response_json.get("model", model)

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
        Get the complete URL for OpenRouter image generation.
        
        OpenRouter uses chat completions endpoint for image generation.
        Default: https://openrouter.ai/api/v1/chat/completions
        """
        if api_base:
            if not api_base.endswith("/chat/completions"):
                api_base = api_base.rstrip("/")
                return f"{api_base}/chat/completions"
            return api_base
        
        return "https://openrouter.ai/api/v1/chat/completions"

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
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("OPENROUTER_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
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
        Transform image generation request to OpenRouter chat completion format.
        
        Args:
            model: The model name
            prompt: The image generation prompt
            optional_params: Optional parameters (including image_config)
            litellm_params: LiteLLM parameters
            headers: Request headers
            
        Returns:
            dict: Request body in chat completion format with image_config
        """
        request_body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        # These will be passed through to OpenRouter
        for key, value in optional_params.items():
            if key not in ["model", "messages", "modalities"]:
                request_body[key] = value
        
        return request_body

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
        Transform OpenRouter chat completion response to ImageResponse format.
        
        Extracts images from the message content and maps usage/cost information.
        
        Args:
            model: The model name
            raw_response: Raw HTTP response from OpenRouter
            model_response: ImageResponse object to populate
            logging_obj: Logging object
            request_data: Original request data
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            encoding: Encoding
            api_key: API key
            json_mode: JSON mode flag
            
        Returns:
            ImageResponse: Populated image response
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise OpenRouterException(
                message=f"Error parsing OpenRouter response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        if not model_response.data:
            model_response.data = []
        
        try:
            choices = response_json.get("choices", [])
            
            for choice in choices:
                message = choice.get("message", {})
                images = message.get("images", [])
                
                for image_data in images:
                    image_url_obj = image_data.get("image_url", {})
                    image_url = image_url_obj.get("url")
                    
                    if image_url:
                        if image_url.startswith("data:"):
                            # Extract base64 data
                            # Format: data:image/png;base64,<base64_data>
                            parts = image_url.split(",", 1)
                            b64_data = parts[1] if len(parts) > 1 else None
                            
                            model_response.data.append(
                                ImageObject(
                                    b64_json=b64_data,
                                    url=None,
                                    revised_prompt=None,
                                )
                            )
                        else:
                            model_response.data.append(
                                ImageObject(
                                    b64_json=None,
                                    url=image_url,
                                    revised_prompt=None,
                                )
                            )
            
            # Extract and set usage and cost information
            self._set_usage_and_cost(model_response, response_json, model)
            
            return model_response
            
        except Exception as e:
            raise OpenRouterException(
                message=f"Error transforming OpenRouter image generation response: {str(e)}",
                status_code=500,
                headers={},
            )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """Get the appropriate error class for OpenRouter errors."""
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
