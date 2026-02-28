"""
OpenRouter Image Edit Support

OpenRouter provides image editing through chat completion endpoints.
The source image is sent as a base64 data URL in the message content,
and the response contains edited images in the message's images array.

Request format:
{
    "model": "google/gemini-2.5-flash-image",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Edit this image by..."}
        ]
    }],
    "modalities": ["image", "text"]
}

Response format:
{
    "choices": [{
        "message": {
            "content": "Here is the edited image.",
            "role": "assistant",
            "images": [{
                "image_url": {"url": "data:image/png;base64,..."},
                "type": "image_url"
            }]
        }
    }],
    "usage": {
        "completion_tokens": 1299,
        "prompt_tokens": 300,
        "total_tokens": 1599,
        "completion_tokens_details": {"image_tokens": 1290},
        "cost": 0.0387243
    }
}
"""

import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterImageEditConfig(BaseImageEditConfig):
    """
    Configuration for OpenRouter image editing via chat completions.

    OpenRouter uses the chat completions endpoint for image editing.
    The source image is sent as a base64 data URL in the message content,
    and the response contains edited images in the message's images array.
    """

    def get_supported_openai_params(self, model: str) -> list:
        return ["size", "quality", "n"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        supported_params = self.get_supported_openai_params(model)
        mapped_params: Dict[str, Any] = {}

        for key, value in image_edit_optional_params.items():
            if key in supported_params:
                if key == "size":
                    if "image_config" not in mapped_params:
                        mapped_params["image_config"] = {}
                    mapped_params["image_config"]["aspect_ratio"] = self._map_size_to_aspect_ratio(value)
                elif key == "quality":
                    image_size = self._map_quality_to_image_size(value)
                    if image_size:
                        if "image_config" not in mapped_params:
                            mapped_params["image_config"] = {}
                        mapped_params["image_config"]["image_size"] = image_size
                else:
                    mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
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

    def use_multipart_form_data(self) -> bool:
        """OpenRouter uses JSON requests, not multipart/form-data."""
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base:
            if not api_base.endswith("/chat/completions"):
                api_base = api_base.rstrip("/")
                return f"{api_base}/chat/completions"
            return api_base
        return "https://openrouter.ai/api/v1/chat/completions"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: Optional[str],
        image: Optional[FileTypes],
        image_edit_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        content_parts: List[Dict[str, Any]] = []

        # Add source image(s) as base64 data URLs
        if image is not None:
            images = image if isinstance(image, list) else [image]
            for img in images:
                if img is None:
                    continue
                mime_type = ImageEditRequestUtils.get_image_content_type(img)
                image_bytes = self._read_image_bytes(img)
                b64_data = base64.b64encode(image_bytes).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_data}"
                        },
                    }
                )

        # Add the text prompt
        if prompt:
            content_parts.append({"type": "text", "text": prompt})

        request_body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content_parts,
                }
            ],
            "modalities": ["image", "text"],
        }

        # Add mapped optional params (image_config, n, etc.)
        for key, value in image_edit_optional_request_params.items():
            if key not in ("model", "messages", "modalities"):
                request_body[key] = value

        empty_files = cast(RequestFiles, [])
        return request_body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise OpenRouterException(
                message=f"Error parsing OpenRouter response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response = ImageResponse()
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
                            # Extract base64 data from data URL
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

            self._set_usage_and_cost(model_response, response_json, model)
            return model_response

        except Exception as e:
            raise OpenRouterException(
                message=f"Error transforming OpenRouter image edit response: {str(e)}",
                status_code=500,
                headers={},
            )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    # Private helper methods

    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """
        Map OpenAI size format to OpenRouter aspect_ratio format.

        Uses the same mapping as image generation since OpenRouter
        handles both through the same chat completions endpoint.
        """
        size_to_aspect_ratio = {
            "256x256": "1:1",
            "512x512": "1:1",
            "1024x1024": "1:1",
            "1536x1024": "3:2",
            "1792x1024": "16:9",
            "1024x1536": "2:3",
            "1024x1792": "9:16",
            "auto": "1:1",
        }
        return size_to_aspect_ratio.get(size, "1:1")

    def _map_quality_to_image_size(self, quality: str) -> Optional[str]:
        """
        Map OpenAI quality to OpenRouter image_size format.

        Uses the same mapping as image generation since OpenRouter
        handles both through the same chat completions endpoint.
        """
        quality_to_image_size = {
            "low": "1K",
            "standard": "1K",
            "medium": "2K",
            "high": "4K",
            "hd": "4K",
            "auto": "1K",
        }
        return quality_to_image_size.get(quality)

    def _set_usage_and_cost(
        self,
        model_response: ImageResponse,
        response_json: dict,
        model: str,
    ) -> None:
        """Extract and set usage and cost information from OpenRouter response."""
        usage_data = response_json.get("usage", {})
        if usage_data:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)

            completion_tokens_details = usage_data.get("completion_tokens_details", {})
            image_tokens = completion_tokens_details.get("image_tokens", 0)

            # For image edit, input may include image tokens
            input_image_tokens = 0
            prompt_tokens_details = usage_data.get("prompt_tokens_details", {})
            if prompt_tokens_details:
                input_image_tokens = prompt_tokens_details.get("image_tokens", 0)

            model_response.usage = ImageUsage(
                input_tokens=prompt_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=input_image_tokens,
                    text_tokens=prompt_tokens - input_image_tokens,
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

    def _read_image_bytes(self, image: FileTypes) -> bytes:
        """Read raw bytes from various image input types."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, BytesIO):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, BufferedReader):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        raise ValueError("Unsupported image type for OpenRouter image edit.")
