"""
DashScope Image Generation Configuration

Handles transformation between OpenAI-compatible format and DashScope multimodal-generation API.

API endpoint: POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

Request format:
{
    "model": "qwen-image-2.0-pro",
    "input": {
        "messages": [{"role": "user", "content": [{"text": "<prompt>"}]}]
    },
    "parameters": {"size": "1024*1024", ...}
}

Response format:
{
    "output": {
        "choices": [{"message": {"content": [{"image": "<url>"}]}}]
    },
    "usage": {"input_tokens": 0, "output_tokens": 0, "width": 1024, "height": 1024, "image_count": 1}
}
"""

from typing import TYPE_CHECKING, Any, Final

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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_API_BASE: Final = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# Maps OpenAI size strings (WxH) to DashScope size strings (W*H)
OPENAI_TO_DASHSCOPE_SIZE: Final[dict] = {
    "256x256": "256*256",
    "512x512": "512*512",
    "1024x1024": "1024*1024",
    "1792x1024": "1792*1024",
    "1024x1792": "1024*1792",
    "2048x2048": "2048*2048",
}


class DashScopeImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for DashScope image generation (qwen-image-2.0, qwen-image-2.0-pro).
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        mapped: Final[dict] = {}
        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported_params:
                continue
            if k == "size":
                # Convert "WxH" → "W*H"
                mapped["size"] = OPENAI_TO_DASHSCOPE_SIZE.get(v, v.replace("x", "*"))
            elif k == "n":
                mapped["image_count"] = v
        return mapped

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return api_base or get_secret_str("DASHSCOPE_API_BASE_IMAGE") or DEFAULT_API_BASE

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
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
        Transform OpenAI-style image generation request to DashScope multimodal-generation format.
        """
        parameters: Final[dict] = {}
        for k, v in optional_params.items():
            parameters[k] = v

        return {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": parameters,
        }

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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """
        Transform DashScope response to litellm ImageResponse.

        DashScope response: output.choices[0].message.content[0].image
        OpenAI response:    data[0].url
        """
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # DashScope can return API-level errors in a 200 response body.
        # Example: {"code": "InvalidParameter", "message": "Size not supported"}
        if "code" in response_data and "output" not in response_data:
            raise self.get_error_class(
                error_message=str(response_data.get("message", response_data)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        choices: Final = response_data.get("output", {}).get("choices", [])
        for choice in choices:
            content_list = choice.get("message", {}).get("content", [])
            for content_item in content_list:
                image_url = content_item.get("image")
                if image_url:
                    model_response.data.append(ImageObject(url=image_url))

        return model_response
