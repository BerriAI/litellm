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
    "parameters": {"size": "1024*1024", "n": 1, ...}
}

Response format:
{
    "output": {
        "choices": [{"message": {"content": [{"image": "<url>"}]}}]
    },
    "usage": {"output_width": 1024, "output_height": 1024, "output_image_count": 1}
}
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    DASHSCOPE_CHAT_COMPATIBLE_PATH,
    DASHSCOPE_IMAGE_PATH,
    is_dashscope_native_endpoint_url,
    resolve_dashscope_endpoints,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(Iterable[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_DECODED_JSON: Final = TypeAdapter(object)

DEFAULT_API_BASE: Final = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

# Maps OpenAI size strings (WxH) to DashScope size strings (W*H)
OPENAI_TO_DASHSCOPE_SIZE: Final[Mapping[str, str]] = {
    "256x256": "256*256",
    "512x512": "512*512",
    "1024x1024": "1024*1024",
    "1792x1024": "1792*1024",
    "1024x1792": "1024*1792",
    "2048x2048": "2048*2048",
}


def _message_image_urls(message: Mapping[str, object]) -> Iterator[str]:
    contents: Final = _JSON_OBJECTS.validate_python(message.get("content", []))
    return (url for url in (item.get("image") for item in contents) if isinstance(url, str) and url)


def _image_urls(response_object: Mapping[str, object]) -> Iterator[str]:
    output: Final = _JSON_OBJECT.validate_python(response_object.get("output", {}))
    choices: Final = _JSON_OBJECTS.validate_python(output.get("choices", []))
    for choice in choices:
        yield from _message_image_urls(_JSON_OBJECT.validate_python(choice.get("message", {})))


def _dashscope_size(size: object) -> object:
    if not isinstance(size, str):
        return size
    return OPENAI_TO_DASHSCOPE_SIZE.get(size, size.replace("x", "*"))


class DashScopeImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for DashScope image generation (qwen-image-2.0, qwen-image-2.0-pro,
    qwen-image-3.0, qwen-image-3.0-pro).
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params: Final = self.get_supported_openai_params(model)
        requested: Final = {
            k: v for k, v in non_default_params.items() if k in supported_params and k not in optional_params
        }
        return {k: _dashscope_size(v) if k == "size" else v for k, v in requested.items()}

    def _resolve_api_key(self, api_key: str | None) -> str:
        resolved_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        return resolved_api_key

    def _image_api_base_env(self) -> str | None:
        return get_secret_str("DASHSCOPE_API_BASE_IMAGE")

    def _default_image_api_base(self) -> str:
        return DEFAULT_API_BASE

    def _configured_chat_api_base(self) -> str | None:
        return get_secret_str("DASHSCOPE_API_BASE")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        image_env_api_base: Final = self._image_api_base_env()
        configured_api_base: Final = api_base or self._configured_chat_api_base()
        if not configured_api_base or configured_api_base.rstrip("/").endswith(DASHSCOPE_CHAT_COMPATIBLE_PATH):
            return image_env_api_base or self._default_image_api_base()
        if is_dashscope_native_endpoint_url(configured_api_base):
            return configured_api_base
        endpoints: Final = resolve_dashscope_endpoints(configured_api_base)
        if endpoints is None:
            return configured_api_base
        if not api_base and image_env_api_base:
            return image_env_api_base
        return f"{endpoints.origin}{DASHSCOPE_IMAGE_PATH}"

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        headers["Authorization"] = f"Bearer {self._resolve_api_key(api_key)}"
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
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: "Tokenizer | None",
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
            response_data: Final = _DECODED_JSON.validate_python(raw_response.json())
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # DashScope can return API-level errors in a 200 response body.
        # Example: {"code": "InvalidParameter", "message": "Size not supported"}
        response_object: Final = _JSON_OBJECT.validate_python(response_data)
        if "code" in response_object and "output" not in response_object:
            raise self.get_error_class(
                error_message=str(response_object.get("message", response_object)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        images: Final = [*(model_response.data or ()), *(ImageObject(url=url) for url in _image_urls(response_object))]
        usage: Final = response_object.get("usage")
        usage_fields: Final[Mapping[str, object]] = usage if isinstance(usage, Mapping) else {}
        width: Final = usage_fields.get("output_width", usage_fields.get("width"))
        height: Final = usage_fields.get("output_height", usage_fields.get("height"))
        image_response: Final = model_response.model_copy(
            update={
                "data": images,
                **({"size": f"{width}x{height}"} if isinstance(width, int) and isinstance(height, int) else {}),
            }
        )
        input_image_count: Final = usage_fields.get("input_image_count")
        if isinstance(input_image_count, int) and input_image_count > 0:
            image_response.hidden_params = {
                **model_response.hidden_params,
                "billable_input_image_count": input_image_count,
            }
        return image_response
