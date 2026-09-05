"""
MiniMax Image Generation Configuration

Maps OpenAI image generation params to the MiniMax image generation API.

API endpoint: POST https://api.minimax.io/v1/image_generation

Request format:
{
    "model": "image-01",
    "prompt": "<prompt>",
    "n": 1,
    "aspect_ratio": "1:1",
    "response_format": "url"
}

Response format:
{
    "data": {"image_urls": ["<url>"], "image_base64": ["<base64>"]},
    "metadata": {"success_count": 1, "failed_count": 0},
    "base_resp": {"status_code": 0, "status_msg": "success"}
}

Reference: https://platform.minimax.io/docs/api-reference/image-generation-t2i
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
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

    LiteLLMLoggingObj = _LiteLLMLoggingObj  # rebind-ok: conditional runtime/type-checking alias
else:
    LiteLLMLoggingObj = object  # rebind-ok: conditional runtime/type-checking alias

DEFAULT_API_BASE: Final = "https://api.minimax.io"
IMAGE_GENERATION_ENDPOINT: Final = "/v1/image_generation"

# OpenAI uses "b64_json", MiniMax uses "base64".
OPENAI_TO_MINIMAX_RESPONSE_FORMAT: Final = MappingProxyType({"b64_json": "base64"})


class MinimaxImageGenerationException(BaseLLMException):
    """Exception raised for MiniMax image generation API errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,  # mutable-ok: BaseLLMException interface requires dict
    ) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for MiniMax image generation models (image-01, image-01-live).
    """

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig returns list
        return [  # mutable-ok: BaseImageGenerationConfig returns a mutable parameter list
            "n",
            "size",
            "response_format",
            "seed",
            "user",
            "aspect_ratio",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        optional_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig interface returns dict
        """
        Map OpenAI image generation params to MiniMax params.

        - `size` (WxH) is expanded to `width` and `height`
        - `response_format` "b64_json" is mapped to "base64"
        - remaining supported params are passed through
        """
        supported_params: Final = self.get_supported_openai_params(model)
        mapped_params: Final = dict(  # mutable-ok: provider interface returns a mutable request mapping
            optional_params
        )
        for k, v in non_default_params.items():
            if k in mapped_params:
                continue
            if k not in supported_params:
                continue
            if k == "size":
                width, height = self._parse_size(v)
                if width is not None and height is not None:
                    mapped_params["width"] = width
                    mapped_params["height"] = height
            elif k == "response_format":
                mapped_params["response_format"] = OPENAI_TO_MINIMAX_RESPONSE_FORMAT.get(v, v)
            else:
                mapped_params[k] = v
        return mapped_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        litellm_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        stream: bool | None = None,
    ) -> str:
        """
        Build the MiniMax image generation endpoint URL.
        """
        configured_base_url: Final = api_base or get_secret_str("MINIMAX_API_BASE") or DEFAULT_API_BASE
        endpoint_base_url: Final = configured_base_url.rstrip("/").removesuffix(IMAGE_GENERATION_ENDPOINT)
        versioned_base_url: Final = (
            endpoint_base_url if endpoint_base_url.endswith("/v1") else f"{endpoint_base_url}/v1"
        )
        return f"{versioned_base_url}/image_generation"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseImageGenerationConfig interface requires list
        optional_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        litellm_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig interface returns dict
        """
        Validate the MiniMax environment and set auth headers.
        """
        final_api_key: Final[str | None] = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
        if not final_api_key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )
        validated_headers: Final = dict(  # mutable-ok: provider interface returns fresh mutable headers
            headers
        )
        validated_headers["Authorization"] = f"Bearer {final_api_key}"
        validated_headers["Content-Type"] = "application/json"
        return validated_headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        litellm_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        headers: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
    ) -> dict:  # mutable-ok: BaseImageGenerationConfig interface returns dict
        """
        Build the MiniMax image generation request body.
        """
        request_data: Final = {  # mutable-ok: request payload is assembled before being returned
            "model": model,
            "prompt": prompt,
        }

        for k, v in optional_params.items():
            if v is None:
                continue
            if k in ("extra_headers", "extra_body", "user"):
                continue
            request_data[k] = v

        extra_body: Final = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            request_data.update((k, v) for k, v in extra_body.items() if v is not None)

        return request_data

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        optional_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        litellm_params: dict,  # mutable-ok: BaseImageGenerationConfig interface requires dict
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """
        Transform the MiniMax response into a litellm ImageResponse.

        MiniMax returns images under `data.image_urls` (response_format=url) or
        `data.image_base64` (response_format=base64).
        """
        try:
            response_data: Final = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Failed to parse MiniMax image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        base_resp: Final = response_data.get("base_resp") or MappingProxyType({})
        status_code: Final = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            raise self.get_error_class(
                error_message=str(base_resp.get("status_msg") or response_data),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []  # mutable-ok: ImageResponse schema requires list  # rebind-ok: transformer populates the supplied response

        data: Final = response_data.get("data") or MappingProxyType({})
        for image_url in data.get("image_urls") or ():
            model_response.data.append(ImageObject(url=image_url))
        for image_base64 in data.get("image_base64") or ():
            model_response.data.append(ImageObject(b64_json=image_base64))

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseImageGenerationConfig interface requires dict
    ) -> BaseLLMException:
        return MinimaxImageGenerationException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @staticmethod
    def _parse_size(size: object) -> tuple[int | None, int | None]:
        """
        Parse an OpenAI `WxH` size string into width/height integers.

        MiniMax accepts width/height in [512, 2048] divisible by 8.
        """
        if not isinstance(size, str) or "x" not in size:
            return None, None
        parts: Final = size.split("x")
        if len(parts) != 2:
            return None, None
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
        if width < 512 or width > 2048 or height < 512 or height > 2048:
            return None, None
        if width % 8 != 0 or height % 8 != 0:
            return None, None
        return width, height
