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

from typing import TYPE_CHECKING, Any

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

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_API_BASE = "https://api.minimax.io"
IMAGE_GENERATION_ENDPOINT = "/v1/image_generation"

# OpenAI uses "b64_json", MiniMax uses "base64".
OPENAI_TO_MINIMAX_RESPONSE_FORMAT = {
    "b64_json": "base64",
}


class MinimaxImageGenerationException(BaseLLMException):
    """Exception raised for MiniMax image generation API errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class MinimaxImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for MiniMax image generation models (image-01, image-01-live).
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size", "response_format", "seed", "user", "aspect_ratio"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI image generation params to MiniMax params.

        - `size` (WxH) is expanded to `width` and `height`
        - `response_format` "b64_json" is mapped to "base64"
        - remaining supported params are passed through
        """
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported_params:
                continue
            if k == "size":
                width, height = self._parse_size(v)
                if width is not None and height is not None:
                    optional_params["width"] = width
                    optional_params["height"] = height
            elif k == "response_format":
                optional_params["response_format"] = OPENAI_TO_MINIMAX_RESPONSE_FORMAT.get(v, v)
            else:
                optional_params[k] = v
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Build the MiniMax image generation endpoint URL.
        """
        base_url: str = api_base or get_secret_str("MINIMAX_API_BASE") or DEFAULT_API_BASE
        base_url = base_url.rstrip("/")
        if base_url.endswith(IMAGE_GENERATION_ENDPOINT):
            base_url = base_url[: -len(IMAGE_GENERATION_ENDPOINT)]
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return f"{base_url}/image_generation"

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
        """
        Validate the MiniMax environment and set auth headers.
        """
        final_api_key: str | None = api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key
        if not final_api_key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )
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
        Build the MiniMax image generation request body.
        """
        request_data: dict = {
            "model": model,
            "prompt": prompt,
        }

        for k, v in optional_params.items():
            if v is None:
                continue
            if k in {"extra_headers", "extra_body", "user"}:
                continue
            request_data[k] = v

        extra_body = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            request_data.update({k: v for k, v in extra_body.items() if v is not None})

        return request_data

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
        Transform the MiniMax response into a litellm ImageResponse.

        MiniMax returns images under `data.image_urls` (response_format=url) or
        `data.image_base64` (response_format=base64).
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse MiniMax image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        base_resp = response_data.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            raise self.get_error_class(
                error_message=str(base_resp.get("status_msg") or response_data),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        data = response_data.get("data") or {}
        for image_url in data.get("image_urls") or []:
            model_response.data.append(ImageObject(url=image_url))
        for image_base64 in data.get("image_base64") or []:
            model_response.data.append(ImageObject(b64_json=image_base64))

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,
    ) -> BaseLLMException:
        return MinimaxImageGenerationException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @staticmethod
    def _parse_size(size: Any) -> tuple[int | None, int | None]:
        """
        Parse an OpenAI `WxH` size string into width/height integers.

        MiniMax accepts width/height in [512, 2048] divisible by 8.
        """
        if not isinstance(size, str) or "x" not in size:
            return None, None
        parts = size.split("x")
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
