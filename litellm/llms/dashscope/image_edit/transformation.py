"""
DashScope Image Edit Configuration

Handles transformation between OpenAI-compatible image edit format and the DashScope
multimodal-generation API.

API endpoint: POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

Request format:
{
    "model": "qwen-image-edit-plus",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"image": "data:image/png;base64,<...>"},
                    {"text": "<prompt>"}
                ]
            }
        ]
    },
    "parameters": {"n": 1, "size": "1024*1024", ...}
}

Response format:
{
    "output": {
        "choices": [{"message": {"content": [{"image": "<url>"}]}}]
    },
    "usage": {"image_count": 1, ...}
}
"""

import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, cast

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.dashscope.image_generation.transformation import (
    DEFAULT_API_BASE,
    OPENAI_TO_DASHSCOPE_SIZE,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class DashScopeImageEditConfig(BaseImageEditConfig):
    """
    Configuration for DashScope image editing (qwen-image-edit-plus).
    """

    SUPPORTED_PARAMS: tuple[str, ...] = ("n", "size")

    def get_supported_openai_params(self, model: str) -> list:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return {
            k: (OPENAI_TO_DASHSCOPE_SIZE.get(v, v.replace("x", "*")) if k == "size" else v)
            for k, v in dict(image_edit_optional_params).items()
            if v is not None and k in self.SUPPORTED_PARAMS
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        return {
            **headers,
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
        }

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return api_base or get_secret_str("DASHSCOPE_API_BASE_IMAGE") or DEFAULT_API_BASE

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        image_parts = self._prepare_image_parts(image) if image else []
        if not image_parts:
            raise ValueError("DashScope image edit requires at least one image.")

        content = [*image_parts, *([{"text": prompt}] if prompt else [])]
        request_body = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": dict(image_edit_optional_request_params),
        }
        return request_body, cast(RequestFiles, [])  # cast-ok: JSON provider sends no multipart files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope image edit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "code" in response_data and "output" not in response_data:
            raise self.get_error_class(
                error_message=str(response_data.get("message", response_data)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        data = [
            ImageObject(url=content_item["image"])
            for choice in response_data.get("output", {}).get("choices", [])
            for content_item in choice.get("message", {}).get("content", [])
            if content_item.get("image")
        ]

        model_response = ImageResponse()
        model_response.data = cast(list[OpenAIImage], data)  # cast-ok: ImageObject satisfies OpenAIImage shape
        return model_response

    def _prepare_image_parts(self, image: FileTypes | list[FileTypes]) -> list[dict[str, str]]:
        images = image if isinstance(image, list) else [image]
        return [
            {
                "image": "data:{};base64,{}".format(
                    ImageEditRequestUtils.get_image_content_type(img),
                    base64.b64encode(self._read_all_bytes(img)).decode("utf-8"),
                )
            }
            for img in images
            if img is not None
        ]

    def _read_all_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, (BytesIO, BufferedReader)):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        raise ValueError("Unsupported image type for DashScope image edit.")
