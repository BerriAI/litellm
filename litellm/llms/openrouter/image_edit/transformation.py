"""
OpenRouter Image Edit Support

OpenRouter edits images through the same endpoint it generates them from,
``POST https://openrouter.ai/api/v1/images``. The source image travels in
``input_references`` as a base64 data URL. Models whose only output modality is ``image``
(``openai/gpt-image-*``, ``bytedance-seed/*``, ...) are reachable only there, and asking
``/chat/completions`` for ``modalities: ["image", "text"]`` on one of them answers with
"No endpoints found that support the requested output modalities".

Request format:
{
    "model": "openai/gpt-image-2",
    "prompt": "Add a sunset behind the mountain",
    "input_references": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ],
    "aspect_ratio": "1:1"
}

See ``litellm/llms/openrouter/image_api.py`` for the response format and the parameter
mapping shared with image generation.
"""

import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Union, cast

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_api import (
    NON_BODY_PARAMS,
    apply_images_response,
    map_image_params,
    parse_images_response,
    resolve_images_url,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterImageEditConfig(BaseImageEditConfig):
    """Maps ``/v1/images/edits`` onto OpenRouter's ``/api/v1/images`` endpoint."""

    def get_supported_openai_params(self, model: str) -> list:
        return ["background", "n", "quality", "response_format", "size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return map_image_params(params=image_edit_optional_params, model=model, drop_params=drop_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
            }
        )
        return headers

    def use_multipart_form_data(self) -> bool:
        """OpenRouter's image API takes JSON, not multipart/form-data."""
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return resolve_images_url(api_base)

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        images = [img for img in (image if isinstance(image, list) else [image]) if img is not None]
        if not images:
            raise ValueError("An image is required to edit; OpenRouter has no image-less edit mode.")

        request_body = {
            "model": model,
            **({"prompt": prompt} if prompt is not None else {}),
            **{key: value for key, value in image_edit_optional_request_params.items() if key not in NON_BODY_PARAMS},
            "input_references": [self._input_reference(img) for img in images],
        }
        return request_body, cast(RequestFiles, [])

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        return apply_images_response(
            parsed=parse_images_response(raw_response),
            model_response=ImageResponse(),
            model=model,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def _input_reference(self, image: FileTypes) -> dict:
        mime_type = ImageEditRequestUtils.get_image_content_type(image)
        b64_data = base64.b64encode(self._read_image_bytes(image)).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
        }

    def _read_image_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, (BytesIO, BufferedReader)):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        raise ValueError("Unsupported image type for OpenRouter image edit.")
