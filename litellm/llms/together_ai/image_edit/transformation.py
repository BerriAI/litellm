import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Union

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.together_ai.common_utils import (
    TogetherAIException,
    get_together_ai_images_generations_url,
    map_openai_image_param_to_together_ai,
    resolve_together_ai_api_key,
    together_ai_image_data_to_image_objects,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class TogetherAIImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["n", "size", "response_format"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return dict(
            mapped_item
            for key, value in image_edit_optional_params.items()
            for mapped_item in map_openai_image_param_to_together_ai(key, value)
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key = resolve_together_ai_api_key(api_key)
        if resolved_api_key is None:
            raise TogetherAIException(
                message="Together AI API key is not set. Set TOGETHERAI_API_KEY or pass api_key.",
                status_code=401,
                headers={},
            )
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return get_together_ai_images_generations_url(api_base)

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        images = image if isinstance(image, list) else ([image] if image is not None else [])
        data_urls = tuple(self._to_data_url(img) for img in images if img is not None)
        if len(data_urls) == 0:
            raise TogetherAIException(
                message="image is required for Together AI image edit requests.",
                status_code=400,
                headers={},
            )
        if len(data_urls) > 1:
            raise TogetherAIException(
                message="Together AI image edit supports a single input image per request.",
                status_code=400,
                headers={},
            )
        prompt_params = {"prompt": prompt} if prompt is not None else {}
        optional_body_params = {
            k: v for k, v in image_edit_optional_request_params.items() if k not in ("extra_headers", "extra_body")
        }
        extra_body = image_edit_optional_request_params.get("extra_body") or {}
        request_body: dict[str, Any] = {
            "model": model,
            "image_url": data_urls[0],
            **prompt_params,
            **optional_body_params,
            **extra_body,
        }
        empty_files: RequestFiles = []
        return request_body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        try:
            response_json = raw_response.json()
        except ValueError as e:
            raise TogetherAIException(
                message=f"Error parsing Together AI image edit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        model_response = ImageResponse()
        model_response.data = together_ai_image_data_to_image_objects(response_json)
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TogetherAIException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def _to_data_url(self, image: FileTypes) -> str:
        mime_type = ImageEditRequestUtils.get_image_content_type(image)
        encoded = base64.b64encode(self._read_image_bytes(image)).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _read_image_bytes(self, image: FileTypes) -> bytes:
        if isinstance(image, bytes):
            return image
        if isinstance(image, bytearray):
            return bytes(image)
        if isinstance(image, (BytesIO, BufferedReader)):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        if isinstance(image, tuple) and len(image) >= 2 and image[1] is not None:
            return self._read_image_bytes(image[1])
        raise TogetherAIException(
            message=f"Unsupported image type for Together AI image edit: {type(image)}",
            status_code=400,
            headers={},
        )
