import base64
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, cast

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.gemini.common_utils import (
    get_gemini_image_generation_config,
    map_openai_image_params_to_gemini,
)
from litellm.llms.gemini.image_usage_transformation import (
    transform_gemini_image_usage,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    FileTypes,
    ImageObject,
    ImageResponse,
    OpenAIImage,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GeminiImageEditConfig(BaseImageEditConfig):
    DEFAULT_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    SUPPORTED_PARAMS: list[str] = ["n", "size", "imageConfig"]

    def get_supported_openai_params(self, model: str) -> list[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, Any]:
        return map_openai_image_params_to_gemini(
            params=image_edit_optional_params,
            model=model,
            supported_params=self.get_supported_openai_params(model),
            parse_image_config_string=True,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final[str | None] = api_key or get_secret_str("GEMINI_API_KEY")
        if not final_api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        headers["x-goog-api-key"] = final_api_key
        headers["Content-Type"] = "application/json"
        return headers

    def use_multipart_form_data(self) -> bool:
        """Gemini uses JSON requests, not multipart/form-data."""
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url = api_base or get_secret_str("GEMINI_API_BASE") or self.DEFAULT_BASE_URL
        base_url = base_url.rstrip("/")
        return f"{base_url}/models/{model}:generateContent"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict[str, Any], RequestFiles | None]:
        inline_parts: Final = self._prepare_inline_image_parts(image) if image else []
        if not inline_parts:
            raise ValueError("Gemini image edit requires at least one image.")

        # Build parts list with image and prompt (if provided)
        parts: Final = inline_parts.copy()
        if prompt is not None and prompt != "":
            parts.append({"text": prompt})

        contents: Final = [
            {
                "parts": parts,
            }
        ]

        request_body: Final[dict[str, Any]] = {"contents": contents}

        request_body["generationConfig"] = get_gemini_image_generation_config(
            model=model,
            optional_params=image_edit_optional_request_params,
        )

        empty_files: Final = cast(RequestFiles, [])
        return request_body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        model_response: Final = ImageResponse()
        try:
            response_json: Final = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error transforming image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        candidates: Final = response_json.get("candidates", [])
        data_list: Final[list[ImageObject]] = []

        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    data_list.append(
                        ImageObject(
                            b64_json=inline_data["data"],
                            url=None,
                        )
                    )

        model_response.data = cast(list[OpenAIImage], data_list)
        if "usageMetadata" in response_json:
            model_response.usage = transform_gemini_image_usage(response_json["usageMetadata"])
        return model_response

    def _prepare_inline_image_parts(self, image: FileTypes | list[FileTypes]) -> list[dict[str, Any]]:
        images: list[FileTypes]
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        inline_parts: Final[list[dict[str, Any]]] = []
        for img in images:
            if img is None:
                continue

            mime_type = ImageEditRequestUtils.get_image_content_type(img)
            image_bytes = self._read_all_bytes(img)
            inline_parts.append(
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                }
            )

        return inline_parts

    def _read_all_bytes(self, image: FileTypes) -> bytes:
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
        raise ValueError("Unsupported image type for Gemini image edit.")
