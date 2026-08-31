import base64
import json
import os
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.vertex_ai import (
    GenerateContentResponseBody,
    HttpxContentType,
    HttpxPartType,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _GenerateContentSource(Protocol):
    """An HTTP response whose JSON body is a Gemini ``generateContent`` result."""

    def json(self) -> GenerateContentResponseBody: ...


def _generate_content_payload(response: _GenerateContentSource) -> GenerateContentResponseBody:
    return response.json()


class VertexAIGeminiImageEditConfig(BaseImageEditConfig, VertexLLM):
    """
    Vertex AI Gemini Image Edit Configuration

    Uses generateContent API for Gemini models on Vertex AI
    """

    SUPPORTED_PARAMS: list[str] = ["size"]

    def __init__(self) -> None:
        BaseImageEditConfig.__init__(self)
        VertexLLM.__init__(self)

    def get_supported_openai_params(self, model: str) -> list[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, str]:
        supported_params: Final = self.get_supported_openai_params(model)
        if "size" not in supported_params or "size" not in image_edit_optional_params:
            return {}

        size: Final = image_edit_optional_params.get("size")
        return {"aspectRatio": self._map_size_to_aspect_ratio(size or "")}

    def _resolve_vertex_project(self) -> str | None:
        return (
            getattr(self, "_vertex_project", None)
            or os.environ.get("VERTEXAI_PROJECT")
            or getattr(litellm, "vertex_project", None)
            or get_secret_str("VERTEXAI_PROJECT")
        )

    def _resolve_vertex_location(self) -> str | None:
        return (
            getattr(self, "_vertex_location", None)
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or getattr(litellm, "vertex_location", None)
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )

    def _resolve_vertex_credentials(self) -> str | None:
        return (
            getattr(self, "_vertex_credentials", None)
            or os.environ.get("VERTEXAI_CREDENTIALS")
            or getattr(litellm, "vertex_credentials", None)
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        headers = headers or {}
        litellm_params = litellm_params or {}

        # If a custom api_base is provided, skip credential validation
        # This allows users to use proxies or mock endpoints without needing Vertex AI credentials
        _api_base: Final = litellm_params.get("api_base") or api_base
        if _api_base is not None:
            return headers

        # First check litellm_params (where vertex_ai_project/vertex_ai_credentials are passed)
        # then fall back to environment variables and other sources
        vertex_project: Final = self.safe_get_vertex_ai_project(litellm_params) or self._resolve_vertex_project()
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params) or self._resolve_vertex_credentials()
        access_token, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        return self.set_headers(access_token, headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        """
        Get the complete URL for Vertex AI Gemini generateContent API
        """
        # Use the model name as provided, handling vertex_ai prefix
        model_name = model
        if model.startswith("vertex_ai/"):
            model_name = model.replace("vertex_ai/", "")

        # If a custom api_base is provided, use it directly
        # This allows users to use proxies or mock endpoints
        if api_base:
            return api_base.rstrip("/")

        # First check litellm_params (where vertex_ai_project/vertex_ai_location are passed)
        # then fall back to environment variables and other sources
        vertex_project: Final = self.safe_get_vertex_ai_project(litellm_params) or self._resolve_vertex_project()
        vertex_location: Final = self.safe_get_vertex_ai_location(litellm_params) or self._resolve_vertex_location()

        if not vertex_project or not vertex_location:
            raise ValueError("vertex_project and vertex_location are required for Vertex AI")

        base_url: Final = get_vertex_base_url(vertex_location)

        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model_name}:generateContent"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, object], RequestFiles | None]:
        inline_parts: Final = self._prepare_inline_image_parts(image) if image else []
        if not inline_parts:
            raise ValueError("Vertex AI Gemini image edit requires at least one image.")

        # Build parts list with image and prompt (if provided)
        text_parts: Final[list[HttpxPartType]] = [{"text": prompt}] if prompt is not None and prompt != "" else []
        parts: Final[list[HttpxPartType]] = [*inline_parts, *text_parts]

        # Correct format for Vertex AI Gemini image editing
        contents: Final[dict[str, object]] = {"role": "USER", "parts": parts}

        # Add image-specific configuration
        image_config: Final = (
            {"aspect_ratio": image_edit_optional_request_params["aspectRatio"]}
            if "aspectRatio" in image_edit_optional_request_params
            else None
        )

        # Generation config with proper structure for image editing
        generation_config: Final[dict[str, object]] = {
            key: value for key, value in (("response_modalities", ["IMAGE"]), ("image_config", image_config)) if value
        }

        request_body: Final[dict[str, object]] = {"contents": contents, "generationConfig": generation_config}

        payload: Final = json.dumps(request_body)
        empty_files: Final = cast(RequestFiles, [])
        return cast(tuple[dict[str, Any], RequestFiles | None], (payload, empty_files))

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        model_response: Final = ImageResponse()
        try:
            response_json: Final = _generate_content_payload(raw_response)
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error transforming image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        candidates: Final = response_json.get("candidates", [])
        contents: Final[list[HttpxContentType]] = [
            candidate["content"] for candidate in candidates if "content" in candidate
        ]
        parts: Final[list[HttpxPartType]] = [part for content in contents for part in content.get("parts", [])]
        data_list: Final[list[ImageObject]] = [
            ImageObject(b64_json=b64_json, url=None)
            for part in parts
            if (inline_data := part.get("inlineData")) and (b64_json := inline_data.get("data"))
        ]

        model_response.data = cast(list[OpenAIImage], data_list)
        return model_response

    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """Map OpenAI size format to Gemini aspect ratio format"""
        aspect_ratio_map: Final = {
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1280x896": "4:3",
            "896x1280": "3:4",
        }
        return aspect_ratio_map.get(size, "1:1")

    def _prepare_inline_image_parts(self, image: FileTypes | list[FileTypes]) -> list[HttpxPartType]:
        images: Final[list[FileTypes]] = image if isinstance(image, list) else [image]
        return [
            {
                "inlineData": {
                    "mimeType": ImageEditRequestUtils.get_image_content_type(img),
                    "data": base64.b64encode(self._read_all_bytes(img)).decode("utf-8"),
                }
            }
            for img in images
            if img is not None
        ]

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
        raise ValueError("Unsupported image type for Vertex AI Gemini image edit.")
