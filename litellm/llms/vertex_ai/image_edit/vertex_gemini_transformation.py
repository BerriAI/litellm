import base64
import json
import os
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

import litellm

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse, OpenAIImage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexAIGeminiImageEditConfig(BaseImageEditConfig, VertexLLM):
    """
    Vertex AI Gemini Image Edit Configuration
    
    Uses generateContent API for Gemini models on Vertex AI
    """
    SUPPORTED_PARAMS: List[str] = ["size"]

    def __init__(self) -> None:
        BaseImageEditConfig.__init__(self)
        VertexLLM.__init__(self)

    def get_supported_openai_params(self, model: str) -> List[str]:
        return list(self.SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        supported_params = self.get_supported_openai_params(model)
        filtered_params = {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported_params
        }

        mapped_params: Dict[str, Any] = {}

        if "size" in filtered_params:
            mapped_params["aspectRatio"] = self._map_size_to_aspect_ratio(
                filtered_params["size"]  # type: ignore[arg-type]
            )

        return mapped_params

    def _resolve_vertex_project(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_project", None)
            or os.environ.get("VERTEXAI_PROJECT")
            or getattr(litellm, "vertex_project", None)
            or get_secret_str("VERTEXAI_PROJECT")
        )

    def _resolve_vertex_location(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_location", None)
            or os.environ.get("VERTEXAI_LOCATION")
            or os.environ.get("VERTEX_LOCATION")
            or getattr(litellm, "vertex_location", None)
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )

    def _resolve_vertex_credentials(self) -> Optional[str]:
        return (
            getattr(self, "_vertex_credentials", None)
            or os.environ.get("VERTEXAI_CREDENTIALS")
            or getattr(litellm, "vertex_credentials", None)
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        headers = headers or {}
        vertex_project = self._resolve_vertex_project()
        vertex_credentials = self._resolve_vertex_credentials()
        access_token, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        return self.set_headers(access_token, headers)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Vertex AI Gemini generateContent API
        """
        vertex_project = self._resolve_vertex_project()
        vertex_location = self._resolve_vertex_location()

        if not vertex_project or not vertex_location:
            raise ValueError("vertex_project and vertex_location are required for Vertex AI")

        # Use the model name as provided, handling vertex_ai prefix
        model_name = model
        if model.startswith("vertex_ai/"):
            model_name = model.replace("vertex_ai/", "")

        if api_base:
            base_url = api_base.rstrip("/")
        else:
            base_url = f"https://{vertex_location}-aiplatform.googleapis.com"

        return f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model_name}:generateContent"

    def transform_image_edit_request(  # type: ignore[override]
        self,
        model: str,
        prompt: str,
        image: FileTypes,
        image_edit_optional_request_params: Dict[str, Any],
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict[str, Any], Optional[RequestFiles]]:
        inline_parts = self._prepare_inline_image_parts(image)
        if not inline_parts:
            raise ValueError("Vertex AI Gemini image edit requires at least one image.")

        # Correct format for Vertex AI Gemini image editing
        contents = {
            "role": "USER",
            "parts": inline_parts + [{"text": prompt}]
        }

        request_body: Dict[str, Any] = {"contents": contents}

        # Generation config with proper structure for image editing
        generation_config: Dict[str, Any] = {
            "response_modalities": ["IMAGE"]
        }

        # Add image-specific configuration
        image_config: Dict[str, Any] = {}
        if "aspectRatio" in image_edit_optional_request_params:
            image_config["aspect_ratio"] = image_edit_optional_request_params["aspectRatio"]
        
        if image_config:
            generation_config["image_config"] = image_config

        request_body["generationConfig"] = generation_config

        payload: Any = json.dumps(request_body)
        empty_files = cast(RequestFiles, [])
        return cast(Tuple[Dict[str, Any], Optional[RequestFiles]], (payload, empty_files))

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ImageResponse:
        model_response = ImageResponse()
        try:
            response_json = raw_response.json()
        except Exception as exc:
            raise self.get_error_class(
                error_message=f"Error transforming image edit response: {exc}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        candidates = response_json.get("candidates", [])
        data_list: List[ImageObject] = []

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

        model_response.data = cast(List[OpenAIImage], data_list)
        return model_response

    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """Map OpenAI size format to Gemini aspect ratio format"""
        aspect_ratio_map = {
            "1024x1024": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1280x896": "4:3",
            "896x1280": "3:4",
        }
        return aspect_ratio_map.get(size, "1:1")

    def _prepare_inline_image_parts(
        self, image: Union[FileTypes, List[FileTypes]]
    ) -> List[Dict[str, Any]]:
        images: List[FileTypes]
        if isinstance(image, list):
            images = image
        else:
            images = [image]

        inline_parts: List[Dict[str, Any]] = []
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
        raise ValueError("Unsupported image type for Vertex AI Gemini image edit.")
