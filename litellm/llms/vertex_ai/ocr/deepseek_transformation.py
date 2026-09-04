"""
Vertex AI DeepSeek OCR transformation implementation.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRPage,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

VERTEX_AI_DEEPSEEK_OCR_API_KEY_ENV_VAR: Final = "VERTEX_AI_API_KEY"

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class _DeepSeekOCRMessage(BaseModel):
    content: str | Mapping[str, object]


class _DeepSeekOCRChoice(BaseModel):
    message: _DeepSeekOCRMessage


class _DeepSeekOCRProviderResponse(BaseModel):
    choices: tuple[_DeepSeekOCRChoice, ...]
    usage: OCRUsageInfo | None = None


class _DeepSeekOCRContent(BaseModel):
    pages: tuple[OCRPage, ...] | None = None
    model: str | None = None
    document_annotation: object | None = None


def _parse_ocr_content(content: str | Mapping[str, object]) -> _DeepSeekOCRContent | None:
    if isinstance(content, Mapping):
        return _DeepSeekOCRContent.model_validate(content)
    if not content.strip().startswith("{"):
        return None
    try:
        return _DeepSeekOCRContent.model_validate_json(content)
    except ValidationError:
        return None


class VertexAIDeepSeekOCRConfig(BaseOCRConfig):
    """
    Vertex AI DeepSeek OCR transformation configuration.

    This transformation converts standard LiteLLM OCR requests to the
    Vertex AI DeepSeek OCR OpenAPI endpoint shape and normalizes the response.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vertex_base = VertexBase()

    def get_api_key_env_var(self) -> str | None:
        return VERTEX_AI_DEEPSEEK_OCR_API_KEY_ENV_VAR

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Validate environment and return headers for Vertex AI OCR.

        Vertex AI uses Bearer token authentication with access token from credentials.
        """
        if api_key is not None:
            return {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **headers,
            }

        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        litellm_params = litellm_params or {}

        vertex_project: Final = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
        vertex_credentials: Final = VertexBase.safe_get_vertex_ai_credentials(litellm_params=litellm_params)

        # Get access token from Vertex credentials
        access_token, project_id = self.vertex_base.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Vertex AI DeepSeek OCR endpoint.

        Args:
            api_base: Vertex AI API base URL (optional)
            model: Model name (e.g., "deepseek-ai/deepseek-ocr-maas")
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters containing vertex_project, vertex_location

        Returns: Complete URL for Vertex AI OCR endpoint
        """
        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        litellm_params = litellm_params or {}

        vertex_project: Final = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
        vertex_location = VertexBase.safe_get_vertex_ai_location(litellm_params=litellm_params)

        if vertex_project is None:
            raise ValueError(
                "Missing vertex_project - Set VERTEXAI_PROJECT environment variable or pass vertex_project parameter"
            )

        if vertex_location is None:
            vertex_location = "us-central1"

        # Get API base URL
        if api_base is None:
            api_base = "https://aiplatform.googleapis.com"

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")

        return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi/chat/completions"

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request for Vertex AI DeepSeek OCR.

        Converts OCR document format to the Vertex AI DeepSeek OCR payload:
        - Input: {"type": "image_url", "image_url": "gs://..."}
        - Output: {"model": "deepseek-ai/deepseek-ocr-maas", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "gs://..."}]}]}

        Args:
            model: Model name (e.g., "deepseek-ai/deepseek-ocr-maas")
            document: Document dict from user (Mistral OCR format)
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments

        Returns:
            OCRRequestData with JSON data for the DeepSeek OCR endpoint
        """
        verbose_logger.debug("Vertex AI DeepSeek OCR transform_ocr_request (sync) called")

        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")

        # Extract document type and URL
        doc_type: Final = document.get("type")
        image_url = None
        document_url = None

        if doc_type == "image_url":
            image_url = document.get("image_url", "")
        elif doc_type == "document_url":
            document_url = document.get("document_url", "")
        else:
            raise ValueError(f"Unsupported document type: {doc_type}. Expected 'image_url' or 'document_url'")

        # Build DeepSeek OCR message content
        content_item = {}
        if image_url:
            content_item = {"type": "image_url", "image_url": image_url}
        elif document_url:
            # For document URLs, we use image_url type as well (Vertex AI supports both)
            content_item = {"type": "image_url", "image_url": document_url}

        # Build DeepSeek OCR request
        provider_model: Final = model if model.startswith("deepseek-ai/") else f"deepseek-ai/{model}"
        data: Final = {
            "model": provider_model,
            "messages": [{"role": "user", "content": [content_item]}],
        }

        # Add optional parameters (stream, temperature, etc.)
        deepseek_ocr_params: Final = {}
        for key, value in optional_params.items():
            if key in ["stream", "temperature", "max_tokens", "top_p", "n", "stop"]:
                deepseek_ocr_params[key] = value

        data.update(deepseek_ocr_params)

        verbose_logger.debug("Vertex AI DeepSeek OCR: Transformed request")

        return OCRRequestData(data=data, files=None)

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request for Vertex AI DeepSeek OCR (async).

        Same as sync version - no async-specific logic needed.

        Args:
            model: Model name
            document: Document dict from user
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments

        Returns:
            OCRRequestData with JSON data for the DeepSeek OCR endpoint
        """
        return self.transform_ocr_request(
            model=model,
            document=document,
            optional_params=optional_params,
            headers=headers,
            **kwargs,
        )

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> OCRResponse:
        """
        Transform Vertex AI DeepSeek OCR response to OCR format.

        Vertex AI DeepSeek OCR returns an OpenAPI response:
        {
            "id": "...",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<OCR result as JSON string or markdown>"
                }
            }],
            "usage": {...}
        }

        We need to extract the content and convert it to OCRResponse format.

        Args:
            model: Model name
            raw_response: Raw HTTP response from Vertex AI
            logging_obj: Logging object
            **kwargs: Additional arguments

        Returns:
            OCRResponse in standard format
        """
        verbose_logger.debug("Vertex AI DeepSeek OCR transform_ocr_response called")
        verbose_logger.debug("Raw response: %s", raw_response.text)

        try:
            provider_response: Final = _DeepSeekOCRProviderResponse.model_validate(raw_response.json())
            choices: Final = provider_response.choices
            if not choices:
                raise ValueError("No choices in DeepSeek OCR response")

            content: Final = choices[0].message.content
            if not content:
                raise ValueError("No content in DeepSeek OCR response")

            ocr_content: Final = _parse_ocr_content(content)
            fallback_markdown: Final = content if isinstance(content, str) else json.dumps(content)
            pages: Final = (
                list(ocr_content.pages)
                if ocr_content is not None and ocr_content.pages
                else [OCRPage(index=0, markdown=fallback_markdown)]
            )
            response_model: Final = ocr_content.model if ocr_content is not None and ocr_content.model else model
            document_annotation: Final = ocr_content.document_annotation if ocr_content is not None else None

            return OCRResponse(
                pages=pages,
                model=response_model,
                document_annotation=document_annotation,
                usage_info=provider_response.usage,
                object="ocr",
            )

        except Exception as e:
            verbose_logger.error("Error parsing Vertex AI DeepSeek OCR response: %s", e)
            raise

    async def async_transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> OCRResponse:
        """
        Async transform Vertex AI DeepSeek OCR response to OCR format.

        Same as sync version - no async-specific logic needed.

        Args:
            model: Model name
            raw_response: Raw HTTP response
            logging_obj: Logging object
            **kwargs: Additional arguments

        Returns:
            OCRResponse in standard format
        """
        return self.transform_ocr_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
            **kwargs,
        )
