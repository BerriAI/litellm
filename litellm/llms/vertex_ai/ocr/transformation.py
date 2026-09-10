"""
Vertex AI Mistral OCR transformation implementation.
"""

from collections.abc import Callable, Mapping
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.base_llm.ocr.transformation import DocumentType, OCRRequestData, RustOCRConfig
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

VERTEX_AI_OCR_API_KEY_ENV_VAR: Final = "VERTEX_AI_API_KEY"


def vertex_rust_ocr_config(kwargs: Mapping[str, object], resolve_secret: Callable[[str], str | None]) -> RustOCRConfig:
    project: Final = (
        kwargs.get("vertex_project")
        or kwargs.get("vertex_ai_project")
        or litellm.vertex_project
        or resolve_secret("VERTEXAI_PROJECT")
    )
    location: Final = (
        kwargs.get("vertex_location")
        or kwargs.get("vertex_ai_location")
        or litellm.vertex_location
        or resolve_secret("VERTEXAI_LOCATION")
        or resolve_secret("VERTEX_LOCATION")
    )
    return RustOCRConfig(
        config_fields=frozenset(
            {
                "vertex_credentials",
                "vertex_ai_credentials",
                "vertex_project",
                "vertex_ai_project",
                "vertex_location",
                "vertex_ai_location",
            }
        ),
        extra_params=tuple(
            (key, value)
            for key, value in (("vertex_project", project), ("vertex_location", location))
            if value is not None
        ),
    )


class VertexAIOCRConfig(MistralOCRConfig):
    """
    Vertex AI Mistral OCR transformation configuration.

    Vertex AI uses Mistral's OCR API format through the Mistral publisher endpoint.
    Inherits transformation logic from MistralOCRConfig since they use the same format.

    Reference: Vertex AI Mistral OCR documentation

    Important: Vertex AI OCR only supports base64 data URIs (data:image/..., data:application/pdf;base64,...).
    Regular URLs are not supported.
    """

    def get_rust_ocr_config(
        self, kwargs: Mapping[str, object], resolve_secret: Callable[[str], str | None]
    ) -> RustOCRConfig | None:
        return vertex_rust_ocr_config(kwargs, resolve_secret)

    def __init__(self) -> None:
        super().__init__()
        self.vertex_base = VertexBase()

    def get_api_key_env_var(self) -> str | None:
        return VERTEX_AI_OCR_API_KEY_ENV_VAR

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
        Get complete URL for Vertex AI OCR endpoint.

        Vertex AI endpoint format:
        https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/mistralai/ocr

        Args:
            api_base: Vertex AI API base URL (optional)
            model: Model name (not used in URL construction)
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
            api_base = get_vertex_base_url(vertex_location)

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")

        # Vertex AI OCR endpoint format for Mistral publisher
        # Format: https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/mistralai/models/{model}:rawPredict
        return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/mistralai/models/{model}:rawPredict"

    def _convert_url_to_data_uri_sync(self, url: str) -> str:
        """
        Synchronously convert a URL to a base64 data URI.

        Vertex AI OCR doesn't have internet access, so we need to fetch URLs
        and convert them to base64 data URIs.

        Args:
            url: The URL to convert

        Returns:
            Base64 data URI string
        """
        verbose_logger.debug("Vertex AI OCR: Converting URL to base64 data URI (sync)")

        # Fetch and convert to base64 data URI
        # convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri: Final = convert_url_to_base64(url=url)

        verbose_logger.debug("Vertex AI OCR: Converted URL to data URI (length: %s)", len(data_uri))

        return data_uri

    async def _convert_url_to_data_uri_async(self, url: str) -> str:
        """
        Asynchronously convert a URL to a base64 data URI.

        Vertex AI OCR doesn't have internet access, so we need to fetch URLs
        and convert them to base64 data URIs.

        Args:
            url: The URL to convert

        Returns:
            Base64 data URI string
        """
        verbose_logger.debug("Vertex AI OCR: Converting URL to base64 data URI (async)")

        # Fetch and convert to base64 data URI asynchronously
        # async_convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri: Final = await async_convert_url_to_base64(url=url)

        verbose_logger.debug("Vertex AI OCR: Converted URL to data URI (length: %s)", len(data_uri))

        return data_uri

    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request for Vertex AI, converting URLs to base64 data URIs (sync).

        Vertex AI OCR doesn't have internet access, so we automatically fetch
        any URLs and convert them to base64 data URIs synchronously.

        Args:
            model: Model name
            document: Document dict from user
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments

        Returns:
            OCRRequestData with JSON data
        """
        verbose_logger.debug("Vertex AI OCR transform_ocr_request (sync) called")

        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")

        # Check if we need to convert URL to base64
        doc_type: Final = document.get("type")
        transformed_document: Final = document.copy()

        if doc_type == "document_url":
            document_url: Final = document.get("document_url", "")
            # If it's not already a data URI, convert it
            if document_url and not document_url.startswith("data:"):
                verbose_logger.debug("Vertex AI OCR: Converting document URL to base64 data URI (sync)")
                data_uri = self._convert_url_to_data_uri_sync(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url: Final = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug("Vertex AI OCR: Converting image URL to base64 data URI (sync)")
                data_uri = self._convert_url_to_data_uri_sync(url=image_url)
                transformed_document["image_url"] = data_uri

        # Call parent's transform to build the request
        return super().transform_ocr_request(
            model=model,
            document=transformed_document,
            optional_params=optional_params,
            headers=headers,
            **kwargs,
        )

    async def async_transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request for Vertex AI, converting URLs to base64 data URIs (async).

        Vertex AI OCR doesn't have internet access, so we automatically fetch
        any URLs and convert them to base64 data URIs asynchronously.

        Args:
            model: Model name
            document: Document dict from user
            optional_params: Already mapped optional parameters
            headers: Request headers
            **kwargs: Additional arguments

        Returns:
            OCRRequestData with JSON data
        """
        verbose_logger.debug("Vertex AI OCR async_transform_ocr_request - model: %s", model)

        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")

        # Check if we need to convert URL to base64
        doc_type: Final = document.get("type")
        transformed_document: Final = document.copy()

        if doc_type == "document_url":
            document_url: Final = document.get("document_url", "")
            # If it's not already a data URI, convert it
            if document_url and not document_url.startswith("data:"):
                verbose_logger.debug("Vertex AI OCR: Converting document URL to base64 data URI (async)")
                data_uri = await self._convert_url_to_data_uri_async(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url: Final = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug("Vertex AI OCR: Converting image URL to base64 data URI (async)")
                data_uri = await self._convert_url_to_data_uri_async(url=image_url)
                transformed_document["image_url"] = data_uri

        # Call parent's transform to build the request
        return super().transform_ocr_request(
            model=model,
            document=transformed_document,
            optional_params=optional_params,
            headers=headers,
            **kwargs,
        )
