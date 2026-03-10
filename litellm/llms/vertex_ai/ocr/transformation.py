"""
Vertex AI Mistral OCR transformation implementation.
"""
from typing import Dict, Optional

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.base_llm.ocr.transformation import DocumentType, OCRRequestData
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class VertexAIOCRConfig(MistralOCRConfig):
    """
    Vertex AI Mistral OCR transformation configuration.
    
    Vertex AI uses Mistral's OCR API format through the Mistral publisher endpoint.
    Inherits transformation logic from MistralOCRConfig since they use the same format.
    
    Reference: Vertex AI Mistral OCR documentation
    
    Important: Vertex AI OCR only supports base64 data URIs (data:image/..., data:application/pdf;base64,...).
    Regular URLs are not supported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vertex_base = VertexBase()

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers for Vertex AI OCR.
        
        Vertex AI uses Bearer token authentication with access token from credentials.
        """
        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        litellm_params = litellm_params or {}
        
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
        vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(litellm_params=litellm_params)
        
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
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: Optional[dict] = None,
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
        
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params=litellm_params)
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
        verbose_logger.debug(f"Vertex AI OCR: Converting URL to base64 data URI (sync): {url}")
        
        # Fetch and convert to base64 data URI
        # convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri = convert_url_to_base64(url=url)
        
        verbose_logger.debug(f"Vertex AI OCR: Converted URL to data URI (length: {len(data_uri)})")
        
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
        verbose_logger.debug(f"Vertex AI OCR: Converting URL to base64 data URI (async): {url}")
        
        # Fetch and convert to base64 data URI asynchronously
        # async_convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri = await async_convert_url_to_base64(url=url)
        
        verbose_logger.debug(f"Vertex AI OCR: Converted URL to data URI (length: {len(data_uri)})")
        
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
        doc_type = document.get("type")
        transformed_document = document.copy()
        
        if doc_type == "document_url":
            document_url = document.get("document_url", "")
            # If it's not already a data URI, convert it
            if document_url and not document_url.startswith("data:"):
                verbose_logger.debug(
                    "Vertex AI OCR: Converting document URL to base64 data URI (sync)"
                )
                data_uri = self._convert_url_to_data_uri_sync(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug(
                    "Vertex AI OCR: Converting image URL to base64 data URI (sync)"
                )
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
        verbose_logger.debug(f"Vertex AI OCR async_transform_ocr_request - model: {model}")
        
        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")
        
        # Check if we need to convert URL to base64
        doc_type = document.get("type")
        transformed_document = document.copy()
        
        if doc_type == "document_url":
            document_url = document.get("document_url", "")
            # If it's not already a data URI, convert it
            if document_url and not document_url.startswith("data:"):
                verbose_logger.debug(
                    "Vertex AI OCR: Converting document URL to base64 data URI (async)"
                )
                data_uri = await self._convert_url_to_data_uri_async(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug(
                    "Vertex AI OCR: Converting image URL to base64 data URI (async)"
                )
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

