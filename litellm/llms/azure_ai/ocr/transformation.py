"""
Azure AI OCR transformation implementation.
"""
from typing import Dict, Optional

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_url_to_base64,
    convert_url_to_base64,
)
from litellm.llms.base_llm.ocr.transformation import DocumentType, OCRRequestData
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.secret_managers.main import get_secret_str


class AzureAIOCRConfig(MistralOCRConfig):
    """
    Azure AI OCR transformation configuration.
    
    Azure AI uses Mistral's OCR API but with a different endpoint format.
    Inherits transformation logic from MistralOCRConfig since they use the same format.
    
    Reference: Azure AI Foundry OCR documentation
    
    Important: Azure AI only supports base64 data URIs (data:image/..., data:application/pdf;base64,...).
    Regular URLs are not supported.
    """

    def __init__(self) -> None:
        super().__init__()

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers for Azure AI OCR.
        
        Azure AI uses Bearer token authentication with AZURE_AI_API_KEY.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = get_secret_str("AZURE_AI_API_KEY")

        if api_key is None:
            raise ValueError(
                "Missing Azure AI API Key - A call is being made to Azure AI but no key is set either in the environment variables or via params"
            )

        # Validate API base is provided
        if api_base is None:
            api_base = get_secret_str("AZURE_AI_API_BASE")
        
        if api_base is None:
            raise ValueError(
                "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Azure AI OCR endpoint.
        
        Azure AI endpoint format: https://<api_base>/providers/mistral/azure/ocr
        
        Args:
            api_base: Azure AI API base URL
            model: Model name (not used in URL construction)
            optional_params: Optional parameters
            
        Returns: Complete URL for Azure AI OCR endpoint
        """
        if api_base is None:
            raise ValueError(
                "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter"
            )

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")
        
        # Azure AI OCR endpoint format
        return f"{api_base}/providers/mistral/azure/ocr"

    def _convert_url_to_data_uri_sync(self, url: str) -> str:
        """
        Synchronously convert a URL to a base64 data URI.
        
        Azure AI OCR doesn't have internet access, so we need to fetch URLs
        and convert them to base64 data URIs.
        
        Args:
            url: The URL to convert
            
        Returns:
            Base64 data URI string
        """
        verbose_logger.debug(f"Azure AI OCR: Converting URL to base64 data URI (sync): {url}")
        
        # Fetch and convert to base64 data URI
        # convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri = convert_url_to_base64(url=url)
        
        verbose_logger.debug(f"Azure AI OCR: Converted URL to data URI (length: {len(data_uri)})")
        
        return data_uri

    async def _convert_url_to_data_uri_async(self, url: str) -> str:
        """
        Asynchronously convert a URL to a base64 data URI.
        
        Azure AI OCR doesn't have internet access, so we need to fetch URLs
        and convert them to base64 data URIs.
        
        Args:
            url: The URL to convert
            
        Returns:
            Base64 data URI string
        """
        verbose_logger.debug(f"Azure AI OCR: Converting URL to base64 data URI (async): {url}")
        
        # Fetch and convert to base64 data URI asynchronously
        # async_convert_url_to_base64 already returns a full data URI like "data:image/jpeg;base64,..."
        data_uri = await async_convert_url_to_base64(url=url)
        
        verbose_logger.debug(f"Azure AI OCR: Converted URL to data URI (length: {len(data_uri)})")
        
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
        Transform OCR request for Azure AI, converting URLs to base64 data URIs (sync).
        
        Azure AI OCR doesn't have internet access, so we automatically fetch
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
        verbose_logger.debug(f"Azure AI OCR transform_ocr_request (sync) - model: {model}")
        
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
                    "Azure AI OCR: Converting document URL to base64 data URI (sync)"
                )
                data_uri = self._convert_url_to_data_uri_sync(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug(
                    "Azure AI OCR: Converting image URL to base64 data URI (sync)"
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
        Transform OCR request for Azure AI, converting URLs to base64 data URIs (async).
        
        Azure AI OCR doesn't have internet access, so we automatically fetch
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
        verbose_logger.debug(f"Azure AI OCR async_transform_ocr_request - model: {model}")
        
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
                    "Azure AI OCR: Converting document URL to base64 data URI (async)"
                )
                data_uri = await self._convert_url_to_data_uri_async(url=document_url)
                transformed_document["document_url"] = data_uri
        elif doc_type == "image_url":
            image_url = document.get("image_url", "")
            # If it's not already a data URI, convert it
            if image_url and not image_url.startswith("data:"):
                verbose_logger.debug(
                    "Azure AI OCR: Converting image URL to base64 data URI (async)"
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

