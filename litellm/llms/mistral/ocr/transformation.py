"""
Mistral OCR transformation implementation.
"""
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.ocr.transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRRequestData,
    OCRResponse,
)
from litellm.secret_managers.main import get_secret_str


class MistralOCRConfig(BaseOCRConfig):
    """
    Mistral OCR transformation configuration.
    
    Reference: https://docs.mistral.ai/api/#tag/ocr
    """

    def __init__(self) -> None:
        super().__init__()

    def get_supported_ocr_params(self, model: str) -> list:
        """
        Get supported OCR parameters for Mistral OCR.
        
        Mistral OCR supports:
        - pages: List of page numbers to process
        - include_image_base64: Whether to include base64 encoded images
        - image_limit: Maximum number of images to return
        - image_min_size: Minimum size of images to include
        - bbox_annotation_format: Format for bounding box annotations
        - document_annotation_format: Format for document annotations
        """
        return [
            "pages",
            "include_image_base64",
            "image_limit",
            "image_min_size",
            "bbox_annotation_format",
            "document_annotation_format",
        ]
    
    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        """
        Map OCR parameters to Mistral-specific format.
        
        Mistral accepts these parameters directly, so no transformation needed.
        Just filter out unsupported params.
        """
        supported_params = self.get_supported_ocr_params(model=model)
        
        # Only include params that are in the supported list
        mapped_params = {}
        for param, value in non_default_params.items():
            if param in supported_params:
                mapped_params[param] = value
        
        return mapped_params

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
        Validate environment and return headers for Mistral OCR.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = (
                get_secret_str("MISTRAL_API_KEY")
            )

        if api_key is None:
            raise ValueError(
                "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            **headers,
        }
        
        # Don't set Content-Type for multipart/form-data - httpx will handle it

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
        Get complete URL for Mistral OCR endpoint.
        
        Returns: https://api.mistral.ai/v1/ocr
        """
        if api_base is None:
            api_base = "https://api.mistral.ai/v1"

        # Ensure no trailing slash
        api_base = api_base.rstrip("/")
        
        # Remove /v1 if it's already in the base to avoid duplication
        if api_base.endswith("/v1"):
            return f"{api_base}/ocr"

        return f"{api_base}/v1/ocr"


    def transform_ocr_request(
        self,
        model: str,
        document: DocumentType,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request to Mistral-specific format.
        
        Mistral OCR API accepts:
        {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "<https-url or data-uri>"
            },
            "pages": [0],  # optional
            "include_image_base64": false,  # optional
            ...
        }
        
        Args:
            model: Model name (e.g., "mistral-ocr-latest")
            document: Document dict from user (Mistral format) - already validated in main.py
            optional_params: Already mapped optional parameters
            headers: Request headers
            
        Returns:
            OCRRequestData with JSON data
        """
        verbose_logger.debug(f"Mistral OCR transform_ocr_request - model: {model}")
        
        # Document parameter is the Mistral-format dict from the user
        # Just pass it through as-is to the Mistral API
        if not isinstance(document, dict):
            raise ValueError(f"Expected document dict, got {type(document)}")
        
        # Build request data - use document dict directly
        data = {
            "model": model,
            "document": document,  # Pass through the Mistral-format document dict
        }
        
        # Add all optional parameters from the already-mapped optional_params
        data.update(optional_params)
        
        # No multipart files - using JSON
        return OCRRequestData(data=data, files=None)

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        **kwargs,
    ) -> OCRResponse:
        """
        Return Mistral OCR response in native format.
        
        Mistral OCR is the standard format for LiteLLM OCR responses.
        No transformation needed - return native response.
        
        Mistral OCR returns:
        {
            "pages": [
                {
                    "index": 0,
                    "markdown": "extracted text content",
                    "images": [...],
                    "dimensions": {...}
                },
                ...
            ],
            "model": "mistral-ocr-2505-completion",
            "document_annotation": null,
            "usage_info": {...}
        }
        """
        try:
            response_json = raw_response.json()
            
            verbose_logger.debug(f"Mistral OCR response keys: {response_json.keys()}")
            
            # Return native Mistral format - no transformation
            return OCRResponse(
                pages=response_json.get("pages", []),
                model=response_json.get("model", model),
                document_annotation=response_json.get("document_annotation"),
                usage_info=response_json.get("usage_info"),
                object="ocr",
            )
        except Exception as e:
            verbose_logger.error(f"Error parsing Mistral OCR response: {e}")
            raise e

