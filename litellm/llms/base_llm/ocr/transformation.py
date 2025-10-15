"""
Base OCR transformation configuration.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import FileTypes

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OCRPageDimensions(BaseModel):
    """Page dimensions from OCR response."""
    dpi: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None


class OCRPageImage(BaseModel):
    """Image extracted from OCR page."""
    image_base64: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    
    model_config = {"extra": "allow"}


class OCRPage(BaseModel):
    """Single page from OCR response."""
    index: int
    markdown: str
    images: Optional[List[OCRPageImage]] = None
    dimensions: Optional[OCRPageDimensions] = None
    
    model_config = {"extra": "allow"}


class OCRUsageInfo(BaseModel):
    """Usage information from OCR response."""
    pages_processed: Optional[int] = None
    doc_size_bytes: Optional[int] = None
    
    model_config = {"extra": "allow"}


class OCRResponse(BaseModel):
    """
    Standard OCR response format.
    Standardized to Mistral OCR format - other providers should transform to this format.
    """
    pages: List[OCRPage]
    model: str
    document_annotation: Optional[Any] = None
    usage_info: Optional[OCRUsageInfo] = None
    object: str = "ocr"
    
    model_config = {"extra": "allow"}


class OCRRequestData(BaseModel):
    """OCR request data structure."""
    data: Optional[Union[Dict, bytes]] = None
    files: Optional[Dict[str, Any]] = None


class BaseOCRConfig:
    """
    Base configuration for OCR transformations.
    Handles provider-agnostic OCR operations.
    """

    def __init__(self) -> None:
        pass

    def get_supported_ocr_params(self, model: str) -> list:
        """
        Get supported OCR parameters for this provider.
        Override this method in provider-specific implementations.
        """
        return []

    def map_ocr_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
    ) -> dict:
        """Map OCR parameters to provider-specific parameters."""
        return optional_params

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        Override in provider-specific implementations.
        """
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: dict,
        **kwargs,
    ) -> str:
        """
        Get complete URL for OCR endpoint.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("get_complete_url must be implemented by provider")

    def transform_ocr_request(
        self,
        model: str,
        image: FileTypes,
        optional_params: dict,
        headers: dict,
        **kwargs,
    ) -> OCRRequestData:
        """
        Transform OCR request to provider-specific format.
        Override in provider-specific implementations.
        
        Args:
            model: Model name
            image: Image file to process (can be file path, bytes, file object, etc.)
            optional_params: Optional parameters for the request
            headers: Request headers
            
        Returns:
            OCRRequestData with data and files fields
        """
        raise NotImplementedError("transform_ocr_request must be implemented by provider")

    def transform_ocr_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> OCRResponse:
        """
        Transform provider-specific OCR response to standard format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError("transform_ocr_response must be implemented by provider")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

