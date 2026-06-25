"""OCR response models shared by the Python public API and proxy."""

from typing import Any, Dict, List

from pydantic import PrivateAttr

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class OCRPageDimensions(LiteLLMPydanticObjectBase):
    """Page dimensions from OCR response."""

    dpi: int | None = None
    height: int | None = None
    width: int | None = None


class OCRPageImage(LiteLLMPydanticObjectBase):
    """Image extracted from OCR page."""

    image_base64: str | None = None
    bbox: Dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class OCRPage(LiteLLMPydanticObjectBase):
    """Single page from OCR response."""

    index: int
    markdown: str
    images: List[OCRPageImage] | None = None
    dimensions: OCRPageDimensions | None = None

    model_config = {"extra": "allow"}


class OCRUsageInfo(LiteLLMPydanticObjectBase):
    """Usage information from OCR response."""

    pages_processed: int | None = None
    credits: float | None = None
    doc_size_bytes: int | None = None

    model_config = {"extra": "allow"}


class OCRResponse(LiteLLMPydanticObjectBase):
    """
    Standard OCR response format.
    Standardized to Mistral OCR format - other providers should transform to this format.
    """

    pages: List[OCRPage]
    model: str
    document_annotation: Any | None = None
    usage_info: OCRUsageInfo | None = None
    object: str = "ocr"

    model_config = {"extra": "allow"}

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)
