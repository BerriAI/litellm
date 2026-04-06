"""Base OCR transformation module."""
from .transformation import (
    BaseOCRConfig,
    DocumentType,
    OCRPage,
    OCRPageDimensions,
    OCRPageImage,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)

__all__ = [
    "BaseOCRConfig",
    "DocumentType",
    "OCRResponse",
    "OCRPage",
    "OCRPageDimensions",
    "OCRPageImage",
    "OCRUsageInfo",
    "OCRRequestData",
]
