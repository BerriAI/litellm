"""Base OCR transformation module."""
from .transformation import (
    BaseOCRConfig,
    OCRPage,
    OCRPageDimensions,
    OCRPageImage,
    OCRRequestData,
    OCRResponse,
    OCRUsageInfo,
)

__all__ = [
    "BaseOCRConfig",
    "OCRResponse",
    "OCRPage",
    "OCRPageDimensions",
    "OCRPageImage",
    "OCRUsageInfo",
    "OCRRequestData",
]
