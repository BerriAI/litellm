"""Azure AI OCR module."""
from .common_utils import get_azure_ai_ocr_config
from .document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)
from .transformation import AzureAIOCRConfig

__all__ = [
    "AzureAIOCRConfig",
    "AzureDocumentIntelligenceOCRConfig",
    "get_azure_ai_ocr_config",
]

