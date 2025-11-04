"""Azure AI OCR module."""
from .document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)
from .transformation import AzureAIOCRConfig

__all__ = ["AzureAIOCRConfig", "AzureDocumentIntelligenceOCRConfig"]

