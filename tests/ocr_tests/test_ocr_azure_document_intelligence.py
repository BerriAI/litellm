"""
Test OCR functionality with Azure Document Intelligence API.

Azure Document Intelligence provides advanced document analysis capabilities
using the v4.0 (2024-11-30) API.
"""
import os

import pytest

from base_ocr_unit_tests import BaseOCRTest


class TestAzureDocumentIntelligenceOCR(BaseOCRTest):
    """
    Test class for Azure Document Intelligence OCR functionality.
    
    Inherits from BaseOCRTest and provides Azure Document Intelligence-specific configuration.
    
    Tests the azure_ai/doc-intelligence/<model> provider route.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Azure Document Intelligence.
        
        Uses prebuilt-layout model which is closest to Mistral OCR format.
        """
        # Check for required environment variables
        api_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if not api_key or not endpoint:
            pytest.skip(
                "AZURE_DOCUMENT_INTELLIGENCE_API_KEY and AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "environment variables are required for Azure Document Intelligence tests"
            )

        return {
            "model": "azure_ai/doc-intelligence/prebuilt-layout",
            "api_key": api_key,
            "api_base": endpoint,
        }

