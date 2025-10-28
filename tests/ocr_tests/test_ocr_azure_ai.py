"""
Test OCR functionality with Azure AI API.

Note: Azure AI OCR automatically converts URLs to base64 data URIs since
the Azure AI endpoint doesn't have internet access.
"""
import os
from base_ocr_unit_tests import BaseOCRTest

class TestAzureAIOCR(BaseOCRTest):
    """
    Test class for Azure AI OCR functionality.
    Inherits from BaseOCRTest and provides Azure AI-specific configuration.
    
    Note: For Azure AI, LiteLLM will automatically convert URLs to base64 data URIs before
    sending to the API, since Azure AI OCR endpoint doesn't have internet access.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Azure AI.
        """
        return {
            "model": "azure_ai/mistral-document-ai-2505",
            "api_key": os.getenv("AZURE_AI_API_KEY_MISTRAL"),
            "api_base": os.getenv("AZURE_AI_API_BASE_MISTRAL"),
        }
