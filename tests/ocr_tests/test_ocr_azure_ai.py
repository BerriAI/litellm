"""
Test OCR functionality with Azure AI API.
"""
import os
from tests.ocr_tests.test_ocr_mistral import TestMistralOCR

class TestAzureAIOCR(TestMistralOCR):
    """
    Test class for Azure AI OCR functionality.
    Inherits from TestMistralOCR and overrides provider-specific methods.
    
    Note: Azure AI only supports base64 data URIs, not regular URLs.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Azure AI.
        """
        return {
            "model": "azure_ai/mistral-document-ai-2505",
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
        }
