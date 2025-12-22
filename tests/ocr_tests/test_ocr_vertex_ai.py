"""
Test OCR functionality with Vertex AI OCR APIs (Mistral and DeepSeek).

Note: Vertex AI OCR automatically converts URLs to base64 data URIs since
the Vertex AI endpoint doesn't have internet access.
"""
import os
import json
import tempfile
import pytest
from base_ocr_unit_tests import BaseOCRTest


def load_vertex_ai_credentials():
    """Load Vertex AI credentials for tests"""
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/vertex_key.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("VERTEX_AI_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("VERTEX_AI_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)


class TestVertexAIMistralOCR(BaseOCRTest):
    """
    Test class for Vertex AI Mistral OCR functionality.
    Inherits from BaseOCRTest and provides Vertex AI-specific configuration.
    
    Note: For Vertex AI, LiteLLM will automatically convert URLs to base64 data URIs before
    sending to the API, since Vertex AI OCR endpoint doesn't have internet access.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Vertex AI Mistral OCR.
        """
        load_vertex_ai_credentials()
        return {
            "model": "vertex_ai/mistral-ocr-2505",
            "vertex_location": "us-central1",
        }


class TestVertexAIDeepSeekOCR(BaseOCRTest):
    """
    Test class for Vertex AI DeepSeek OCR functionality.
    Inherits from BaseOCRTest and provides Vertex AI-specific configuration.
    
    Note: DeepSeek OCR uses the chat completion API format through the openapi endpoint.
    Note: DeepSeek OCR does not support PDF URLs - only image URLs and base64 data.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Vertex AI DeepSeek OCR.
        """
        load_vertex_ai_credentials()
        return {
            "model": "vertex_ai/deepseek-ocr-maas",
            "vertex_location": "us-central1",
        }

    # Skip PDF URL tests for DeepSeek OCR as it doesn't support PDF URLs
    @pytest.mark.skip(reason="DeepSeek OCR does not support PDF URLs")
    async def test_basic_ocr_with_url(self, sync_mode):
        """Skip this test for DeepSeek OCR - PDF URLs not supported"""
        pass

    @pytest.mark.skip(reason="DeepSeek OCR does not support PDF URLs")
    def test_ocr_response_structure(self):
        """Skip this test for DeepSeek OCR - PDF URLs not supported"""
        pass


def test_vertex_ai_ocr_routing():
    """
    Test that Vertex AI OCR routing correctly selects the right config based on model name.
    """
    from litellm.llms.vertex_ai.ocr.common_utils import get_vertex_ai_ocr_config
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig
    from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig
    
    # Test DeepSeek OCR routing
    deepseek_config = get_vertex_ai_ocr_config("vertex_ai/deepseek-ocr-maas")
    assert isinstance(deepseek_config, VertexAIDeepSeekOCRConfig), \
        "DeepSeek model should route to VertexAIDeepSeekOCRConfig"
    
    # Test Mistral OCR routing (should use default VertexAIOCRConfig)
    mistral_config = get_vertex_ai_ocr_config("vertex_ai/mistral-ocr-2505")
    assert isinstance(mistral_config, VertexAIOCRConfig), \
        "Mistral model should route to VertexAIOCRConfig"
    
    # Test other DeepSeek variants
    deepseek_variant = get_vertex_ai_ocr_config("vertex_ai/deepseek-ocr-maas")
    assert isinstance(deepseek_variant, VertexAIDeepSeekOCRConfig), \
        "DeepSeek variant should route to VertexAIDeepSeekOCRConfig"

