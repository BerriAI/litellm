"""
Test OCR functionality with Vertex AI Mistral OCR API.

Note: Vertex AI OCR automatically converts URLs to base64 data URIs since
the Vertex AI endpoint doesn't have internet access.
"""
import os
import json
import tempfile
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

class TestVertexAIOCR(BaseOCRTest):
    """
    Test class for Vertex AI Mistral OCR functionality.
    Inherits from BaseOCRTest and provides Vertex AI-specific configuration.
    
    Note: For Vertex AI, LiteLLM will automatically convert URLs to base64 data URIs before
    sending to the API, since Vertex AI OCR endpoint doesn't have internet access.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Vertex AI.
        """
        load_vertex_ai_credentials()
        return {
            "model": "vertex_ai/mistral-ocr-2505",
            "vertex_location": "us-central1",
        }

