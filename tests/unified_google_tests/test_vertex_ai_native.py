from base_google_genai_proxy_sdk_test import BaseGoogleGenAIProxySDKTest
from base_google_test import BaseGoogleGenAITest


class TestVertexAIGenerateContent(BaseGoogleGenAITest, BaseGoogleGenAIProxySDKTest):
    """Test Vertex AI"""

    @property
    def model_config(self):
        return {
            "model": "vertex_ai/gemini-3.5-flash-lite",
            "vertex_location": "global",
        }

    @property
    def proxy_model_name(self) -> str:
        return "vertex-gemini-3.5-flash-lite"
