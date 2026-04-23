from base_google_test import BaseGoogleGenAITest

class TestVertexAIGenerateContent(BaseGoogleGenAITest):
    """Test Vertex AI"""

    @property
    def model_config(self):
        return {
            "model": "vertex_ai/gemini-2.5-flash-lite",
        }