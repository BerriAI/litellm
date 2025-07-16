from base_google_test import BaseGoogleGenAITest

class TestGoogleGenAIStudio(BaseGoogleGenAITest):
    """Test Google GenAI Studio"""

    @property
    def model_config(self):
        return {
            "model": "gemini/gemini-1.5-flash",
        }