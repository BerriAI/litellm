from base_google_test import BaseGoogleGenAITest

class TestOpenAIFormatGoogleGenAITest(BaseGoogleGenAITest):
    """Test OpenAI Format Google GenAI"""

    @property
    def model_config(self):
        return {
            "model": "openai/gpt-4o-mini",
        }