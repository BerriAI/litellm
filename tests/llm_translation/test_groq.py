from base_llm_unit_tests import BaseLLMChatTest
from base_audio_transcription_unit_tests import BaseLLMAudioTranscriptionTest
import litellm


class TestGroq(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "groq/llama-3.1-70b-versatile",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

class TestGroqAudioTranscription(BaseLLMAudioTranscriptionTest):
    def get_base_audio_transcription_call_args(self) -> dict:
        return {
            "model": "groq/whisper-large-v3",
            "api_base": "https://api.groq.com/openai/v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.GROQ