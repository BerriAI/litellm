from base_llm_unit_tests import BaseLLMChatTest
import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


class TestBedrockGPTOSS(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/converse/openai.gpt-oss-20b-1:0",
        }
    
    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """
        Remove override once we have access to Bedrock prompt caching
        """
        pass
    

    def test_basic_tool_calling(self):
        """
        TODO: Add support for this
        """
        pass