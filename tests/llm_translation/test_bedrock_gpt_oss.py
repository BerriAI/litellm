from base_llm_unit_tests import BaseLLMChatTest
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


class TestBedrockGPTOSS(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
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

    async def test_completion_cost(self):
        """
        Bedrock GPT-OSS models are flaky and occasionally report 0 token counts in api response
        """
        pass

    @pytest.mark.parametrize("model", [
        "bedrock/openai.gpt-oss-20b-1:0",
        "bedrock/openai.gpt-oss-120b-1:0",
    ])
    def test_reasoning_effort_transformation_gpt_oss(self, model):
        """Test that reasoning_effort is handled correctly for GPT-OSS models."""
        config = AmazonConverseConfig()

        # Test GPT-OSS model - should keep reasoning_effort as-is
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )
        
        # GPT-OSS should have reasoning_effort in result, not thinking
        assert "reasoning_effort" in result
        assert result["reasoning_effort"] == "low"
        assert "thinking" not in result
