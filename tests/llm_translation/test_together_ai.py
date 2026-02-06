"""
Test TogetherAI LLM
"""

from base_llm_unit_tests import BaseLLMChatTest
import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
import pytest


class TestTogetherAI(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    @pytest.mark.parametrize(
        "model, expected_bool",
        [
            ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", True),
            ("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", False),
        ],
    )
    def test_get_supported_response_format_together_ai(
        self, model: str, expected_bool: bool
    ) -> None:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        optional_params = litellm.get_supported_openai_params(
            model, custom_llm_provider="together_ai"
        )
        # Mapped provider
        assert isinstance(optional_params, list)

        if expected_bool:
            assert "response_format" in optional_params
            assert "tools" in optional_params
        else:
            assert "response_format" not in optional_params
            assert "tools" not in optional_params
