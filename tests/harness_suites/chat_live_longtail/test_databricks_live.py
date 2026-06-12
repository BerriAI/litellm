import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest


@pytest.mark.skip(reason="Databricks rate limit errors")
class TestDatabricksCompletion(BaseLLMChatTest, BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "databricks/databricks-claude-3-7-sonnet"}

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "databricks/databricks-claude-3-7-sonnet",
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        }

    def test_pdf_handling(self, pdf_messages):
        pytest.skip("Databricks does not support PDF handling")
