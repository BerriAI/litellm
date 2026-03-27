import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cohere.chat.transformation import CohereChatConfig


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}


class TestCohereV2StripFields:
    """Regression tests for #24031: Cohere v2 API rejects ``index`` on
    tool_calls and ``name`` on tool-role messages."""

    def test_strip_index_from_tool_calls(self):
        from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            }
        ]
        cleaned = CohereV2ChatConfig._strip_cohere_unsupported_fields(messages)
        tc = cleaned[0]["tool_calls"][0]
        assert "index" not in tc
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "get_time"

    def test_strip_name_from_tool_message(self):
        from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig

        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_time",
                "content": "12:00",
            }
        ]
        cleaned = CohereV2ChatConfig._strip_cohere_unsupported_fields(messages)
        assert "name" not in cleaned[0]
        assert cleaned[0]["content"] == "12:00"
        assert cleaned[0]["tool_call_id"] == "call_1"

    def test_non_tool_messages_unchanged(self):
        from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        cleaned = CohereV2ChatConfig._strip_cohere_unsupported_fields(messages)
        assert cleaned == messages
