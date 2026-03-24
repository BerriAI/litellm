import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig


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


class TestCohereV2Transform:
    def setup_method(self):
        self.config = CohereV2ChatConfig()
        self.model = "command-r-08-2024"

    def _make_transform_request(self, messages):
        with patch.object(
            self.config.__class__.__bases__[0],
            "transform_request",
            return_value={"model": self.model, "messages": messages},
        ):
            return self.config.transform_request(
                model=self.model,
                messages=messages,
                optional_params={},
                litellm_params={},
                headers={},
            )

    def test_strips_index_from_assistant_tool_calls(self):
        """Cohere v2 rejects 'index' in tool_calls — it must be stripped before sending."""
        messages = [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = self._make_transform_request(messages)
        assistant_msg = result["messages"][1]
        assert "index" not in assistant_msg["tool_calls"][0]
        assert assistant_msg["tool_calls"][0]["id"] == "call_abc"

    def test_strips_name_from_tool_result_messages(self):
        """Cohere v2 rejects 'name' in tool result messages — it must be stripped."""
        messages = [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "name": "get_time",
                "content": "12:00",
            },
        ]
        result = self._make_transform_request(messages)
        tool_msg = result["messages"][1]
        assert "name" not in tool_msg
        assert tool_msg["tool_call_id"] == "call_abc"
        assert tool_msg["content"] == "12:00"

    def test_preserves_messages_without_offending_fields(self):
        """Messages that don't have index or name are passed through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self._make_transform_request(messages)
        assert result["messages"] == messages
