import os
import sys


import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# ) # noqa
# )  # Adds the parent directory to the system path

import litellm
from base_llm_unit_tests import BaseLLMChatTest
from litellm.llms.groq.chat.transformation import GroqChatConfig

class TestGroq(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "groq/llama-3.3-70b-versatile",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_tool_call_with_empty_enum_property(self):
        pass

    @pytest.mark.parametrize("model", ["groq/qwen/qwen3-32b", "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"])
    def test_reasoning_effort_in_supported_params(self, model):
        """Test that reasoning_effort is in the list of supported parameters for Groq"""
        supported_params = GroqChatConfig().get_supported_openai_params(model=model)
        assert "reasoning_effort" in supported_params


class TestGroqStructuredOutputs:
    """
    Tests for Groq structured outputs handling.
    Related issues:
    - https://github.com/BerriAI/litellm/issues/11001
    - https://github.com/openai/openai-agents-python/issues/2140
    """

    def test_structured_output_with_tools_raises_error_for_non_native_models(self):
        """
        Test that using structured outputs + tools with models that don't support
        native json_schema raises a clear error message.

        Groq does not support structured outputs + tools together.
        See: https://console.groq.com/docs/structured-outputs
        "Streaming and tool use are not currently supported with Structured Outputs"
        """
        config = GroqChatConfig()

        # Model that doesn't support native json_schema
        model = "llama-3.3-70b-versatile"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        }

        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model=model,
                drop_params=False,
            )

        assert "does not support native structured outputs" in str(exc_info.value)
        assert "incompatible with user-provided tools" in str(exc_info.value)

    def test_structured_output_without_tools_uses_workaround_for_non_native_models(self):
        """
        Test that structured outputs without tools works using the json_tool_call workaround
        for models that don't support native json_schema.
        """
        config = GroqChatConfig()

        model = "llama-3.3-70b-versatile"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            }
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )

        # Should use the workaround (json_tool_call)
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "json_tool_call"
        assert result["tool_choice"]["function"]["name"] == "json_tool_call"
        assert result.get("json_mode") is True

    def test_structured_output_passes_through_for_native_models(self):
        """
        Test that structured outputs pass through directly for models that
        support native json_schema (e.g., gpt-oss-120b).
        """
        config = GroqChatConfig()

        # Model that supports native json_schema
        model = "openai/gpt-oss-120b"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"]
                    }
                }
            }
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )

        # Should NOT use the workaround - response_format should pass through
        # The workaround sets json_mode=True, so if it's not set, we know it passed through
        assert result.get("json_mode") is not True
        # Should not have the json_tool_call tool
        if "tools" in result:
            tool_names = [t.get("function", {}).get("name") for t in result["tools"]]
            assert "json_tool_call" not in tool_names
