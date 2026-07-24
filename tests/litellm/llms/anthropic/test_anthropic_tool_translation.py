"""
Tests for Anthropic -> OpenAI tool translation in the experimental
pass-through adapter.

Related issue: https://github.com/BerriAI/litellm/issues/34510
"""

import copy

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)


class TestTranslateAnthropicToolsToOpenai:
    """Regression tests for translate_anthropic_tools_to_openai."""

    def setup_method(self):
        self.adapter = LiteLLMAnthropicMessagesAdapter()

    def test_does_not_mutate_callers_input_schema(self):
        """#34510: extra tool kwargs must not leak into the source dict."""
        tool = {
            "name": "computer",
            "description": "computer use tool",
            "input_schema": {"type": "object", "properties": {}},
            # Extra top-level key, as sent by Anthropic for the computer tool.
            "display_width_px": 1024,
        }
        # Keep a pristine copy of the caller's schema to compare against.
        schema_before = copy.deepcopy(tool["input_schema"])

        self.adapter.translate_anthropic_tools_to_openai([tool])

        # The caller's original input_schema must be left completely untouched.
        assert tool["input_schema"] == schema_before
        assert "display_width_px" not in tool["input_schema"]

    def test_extra_kwargs_still_propagate_to_parameters(self):
        """The fix must still carry intentional vendor kwargs to `parameters`."""
        tool = {
            "name": "computer",
            "description": "computer use tool",
            "input_schema": {"type": "object", "properties": {}},
            "display_width_px": 1024,
        }

        translated, _ = self.adapter.translate_anthropic_tools_to_openai([tool])

        params = translated[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert params["display_width_px"] == 1024

    def test_reuse_is_safe(self):
        """Reusing the same tool list (as a guardrail + real pass) is idempotent."""
        shared_tool = {
            "name": "computer",
            "input_schema": {"type": "object", "properties": {}},
            "display_width_px": 800,
        }
        tools = [shared_tool]

        first_pass, _ = self.adapter.translate_anthropic_tools_to_openai(tools)
        second_pass, _ = self.adapter.translate_anthropic_tools_to_openai(tools)

        # Source untouched after repeated translation.
        assert shared_tool["input_schema"] == {"type": "object", "properties": {}}
        # Both passes produce equivalent, correct output.
        assert first_pass[0]["function"]["parameters"] == second_pass[0]["function"]["parameters"]
