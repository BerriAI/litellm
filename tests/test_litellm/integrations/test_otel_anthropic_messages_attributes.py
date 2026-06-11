"""
Regression for #30121.

OpenTelemetry.set_attributes() missed both gen_ai.input.messages and
gen_ai.output.messages for the Anthropic-native /v1/messages route
(call_type="anthropic_messages") because:

  - Input gate checked ``kwargs.get("messages")`` only, but the
    anthropic_messages handler stores the messages on
    ``optional_params["messages"]``.
  - Output gate had branches for OpenAI ``choices`` and Responses API
    ``output`` but none for the Anthropic top-level ``content`` block list.

Span otherwise had cost/usage/model fine — only the prompt/completion
content was missing in Langfuse, Phoenix, Arize, etc.
"""

import unittest
from unittest.mock import MagicMock

from litellm.integrations.opentelemetry import OpenTelemetry


def _base_kwargs() -> dict:
    return {
        "model": "claude-opus-4-7",
        "litellm_params": {"custom_llm_provider": "anthropic"},
        "standard_logging_object": {
            "id": "test-id",
            "call_type": "anthropic_messages",
            "metadata": {},
        },
    }


def _attr_set(mock_span: MagicMock) -> dict:
    out = {}
    for c in mock_span.set_attribute.call_args_list:
        args, _ = c
        out[args[0]] = args[1]
    return out


class TestOtelAnthropicMessagesInput(unittest.TestCase):
    def test_messages_on_optional_params_populate_input_messages(self):
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        kwargs["optional_params"] = {
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": "ok"},
            ]
        }

        response_obj = {
            "content": [{"type": "text", "text": "done"}],
            "role": "assistant",
            "stop_reason": "end_turn",
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        attrs = _attr_set(mock_span)
        self.assertIn("gen_ai.input.messages", attrs)
        self.assertIn("list files", attrs["gen_ai.input.messages"])

    def test_kwargs_messages_still_win_when_both_present(self):
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        kwargs["messages"] = [{"role": "user", "content": "from kwargs"}]
        kwargs["optional_params"] = {
            "messages": [{"role": "user", "content": "from optional"}]
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj={"content": []})

        attrs = _attr_set(mock_span)
        self.assertIn("gen_ai.input.messages", attrs)
        self.assertIn("from kwargs", attrs["gen_ai.input.messages"])
        self.assertNotIn("from optional", attrs["gen_ai.input.messages"])


class TestOtelAnthropicMessagesOutput(unittest.TestCase):
    def test_anthropic_content_blocks_populate_output_messages(self):
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        kwargs["optional_params"] = {"messages": [{"role": "user", "content": "hi"}]}

        response_obj = {
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_use",
                    "id": "tool_abc",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
            ],
            "role": "assistant",
            "stop_reason": "tool_use",
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        attrs = _attr_set(mock_span)
        self.assertIn("gen_ai.output.messages", attrs)
        out = attrs["gen_ai.output.messages"]
        self.assertIn("hello", out)
        self.assertIn("bash", out)
        self.assertIn("tool_abc", out)

        self.assertIn("gen_ai.response.finish_reasons", attrs)
        self.assertIn("tool_use", attrs["gen_ai.response.finish_reasons"])

    def test_thinking_block_is_serialised_as_text_part(self):
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        response_obj = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "deliberating...",
                    "signature": "sig",
                },
                {"type": "text", "text": "answer"},
            ],
            "role": "assistant",
            "stop_reason": "end_turn",
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        attrs = _attr_set(mock_span)
        self.assertIn("gen_ai.output.messages", attrs)
        out = attrs["gen_ai.output.messages"]
        self.assertIn("deliberating...", out)
        self.assertIn("answer", out)

    def test_empty_or_unrecognised_content_blocks_skip_output_messages_emit(self):
        """Greptile flagged the unconditional emit as inconsistent with the
        Responses API branch, which guards ``if output_messages:``. Mirror
        that guard so a content list of only unknown block types doesn't
        leave a blank assistant-message entry in observability tools."""
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        response_obj = {
            "content": [
                {"type": "future_block_type_unknown_to_litellm", "payload": "..."},
            ],
            "role": "assistant",
            "stop_reason": "end_turn",
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        attrs = _attr_set(mock_span)
        self.assertNotIn("gen_ai.output.messages", attrs)
        # stop_reason still emits even with empty parts.
        self.assertIn("gen_ai.response.finish_reasons", attrs)

    def test_choices_still_takes_precedence_over_content_for_openai_shape(self):
        otel = OpenTelemetry()
        mock_span = MagicMock()

        kwargs = _base_kwargs()
        kwargs["standard_logging_object"]["call_type"] = "completion"
        kwargs["messages"] = [{"role": "user", "content": "hi"}]

        # response with BOTH choices (openai shape) and a spurious content
        # field — choices branch must win to preserve the openai contract.
        response_obj = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "openai-out"},
                    "finish_reason": "stop",
                }
            ],
            "content": [{"type": "text", "text": "anthropic-out"}],
        }

        otel.set_attributes(span=mock_span, kwargs=kwargs, response_obj=response_obj)

        attrs = _attr_set(mock_span)
        self.assertIn("gen_ai.output.messages", attrs)
        out = attrs["gen_ai.output.messages"]
        self.assertIn("openai-out", out)
        self.assertNotIn("anthropic-out", out)
