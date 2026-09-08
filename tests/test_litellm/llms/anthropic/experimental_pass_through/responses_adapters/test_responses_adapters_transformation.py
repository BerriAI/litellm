"""
Tests for LiteLLMAnthropicToResponsesAPIAdapter
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/transformation.py)
"""

import json
import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    TOOL_RESULT_IMAGE_BOUNDARY,
    TOOL_RESULT_IMAGE_PLACEHOLDER,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
)
from litellm.types.llms.openai import ResponseAPIUsage


def _make_request(**overrides) -> AnthropicMessagesRequest:
    base: dict = {
        "model": "openai.gpt-5.1-codex",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
    }
    base.update(overrides)
    return AnthropicMessagesRequest(**base)


_ADAPTER = LiteLLMAnthropicToResponsesAPIAdapter()


# ---------------------------------------------------------------------------
# context_management conversion
# ---------------------------------------------------------------------------


class TestContextManagementConversion:
    """Anthropic dict -> OpenAI array conversion for context_management."""

    def test_compact_edit_converted_to_array(self):
        """compact_20260112 with trigger maps to OpenAI compaction entry."""
        cm = {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 150000},
                }
            ]
        }
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction", "compact_threshold": 150000}]

    def test_compact_edit_without_trigger(self):
        """compact_20260112 without a trigger still maps to a compaction entry."""
        cm = {"edits": [{"type": "compact_20260112"}]}
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction"}]

    def test_unknown_edit_type_is_dropped(self):
        """Anthropic-only edit types (e.g. clear_thinking) are silently dropped."""
        cm = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result is None

    def test_mixed_edits_only_known_types_kept(self):
        """Only compact_20260112 is converted; unknown types are dropped."""
        cm = {
            "edits": [
                {"type": "clear_thinking_20251015", "keep": "all"},
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 200000},
                },
            ]
        }
        result = _ADAPTER.translate_context_management_to_responses_api(cm)
        assert result == [{"type": "compaction", "compact_threshold": 200000}]

    def test_non_dict_returns_none(self):
        result = _ADAPTER.translate_context_management_to_responses_api([])  # type: ignore
        assert result is None

    def test_translate_request_includes_context_management(self):
        """translate_request converts context_management and sets it on kwargs."""
        req = _make_request(
            context_management={
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 100000},
                    }
                ]
            }
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["context_management"] == [{"type": "compaction", "compact_threshold": 100000}]

    def test_translate_request_drops_anthropic_only_context_management(self):
        """context_management with only unknown edit types is omitted from kwargs."""
        req = _make_request(context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]})
        kwargs = _ADAPTER.translate_request(req)
        assert "context_management" not in kwargs


# ---------------------------------------------------------------------------
# structured output via output_config
# ---------------------------------------------------------------------------


class TestOutputConfigStructuredOutput:
    """output_config.format.json_schema -> OpenAI text.format conversion."""

    _SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["name", "email"],
        "additionalProperties": False,
    }

    def test_output_config_format_json_schema_converted(self):
        """output_config.format.json_schema is converted to OpenAI text.format, defaulting strict to False."""
        req = _make_request(output_config={"format": {"type": "json_schema", "schema": self._SCHEMA}})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" in kwargs
        fmt = kwargs["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"] == self._SCHEMA
        assert fmt["strict"] is False
        assert fmt["name"] == "structured_output"

    def test_output_config_format_explicit_strict_true_is_preserved(self):
        """Nested output_config.format with explicit strict=True is preserved."""
        req = _make_request(
            output_config={"format": {"type": "json_schema", "schema": self._SCHEMA, "strict": True}}
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["text"]["format"]["strict"] is True

    def test_output_config_without_format_does_not_set_text(self):
        """output_config with only non-format keys doesn't produce text.format."""
        req = _make_request(output_config={"effort": "high"})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" not in kwargs

    def test_output_format_still_works(self):
        """The original output_format field still takes precedence when present, defaulting strict to False."""
        req = _make_request(output_format={"type": "json_schema", "schema": self._SCHEMA})
        kwargs = _ADAPTER.translate_request(req)
        assert "text" in kwargs
        assert kwargs["text"]["format"]["type"] == "json_schema"
        assert kwargs["text"]["format"]["strict"] is False

    def test_output_format_explicit_strict_false_is_preserved(self):
        """output_format with an explicit strict=False is preserved as False."""
        req = _make_request(output_format={"type": "json_schema", "schema": self._SCHEMA, "strict": False})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["text"]["format"]["strict"] is False

    def test_output_format_explicit_strict_true_is_preserved(self):
        """output_format with an explicit strict=True is preserved as True."""
        req = _make_request(output_format={"type": "json_schema", "schema": self._SCHEMA, "strict": True})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["text"]["format"]["strict"] is True

    def test_output_format_takes_precedence_over_output_config(self):
        """output_format takes precedence over output_config.format, for both schema and strict."""
        other_schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
        req = _make_request(
            output_format={"type": "json_schema", "schema": self._SCHEMA, "strict": False},
            output_config={"format": {"type": "json_schema", "schema": other_schema, "strict": True}},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["text"]["format"]["schema"] == self._SCHEMA
        assert kwargs["text"]["format"]["strict"] is False

    def test_optional_property_stays_out_of_required_list(self):
        """A property absent from required must stay absent from required in the translated schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        req = _make_request(output_format={"type": "json_schema", "schema": schema})
        kwargs = _ADAPTER.translate_request(req)
        fmt_schema = kwargs["text"]["format"]["schema"]
        assert fmt_schema["required"] == ["name"]
        assert "nickname" not in fmt_schema["required"]
        assert fmt_schema["additionalProperties"] is False

    def test_translate_request_does_not_mutate_input_schema(self):
        """translate_request must not mutate the caller's output_format or schema dicts."""
        schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
        output_format = {"type": "json_schema", "schema": schema, "strict": False}
        req = _make_request(output_format=output_format)
        snapshot = json.loads(json.dumps(output_format))

        _ADAPTER.translate_request(req)

        assert output_format == snapshot
        assert req["output_format"] == snapshot


# ---------------------------------------------------------------------------
# translate_messages_to_responses_input
# ---------------------------------------------------------------------------


# Helper: cast plain dicts to the expected type so call sites stay clean.
def _translate_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    return _ADAPTER.translate_messages_to_responses_input(messages)  # type: ignore[arg-type]


class TestTranslateMessagesToResponsesInput:
    """Anthropic messages list -> OpenAI Responses API input items."""

    def test_user_string_content(self):
        """Plain string user message becomes a message with input_text."""
        messages = [{"role": "user", "content": "Hello world"}]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello world"}],
            }
        ]

    def test_user_list_text_block(self):
        """User message with text content block maps to input_text."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 2+2?"}],
            }
        ]

    def test_user_multiple_text_blocks(self):
        """Multiple text blocks in a user message are all converted."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "input_text", "text": "First part."},
            {"type": "input_text", "text": "Second part."},
        ]

    @pytest.mark.parametrize(
        "system_content",
        [
            "Use the corrected result.",
            [{"type": "text", "text": "Use the corrected result."}],
            [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                {"type": "text", "text": "Use the corrected result."},
            ],
        ],
    )
    def test_midturn_system_correction_stays_system_in_sequence(self, system_content: object):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01234",
                        "name": "get_weather",
                        "input": {"location": "Boston"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01234",
                        "content": "Rainy, 55°F",
                    }
                ],
            },
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Continue."},
        ]

        result = _translate_messages(messages)

        assert result == [
            {
                "type": "function_call",
                "call_id": "toolu_01234",
                "name": "get_weather",
                "arguments": '{"location": "Boston"}',
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_01234",
                "output": "Rainy, 55°F",
            },
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "Use the corrected result."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue."}],
            },
        ]

    def test_midturn_system_correction_keeps_multiple_text_blocks(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "First correction."},
                    {"type": "text", "text": "Second correction."},
                ],
            }
        ]

        assert _translate_messages(messages) == [
            {
                "type": "message",
                "role": "system",
                "content": [
                    {"type": "input_text", "text": "First correction."},
                    {"type": "input_text", "text": "Second correction."},
                ],
            }
        ]

    @pytest.mark.parametrize(
        "system_content",
        [
            "",
            [{"type": "text", "text": ""}],
            [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}],
            None,
        ],
    )
    def test_empty_or_unsupported_midturn_system_correction_is_dropped(self, system_content: object):
        messages = [{"role": "system", "content": system_content}]

        assert _translate_messages(messages) == []

    def test_user_base64_image(self):
        """User message with base64 image source becomes input_image with data URL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc123",
                        },
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == [{"type": "input_image", "image_url": "data:image/png;base64,abc123"}]

    def test_user_url_image(self):
        """User message with URL image source becomes input_image with the URL."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/img.jpg"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["content"] == [{"type": "input_image", "image_url": "https://example.com/img.jpg"}]

    def test_user_base64_image_empty_data_skipped(self):
        """Base64 image with empty data is skipped (no URL can be formed)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "",
                        },
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        # No user_parts -> no message item appended
        assert result == []

    def test_user_tool_result_string_content(self):
        """tool_result with string content becomes function_call_output."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": "42 degrees",
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "42 degrees",
            }
        ]

    def test_user_tool_result_list_content(self):
        """tool_result with list of text blocks is joined into a single string."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_xyz",
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ],
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["output"] == "Line 1\nLine 2"

    def test_user_tool_result_null_content(self):
        """tool_result with null content becomes empty string output."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_null", "content": None}],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["output"] == ""

    def test_assistant_string_content(self):
        """Plain string assistant message becomes a message with output_text."""
        messages = [{"role": "assistant", "content": "I can help with that."}]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I can help with that."}],
            }
        ]

    def test_assistant_text_block(self):
        """Assistant message with text block maps to output_text."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Here is the answer."}],
            }
        ]
        result = _translate_messages(messages)
        assert result[0]["content"] == [{"type": "output_text", "text": "Here is the answer."}]

    def test_assistant_tool_use_becomes_function_call(self):
        """Assistant tool_use block becomes a top-level function_call item."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"location": "Boston"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "function_call",
                "call_id": "toolu_01",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }
        ]

    def test_assistant_thinking_block_becomes_reasoning_item(self):
        """Assistant thinking block becomes a reasoning item, never visible assistant prose."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Let me reason step by step."}],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Let me reason step by step."}],
            }
        ]

    def test_reasoning_item_carries_no_id(self):
        """A fabricated reasoning id 404s upstream, so the item must go out without one."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "Private reasoning.", "signature": "rs_abc123"}],
            }
        ]
        result = _translate_messages(messages)
        assert "id" not in result[0]

    def test_consecutive_thinking_blocks_become_one_reasoning_item(self):
        """Summary parts of one upstream reasoning item are regrouped into that item."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "First part."},
                    {"type": "thinking", "thinking": "Second part."},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "First part."},
                    {"type": "summary_text", "text": "Second part."},
                ],
            }
        ]

    def test_a_tool_call_splits_the_reasoning_items_around_it(self):
        """Thinking on either side of a tool call belongs to two different reasoning items."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Before the call."},
                    {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Denver"}},
                    {"type": "thinking", "thinking": "After the call."},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert [item["type"] for item in result] == ["reasoning", "function_call", "reasoning"]
        assert result[0]["summary"] == [{"type": "summary_text", "text": "Before the call."}]
        assert result[2]["summary"] == [{"type": "summary_text", "text": "After the call."}]

    def test_thinking_and_text_stay_separate(self):
        """The visible answer stays the only thing in the assistant message."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "The user wants Denver."},
                    {"type": "text", "text": "Denver is the best pick."},
                ],
            }
        ]
        result = _translate_messages(messages)
        assert [item["type"] for item in result] == ["reasoning", "message"]
        assert result[1]["content"] == [{"type": "output_text", "text": "Denver is the best pick."}]

    def test_assistant_empty_thinking_block_skipped(self):
        """Assistant thinking block with empty thinking text is skipped."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": ""}],
            }
        ]
        result = _translate_messages(messages)
        assert result == []

    def test_mixed_messages_ordering(self):
        """Full multi-turn conversation is converted in order."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_02",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_02",
                        "content": "Sunny, 72F",
                    }
                ],
            },
            {"role": "assistant", "content": "It's sunny and 72°F in NYC."},
        ]
        result = _translate_messages(messages)
        types = [item["type"] for item in result]
        assert types == ["message", "function_call", "function_call_output", "message"]

    def test_user_text_and_image_mixed(self):
        """User message with both text and image produces both parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
                    },
                ],
            }
        ]
        result = _translate_messages(messages)
        assert len(result) == 1
        assert result[0]["content"][0] == {
            "type": "input_text",
            "text": "Describe this image:",
        }
        assert result[0]["content"][1] == {
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg",
        }

    def test_unknown_image_source_type_skipped(self):
        """Image block with unknown source type is silently skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "file_path", "path": "/tmp/img.png"},
                    }
                ],
            }
        ]
        result = _translate_messages(messages)
        assert result == []


# ---------------------------------------------------------------------------
# translate_tools_to_responses_api
# ---------------------------------------------------------------------------


class TestTranslateToolsToResponsesAPI:
    """Anthropic tool definitions -> Responses API function tools."""

    def test_regular_tool_with_description_and_schema(self):
        """Standard tool with description and input_schema is converted to function."""
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [
            {
                "type": "function",
                "name": "get_weather",
                "strict": False,
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

    def test_tool_with_optional_properties_stays_non_strict(self):
        """Regression: an unset Anthropic `strict` must not become the Responses strict default,
        which would rewrite `required` to include every optional property."""
        tools: List[AllAnthropicToolsValues] = [
            {
                "name": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "cursor": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

        result = _ADAPTER.translate_tools_to_responses_api(tools)

        assert result[0]["strict"] is False
        assert result[0]["parameters"]["required"] == ["query"]

    def test_tool_forwards_explicit_strict_true(self):
        """An explicit Anthropic `strict: True` still reaches Responses as True."""
        tools: List[AllAnthropicToolsValues] = [
            {
                "name": "search",
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

        result = _ADAPTER.translate_tools_to_responses_api(tools)

        assert result == [
            {
                "type": "function",
                "name": "search",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def test_tool_without_description(self):
        """Tool without a description omits the description key."""
        tools = [{"name": "ping", "input_schema": {"type": "object", "properties": {}}}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "ping"
        assert "description" not in result[0]

    def test_tool_without_input_schema(self):
        """Tool without input_schema omits the parameters key."""
        tools = [{"name": "no_schema_tool", "description": "Does something."}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result[0]["type"] == "function"
        assert "parameters" not in result[0]

    def test_web_search_tool_by_name(self):
        """Tool named 'web_search' maps to web_search_preview."""
        tools = [{"name": "web_search", "type": "custom"}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [{"type": "web_search_preview"}]

    def test_web_search_tool_by_type_prefix(self):
        """Tool with type starting with 'web_search' maps to web_search_preview."""
        tools = [{"name": "search", "type": "web_search_20250305"}]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert result == [{"type": "web_search_preview"}]

    def test_multiple_tools_order_preserved(self):
        """Multiple tools are converted in order."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "web_search", "type": "custom"},
            {"name": "tool_b", "description": "B"},
        ]
        result = _ADAPTER.translate_tools_to_responses_api(tools)  # type: ignore[arg-type]
        assert len(result) == 3
        assert result[0]["name"] == "tool_a"
        assert result[1] == {"type": "web_search_preview"}
        assert result[2]["name"] == "tool_b"

    def test_empty_tools_list(self):
        """Empty tools list returns empty list."""
        assert _ADAPTER.translate_tools_to_responses_api([]) == []


# ---------------------------------------------------------------------------
# translate_tool_choice_to_responses_api
# ---------------------------------------------------------------------------


class TestTranslateToolChoiceToResponsesAPI:
    """Anthropic tool_choice -> Responses API tool_choice.

    The Responses API's tool_choice schema (openai.types.responses.tool_choice_options)
    is a bare Literal["none", "auto", "required"] for these simple cases - not an
    object like {"type": "auto"}. Sending the object shape to an OpenAI-compatible
    server gets rejected with a pydantic validation error.
    """

    def test_auto_maps_to_bare_string_auto(self):
        assert _ADAPTER.translate_tool_choice_to_responses_api({"type": "auto"}) == "auto"

    def test_any_maps_to_bare_string_required(self):
        assert _ADAPTER.translate_tool_choice_to_responses_api({"type": "any"}) == "required"

    def test_none_maps_to_bare_string_none(self):
        assert _ADAPTER.translate_tool_choice_to_responses_api({"type": "none"}) == "none"

    def test_specific_tool_maps_to_function(self):
        result = _ADAPTER.translate_tool_choice_to_responses_api({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "name": "get_weather"}


# ---------------------------------------------------------------------------
# translate_thinking_to_reasoning
# ---------------------------------------------------------------------------


class TestTranslateThinkingToReasoning:
    """Anthropic thinking param -> Responses API reasoning param."""

    def test_budget_high_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 10000})
        # Default (reasoning_auto_summary=False): only effort, no summary
        assert result == {"effort": "high"}
        assert result is not None and "summary" not in result

    def test_budget_above_threshold_high_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 50000})
        assert result is not None
        assert result["effort"] == "high"
        assert "summary" not in result

    def test_budget_medium_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            }
        )
        assert result == {"effort": "medium"}
        assert result is not None and "summary" not in result

    def test_budget_low_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            }
        )
        assert result == {"effort": "low"}
        assert result is not None and "summary" not in result

    def test_budget_minimal_effort(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 500})
        assert result == {"effort": "minimal"}
        assert result is not None and "summary" not in result

    def test_budget_at_exact_thresholds(self):
        result_high = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            }
        )
        assert result_high is not None
        assert result_high["effort"] == "high"
        result_medium = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            }
        )
        assert result_medium is not None
        assert result_medium["effort"] == "medium"
        assert "summary" not in result_medium
        result_low = _ADAPTER.translate_thinking_to_reasoning(
            {
                "type": "enabled",
                "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            }
        )
        assert result_low is not None
        assert result_low["effort"] == "low"
        assert "summary" not in result_low

    def test_disabled_type_returns_none(self):
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "disabled"})
        assert result is None

    def test_non_dict_returns_none(self):
        result = _ADAPTER.translate_thinking_to_reasoning("enabled")  # type: ignore
        assert result is None

    def test_missing_budget_defaults_to_minimal(self):
        """Missing budget_tokens defaults to 0, below the low threshold -> minimal."""
        result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled"})
        assert result == {"effort": "minimal"}
        assert result is not None and "summary" not in result

    def test_summary_added_when_auto_summary_enabled(self):
        """When reasoning_auto_summary is True, summary='detailed' is included."""
        import litellm

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = True
            result = _ADAPTER.translate_thinking_to_reasoning({"type": "enabled", "budget_tokens": 10000})
            assert result == {"effort": "high", "summary": "detailed"}
        finally:
            litellm.reasoning_auto_summary = original

    def test_summary_added_when_env_var_set(self, monkeypatch):
        """When LITELLM_REASONING_AUTO_SUMMARY env var is true, summary is included."""
        import litellm

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")
            result = _ADAPTER.translate_thinking_to_reasoning(
                {
                    "type": "enabled",
                    "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                }
            )
            assert result == {"effort": "medium", "summary": "detailed"}
        finally:
            litellm.reasoning_auto_summary = original
            os.environ.pop("LITELLM_REASONING_AUTO_SUMMARY", None)


# ---------------------------------------------------------------------------
# translate_request – broader coverage
# ---------------------------------------------------------------------------


class TestTranslateRequestBroaderCoverage:
    """Full translate_request call: field-by-field mapping verification."""

    def test_model_and_input_always_present(self):
        req = _make_request()
        kwargs = _ADAPTER.translate_request(req)
        assert "model" in kwargs
        assert "input" in kwargs

    def test_system_string_becomes_instructions(self):
        req = _make_request(system="You are a helpful assistant.")
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "You are a helpful assistant."

    def test_top_level_system_and_midturn_correction_are_not_duplicated(self):
        """
        Request level: the trusted top-level prompt goes to `instructions` only, and the
        in-sequence correction stays a `role: "system"` input item in its original position.
        Neither appears twice, and the surrounding turns keep their order.
        """
        req = _make_request(
            system="Trusted top-level prompt.",
            messages=[
                {"role": "user", "content": "First question."},
                {"role": "system", "content": "Use the corrected result."},
                {"role": "user", "content": "Continue."},
            ],
        )

        kwargs = _ADAPTER.translate_request(req)

        assert kwargs["instructions"] == "Trusted top-level prompt."
        assert kwargs["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "First question."}],
            },
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "Use the corrected result."}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue."}],
            },
        ]

    def test_system_list_of_text_blocks_joined(self):
        req = _make_request(
            system=[
                {"type": "text", "text": "Be concise."},
                {"type": "text", "text": "Be helpful."},
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "Be concise.\nBe helpful."

    def test_system_list_skips_non_text_blocks(self):
        req = _make_request(
            system=[
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only text matters."},
            ]
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["instructions"] == "Only text matters."

    def test_max_tokens_mapped_to_max_output_tokens(self):
        req = _make_request(max_tokens=512)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["max_output_tokens"] == 512

    def test_temperature_passed_through(self):
        req = _make_request(temperature=0.7)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["temperature"] == 0.7

    def test_top_p_passed_through(self):
        req = _make_request(top_p=0.9)
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["top_p"] == 0.9

    def test_tools_translated(self):
        req = _make_request(tools=[{"name": "calculator", "description": "Does math.", "input_schema": {}}])
        kwargs = _ADAPTER.translate_request(req)
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "calculator"

    def test_tool_choice_translated(self):
        req = _make_request(
            tools=[{"name": "do_thing"}],
            tool_choice={"type": "tool", "name": "do_thing"},
        )
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["tool_choice"] == {"type": "function", "name": "do_thing"}

    def test_thinking_translated_to_reasoning(self):
        req = _make_request(thinking={"type": "enabled", "budget_tokens": 12000})
        kwargs = _ADAPTER.translate_request(req)
        # reasoning_auto_summary is False by default, so no summary key
        assert kwargs["reasoning"] == {"effort": "high"}
        assert "summary" not in kwargs["reasoning"]

    def test_disabled_thinking_not_included_in_kwargs(self):
        req = _make_request(thinking={"type": "disabled"})
        kwargs = _ADAPTER.translate_request(req)
        assert "reasoning" not in kwargs

    def test_metadata_user_id_mapped_to_user(self):
        req = _make_request(metadata={"user_id": "user-42"})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["user"] == "user-42"

    def test_metadata_user_id_truncated_to_64_chars(self):
        long_id = "x" * 100
        req = _make_request(metadata={"user_id": long_id})
        kwargs = _ADAPTER.translate_request(req)
        assert len(kwargs["user"]) == 64

    def test_metadata_user_id_mapped_to_prompt_cache_key(self):
        req = _make_request(metadata={"user_id": "user-42"})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["prompt_cache_key"] == "user-42"

    def test_metadata_user_id_prompt_cache_key_truncated_to_first_64_chars(self):
        long_id = "".join(str(i % 10) for i in range(100))
        req = _make_request(metadata={"user_id": long_id})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["prompt_cache_key"] == long_id[:64]
        assert len(kwargs["prompt_cache_key"]) == 64

    def test_metadata_empty_user_id_sets_no_prompt_cache_key(self):
        req = _make_request(metadata={"user_id": ""})
        kwargs = _ADAPTER.translate_request(req)
        assert kwargs["user"] == ""
        assert "prompt_cache_key" not in kwargs

    def test_metadata_null_user_id_sets_no_prompt_cache_key(self):
        req = _make_request(metadata={"user_id": None})
        kwargs = _ADAPTER.translate_request(req)
        assert "prompt_cache_key" not in kwargs

    def test_no_optional_fields_does_not_add_spurious_keys(self):
        req = _make_request()
        kwargs = _ADAPTER.translate_request(req)
        for key in (
            "instructions",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "reasoning",
            "text",
            "context_management",
            "user",
            "prompt_cache_key",
        ):
            assert key not in kwargs, f"unexpected key: {key}"


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


def _make_mock_response(
    output: list,
    status: str = "completed",
    response_id: str = "resp_001",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> MagicMock:
    """Build a minimal mock ResponsesAPIResponse."""
    usage = ResponseAPIUsage(
        input_tokens=input_tokens,
        input_tokens_details={
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
        },
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )

    resp = MagicMock()
    resp.id = response_id
    resp.model = model
    resp.status = status
    resp.output = output
    resp.usage = usage
    return resp


def _make_output_message(texts: List[str]) -> MagicMock:
    """Build a mock ResponseOutputMessage with output_text parts."""
    from openai.types.responses import ResponseOutputMessage  # type: ignore[import]

    parts = []
    for t in texts:
        part = MagicMock()
        part.type = "output_text"
        part.text = t
        parts.append(part)

    msg = MagicMock(spec=ResponseOutputMessage)
    msg.content = parts
    return msg


def _make_function_call_item(call_id: str, name: str, arguments: str) -> MagicMock:
    """Build a mock ResponseFunctionToolCall."""
    from openai.types.responses import ResponseFunctionToolCall  # type: ignore[import]

    item = MagicMock(spec=ResponseFunctionToolCall)
    item.call_id = call_id
    item.id = call_id
    item.name = name
    item.arguments = arguments
    return item


def _make_reasoning_item(summaries: List[str], item_id: str = "rs_test_1") -> MagicMock:
    """Build a mock ResponseReasoningItem."""
    from openai.types.responses import ResponseReasoningItem  # type: ignore[import]

    summary_mocks = []
    for text in summaries:
        s = MagicMock()
        s.text = text
        summary_mocks.append(s)

    item = MagicMock(spec=ResponseReasoningItem)
    item.id = item_id
    item.summary = summary_mocks
    return item


class TestTranslateResponse:
    """Responses API -> AnthropicMessagesResponse conversion."""

    def test_output_text_message_becomes_text_block(self):
        """ResponseOutputMessage with output_text parts -> Anthropic text content."""
        response = _make_mock_response(output=[_make_output_message(["Hello!"])])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"

    def test_multiple_text_parts(self):
        """Multiple output_text parts become multiple text content blocks."""
        response = _make_mock_response(output=[_make_output_message(["Part 1", "Part 2"])])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["text"] == "Part 1"
        assert result["content"][1]["text"] == "Part 2"

    def test_function_call_becomes_tool_use(self):
        """ResponseFunctionToolCall -> Anthropic tool_use content block."""
        fc = _make_function_call_item("call_99", "get_weather", '{"city": "NYC"}')
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_99"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "NYC"}

    def test_function_call_sets_stop_reason_tool_use(self):
        """Presence of a function_call sets stop_reason to 'tool_use'."""
        fc = _make_function_call_item("call_1", "tool_a", "{}")
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "tool_use"

    def test_text_only_stop_reason_end_turn(self):
        """Text-only response has stop_reason 'end_turn'."""
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "end_turn"

    def test_incomplete_status_sets_max_tokens(self):
        """status='incomplete' overrides stop_reason to 'max_tokens'."""
        response = _make_mock_response(
            output=[_make_output_message(["Truncated..."])],
            status="incomplete",
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["stop_reason"] == "max_tokens"

    def test_reasoning_item_becomes_thinking_block(self):
        """ResponseReasoningItem summaries -> Anthropic thinking content blocks."""
        reasoning = _make_reasoning_item(["Step 1: analyze. Step 2: conclude."])
        response = _make_mock_response(output=[reasoning])
        result: Any = _ADAPTER.translate_response(response)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "thinking"
        assert "Step 1" in result["content"][0]["thinking"]

    def test_empty_reasoning_summary_skipped(self):
        """Reasoning item with empty text summary is not added to content."""
        reasoning = _make_reasoning_item([""])
        response = _make_mock_response(output=[reasoning])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == []

    def test_null_summary_text_skipped_rather_than_stringified(self):
        """A summary part whose text is null must not reach the client as the word "None"."""
        response = _make_mock_response(
            output=[
                {
                    "type": "reasoning",
                    "id": "rs_null_1",
                    "summary": [{"type": "summary_text", "text": None}],
                }
            ]
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == []

    def test_reasoning_item_id_never_becomes_a_thinking_signature(self):
        """Only Anthropic can sign a thinking block, so a stand-in signature is never invented."""
        reasoning = _make_reasoning_item(["Part one.", "Part two."], item_id="rs_abc123")
        response = _make_mock_response(output=[reasoning])
        result: Any = _ADAPTER.translate_response(response)
        assert [block["signature"] for block in result["content"]] == [None, None]

    def test_dict_reasoning_item_becomes_thinking_block(self):
        """A reasoning item arriving as a plain dict is kept, not dropped."""
        response = _make_mock_response(
            output=[
                {
                    "type": "reasoning",
                    "id": "rs_dict_1",
                    "summary": [{"type": "summary_text", "text": "Weighing the options."}],
                }
            ]
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == [
            {"type": "thinking", "thinking": "Weighing the options.", "signature": None}
        ]

    def test_thinking_blocks_are_dropped_when_replayed_to_anthropic(self):
        """Replaying this turn to an Anthropic model must not send a signature it cannot verify."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            _drop_unsignable_thinking_blocks,
        )

        response = _make_mock_response(output=[_make_reasoning_item(["Part one."], item_id="rs_abc123")])
        result: Any = _ADAPTER.translate_response(response)
        assert _drop_unsignable_thinking_blocks(result["content"]) == []

    def test_usage_mapped_correctly(self):
        """Input/output tokens from ResponseAPIUsage are mapped to AnthropicUsage."""
        response = _make_mock_response(
            output=[_make_output_message(["OK"])],
            input_tokens=200,
            output_tokens=75,
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"]["input_tokens"] == 200
        assert result["usage"]["output_tokens"] == 75

    def test_cache_tokens_mapped_to_anthropic_usage(self):
        """Cache reads/writes reported by the Responses API must survive the
        Anthropic mapping, and input_tokens must exclude them so spend is not
        billed at the uncached input rate."""
        response = _make_mock_response(
            output=[_make_output_message(["OK"])],
            input_tokens=4017,
            output_tokens=5,
            cached_tokens=4004,
            cache_write_tokens=10,
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["usage"] == {
            "input_tokens": 3,
            "output_tokens": 5,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 4004,
        }

    def test_missing_usage_maps_to_zero_tokens(self):
        """A response without a usage object must map to zeroed Anthropic usage."""
        assert LiteLLMAnthropicToResponsesAPIAdapter.translate_responses_api_usage_to_anthropic_usage(None) == {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def test_model_and_id_preserved(self):
        """Model and response ID from the Responses API are forwarded."""
        response = _make_mock_response(
            output=[_make_output_message(["Hi"])],
            response_id="resp_xyz",
            model="gpt-4-turbo",
        )
        result: Any = _ADAPTER.translate_response(response)
        assert result["id"] == "resp_xyz"
        assert result["model"] == "gpt-4-turbo"

    def test_role_is_always_assistant(self):
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["role"] == "assistant"

    def test_type_is_always_message(self):
        response = _make_mock_response(output=[_make_output_message(["Hi"])])
        result: Any = _ADAPTER.translate_response(response)
        assert result["type"] == "message"

    def test_empty_output_list(self):
        """Empty output list produces empty content with 'end_turn' stop reason."""
        response = _make_mock_response(output=[])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"

    def test_function_call_with_invalid_json_arguments(self):
        """Invalid JSON in function_call arguments falls back to empty dict."""
        fc = _make_function_call_item("call_bad", "broken_tool", "not-valid-json")
        response = _make_mock_response(output=[fc])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["input"] == {}

    def test_dict_output_message_item(self):
        """Dict-shaped output message (type=message) is also handled."""
        output_item = {
            "type": "message",
            "content": [{"type": "output_text", "text": "Dict-based response"}],
        }
        response = _make_mock_response(output=[output_item])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Dict-based response"

    def test_dict_function_call_item(self):
        """Dict-shaped function_call item is converted to tool_use block."""
        output_item = {
            "type": "function_call",
            "call_id": "call_dict_1",
            "name": "search",
            "arguments": '{"query": "cats"}',
        }
        response = _make_mock_response(output=[output_item])
        result: Any = _ADAPTER.translate_response(response)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "search"
        assert result["content"][0]["input"] == {"query": "cats"}
        assert result["stop_reason"] == "tool_use"

    def test_mixed_reasoning_text_and_tool_use(self):
        """Reasoning + text + tool_use in one response all convert correctly."""
        reasoning = _make_reasoning_item(["Thinking..."])
        text_msg = _make_output_message(["Here is my answer."])
        fc = _make_function_call_item("call_mix", "lookup", '{"id": 1}')
        response = _make_mock_response(output=[reasoning, text_msg, fc])
        result: Any = _ADAPTER.translate_response(response)
        types = [b["type"] for b in result["content"]]
        assert "thinking" in types
        assert "text" in types
        assert "tool_use" in types
        assert result["stop_reason"] == "tool_use"


class TestToolResultImages:
    """Images inside tool_result blocks must survive translation: the
    function_call_output carries a text placeholder and the image is sent as an
    input_image part in a user message emitted after the tool outputs."""

    B64_DATA = "iVBORw0KGgoAAAANSUhEUg=="
    DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    HTTP_URL = "https://example.com/screenshot.png"

    def _messages(self, tool_result_content):
        return [
            {"role": "user", "content": "read the screenshot"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_01", "name": "read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": tool_result_content}
                ],
            },
        ]

    def _translate(self, tool_result_content):
        return _ADAPTER.translate_messages_to_responses_input(self._messages(tool_result_content))

    @staticmethod
    def _input_images(items):
        return [
            part
            for item in items
            if item.get("type") == "message" and item.get("role") == "user"
            for part in item.get("content", [])
            if part.get("type") == "input_image"
        ]

    @staticmethod
    def _image_message(items):
        return next(
            item
            for item in items
            if item.get("type") == "message"
            and any(part.get("type") == "input_image" for part in item.get("content", []))
        )

    def test_base64_image_survives(self):
        items = self._translate(
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}}]
        )

        images = self._input_images(items)
        assert len(images) == 1
        assert images[0]["image_url"] == self.DATA_URI

        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "toolu_01"
        assert "image" in outputs[0]["output"]

    def test_url_image_survives(self):
        items = self._translate([{"type": "image", "source": {"type": "url", "url": self.HTTP_URL}}])

        images = self._input_images(items)
        assert len(images) == 1
        assert images[0]["image_url"] == self.HTTP_URL

    def test_text_and_image_keeps_text_in_output(self):
        items = self._translate(
            [
                {"type": "text", "text": "screenshot saved"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}},
            ]
        )

        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert outputs[0]["output"].startswith("screenshot saved")
        assert len(self._input_images(items)) == 1

    def test_two_images_both_survive(self):
        items = self._translate(
            [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}},
                {"type": "image", "source": {"type": "url", "url": self.HTTP_URL}},
            ]
        )

        images = self._input_images(items)
        assert [img["image_url"] for img in images] == [self.DATA_URI, self.HTTP_URL]

    def test_image_user_message_comes_after_function_call_output(self):
        items = self._translate(
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}}]
        )

        fco_index = next(i for i, item in enumerate(items) if item.get("type") == "function_call_output")
        assert fco_index < items.index(self._image_message(items))

    def test_boundary_text_precedes_hoisted_images(self):
        items = self._translate(
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}}]
        )

        assert self._image_message(items)["content"] == [
            {"type": "input_text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
            {"type": "input_image", "image_url": self.DATA_URI},
        ]

    def test_sibling_user_blocks_stay_out_of_boundary_message(self):
        messages = self._messages(
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.B64_DATA}}]
        )
        messages[-1]["content"].append({"type": "text", "text": "what changed?"})

        items = _ADAPTER.translate_messages_to_responses_input(messages)

        assert self._image_message(items)["content"] == [
            {"type": "input_text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
            {"type": "input_image", "image_url": self.DATA_URI},
        ]
        assert any(
            part == {"type": "input_text", "text": "what changed?"}
            for item in items
            if item.get("type") == "message"
            for part in item.get("content", [])
        )

    def test_text_only_tool_result_unchanged(self):
        items = self._translate([{"type": "text", "text": "plain result"}])

        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert outputs[0]["output"] == "plain result"
        assert self._input_images(items) == []

    def test_image_without_source_dict_keeps_plain_text_output(self):
        items = self._translate(
            [
                {"type": "text", "text": "screenshot saved"},
                {"type": "image", "source": self.HTTP_URL},
            ]
        )

        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert outputs[0]["output"] == "screenshot saved"
        assert self._input_images(items) == []


class TestToolResultDocuments:
    """Documents inside tool_result blocks must survive translation (LIT-6135):
    the function_call_output output becomes a list of parts carrying the joined
    text as input_text and each document as an input_file. Without documents the
    output stays the plain string it always was."""

    PDF_B64 = "JVBERi0xLjQKJSBQT05H"
    PDF_DATA_URI = "data:application/pdf;base64,JVBERi0xLjQKJSBQT05H"
    PDF_URL = "https://example.com/report.pdf"
    PNG_B64 = "iVBORw0KGgoAAAANSUhEUg=="

    def _messages(self, tool_result_content):
        return [
            {"role": "user", "content": "read the pdf"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_01", "name": "read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": tool_result_content}
                ],
            },
        ]

    def _translate(self, tool_result_content):
        return _ADAPTER.translate_messages_to_responses_input(self._messages(tool_result_content))

    @staticmethod
    def _tool_output(items):
        return next(item for item in items if item.get("type") == "function_call_output")["output"]

    def _base64_document(self, **extra):
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": self.PDF_B64},
            **extra,
        }

    def test_text_and_base64_document_produce_part_list(self):
        output = self._tool_output(
            self._translate([{"type": "text", "text": "PDF file read: mystery.pdf"}, self._base64_document()])
        )
        assert output == [
            {"type": "input_text", "text": "PDF file read: mystery.pdf"},
            {"type": "input_file", "filename": "document.pdf", "file_data": self.PDF_DATA_URI},
        ]

    def test_document_only_produces_single_file_part(self):
        output = self._tool_output(self._translate([self._base64_document()]))
        assert output == [{"type": "input_file", "filename": "document.pdf", "file_data": self.PDF_DATA_URI}]

    def test_document_title_becomes_filename(self):
        output = self._tool_output(self._translate([self._base64_document(title="quarterly-report.pdf")]))
        assert output == [
            {"type": "input_file", "filename": "quarterly-report.pdf", "file_data": self.PDF_DATA_URI}
        ]

    def test_url_document_becomes_file_url_part(self):
        output = self._tool_output(
            self._translate([{"type": "document", "source": {"type": "url", "url": self.PDF_URL}}])
        )
        assert output == [{"type": "input_file", "file_url": self.PDF_URL}]

    def test_document_with_empty_data_falls_back_to_string_output(self):
        output = self._tool_output(
            self._translate(
                [
                    {"type": "text", "text": "PDF file read"},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ""}},
                ]
            )
        )
        assert output == "PDF file read"

    def test_document_without_source_dict_keeps_string_output(self):
        output = self._tool_output(
            self._translate([{"type": "text", "text": "stub"}, {"type": "document", "source": self.PDF_URL}])
        )
        assert output == "stub"

    def test_text_only_tool_result_keeps_plain_string_output(self):
        output = self._tool_output(self._translate([{"type": "text", "text": "plain result"}]))
        assert output == "plain result"

    def test_file_id_source_document_keeps_string_output(self):
        output = self._tool_output(
            self._translate(
                [
                    {"type": "text", "text": "stub"},
                    {"type": "document", "source": {"type": "file", "file_id": "file_abc123"}},
                ]
            )
        )
        assert output == "stub"

    def test_url_source_without_url_keeps_string_output(self):
        output = self._tool_output(
            self._translate([{"type": "text", "text": "stub"}, {"type": "document", "source": {"type": "url"}}])
        )
        assert output == "stub"

    def test_text_image_and_document_mix(self):
        items = self._translate(
            [
                {"type": "text", "text": "captured"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": self.PNG_B64}},
                self._base64_document(),
            ]
        )

        output = self._tool_output(items)
        assert output == [
            {"type": "input_text", "text": f"captured\n{TOOL_RESULT_IMAGE_PLACEHOLDER}"},
            {"type": "input_file", "filename": "document.pdf", "file_data": self.PDF_DATA_URI},
        ]

        image_message = next(
            item
            for item in items
            if item.get("type") == "message"
            and any(part.get("type") == "input_image" for part in item.get("content", []))
        )
        assert image_message["content"] == [
            {"type": "input_text", "text": TOOL_RESULT_IMAGE_BOUNDARY},
            {"type": "input_image", "image_url": f"data:image/png;base64,{self.PNG_B64}"},
        ]


class TestUserContentDocuments:
    """Documents in plain user content must survive translation (LIT-6144): each
    document block becomes an input_file part of the user message, in block order,
    exactly like image blocks become input_image parts. Untranslatable documents
    are dropped without disturbing the surrounding parts."""

    PDF_B64 = "JVBERi0xLjQKJSBQT05H"
    PDF_DATA_URI = "data:application/pdf;base64,JVBERi0xLjQKJSBQT05H"
    PDF_URL = "https://example.com/report.pdf"
    EXPLICIT = {"mode": "explicit"}

    def _translate(self, user_content):
        return _ADAPTER.translate_messages_to_responses_input([{"role": "user", "content": user_content}])

    @staticmethod
    def _user_content(items):
        return next(item for item in items if item.get("type") == "message" and item.get("role") == "user")["content"]

    def _base64_document(self, **extra):
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": self.PDF_B64},
            **extra,
        }

    def test_document_then_text_keeps_block_order(self):
        content = self._user_content(
            self._translate([self._base64_document(), {"type": "text", "text": "what does the pdf say?"}])
        )
        assert content == [
            {"type": "input_file", "filename": "document.pdf", "file_data": self.PDF_DATA_URI},
            {"type": "input_text", "text": "what does the pdf say?"},
        ]

    def test_document_title_becomes_filename(self):
        content = self._user_content(self._translate([self._base64_document(title="quarterly-report.pdf")]))
        assert content == [
            {"type": "input_file", "filename": "quarterly-report.pdf", "file_data": self.PDF_DATA_URI}
        ]

    def test_url_document_becomes_file_url_part(self):
        content = self._user_content(
            self._translate([{"type": "document", "source": {"type": "url", "url": self.PDF_URL}}])
        )
        assert content == [{"type": "input_file", "file_url": self.PDF_URL}]

    def test_document_only_content_still_produces_user_message(self):
        content = self._user_content(self._translate([self._base64_document()]))
        assert content == [{"type": "input_file", "filename": "document.pdf", "file_data": self.PDF_DATA_URI}]

    def test_empty_base64_data_drops_only_the_document_part(self):
        content = self._user_content(
            self._translate(
                [
                    {"type": "text", "text": "still here"},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ""}},
                ]
            )
        )
        assert content == [{"type": "input_text", "text": "still here"}]

    def test_non_dict_source_drops_only_the_document_part(self):
        content = self._user_content(
            self._translate([{"type": "text", "text": "still here"}, {"type": "document", "source": self.PDF_URL}])
        )
        assert content == [{"type": "input_text", "text": "still here"}]

    def test_document_breakpoint_rides_on_the_file_part(self):
        content = self._user_content(
            self._translate([self._base64_document(prompt_cache_breakpoint=self.EXPLICIT)])
        )
        assert content == [
            {
                "type": "input_file",
                "filename": "document.pdf",
                "file_data": self.PDF_DATA_URI,
                "prompt_cache_breakpoint": self.EXPLICIT,
            }
        ]


def _contains_key(value, key) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


class TestPromptCacheBreakpointToResponses:
    """OpenAI `prompt_cache_breakpoint` markers ride through the /v1/messages -> Responses bridge (#37509)."""

    EXPLICIT = {"mode": "explicit"}

    def test_system_with_breakpoint_becomes_leading_developer_message(self):
        request = _make_request(
            model="openai/gpt-5.6",
            system=[
                {"type": "text", "text": "Be concise."},
                {"type": "text", "text": "Be helpful.", "prompt_cache_breakpoint": self.EXPLICIT},
            ],
        )
        kwargs = _ADAPTER.translate_request(request)
        assert "instructions" not in kwargs
        assert kwargs["input"] == [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "Be concise."},
                    {"type": "input_text", "text": "Be helpful.", "prompt_cache_breakpoint": self.EXPLICIT},
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        ]

    def test_system_without_breakpoint_still_becomes_instructions(self):
        request = _make_request(system=[{"type": "text", "text": "Be concise."}, {"type": "text", "text": "Be helpful."}])
        kwargs = _ADAPTER.translate_request(request)
        assert kwargs["instructions"] == "Be concise.\nBe helpful."
        assert kwargs["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]

    def test_system_string_still_becomes_instructions(self):
        kwargs = _ADAPTER.translate_request(_make_request(system="Be concise."))
        assert kwargs["instructions"] == "Be concise."
        assert kwargs["input"][0]["role"] == "user"

    def test_system_with_breakpoint_skips_non_text_blocks(self):
        request = _make_request(
            system=[
                {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                {"type": "text", "text": "only", "prompt_cache_breakpoint": self.EXPLICIT},
            ]
        )
        kwargs = _ADAPTER.translate_request(request)
        assert kwargs["input"][0] == {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "only", "prompt_cache_breakpoint": self.EXPLICIT}],
        }

    def test_user_text_and_image_blocks_carry_breakpoint(self):
        items = _ADAPTER.translate_messages_to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look", "prompt_cache_breakpoint": self.EXPLICIT},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/a.png"},
                            "prompt_cache_breakpoint": self.EXPLICIT,
                        },
                    ],
                }
            ]
        )
        assert items == [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look", "prompt_cache_breakpoint": self.EXPLICIT},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/a.png",
                        "prompt_cache_breakpoint": self.EXPLICIT,
                    },
                ],
            }
        ]

    def test_user_blocks_without_breakpoint_are_unchanged(self):
        items = _ADAPTER.translate_messages_to_responses_input(
            [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
        )
        assert items == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look"}]}]

    def test_midturn_system_block_carries_breakpoint(self):
        items = _ADAPTER.translate_messages_to_responses_input(
            [{"role": "system", "content": [{"type": "text", "text": "fix", "prompt_cache_breakpoint": self.EXPLICIT}]}]
        )
        assert items == [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "fix", "prompt_cache_breakpoint": self.EXPLICIT}],
            }
        ]

    def test_assistant_and_tool_result_blocks_drop_breakpoint(self):
        items = _ADAPTER.translate_messages_to_responses_input(
            [
                {"role": "user", "content": [{"type": "text", "text": "q"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a", "prompt_cache_breakpoint": self.EXPLICIT},
                        {"type": "tool_use", "id": "toolu_01", "name": "t", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "r",
                            "prompt_cache_breakpoint": self.EXPLICIT,
                        }
                    ],
                },
            ]
        )
        assert len(items) == 4
        assert not _contains_key(items, "prompt_cache_breakpoint")

    def test_prompt_cache_options_forwarded_to_responses_kwargs(self):
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        kwargs = _build_responses_kwargs(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.6",
            extra_kwargs={"prompt_cache_options": {"mode": "explicit"}},
        )
        assert kwargs["prompt_cache_options"] == {"mode": "explicit"}
