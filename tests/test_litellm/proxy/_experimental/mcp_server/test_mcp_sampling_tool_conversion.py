"""
Tests for MCP sampling handler tool_use / tool_result content conversion.

Verifies that multi-turn tool-calling conversations from upstream MCP
servers are faithfully converted to OpenAI format instead of being
reduced to lossy plain-text stubs.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict

from litellm.proxy._experimental.mcp_server.sampling_handler import (
    _convert_mcp_messages_to_openai,
    _convert_single_content,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight MCP type stand-ins
# ---------------------------------------------------------------------------


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(*, name: str, tool_id: str, input_data: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, id=tool_id, input=input_data)


def _tool_result(
    *, tool_use_id: str, content: Any = None, is_error: bool = False
) -> SimpleNamespace:
    if content is None:
        content = []
    return SimpleNamespace(
        type="tool_result", toolUseId=tool_use_id, content=content, isError=is_error
    )


def _sampling_msg(role: str, content: Any) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


# ---------------------------------------------------------------------------
# _convert_single_content — tool_use
# ---------------------------------------------------------------------------


class TestConvertSingleContentToolUse:
    """Tests for the tool_use branch of _convert_single_content."""

    def test_should_produce_function_call_dict(self):
        """tool_use must produce a proper function-call dict, not a text stub."""
        tu = _tool_use(name="get_weather", tool_id="call_123", input_data={"city": "NYC"})
        result = _convert_single_content(tu)

        assert result["_marker_type"] == "tool_use"
        assert result["type"] == "function"
        assert result["id"] == "call_123"
        assert result["function"]["name"] == "get_weather"
        assert json.loads(result["function"]["arguments"]) == {"city": "NYC"}

    def test_should_not_produce_text_stub(self):
        """Regression: the old code produced '[Tool call: get_weather]'."""
        tu = _tool_use(name="get_weather", tool_id="call_1", input_data={})
        result = _convert_single_content(tu)

        # Must NOT be a text content part
        assert result.get("type") != "text"
        assert "Tool call" not in str(result)

    def test_should_handle_empty_input(self):
        tu = _tool_use(name="no_args_tool", tool_id="call_2", input_data={})
        result = _convert_single_content(tu)

        assert json.loads(result["function"]["arguments"]) == {}


# ---------------------------------------------------------------------------
# _convert_single_content — tool_result
# ---------------------------------------------------------------------------


class TestConvertSingleContentToolResult:
    """Tests for the tool_result branch of _convert_single_content."""

    def test_should_produce_tool_role_message(self):
        """tool_result must produce a tool-role dict, not a text content part."""
        tr = _tool_result(
            tool_use_id="call_123",
            content=[_text("Temperature: 72°F")],
        )
        result = _convert_single_content(tr)

        assert result["_marker_type"] == "tool_result"
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert "72°F" in result["content"]

    def test_should_handle_empty_content(self):
        tr = _tool_result(tool_use_id="call_456", content=[])
        result = _convert_single_content(tr)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_456"
        assert result["content"] == ""

    def test_should_concatenate_multiple_text_parts(self):
        tr = _tool_result(
            tool_use_id="call_789",
            content=[_text("Line 1"), _text("Line 2")],
        )
        result = _convert_single_content(tr)
        assert "Line 1" in result["content"]
        assert "Line 2" in result["content"]


# ---------------------------------------------------------------------------
# _convert_mcp_messages_to_openai — multi-turn tool calling
# ---------------------------------------------------------------------------


class TestConvertMcpMessagesMultiTurnTools:
    """End-to-end tests for multi-turn tool-calling message sequences."""

    def test_should_convert_assistant_tool_use_to_tool_calls_array(self):
        """An assistant message with tool_use content should produce
        a proper tool_calls array, not a text stub."""
        messages = [
            _sampling_msg("assistant", _tool_use(
                name="search", tool_id="call_1", input_data={"query": "LiteLLM"}
            )),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert tc["id"] == "call_1"

    def test_should_convert_user_tool_result_to_tool_role_message(self):
        """A user message with tool_result content should produce
        a separate role='tool' message."""
        messages = [
            _sampling_msg("user", _tool_result(
                tool_use_id="call_1",
                content=[_text("Found 42 results")],
            )),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert "42 results" in msg["content"]

    def test_should_handle_full_tool_calling_round_trip(self):
        """Simulate a complete tool-calling conversation:
        user → assistant(tool_use) → user(tool_result) → assistant(text)
        """
        messages = [
            _sampling_msg("user", _text("What's the weather in NYC?")),
            _sampling_msg("assistant", _tool_use(
                name="get_weather", tool_id="call_w1",
                input_data={"city": "NYC"},
            )),
            _sampling_msg("user", _tool_result(
                tool_use_id="call_w1",
                content=[_text("72°F, sunny")],
            )),
            _sampling_msg("assistant", _text("It's 72°F and sunny in NYC!")),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 4

        # 1. User message
        assert result[0]["role"] == "user"

        # 2. Assistant with tool_calls
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather"

        # 3. Tool result
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_w1"

        # 4. Final assistant text
        assert result[3]["role"] == "assistant"
        assert "72°F" in str(result[3]["content"])

    def test_should_handle_mixed_text_and_tool_use_in_assistant(self):
        """An assistant message with both text and tool_use content."""
        messages = [
            _sampling_msg("assistant", [
                _text("Let me check that for you."),
                _tool_use(name="lookup", tool_id="call_lu1", input_data={"id": 42}),
            ]),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["function"]["name"] == "lookup"
        # Text content should also be present
        assert msg.get("content") is not None

    def test_should_handle_multiple_tool_uses_in_single_message(self):
        """Multiple tool_use items in a single assistant message → multiple tool_calls."""
        messages = [
            _sampling_msg("assistant", [
                _tool_use(name="tool_a", tool_id="call_a", input_data={}),
                _tool_use(name="tool_b", tool_id="call_b", input_data={"x": 1}),
            ]),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        msg = result[0]
        assert len(msg["tool_calls"]) == 2
        names = {tc["function"]["name"] for tc in msg["tool_calls"]}
        assert names == {"tool_a", "tool_b"}

    def test_should_handle_multiple_tool_results_in_single_message(self):
        """Multiple tool_result items in a single user message → multiple tool messages."""
        messages = [
            _sampling_msg("user", [
                _tool_result(tool_use_id="call_a", content=[_text("Result A")]),
                _tool_result(tool_use_id="call_b", content=[_text("Result B")]),
            ]),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 2
        assert all(m["role"] == "tool" for m in result)
        ids = {m["tool_call_id"] for m in result}
        assert ids == {"call_a", "call_b"}

    def test_should_preserve_system_prompt(self):
        """System prompt should still be emitted first."""
        messages = [_sampling_msg("user", _text("Hi"))]
        result = _convert_mcp_messages_to_openai(
            messages, system_prompt="You are helpful."
        )

        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."


# ---------------------------------------------------------------------------
# _convert_mcp_messages_to_openai — marker hoisting on unexpected roles
# ---------------------------------------------------------------------------


class TestConvertMcpMessagesMarkerHoisting:
    """The role-matched fast paths only fire for assistant/tool_use and
    user/tool_result. Content that arrives on an unexpected role must still
    be hoisted to the correct message position by the generic fallback,
    not silently dropped or embedded inline as a content part."""

    def test_should_hoist_tool_use_arriving_on_user_role(self):
        messages = [
            _sampling_msg("user", _tool_use(
                name="search", tool_id="call_1", input_data={"q": "x"}
            )),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_should_hoist_tool_result_arriving_on_assistant_role(self):
        messages = [
            _sampling_msg("assistant", _tool_result(
                tool_use_id="call_1", content=[_text("done")]
            )),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert "done" in result[0]["content"]

    def test_should_keep_text_when_hoisting_tool_use_on_user_role(self):
        messages = [
            _sampling_msg("user", [
                _text("here you go"),
                _tool_use(name="lookup", tool_id="call_2", input_data={}),
            ]),
        ]
        result = _convert_mcp_messages_to_openai(messages)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["function"]["name"] == "lookup"
        assert any(
            isinstance(p, dict) and p.get("text") == "here you go"
            for p in msg["content"]
        )
