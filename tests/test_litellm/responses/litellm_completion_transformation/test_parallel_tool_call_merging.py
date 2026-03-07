"""
Tests for merging consecutive function_call items into a single assistant message.

When the Responses API input contains multiple consecutive function_call items
(parallel tool calls from one model turn), they must be merged into a single
assistant message with multiple tool_calls entries.  Otherwise, downstream
converters (e.g. Gemini) fail to match tool results to their originating calls.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _make_function_call(call_id: str, name: str, arguments: str = "{}"):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "id": f"fc_{call_id}",
        "status": "completed",
    }


def _make_function_call_output(call_id: str, output: str = '{"ok":true}'):
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def test_parallel_function_calls_merged_into_single_assistant_message():
    """Two consecutive function_call items should produce one assistant message
    with two tool_calls entries."""
    input_items = [
        {"role": "user", "content": "What's the weather in Paris and London?"},
        _make_function_call("call_1", "get_weather", '{"city":"Paris"}'),
        _make_function_call("call_2", "get_weather", '{"city":"London"}'),
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    # Should have: 1 user message + 1 merged assistant message
    assistant_msgs = [m for m in messages if _role(m) == "assistant"]
    assert len(assistant_msgs) == 1, (
        f"Expected 1 assistant message, got {len(assistant_msgs)}"
    )

    tool_calls = _get_tool_calls(assistant_msgs[0])
    assert len(tool_calls) == 2, (
        f"Expected 2 tool_calls in merged message, got {len(tool_calls)}"
    )

    # Check call IDs preserved
    ids = [_tc_id(tc) for tc in tool_calls]
    assert "call_1" in ids
    assert "call_2" in ids


def test_parallel_function_calls_reindexed():
    """Merged tool_calls should have sequential index values."""
    input_items = [
        {"role": "user", "content": "hi"},
        _make_function_call("call_a", "fn_a"),
        _make_function_call("call_b", "fn_b"),
        _make_function_call("call_c", "fn_c"),
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    assistant_msgs = [m for m in messages if _role(m) == "assistant"]
    assert len(assistant_msgs) == 1

    tool_calls = _get_tool_calls(assistant_msgs[0])
    assert len(tool_calls) == 3

    indices = [_tc_index(tc) for tc in tool_calls]
    assert indices == [0, 1, 2], f"Expected [0, 1, 2], got {indices}"


def test_single_function_call_not_affected():
    """A single function_call should still produce one assistant message with
    one tool_call (no regression)."""
    input_items = [
        {"role": "user", "content": "hi"},
        _make_function_call("call_1", "fn_a"),
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    assistant_msgs = [m for m in messages if _role(m) == "assistant"]
    assert len(assistant_msgs) == 1

    tool_calls = _get_tool_calls(assistant_msgs[0])
    assert len(tool_calls) == 1


def test_parallel_calls_with_outputs_produce_correct_structure():
    """Full round-trip: parallel function_calls followed by their outputs.
    Should produce one assistant message (merged) and two tool messages."""
    input_items = [
        {"role": "user", "content": "hi"},
        _make_function_call("call_1", "fn_a"),
        _make_function_call("call_2", "fn_b"),
        _make_function_call_output("call_1", '{"result":"a"}'),
        _make_function_call_output("call_2", '{"result":"b"}'),
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    assistant_msgs = [m for m in messages if _role(m) == "assistant"]
    tool_msgs = [m for m in messages if _role(m) == "tool"]

    assert len(assistant_msgs) == 1, (
        f"Expected 1 merged assistant message, got {len(assistant_msgs)}"
    )
    assert len(tool_msgs) == 2, (
        f"Expected 2 tool messages, got {len(tool_msgs)}"
    )


def test_non_consecutive_function_calls_not_merged():
    """Function calls separated by a non-assistant message should NOT be merged."""
    input_items = [
        {"role": "user", "content": "hi"},
        _make_function_call("call_1", "fn_a"),
        _make_function_call_output("call_1", '{"result":"a"}'),
        # New user turn, then another function call
        {"role": "user", "content": "now do this"},
        _make_function_call("call_2", "fn_b"),
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )

    assistant_msgs = [m for m in messages if _role(m) == "assistant"]
    # Should be 2 separate assistant messages (not merged)
    assert len(assistant_msgs) == 2, (
        f"Expected 2 assistant messages, got {len(assistant_msgs)}"
    )


# ─── helpers ────────────────────────────────────────────────────────────

def _role(m):
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", "")


def _get_tool_calls(m):
    if isinstance(m, dict):
        return m.get("tool_calls") or []
    return getattr(m, "tool_calls", None) or []


def _tc_id(tc):
    return tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")


def _tc_index(tc):
    return tc.get("index") if isinstance(tc, dict) else getattr(tc, "index", None)
