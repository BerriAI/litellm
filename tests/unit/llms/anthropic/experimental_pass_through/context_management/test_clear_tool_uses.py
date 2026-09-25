"""
Unit tests for the in-gateway `clear_tool_uses_20250919` polyfill editor.
"""

from copy import deepcopy

from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
    CLEARED_TOOL_RESULT_PLACEHOLDER,
)
from litellm.llms.anthropic.experimental_pass_through.context_management.editors.clear_tool_uses import (
    apply_clear_tool_uses_20250919,
)

MODEL = "xai/grok-4"


def _make_pair(tool_use_id: str, result_text: str, location: str = "Mumbai"):
    """Return an (assistant, user) message pair with one tool_use + tool_result."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": "get_weather",
                "input": {"location": location},
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
            }
        ],
    }
    return assistant_msg, user_msg


def _make_history(n_pairs: int, result_filler: str = "x" * 200):
    messages = [{"role": "user", "content": "Compare weather across cities."}]
    for i in range(n_pairs):
        assistant_msg, user_msg = _make_pair(
            tool_use_id=f"toolu_{i:02d}",
            result_text=f"Result {i}: {result_filler}",
            location=f"City{i}",
        )
        messages.append(assistant_msg)
        messages.append(user_msg)
    return messages


def test_below_trigger_returns_unchanged():
    """If trigger threshold isn't exceeded, editor is a no-op."""
    messages = _make_history(n_pairs=2)
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 10_000_000},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is None
    assert new_messages == original


def test_keep_preserves_most_recent_pairs():
    """With keep=2 and 5 pairs, the 3 oldest pairs are cleared."""
    messages = _make_history(n_pairs=5)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 2},
        },
    )
    assert applied is not None
    assert applied["type"] == "clear_tool_uses_20250919"
    assert applied["cleared_tool_uses"] == 3

    # Tool results for the first 3 pairs should be the placeholder, last 2 untouched.
    cleared_ids = {"toolu_00", "toolu_01", "toolu_02"}
    kept_ids = {"toolu_03", "toolu_04"}
    for msg in new_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            if block["tool_use_id"] in cleared_ids:
                assert block["content"] == CLEARED_TOOL_RESULT_PLACEHOLDER
            elif block["tool_use_id"] in kept_ids:
                assert "Result" in block["content"]


def test_tool_use_input_is_not_cleared():
    """clear_tool_inputs defaults to false — tool_use.input must remain intact."""
    messages = _make_history(n_pairs=3)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    # Every tool_use block still has its original `input`.
    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "tool_use":
                assert block["input"] == {"location": block["input"]["location"]}
                assert block["input"]["location"].startswith("City")


def test_message_array_length_and_roles_preserved():
    messages = _make_history(n_pairs=4)
    original_roles = [m["role"] for m in messages]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    assert len(new_messages) == len(messages)
    assert [m["role"] for m in new_messages] == original_roles


def test_defaults_applied_when_knobs_omitted():
    """No trigger/keep specified — defaults are 100k input_tokens / 3 tool_uses."""
    messages = _make_history(n_pairs=2)
    # Below 100k tokens; should not fire.
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={"type": "clear_tool_uses_20250919"},
    )
    assert applied is None
    assert new_messages == messages


def test_tool_uses_trigger_variant():
    """Trigger by raw count of tool_use blocks, not tokens."""
    messages = _make_history(n_pairs=4)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 2},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    # 4 total - 1 kept = 3 cleared
    assert applied["cleared_tool_uses"] == 3


def test_cleared_input_tokens_is_nonnegative():
    messages = _make_history(n_pairs=4)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    assert applied["cleared_input_tokens"] >= 0


def test_ignored_knobs_do_not_alter_behavior():
    """clear_at_least / exclude_tools / clear_tool_inputs are accepted but ignored in v0."""
    messages = _make_history(n_pairs=3)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 999_999_999},
            "exclude_tools": ["get_weather"],
            "clear_tool_inputs": True,
        },
    )
    # Despite clear_at_least being huge, polyfill still applies (knob ignored).
    # Despite clear_tool_inputs=True, inputs are NOT cleared (knob ignored).
    assert applied is not None
    assert applied["cleared_tool_uses"] == 2
    # Ignored knobs surface as warnings on the AppliedEdit so operators can
    # see what was dropped (the v0 polyfill silently dropping them at debug
    # log level made misconfiguration invisible from the response).
    assert set(applied.get("warnings", [])) == {
        "clear_at_least_ignored",
        "exclude_tools_ignored",
        "clear_tool_inputs_ignored",
    }


def test_no_ignored_knobs_omits_warnings_field():
    """When the caller doesn't pass any unsupported knobs, no ``warnings`` are added."""
    messages = _make_history(n_pairs=3)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    assert "warnings" not in applied


def test_tool_result_list_content_shape_preserved():
    """When tool_result.content is a list of blocks, replacement returns a list shape."""
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_a", "name": "f", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_a",
                    "content": [{"type": "text", "text": "huge result"}],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_b", "name": "f", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_b",
                    "content": [{"type": "text", "text": "keep me"}],
                }
            ],
        },
    ]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    cleared_block = new_messages[2]["content"][0]
    assert isinstance(cleared_block["content"], list)
    assert cleared_block["content"][0]["type"] == "text"
    assert cleared_block["content"][0]["text"] == CLEARED_TOOL_RESULT_PLACEHOLDER
