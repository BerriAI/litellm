"""
Unit tests for the in-gateway `clear_tool_uses_20250919` polyfill editor.
"""

from copy import deepcopy

import litellm
from litellm.llms.anthropic.pass_through.context_management.constants import (
    CLEARED_TOOL_RESULT_PLACEHOLDER,
)
from litellm.llms.anthropic.pass_through.context_management.editors.clear_tool_uses import (
    apply_clear_tool_uses_20250919,
)

MODEL = "xai/grok-4"


def _make_pair(tool_use_id: str, result_text: str, location: str = "Mumbai", name: str = "get_weather"):
    """Return an (assistant, user) message pair with one tool_use + tool_result."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
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


def _result_contents(messages):
    """tool_use_id -> tool_result content."""
    return {
        block["tool_use_id"]: block["content"]
        for msg in messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if block.get("type") == "tool_result"
    }


def _cleared_ids(messages):
    return {
        tool_use_id
        for tool_use_id, content in _result_contents(messages).items()
        if content == CLEARED_TOOL_RESULT_PLACEHOLDER
    }


def _result_tokens(text: str) -> int:
    return litellm.token_counter(model=MODEL, text=text)


def test_clear_at_least_clears_in_batches_and_leaves_an_incomplete_batch():
    """Oldest first, a batch closes once it reaches clear_at_least; the rest waits."""
    messages = _make_history(n_pairs=8)
    one_result = _result_tokens(_result_contents(messages)["toolu_00"])
    edit_spec = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 0},
        "keep": {"type": "tool_uses", "value": 1},
        # Three results fill a batch; two don't.
        "clear_at_least": {"type": "input_tokens", "value": 2 * one_result + 1},
    }
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL, messages=messages, tools=None, system=None, edit_spec=edit_spec
    )
    # 7 eligible: two batches of 3; the 7th waits for its batch to fill.
    assert applied is not None
    assert applied["cleared_tool_uses"] == 6
    assert _cleared_ids(new_messages) == {f"toolu_{i:02d}" for i in range(6)}
    assert "warnings" not in applied


def test_clear_at_least_keeps_the_cleared_prefix_stable_between_requests():
    """A new tool use doesn't move the cleared prefix until the next batch fills (prompt-cache friendly)."""
    history = _make_history(n_pairs=10)
    one_result = _result_tokens(_result_contents(history)["toolu_00"])
    edit_spec = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 0},
        "keep": {"type": "tool_uses", "value": 1},
        "clear_at_least": {"type": "input_tokens", "value": 2 * one_result + 1},
    }
    cleared = []
    for n_pairs in (8, 9, 10):
        new_messages, _ = apply_clear_tool_uses_20250919(
            model=MODEL, messages=history[: 1 + 2 * n_pairs], tools=None, system=None, edit_spec=edit_spec
        )
        cleared.append(_cleared_ids(new_messages))
    assert cleared[0] == cleared[1] == {f"toolu_{i:02d}" for i in range(6)}
    assert cleared[2] == {f"toolu_{i:02d}" for i in range(9)}


def test_clear_at_least_out_of_reach_leaves_messages_unchanged():
    messages = _make_history(n_pairs=3)
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 999_999_999},
        },
    )
    assert applied is None
    assert new_messages == original


def test_exclude_tools_are_never_cleared_and_do_not_count_toward_keep():
    """keep counts clearable uses only: with keep=1 the newest get_weather result stays, not the newest search."""
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i, name in enumerate(["get_weather", "search", "get_weather", "get_weather", "search"]):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, name=name))
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["search"],
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_02"}
    assert applied["cleared_tool_uses"] == 2


def test_history_of_only_excluded_tools_is_left_unchanged():
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i in range(4):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, name="search"))
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["search"],
        },
    )
    assert applied is None
    assert new_messages == original


def _tool_inputs(messages):
    return {
        block["id"]: block["input"]
        for msg in messages
        if msg["role"] == "assistant"
        for block in msg["content"]
        if block.get("type") == "tool_use"
    }


def test_clear_tool_inputs_true_empties_the_inputs_of_cleared_uses_only():
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
            "clear_tool_inputs": True,
        },
    )
    assert applied is not None
    assert _tool_inputs(new_messages) == {"toolu_00": {}, "toolu_01": {}, "toolu_02": {"location": "City2"}}
    # The caller's messages are not mutated.
    assert _tool_inputs(messages)["toolu_00"] == {"location": "City0"}


def test_clear_tool_inputs_list_clears_only_the_named_tools():
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i, name in enumerate(["get_weather", "search", "get_weather"]):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, location=f"City{i}", name=name))
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": ["search"],
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01"}
    assert _tool_inputs(new_messages) == {
        "toolu_00": {"location": "City0"},
        "toolu_01": {},
        "toolu_02": {"location": "City2"},
    }


def test_invalid_knobs_are_ignored_with_warnings():
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
            "clear_at_least": {"type": "tool_uses", "value": 5},
            "exclude_tools": "get_weather",
            "clear_tool_inputs": "yes",
        },
    )
    assert applied is not None
    assert applied["cleared_tool_uses"] == 2
    assert _tool_inputs(new_messages)["toolu_00"] == {"location": "City0"}
    assert applied["warnings"] == ["clear_at_least_invalid", "exclude_tools_invalid", "clear_tool_inputs_invalid"]


def test_valid_spec_omits_warnings_field():
    """A spec without invalid fields gets no ``warnings``."""
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
            "content": [{"type": "tool_use", "id": "toolu_a", "name": "f", "input": {}}],
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
            "content": [{"type": "tool_use", "id": "toolu_b", "name": "f", "input": {}}],
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
