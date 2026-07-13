"""
Unit tests for the context_management polyfill dispatcher.
"""

from litellm.llms.anthropic.experimental_pass_through.context_management import (
    apply_context_management,
)

MODEL = "xai/grok-4"


def _history_with_two_tool_pairs():
    return [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "first result",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t2", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": "second result",
                }
            ],
        },
    ]


async def test_unknown_edit_type_is_noop():
    messages = _history_with_two_tool_pairs()
    result = await apply_context_management(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        context_management_spec={
            "edits": [{"type": "totally_not_a_real_edit_20999999"}]
        },
    )
    assert result.applied_edits == []
    assert result.messages == messages


async def test_known_edit_is_applied():
    messages = _history_with_two_tool_pairs()
    result = await apply_context_management(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        context_management_spec={
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "tool_uses", "value": 1},
                    "keep": {"type": "tool_uses", "value": 1},
                }
            ]
        },
    )
    assert len(result.applied_edits) == 1
    assert result.applied_edits[0]["type"] == "clear_tool_uses_20250919"
    assert result.applied_edits[0]["cleared_tool_uses"] == 1


async def test_mixed_known_unknown_only_known_applied():
    messages = _history_with_two_tool_pairs()
    result = await apply_context_management(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        context_management_spec={
            "edits": [
                {"type": "unknown_foo"},
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "tool_uses", "value": 0},
                    "keep": {"type": "tool_uses", "value": 1},
                },
                {"type": "another_unknown"},
            ]
        },
    )
    assert len(result.applied_edits) == 1
    assert result.applied_edits[0]["type"] == "clear_tool_uses_20250919"


async def test_empty_or_missing_edits_list():
    messages = _history_with_two_tool_pairs()
    for spec in [{}, {"edits": None}, {"edits": []}, None]:
        result = await apply_context_management(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            context_management_spec=spec,  # type: ignore[arg-type]
        )
        assert result.applied_edits == []
        assert result.messages == messages


async def test_malformed_edit_entries_are_skipped():
    """Non-dict entries in `edits` list should be silently skipped."""
    messages = _history_with_two_tool_pairs()
    result = await apply_context_management(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        context_management_spec={"edits": ["not a dict", 42, None, {"type": None}]},
    )
    assert result.applied_edits == []
    assert result.messages == messages
