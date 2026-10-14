from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.chat.prompt_cache_breakpoints import (
    CacheBreakpointCapExceeded,
    ClaudeCacheMarks,
    claude_cache_marks,
    reject_prompt_cache_options,
    resolved_claude_cache_marks,
    tools_carry_prompt_cache_breakpoint,
)

EPHEMERAL: Final = {"type": "ephemeral"}
EXPLICIT: Final = {"mode": "explicit"}
EXPLICIT_30M: Final = {"mode": "explicit", "ttl": "30m"}
MODEL: Final = "claude-opus-5-5"


def _text(text: str, **fields: object) -> dict[str, object]:
    return {"type": "text", "text": text, **fields}


def _tool(**fields: object) -> dict[str, object]:
    return {"type": "function", "function": {"name": "lookup", "parameters": {}}, **fields}


def _call(identity: str, **fields: object) -> dict[str, object]:
    return {"id": identity, "type": "function", "function": {"name": "lookup", "arguments": "{}"}, **fields}


def _six_breakpoints() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """A tool definition, a system part, a user part, a tool call, a tool result part and a string message, every
    one marked, in the order Claude reads them"""
    messages: Final = [
        {"role": "system", "content": [_text("rules", prompt_cache_breakpoint=EXPLICIT)]},
        {"role": "user", "content": [_text("hi", prompt_cache_breakpoint=EXPLICIT_30M)]},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_1", prompt_cache_breakpoint=EXPLICIT)]},
        {"role": "tool", "tool_call_id": "call_1", "content": [_text("sunny", prompt_cache_breakpoint=EXPLICIT)]},
        {"role": "user", "content": "done", "prompt_cache_breakpoint": EXPLICIT},
    ]
    return messages, [_tool(prompt_cache_breakpoint=EXPLICIT)]


def test_drop_params_keeps_the_latest_four_marks_in_wire_order() -> None:
    messages, tools = _six_breakpoints()

    marks: Final = claude_cache_marks(messages, tools, automatic_breakpoint=False, drop_params=True)

    assert marks == ClaudeCacheMarks(
        messages=(
            {"role": "system", "content": [_text("rules")]},
            {"role": "user", "content": [_text("hi", cache_control=EPHEMERAL)]},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1", cache_control=EPHEMERAL)]},
            {"role": "tool", "tool_call_id": "call_1", "content": [_text("sunny", cache_control=EPHEMERAL)]},
            {"role": "user", "content": "done", "cache_control": EPHEMERAL},
        ),
        tools=(_tool(),),
    )


def test_four_breakpoints_are_renamed_without_dropping_anything() -> None:
    messages, tools = _six_breakpoints()

    marks: Final = claude_cache_marks(messages[2:], tools, automatic_breakpoint=False, drop_params=False)

    assert marks == ClaudeCacheMarks(
        messages=(
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1", cache_control=EPHEMERAL)]},
            {"role": "tool", "tool_call_id": "call_1", "content": [_text("sunny", cache_control=EPHEMERAL)]},
            {"role": "user", "content": "done", "cache_control": EPHEMERAL},
        ),
        tools=(_tool(cache_control=EPHEMERAL),),
    )


def test_the_automatic_breakpoint_takes_a_slot_and_is_never_dropped() -> None:
    parts: Final = [_text(f"part {index}", prompt_cache_breakpoint=EXPLICIT) for index in range(4)]

    marks: Final = claude_cache_marks(
        [{"role": "user", "content": parts}], None, automatic_breakpoint=True, drop_params=True
    )

    assert marks == ClaudeCacheMarks(
        messages=(
            {
                "role": "user",
                "content": [
                    _text("part 0"),
                    *(_text(f"part {index}", cache_control=EPHEMERAL) for index in range(1, 4)),
                ],
            },
        ),
        tools=(),
    )


def test_without_drop_params_the_excess_is_reported_with_the_count() -> None:
    messages, tools = _six_breakpoints()

    assert claude_cache_marks(messages, tools, automatic_breakpoint=False, drop_params=False) == (
        CacheBreakpointCapExceeded(found=6)
    )


def test_a_request_without_breakpoints_is_forwarded_as_written_even_past_the_cap() -> None:
    messages: Final = [
        {"role": "user", "content": [_text(f"part {index}", cache_control=EPHEMERAL) for index in range(5)]}
    ]
    tools: Final = [_tool(cache_control=EPHEMERAL)]

    marks: Final = claude_cache_marks(messages, tools, automatic_breakpoint=True, drop_params=True)

    assert marks == ClaudeCacheMarks(messages=(messages[0],), tools=(tools[0],))
    assert marks.messages[0] is messages[0]


def test_a_message_level_mark_on_list_content_takes_no_slot_because_claude_never_renders_it() -> None:
    marked_parts: Final = [_text(f"part {index}", prompt_cache_breakpoint=EXPLICIT) for index in range(3)]
    messages: Final = [
        {"role": "user", "content": "first", "prompt_cache_breakpoint": EXPLICIT},
        {"role": "user", "content": [*marked_parts, _text("tail")], "prompt_cache_breakpoint": EXPLICIT},
    ]

    marks: Final = claude_cache_marks(messages, None, automatic_breakpoint=False, drop_params=False)

    assert marks == ClaudeCacheMarks(
        messages=(
            {"role": "user", "content": "first", "cache_control": EPHEMERAL},
            {
                "role": "user",
                "content": [*(_text(f"part {index}", cache_control=EPHEMERAL) for index in range(3)), _text("tail")],
                "cache_control": EPHEMERAL,
            },
        ),
        tools=(),
    )


def test_a_message_level_mark_on_a_tool_result_with_list_content_takes_a_slot() -> None:
    messages: Final = [
        {"role": "user", "content": [_text(f"part {index}", prompt_cache_breakpoint=EXPLICIT) for index in range(4)]},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [_text("sunny")],
            "prompt_cache_breakpoint": EXPLICIT,
        },
    ]

    assert claude_cache_marks(messages, None, automatic_breakpoint=False, drop_params=False) == (
        CacheBreakpointCapExceeded(found=5)
    )


def test_tools_already_mapped_to_the_anthropic_shape_still_arm_the_cap() -> None:
    mapped_tools: Final = [
        {"name": f"tool_{index}", "input_schema": {}, "cache_control": EPHEMERAL} for index in range(4)
    ]
    messages: Final = [{"role": "user", "content": [_text("hi", prompt_cache_breakpoint=EXPLICIT)]}]

    assert claude_cache_marks(
        messages, mapped_tools, automatic_breakpoint=False, drop_params=False, tools_carried_breakpoint=True
    ) == CacheBreakpointCapExceeded(found=5)
    assert claude_cache_marks(
        messages, mapped_tools, automatic_breakpoint=False, drop_params=True, tools_carried_breakpoint=True
    ) == ClaudeCacheMarks(
        messages=({"role": "user", "content": [_text("hi", cache_control=EPHEMERAL)]},),
        tools=({"name": "tool_0", "input_schema": {}}, *mapped_tools[1:]),
    )


def _tool_result_with_two_marked_parts() -> list[dict[str, object]]:
    return [
        {"role": "system", "content": [_text("rules", prompt_cache_breakpoint=EXPLICIT)]},
        {"role": "user", "content": [_text("hi", prompt_cache_breakpoint=EXPLICIT)]},
        {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                _text("sunny", prompt_cache_breakpoint=EXPLICIT),
                _text("warm", prompt_cache_breakpoint=EXPLICIT),
            ],
        },
        {"role": "user", "content": "done", "prompt_cache_breakpoint": EXPLICIT},
    ]


def test_a_tool_result_with_two_marked_parts_takes_one_converse_slot_and_two_anthropic_slots() -> None:
    messages: Final = _tool_result_with_two_marked_parts()

    converse: Final = claude_cache_marks(
        messages, None, automatic_breakpoint=False, drop_params=False, wire="bedrock_converse"
    )

    assert converse == ClaudeCacheMarks(
        messages=(
            {"role": "system", "content": [_text("rules", cache_control=EPHEMERAL)]},
            {"role": "user", "content": [_text("hi", cache_control=EPHEMERAL)]},
            {"role": "assistant", "content": "", "tool_calls": [_call("call_1")]},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [_text("sunny", cache_control=EPHEMERAL), _text("warm", cache_control=EPHEMERAL)],
            },
            {"role": "user", "content": "done", "cache_control": EPHEMERAL},
        ),
        tools=(),
    )
    assert claude_cache_marks(messages, None, automatic_breakpoint=False, drop_params=False) == (
        CacheBreakpointCapExceeded(found=5)
    )


def test_dropping_a_converse_tool_result_slot_clears_every_mark_on_that_message() -> None:
    messages: Final = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                _text("sunny", prompt_cache_breakpoint=EXPLICIT),
                _text("warm", prompt_cache_breakpoint=EXPLICIT),
            ],
            "prompt_cache_breakpoint": EXPLICIT,
        },
        {"role": "user", "content": [_text(f"part {index}", prompt_cache_breakpoint=EXPLICIT) for index in range(4)]},
    ]

    marks: Final = claude_cache_marks(
        messages, None, automatic_breakpoint=False, drop_params=True, wire="bedrock_converse"
    )

    assert marks == ClaudeCacheMarks(
        messages=(
            {"role": "tool", "tool_call_id": "call_1", "content": [_text("sunny"), _text("warm")]},
            {"role": "user", "content": [_text(f"part {index}", cache_control=EPHEMERAL) for index in range(4)]},
        ),
        tools=(),
    )


def test_a_function_level_cache_control_takes_no_converse_slot() -> None:
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {}, "cache_control": EPHEMERAL}}]
    messages: Final = [
        {"role": "user", "content": [_text(f"part {index}", prompt_cache_breakpoint=EXPLICIT) for index in range(4)]}
    ]

    converse: Final = claude_cache_marks(
        messages, tools, automatic_breakpoint=False, drop_params=False, wire="bedrock_converse"
    )

    assert converse == ClaudeCacheMarks(
        messages=(
            {"role": "user", "content": [_text(f"part {index}", cache_control=EPHEMERAL) for index in range(4)]},
        ),
        tools=(tools[0],),
    )
    assert claude_cache_marks(messages, tools, automatic_breakpoint=False, drop_params=False) == (
        CacheBreakpointCapExceeded(found=5)
    )


def test_the_cap_error_is_a_public_bad_request_naming_drop_params() -> None:
    with pytest.raises(litellm.BadRequestError) as caught:
        resolved_claude_cache_marks(CacheBreakpointCapExceeded(found=6), model=MODEL, llm_provider="anthropic")

    assert "Claude accepts at most 4 cache breakpoints, found 6" in str(caught.value)
    assert "drop_params=True" in str(caught.value)
    assert caught.value.status_code == 400
    assert caught.value.llm_provider == "anthropic"


def test_marks_within_the_cap_resolve_to_themselves() -> None:
    marks: Final = ClaudeCacheMarks(messages=(), tools=())

    assert resolved_claude_cache_marks(marks, model=MODEL, llm_provider="anthropic") is marks


def test_prompt_cache_options_are_rejected_unless_drop_params() -> None:
    with pytest.raises(litellm.UnsupportedParamsError) as caught:
        reject_prompt_cache_options(model=MODEL, options={"mode": "explicit"}, drop_params=False)

    assert "prompt_cache_options" in str(caught.value)
    assert "drop_params" in str(caught.value)
    assert caught.value.status_code == 400
    assert reject_prompt_cache_options(model=MODEL, options={"mode": "explicit"}, drop_params=True) is None
    assert reject_prompt_cache_options(model=MODEL, options=None, drop_params=False) is None


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        ([_tool(prompt_cache_breakpoint=EXPLICIT)], True),
        ([{"type": "function", "function": {"name": "lookup", "prompt_cache_breakpoint": EXPLICIT}}], True),
        ((_tool(prompt_cache_breakpoint=EXPLICIT),), True),
        ([_tool(cache_control=EPHEMERAL), _tool()], False),
        (None, False),
        ("not a list", False),
    ],
    ids=["top-level", "under-function", "tuple", "cache-control-only", "none", "opaque"],
)
def test_tools_carry_prompt_cache_breakpoint_reads_both_tool_shapes(tools: object, expected: bool) -> None:
    assert tools_carry_prompt_cache_breakpoint(tools) is expected
