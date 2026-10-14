from typing import Final

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.anthropic.chat.prompt_cache_breakpoints import (
    ClaudePromptCache,
    UnsupportedPromptCacheParam,
    claude_prompt_cache,
    resolved_claude_prompt_cache,
)

_EXPLICIT: Final = {"mode": "explicit"}
_EPHEMERAL: Final = {"type": "ephemeral"}
_ONE_HOUR: Final = {"type": "ephemeral", "ttl": "1h"}


def _cache(
    messages: list[object],
    tools: list[object] | None = None,
    params: dict[str, object] | None = None,
    *,
    drop_params: bool = False,
) -> ClaudePromptCache | UnsupportedPromptCacheParam:
    return claude_prompt_cache(messages, tools, params or {}, drop_params=drop_params)


def _resolved(
    messages: list[object],
    tools: list[object] | None = None,
    params: dict[str, object] | None = None,
    *,
    drop_params: bool = False,
) -> ClaudePromptCache:
    outcome: Final = _cache(messages, tools, params, drop_params=drop_params)
    assert isinstance(outcome, ClaudePromptCache), outcome
    return outcome


def _part(text: str, **marks: object) -> dict[str, object]:
    return {"type": "text", "text": text, **marks}


def test_breakpoints_become_cache_control_on_every_block_that_carries_one():
    messages: Final = [
        {"role": "system", "content": [_part("sys", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": "hi", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "prompt_cache_breakpoint": _EXPLICIT}],
        },
    ]
    tools: Final = [
        {"type": "function", "function": {"name": "a"}, "prompt_cache_breakpoint": _EXPLICIT},
        {"type": "function", "function": {"name": "b", "prompt_cache_breakpoint": _EXPLICIT}},
    ]

    result: Final = _resolved(messages, tools)

    assert list(result.messages) == [
        {"role": "system", "content": [_part("sys", cache_control=_ONE_HOUR)]},
        {"role": "user", "content": "hi", "cache_control": _ONE_HOUR},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "cache_control": _EPHEMERAL}],
        },
    ]
    assert list(result.tools or ()) == [
        {"type": "function", "function": {"name": "a"}, "cache_control": _ONE_HOUR},
        {"type": "function", "function": {"name": "b"}, "cache_control": _ONE_HOUR},
    ]


def test_a_request_without_breakpoints_is_returned_as_written():
    messages: Final = [{"role": "user", "content": [_part("hi", cache_control=_EPHEMERAL)] * 5}]

    result: Final = _resolved(messages, None, {"temperature": 0.1}, drop_params=True)

    assert result == ClaudePromptCache(tuple(messages), None)


def test_a_client_cache_control_on_the_same_block_wins():
    hour: Final = {"type": "ephemeral", "ttl": "1h"}
    messages: Final = [
        {"role": "user", "content": [_part("hi", cache_control=hour, prompt_cache_breakpoint={"ttl": "9d"})]}
    ]

    assert list(_resolved(messages).messages) == [{"role": "user", "content": [_part("hi", cache_control=hour)]}]


@pytest.mark.parametrize(
    "marker, kept",
    [({"mode": "explicit", "ttl": "30m"}, _EPHEMERAL), ({"mode": "sometimes"}, None), ("yes", None)],
)
def test_a_breakpoint_claude_cannot_honor_is_reported_or_adjusted_under_drop_params(marker: object, kept: object):
    messages: Final = [{"role": "user", "content": "hi", "prompt_cache_breakpoint": marker}]

    problem: Final = _cache(messages)
    dropped: Final = _resolved(messages, drop_params=True)

    assert isinstance(problem, UnsupportedPromptCacheParam)
    assert (problem.param, problem.value) == ("prompt_cache_breakpoint", marker)
    assert list(dropped.messages) == [{"role": "user", "content": "hi", **({"cache_control": kept} if kept else {})}]


def test_an_unsupported_breakpoint_raises_a_400_naming_drop_params():
    problem: Final = _cache(
        [{"role": "user", "content": "hi", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"}}]
    )

    with pytest.raises(UnsupportedParamsError) as raised:
        resolved_claude_prompt_cache(problem, model="claude-x", llm_provider="anthropic")

    assert raised.value.status_code == 400
    assert "30m" in raised.value.message and "drop_params" in raised.value.message


def test_drop_params_drops_the_earliest_breakpoints_and_keeps_every_mark_the_client_wrote():
    tools: Final = [{"type": "function", "function": {"name": "t"}, "prompt_cache_breakpoint": _EXPLICIT}]
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(3)]},
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
    ]

    result: Final = _resolved(messages, tools, {"cache_control": _EPHEMERAL}, drop_params=True)

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}}]
    assert list(result.messages) == [
        {
            "role": "user",
            "content": [_part("0"), _part("1", cache_control=_EPHEMERAL), _part("2", cache_control=_EPHEMERAL)],
        },
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
    ]


def test_marks_the_client_wrote_past_the_cap_reach_claude_as_written_under_drop_params():
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(5)]},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s")]},
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(5)]},
    ]


@pytest.mark.parametrize(
    ("later", "params"),
    [
        ({"cache_control": _ONE_HOUR}, {}),
        ({"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}}, {}),
        ({}, {"cache_control": _ONE_HOUR}),
    ],
    ids=["client-mark", "breakpoint", "top-level-cache-control"],
)
def test_a_breakpoint_read_before_a_one_hour_mark_is_sent_as_one_hour(
    later: dict[str, object], params: dict[str, object]
):
    messages: Final = [
        {"role": "user", "content": [_part("doc", prompt_cache_breakpoint=_EXPLICIT), _part("ask", **later)]},
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
    ]

    result: Final = _resolved(messages, params=params)

    ask: Final = _part("ask", cache_control=_ONE_HOUR) if later else _part("ask")
    assert list(result.messages) == [
        {"role": "user", "content": [_part("doc", cache_control=_ONE_HOUR), ask]},
        {"role": "system", "content": [_part("s", cache_control=_ONE_HOUR)]},
    ]


def test_a_breakpoint_read_after_the_last_one_hour_mark_keeps_the_default_ttl():
    messages: Final = [
        {"role": "system", "content": [_part("s", cache_control=_ONE_HOUR)]},
        {"role": "user", "content": [_part("ask", prompt_cache_breakpoint=_EXPLICIT)]},
    ]

    result: Final = _resolved(messages)

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=_ONE_HOUR)]},
        {"role": "user", "content": [_part("ask", cache_control=_EPHEMERAL)]},
    ]


@pytest.mark.parametrize(
    ("system", "ask", "as_written", "under_drop_params"),
    [
        (
            {"cache_control": _EPHEMERAL},
            {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
            ({"cache_control": _EPHEMERAL}, {"cache_control": _ONE_HOUR}),
            ({"cache_control": _EPHEMERAL}, {"cache_control": _EPHEMERAL}),
        ),
        (
            {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "5m"}},
            {"cache_control": _ONE_HOUR},
            ({"cache_control": {"type": "ephemeral", "ttl": "5m"}}, {"cache_control": _ONE_HOUR}),
            ({"cache_control": _ONE_HOUR}, {"cache_control": _ONE_HOUR}),
        ),
    ],
    ids=["one-hour-breakpoint-after-a-five-minute-mark", "five-minute-breakpoint-before-a-one-hour-mark"],
)
@pytest.mark.parametrize("drop_params", [False, True])
def test_a_ttl_the_client_set_out_of_claude_order_is_sent_as_written_or_dropped_under_drop_params(
    system: dict[str, object],
    ask: dict[str, object],
    as_written: tuple[dict[str, object], dict[str, object]],
    under_drop_params: tuple[dict[str, object], dict[str, object]],
    drop_params: bool,
):
    messages: Final = [
        {"role": "system", "content": [_part("s", **system)]},
        {"role": "user", "content": [_part("ask", **ask)]},
    ]

    result: Final = _resolved(messages, drop_params=drop_params)

    expected_system, expected_ask = under_drop_params if drop_params else as_written
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", **expected_system)]},
        {"role": "user", "content": [_part("ask", **expected_ask)]},
    ]


def test_without_drop_params_more_than_four_marks_are_sent_for_claude_to_answer():
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(5)]}
    ]

    result: Final = _resolved(messages)

    assert list(result.messages) == [
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(5)]}
    ]


def test_a_message_level_breakpoint_on_list_content_takes_no_slot_under_drop_params():
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part("u", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "assistant", "content": [_part("a", prompt_cache_breakpoint=_EXPLICIT)]},
        {
            "role": "user",
            "content": [_part("q", prompt_cache_breakpoint=_EXPLICIT)],
            "prompt_cache_breakpoint": _EXPLICIT,
        },
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert result.messages[0] == {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]}


def test_a_message_level_breakpoint_on_a_tool_result_counts_toward_the_cap():
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(4)]},
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r")], "prompt_cache_breakpoint": _EXPLICIT},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages) == [
        {
            "role": "user",
            "content": [_part("0"), *(_part(str(index), cache_control=_EPHEMERAL) for index in range(1, 4))],
        },
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r")], "cache_control": _EPHEMERAL},
    ]


def test_a_tool_marked_at_both_levels_takes_one_slot_under_drop_params():
    tools: Final = [
        {"type": "function", "function": {"name": "t", "cache_control": _EPHEMERAL}, "cache_control": _EPHEMERAL}
    ]
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(3)]}
    ]

    result: Final = _resolved(messages, tools, drop_params=True)

    assert list(result.messages) == [
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(3)]}
    ]


def test_a_null_tool_level_breakpoint_leaves_the_function_breakpoint_in_force():
    tools: Final = [
        {
            "type": "function",
            "prompt_cache_breakpoint": None,
            "function": {"name": "t", "prompt_cache_breakpoint": _EXPLICIT},
        }
    ]

    result: Final = _resolved([{"role": "user", "content": "hi"}], tools)

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}, "cache_control": _EPHEMERAL}]
