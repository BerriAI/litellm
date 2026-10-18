from typing import Final

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.anthropic.chat.prompt_cache_breakpoints import (
    ClaudePromptCache,
    ClaudeRoute,
    UnsupportedPromptCacheParam,
    claude_prompt_cache,
    resolved_claude_prompt_cache,
)
from litellm.types.llms.openai import ChatCompletionCachedContent

_EXPLICIT: Final = {"mode": "explicit"}
_EPHEMERAL: Final = {"type": "ephemeral"}
_ONE_HOUR: Final = {"type": "ephemeral", "ttl": "1h"}


def _cache(
    messages: list[object],
    tools: list[object] | None = None,
    params: dict[str, object] | None = None,
    *,
    drop_params: bool = False,
    tool_config_cache_control: ChatCompletionCachedContent | None = None,
    route: ClaudeRoute = "anthropic",
    adds_a_tool: bool = False,
    pairs_unanswered_tool_calls: bool = False,
) -> ClaudePromptCache | UnsupportedPromptCacheParam:
    return claude_prompt_cache(
        messages,
        tools,
        params or {},
        drop_params=drop_params,
        tool_history_as_text=False,
        tool_config_cache_control=tool_config_cache_control,
        route=route,
        adds_a_tool=adds_a_tool,
        pairs_unanswered_tool_calls=pairs_unanswered_tool_calls,
    )


def _resolved(
    messages: list[object],
    tools: list[object] | None = None,
    params: dict[str, object] | None = None,
    *,
    drop_params: bool = False,
    tool_config_cache_control: ChatCompletionCachedContent | None = None,
    route: ClaudeRoute = "anthropic",
    adds_a_tool: bool = False,
    pairs_unanswered_tool_calls: bool = False,
) -> ClaudePromptCache:
    outcome: Final = _cache(
        messages,
        tools,
        params,
        drop_params=drop_params,
        tool_config_cache_control=tool_config_cache_control,
        route=route,
        adds_a_tool=adds_a_tool,
        pairs_unanswered_tool_calls=pairs_unanswered_tool_calls,
    )
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
    messages: Final = [
        {
            "role": "user",
            "content": [
                _part("hi", cache_control=_ONE_HOUR, prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"})
            ],
        }
    ]

    assert list(_resolved(messages).messages) == [{"role": "user", "content": [_part("hi", cache_control=_ONE_HOUR)]}]


_THIRTY_MINUTES: Final = {"mode": "explicit", "ttl": "30m"}


@pytest.mark.parametrize(
    ("messages", "tools"),
    [
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [_part("hi", cache_control=_EPHEMERAL, prompt_cache_breakpoint=_THIRTY_MINUTES)],
                }
            ],
            None,
            id="same-block",
        ),
        pytest.param(
            [{"role": "user", "content": "hi"}],
            [
                {
                    "type": "function",
                    "cache_control": _EPHEMERAL,
                    "function": {"name": "t", "prompt_cache_breakpoint": _THIRTY_MINUTES},
                }
            ],
            id="function-under-a-marked-tool",
        ),
        pytest.param(
            [{"role": "user", "content": "hi"}],
            [
                {
                    "type": "function",
                    "prompt_cache_breakpoint": _THIRTY_MINUTES,
                    "function": {"name": "t", "cache_control": _EPHEMERAL},
                }
            ],
            id="tool-over-a-marked-function",
        ),
        pytest.param(
            [
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "cache_control": _EPHEMERAL,
                    "content": [_part("r0", prompt_cache_breakpoint=_THIRTY_MINUTES)],
                }
            ],
            None,
            id="part-of-a-marked-tool-result",
        ),
        pytest.param(
            [
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "prompt_cache_breakpoint": _THIRTY_MINUTES,
                    "content": [_part("r0", cache_control=_EPHEMERAL)],
                }
            ],
            None,
            id="tool-result-with-a-marked-part",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "prompt_cache_breakpoint": _THIRTY_MINUTES,
                    "content": [_part("u0", cache_control=_EPHEMERAL)],
                }
            ],
            None,
            id="message-whose-last-part-is-marked",
        ),
    ],
)
def test_a_breakpoint_a_client_cache_control_wins_over_is_still_reported(
    messages: list[object], tools: list[object] | None
):
    outcome: Final = _cache(messages, tools)

    assert isinstance(outcome, UnsupportedPromptCacheParam)
    assert outcome.value == _THIRTY_MINUTES


@pytest.mark.parametrize(
    ("pairs_unanswered_tool_calls", "system_mark"),
    [(False, {"cache_control": _EPHEMERAL}), (True, {})],
    ids=["dropped_unanswered", "answered_with_a_placeholder"],
)
def test_a_breakpoint_on_a_tool_search_call_nobody_answered_takes_a_slot_only_when_the_route_answers_it(
    pairs_unanswered_tool_calls: bool, system_mark: dict[str, object]
):
    search_call: Final = {
        "id": "srvtoolu_01search",
        "type": "function",
        "function": {"name": "tool_search_tool_regex", "arguments": '{"query": "weather"}'},
        "prompt_cache_breakpoint": _EXPLICIT,
    }
    messages: Final = [
        {"role": "system", "content": [_part("sys", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part(f"note {n}", prompt_cache_breakpoint=_EXPLICIT) for n in range(3)]},
        {"role": "assistant", "content": None, "tool_calls": [search_call]},
        {"role": "user", "content": "Which section mentions watchlists?"},
    ]

    result: Final = _resolved(messages, drop_params=True, pairs_unanswered_tool_calls=pairs_unanswered_tool_calls)

    assert result.messages[0] == {"role": "system", "content": [_part("sys", **system_mark)]}


@pytest.mark.parametrize(
    ("route", "system_mark"),
    [("anthropic", _EPHEMERAL), ("converse", _ONE_HOUR)],
    ids=["route_sends_function_marks", "route_drops_function_marks"],
)
def test_a_five_minute_mark_under_a_tools_function_holds_back_an_hour_only_on_a_route_that_sends_it(
    route: ClaudeRoute, system_mark: dict[str, object]
):
    five_minutes: Final = {"type": "ephemeral", "ttl": "5m"}
    tool: Final = {"type": "function", "function": {"name": "t", "cache_control": five_minutes}}

    result: Final = _resolved(
        [
            {"role": "system", "content": [_part("sys", prompt_cache_breakpoint=_EXPLICIT)]},
            {"role": "user", "content": [_part("ask", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
        ],
        [tool],
        route=route,
    )

    assert result.messages[0] == {"role": "system", "content": [_part("sys", cache_control=system_mark)]}
    assert list(result.tools or ()) == [tool]


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
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part("doc", prompt_cache_breakpoint=_EXPLICIT), _part("ask", **later)]},
    ]

    result: Final = _resolved(messages, params=params)

    ask: Final = _part("ask", cache_control=_ONE_HOUR) if later else _part("ask")
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=_ONE_HOUR)]},
        {"role": "user", "content": [_part("doc", cache_control=_ONE_HOUR), ask]},
    ]


def test_a_breakpoint_read_before_a_later_system_messages_one_hour_mark_is_sent_as_one_hour():
    messages: Final = [
        {"role": "system", "content": [_part("s")]},
        {"role": "user", "content": [_part("doc", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "system", "content": [_part("reminder", cache_control=_ONE_HOUR)]},
        {"role": "user", "content": [_part("ask")]},
    ]

    result: Final = _resolved(messages)

    assert list(result.messages)[1] == {"role": "user", "content": [_part("doc", cache_control=_ONE_HOUR)]}


def test_drop_params_drops_a_breakpoint_read_before_a_later_system_message_first():
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(4)]},
        {"role": "system", "content": [_part("reminder", prompt_cache_breakpoint=_EXPLICIT)]},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages) == [
        {
            "role": "user",
            "content": [_part("0"), *(_part(str(index), cache_control=_EPHEMERAL) for index in (1, 2, 3))],
        },
        {"role": "system", "content": [_part("reminder", cache_control=_EPHEMERAL)]},
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


def test_a_message_level_breakpoint_on_list_content_ending_in_a_marked_part_takes_no_extra_slot_under_drop_params():
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


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_a_message_level_breakpoint_on_list_content_reaches_claude_on_its_last_part(role: str):
    messages: Final = [{"role": role, "content": [_part("a"), _part("b")], "prompt_cache_breakpoint": _EXPLICIT}]

    result: Final = _resolved(messages)

    assert list(result.messages) == [{"role": role, "content": [_part("a"), _part("b", cache_control=_EPHEMERAL)]}]


def test_a_message_level_breakpoint_on_a_developer_message_reaches_claude_on_its_text():
    messages: Final = [{"role": "developer", "content": "rules", "prompt_cache_breakpoint": _EXPLICIT}]

    result: Final = _resolved(messages)

    assert list(result.messages) == [{"role": "developer", "content": [_part("rules", cache_control=_EPHEMERAL)]}]


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


def test_a_tool_breakpoint_beside_a_function_cache_control_frees_a_slot_claude_still_reads_under_drop_params():
    tools: Final = [
        {
            "type": "function",
            "prompt_cache_breakpoint": _EXPLICIT,
            "function": {"name": "t", "cache_control": _EPHEMERAL},
        }
    ]
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(3)]},
    ]

    result: Final = _resolved(messages, tools, drop_params=True)

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t", "cache_control": _EPHEMERAL}}]
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s")]},
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(3)]},
    ]


def test_a_message_level_breakpoint_on_a_tool_call_turn_without_content_reaches_claude_on_its_last_call():
    calls: Final = [
        {"id": f"call_{index}", "type": "function", "function": {"name": "t", "arguments": "{}"}} for index in range(2)
    ]
    messages: Final = [
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": None, "tool_calls": calls, "prompt_cache_breakpoint": _EXPLICIT},
    ]

    result: Final = _resolved(messages)

    assert result.messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [calls[0], {**calls[1], "cache_control": _EPHEMERAL}],
    }


def test_a_breakpoint_on_assistant_text_read_before_a_one_hour_tool_call_mark_is_sent_as_one_hour():
    messages: Final = [
        {"role": "user", "content": "Look it up."},
        {
            "role": "assistant",
            "content": "Checking.",
            "prompt_cache_breakpoint": _EXPLICIT,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "t", "arguments": "{}"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "blue"},
    ]

    result: Final = _resolved(messages)

    assistant: Final = result.messages[1]
    assert isinstance(assistant, dict)
    assert assistant["cache_control"] == _ONE_HOUR
    assert assistant["tool_calls"][0]["cache_control"] == _ONE_HOUR


@pytest.mark.parametrize(
    ("message_mark", "part_mark", "sent_message_mark", "sent_part"),
    [
        (
            {"prompt_cache_breakpoint": _EXPLICIT},
            {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
            {"cache_control": _ONE_HOUR},
            _part("r1"),
        ),
        (
            {"cache_control": _EPHEMERAL},
            {"prompt_cache_breakpoint": _EXPLICIT},
            {"cache_control": _EPHEMERAL},
            _part("r1"),
        ),
        (
            {"prompt_cache_breakpoint": _EXPLICIT},
            {"cache_control": _EPHEMERAL},
            {},
            _part("r1", cache_control=_EPHEMERAL),
        ),
    ],
)
def test_a_tool_result_marked_at_both_levels_reaches_claude_with_one_mark(
    message_mark: dict[str, object],
    part_mark: dict[str, object],
    sent_message_mark: dict[str, object],
    sent_part: dict[str, object],
):
    messages: Final = [
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1", **part_mark)], **message_mark},
    ]

    result: Final = _resolved(messages)

    assert list(result.messages) == [
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), sent_part], **sent_message_mark},
    ]


def test_a_part_breakpoint_a_tool_results_own_breakpoint_replaces_is_still_reported():
    messages: Final = [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [_part("r0", prompt_cache_breakpoint={"mode": "explicit", "ttl": "30m"})],
            "prompt_cache_breakpoint": _EXPLICIT,
        },
    ]

    outcome: Final = _cache(messages)

    assert isinstance(outcome, UnsupportedPromptCacheParam)
    assert outcome.value == {"mode": "explicit", "ttl": "30m"}


def test_a_tool_results_marked_parts_count_as_one_mark_on_the_result_under_drop_params():
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part("u1", prompt_cache_breakpoint=_EXPLICIT)]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [_part("r0", prompt_cache_breakpoint=_EXPLICIT), _part("r1", prompt_cache_breakpoint=_EXPLICIT)],
        },
        {"role": "user", "content": [_part("u2", prompt_cache_breakpoint=_EXPLICIT)]},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part("u1", cache_control=_EPHEMERAL)]},
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1")], "cache_control": _EPHEMERAL},
        {"role": "user", "content": [_part("u2", cache_control=_EPHEMERAL)]},
    ]


def test_a_client_marked_part_of_a_tool_result_wins_over_its_breakpoints_under_drop_params():
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part("u1", prompt_cache_breakpoint=_EXPLICIT)]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [_part("r0", cache_control=_EPHEMERAL), _part("r1", prompt_cache_breakpoint=_EXPLICIT)],
        },
        {"role": "user", "content": [_part("u2", prompt_cache_breakpoint=_EXPLICIT)]},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part("u1", cache_control=_EPHEMERAL)]},
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0", cache_control=_EPHEMERAL), _part("r1")]},
        {"role": "user", "content": [_part("u2", cache_control=_EPHEMERAL)]},
    ]


def test_a_part_breakpoint_a_client_marked_part_replaces_is_still_reported():
    messages: Final = [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                _part("r0", cache_control=_EPHEMERAL),
                _part("r1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "30m"}),
            ],
        }
    ]

    outcome: Final = _cache(messages)

    assert isinstance(outcome, UnsupportedPromptCacheParam)
    assert outcome.value == {"mode": "explicit", "ttl": "30m"}


@pytest.mark.parametrize("malformed", [5, [], "", {"mode": "implicit"}])
def test_a_malformed_breakpoint_beside_a_valid_one_on_a_tool_result_keeps_the_valid_one_under_drop_params(
    malformed: object,
):
    messages: Final = [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                _part("r0", prompt_cache_breakpoint=malformed),
                _part("r1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"}),
            ],
        }
    ]

    assert isinstance(_cache(messages), UnsupportedPromptCacheParam)
    assert list(_resolved(messages, drop_params=True).messages) == [
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1")], "cache_control": _ONE_HOUR},
    ]


def test_a_malformed_breakpoint_beside_one_whose_ttl_drop_params_drops_keeps_that_one_under_drop_params():
    messages: Final = [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                _part("r0", prompt_cache_breakpoint={"mode": "implicit"}),
                _part("r1", prompt_cache_breakpoint=_THIRTY_MINUTES),
            ],
        }
    ]

    assert list(_resolved(messages, drop_params=True).messages) == [
        {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1")], "cache_control": _EPHEMERAL},
    ]


@pytest.mark.parametrize("malformed", [5, [], "", {"mode": "implicit"}])
def test_a_malformed_breakpoint_on_a_messages_last_part_keeps_the_messages_own_breakpoint_under_drop_params(
    malformed: object,
):
    messages: Final = [
        {
            "role": "user",
            "content": [_part("u0"), _part("u1", prompt_cache_breakpoint=malformed)],
            "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
        }
    ]

    assert isinstance(_cache(messages), UnsupportedPromptCacheParam)
    assert list(_resolved(messages, drop_params=True).messages) == [
        {"role": "user", "content": [_part("u0"), _part("u1", cache_control=_ONE_HOUR)]},
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


@pytest.mark.parametrize(
    ("between", "ask", "drop_params", "sent_reminder", "sent_ask"),
    [
        pytest.param((), _EXPLICIT, False, _ONE_HOUR, _ONE_HOUR, id="free-mark-takes-one-hour"),
        pytest.param((), _EXPLICIT, True, _ONE_HOUR, _ONE_HOUR, id="free-mark-takes-one-hour-under-drop-params"),
        pytest.param(
            ({"role": "user", "content": "more"}, {"role": "system", "content": "note"}),
            _EXPLICIT,
            False,
            _ONE_HOUR,
            _ONE_HOUR,
            id="past-a-later-system-message",
        ),
        pytest.param(
            ({"role": "user", "content": "more"}, {"role": "assistant", "content": "b"}),
            _EXPLICIT,
            False,
            _ONE_HOUR,
            _EPHEMERAL,
            id="after-the-next-assistant-turn",
        ),
        pytest.param(
            (),
            {"mode": "explicit", "ttl": "5m"},
            True,
            _EPHEMERAL,
            {"type": "ephemeral", "ttl": "5m"},
            id="set-ttls-loosen-under-drop-params",
        ),
    ],
)
def test_marks_on_a_mid_conversation_system_message_and_the_turns_it_may_move_past_settle_together(
    between: tuple[dict[str, object], ...],
    ask: dict[str, object],
    drop_params: bool,
    sent_reminder: dict[str, object],
    sent_ask: dict[str, object],
):
    messages: Final = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "system", "content": [_part("reminder", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
        *between,
        {"role": "user", "content": [_part("ask", prompt_cache_breakpoint=ask)]},
    ]

    result: Final = _resolved(messages, drop_params=drop_params)

    assert (result.messages[3], result.messages[-1]) == (
        {"role": "system", "content": [_part("reminder", cache_control=sent_reminder)]},
        {"role": "user", "content": [_part("ask", cache_control=sent_ask)]},
    )


_FIVE_MINUTES: Final = {"type": "ephemeral", "ttl": "5m"}


@pytest.mark.parametrize(
    ("after_the_answer", "sent"),
    [
        pytest.param(
            [
                {"role": "system", "content": "note"},
                {
                    "role": "user",
                    "content": [
                        _part("p0", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"}),
                        _part("p1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"}),
                        _part("p2", prompt_cache_breakpoint=_EXPLICIT),
                    ],
                },
            ],
            [
                {"role": "system", "content": "note"},
                {
                    "role": "user",
                    "content": [
                        _part("p0", cache_control=_ONE_HOUR),
                        _part("p1", cache_control=_FIVE_MINUTES),
                        _part("p2", cache_control=_EPHEMERAL),
                    ],
                },
            ],
            id="parts-after-an-unmarked-system-message",
        ),
        pytest.param(
            [
                {"role": "system", "content": [_part("s", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
                {"role": "user", "content": [_part("u1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"})]},
                {"role": "user", "content": [_part("u2", prompt_cache_breakpoint=_EXPLICIT)]},
            ],
            [
                {"role": "system", "content": [_part("s", cache_control=_ONE_HOUR)]},
                {"role": "user", "content": [_part("u1", cache_control=_FIVE_MINUTES)]},
                {"role": "user", "content": [_part("u2", cache_control=_EPHEMERAL)]},
            ],
            id="turn-after-a-five-minute-turn",
        ),
        pytest.param(
            [
                {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
                {"role": "user", "content": [_part("u1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
                {"role": "user", "content": [_part("u2", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"})]},
            ],
            [
                {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
                {"role": "user", "content": [_part("u1", cache_control=_ONE_HOUR)]},
                {"role": "user", "content": [_part("u2", cache_control=_FIVE_MINUTES)]},
            ],
            id="system-message-that-may-move-past-a-five-minute-turn",
        ),
    ],
)
def test_a_breakpoint_claude_may_read_after_a_five_minute_mark_keeps_the_default_ttl(
    after_the_answer: list[dict[str, object]], sent: list[dict[str, object]]
):
    messages: Final = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        *after_the_answer,
    ]

    result: Final = _resolved(messages)

    assert list(result.messages[3:]) == sent


def test_drop_params_drops_a_one_hour_ttl_a_moving_system_message_would_put_after_a_five_minute_mark():
    messages: Final = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part("u1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
        {"role": "user", "content": [_part("u2", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"})]},
    ]

    result: Final = _resolved(messages, drop_params=True)

    assert list(result.messages[3:]) == [
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part("u1", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part("u2", cache_control=_FIVE_MINUTES)]},
    ]


@pytest.mark.parametrize(
    ("message", "sent"),
    [
        pytest.param(
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    _part("r0", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"}),
                    _part("r1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"}),
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1")], "cache_control": _ONE_HOUR},
            id="tool-result-parts",
        ),
        pytest.param(
            {
                "role": "user",
                "content": [_part("u0"), _part("u1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "5m"})],
                "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
            },
            {"role": "user", "content": [_part("u0"), _part("u1", cache_control=_ONE_HOUR)]},
            id="message-and-its-last-part",
        ),
    ],
)
def test_breakpoints_joined_into_one_mark_keep_the_longest_ttl_the_caller_chose(
    message: dict[str, object], sent: dict[str, object]
):
    result: Final = _resolved([message])

    assert list(result.messages) == [sent]


def test_a_tools_breakpoint_joined_with_its_functions_keeps_the_hour_the_function_asked_for():
    tools: Final = [
        {
            "type": "function",
            "prompt_cache_breakpoint": _EXPLICIT,
            "function": {"name": "t", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
        }
    ]

    result: Final = _resolved([{"role": "user", "content": "hi"}], tools)

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}, "cache_control": _ONE_HOUR}]


@pytest.mark.parametrize(
    ("messages", "tools", "sent_messages", "sent_tools"),
    [
        pytest.param(
            [
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": [
                        _part("r0", prompt_cache_breakpoint=_THIRTY_MINUTES),
                        _part("r1", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"}),
                    ],
                }
            ],
            None,
            [{"role": "tool", "tool_call_id": "c1", "content": [_part("r0"), _part("r1")], "cache_control": _ONE_HOUR}],
            [],
            id="tool-result-parts",
        ),
        pytest.param(
            [
                {
                    "role": "user",
                    "content": [_part("u0"), _part("u1", prompt_cache_breakpoint=_THIRTY_MINUTES)],
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                }
            ],
            None,
            [{"role": "user", "content": [_part("u0"), _part("u1", cache_control=_ONE_HOUR)]}],
            [],
            id="message-and-its-last-part",
        ),
        pytest.param(
            [{"role": "user", "content": "hi"}],
            [
                {
                    "type": "function",
                    "prompt_cache_breakpoint": _THIRTY_MINUTES,
                    "function": {"name": "t", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
                }
            ],
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "t"}, "cache_control": _ONE_HOUR}],
            id="tool-and-its-function",
        ),
    ],
)
def test_a_breakpoint_whose_ttl_drop_params_drops_never_costs_the_hour_of_the_one_it_joins(
    messages: list[object],
    tools: list[object] | None,
    sent_messages: list[object],
    sent_tools: list[object],
):
    outcome: Final = _cache(messages, tools)
    result: Final = _resolved(messages, tools, drop_params=True)

    assert isinstance(outcome, UnsupportedPromptCacheParam)
    assert outcome.value == _THIRTY_MINUTES
    assert list(result.messages) == sent_messages
    assert list(result.tools or ()) == sent_tools


def test_a_function_breakpoint_the_tools_own_breakpoint_replaces_is_still_reported():
    tools: Final = [
        {
            "type": "function",
            "prompt_cache_breakpoint": _EXPLICIT,
            "function": {"name": "t", "prompt_cache_breakpoint": _THIRTY_MINUTES},
        }
    ]

    outcome: Final = _cache([{"role": "user", "content": "hi"}], tools)

    assert isinstance(outcome, UnsupportedPromptCacheParam)
    assert outcome.value == _THIRTY_MINUTES


def test_a_tool_config_cache_point_takes_one_of_claudes_four_and_is_never_dropped_under_drop_params():
    messages: Final = [
        {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(3)]},
    ]

    result: Final = _resolved(
        messages,
        [{"type": "function", "function": {"name": "t"}}],
        drop_params=True,
        tool_config_cache_control=ChatCompletionCachedContent(type="ephemeral"),
    )

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s")]},
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(3)]},
    ]


@pytest.mark.parametrize(
    ("tool_config", "tool_marker", "ask_marker", "sent_tool", "sent_ask"),
    [
        pytest.param(
            ChatCompletionCachedContent(type="ephemeral"),
            None,
            {"mode": "explicit", "ttl": "1h"},
            {},
            _ONE_HOUR,
            id="free-keeps-a-later-hour",
        ),
        pytest.param(
            ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
            None,
            {"mode": "explicit", "ttl": "1h"},
            {},
            _EPHEMERAL,
            id="five-minutes-loosens-a-later-hour",
        ),
        pytest.param(
            ChatCompletionCachedContent(type="ephemeral", ttl="1h"),
            _EXPLICIT,
            None,
            {"cache_control": _ONE_HOUR},
            None,
            id="one-hour-lifts-an-earlier-tool-mark",
        ),
    ],
)
def test_a_tool_config_cache_point_settles_after_the_tools_under_drop_params(
    tool_config: ChatCompletionCachedContent,
    tool_marker: dict[str, object] | None,
    ask_marker: dict[str, object] | None,
    sent_tool: dict[str, object],
    sent_ask: dict[str, object] | None,
):
    tool: Final = {"type": "function", "function": {"name": "t"}}
    ask: Final = _part("ask") if ask_marker is None else _part("ask", prompt_cache_breakpoint=ask_marker)

    result: Final = _resolved(
        [{"role": "user", "content": [ask]}],
        [tool if tool_marker is None else {**tool, "prompt_cache_breakpoint": tool_marker}],
        drop_params=True,
        tool_config_cache_control=tool_config,
    )

    assert list(result.tools or ()) == [{**tool, **sent_tool}]
    assert list(result.messages) == [
        {"role": "user", "content": [_part("ask") if sent_ask is None else _part("ask", cache_control=sent_ask)]}
    ]


@pytest.mark.parametrize(
    ("tool_fields", "tool_config", "first_ask_marker", "sent_tool_fields", "sent_system", "sent_first_ask"),
    [
        pytest.param(
            {"cache_control": _EPHEMERAL},
            ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
            _EXPLICIT,
            {"cache_control": _EPHEMERAL},
            None,
            _EPHEMERAL,
            id="sent-after-a-tool-mark-without-a-ttl",
        ),
        pytest.param(
            {"cache_control": _ONE_HOUR},
            ChatCompletionCachedContent(type="ephemeral"),
            _EXPLICIT,
            {"cache_control": _ONE_HOUR},
            None,
            _EPHEMERAL,
            id="sent-without-a-ttl-after-a-tool-mark-with-one",
        ),
        pytest.param(
            {"prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}},
            ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
            _EXPLICIT,
            {},
            _EPHEMERAL,
            _EPHEMERAL,
            id="sent-after-a-dropped-tool-mark",
        ),
        pytest.param(
            {"cache_control": _ONE_HOUR},
            ChatCompletionCachedContent(type="ephemeral"),
            {"mode": "explicit", "ttl": "1h"},
            {"cache_control": _ONE_HOUR},
            _ONE_HOUR,
            _ONE_HOUR,
            id="left-out-once-it-settles-to-an-hour-after-a-tool-mark-with-one",
        ),
    ],
)
def test_a_tool_config_cache_point_takes_one_of_claudes_four_only_when_converse_sends_it(
    tool_fields: dict[str, object],
    tool_config: ChatCompletionCachedContent,
    first_ask_marker: dict[str, object],
    sent_tool_fields: dict[str, object],
    sent_system: dict[str, object] | None,
    sent_first_ask: dict[str, object],
):
    tool: Final = {"type": "function", "function": {"name": "t"}}

    result: Final = _resolved(
        [
            {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
            {
                "role": "user",
                "content": [
                    _part("0", prompt_cache_breakpoint=first_ask_marker),
                    _part("1", prompt_cache_breakpoint=_EXPLICIT),
                ],
            },
        ],
        [{**tool, **tool_fields}],
        drop_params=True,
        tool_config_cache_control=tool_config,
        route="converse",
    )

    assert list(result.tools or ()) == [{**tool, **sent_tool_fields}]
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s") if sent_system is None else _part("s", cache_control=sent_system)]},
        {"role": "user", "content": [_part("0", cache_control=sent_first_ask), _part("1", cache_control=_EPHEMERAL)]},
    ]


@pytest.mark.parametrize(
    ("tools", "sent_later_mark"),
    [
        pytest.param(
            [
                {
                    "type": "function",
                    "function": {"name": "t"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                }
            ],
            _ONE_HOUR,
            id="left-out-after-a-function-tool-mark",
        ),
        pytest.param(
            [
                {"type": "function", "function": {"name": "t"}},
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
            ],
            _EPHEMERAL,
            id="sent-after-a-function-tool-when-converse-drops-the-marked-web-search-tool",
        ),
        pytest.param(
            [
                {
                    "type": "function",
                    "function": {"name": "t"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
                {"toolSpec": {"name": "u", "inputSchema": {"json": {"type": "object"}}}},
            ],
            _EPHEMERAL,
            id="sent-after-a-bedrock-tool-spec-passed-through-after-the-marked-tool",
        ),
        pytest.param(
            [
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                }
            ],
            _ONE_HOUR,
            id="left-out-when-converse-drops-every-tool",
        ),
        pytest.param(
            [
                {
                    "type": "function",
                    "function": {"name": "t"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
                {"systemTool": {"name": "bash"}},
            ],
            _EPHEMERAL,
            id="sent-after-a-bedrock-system-tool-listed-after-the-marked-tool",
        ),
        pytest.param(
            [
                {"systemTool": {"name": "bash"}},
                {
                    "type": "function",
                    "function": {"name": "t"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
            ],
            _EPHEMERAL,
            id="sent-after-a-bedrock-system-tool-converse-moves-after-the-marked-tool",
        ),
        pytest.param(
            [
                {"type": "function", "function": {"name": "t"}},
                {
                    "toolSpec": {"name": "u", "inputSchema": {"json": {"type": "object"}}},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
            ],
            _EPHEMERAL,
            id="sent-after-a-bedrock-tool-spec-whose-own-mark-converse-never-sends",
        ),
        pytest.param(
            [{"type": "function", "function": {"name": "t"}}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
            _ONE_HOUR,
            id="left-out-after-a-bedrock-cache-point-with-an-hour",
        ),
        pytest.param(
            [{"type": "function", "function": {"name": "t"}}, {"cachePoint": {"type": "default"}}],
            _EPHEMERAL,
            id="sent-after-a-bedrock-cache-point-without-a-ttl",
        ),
        pytest.param(
            [{"type": "function", "function": {"name": "t"}}, {"cachePoint": {"type": "default", "ttl": "5m"}}],
            _EPHEMERAL,
            id="left-out-after-a-bedrock-cache-point-with-five-minutes",
        ),
        pytest.param(
            [
                {
                    "type": "function",
                    "function": {"name": "t"},
                    "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
                },
                {"cachePoint": {"type": "default"}},
            ],
            _ONE_HOUR,
            id="left-out-after-a-bedrock-cache-point-without-a-ttl-right-after-the-marked-tool",
        ),
    ],
)
def test_later_one_hour_breakpoints_are_shortened_only_when_converses_tool_config_ends_in_a_five_minute_cache_point(
    tools: list[object], sent_later_mark: dict[str, object]
):
    result: Final = _resolved(
        [
            {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
            {"role": "user", "content": [_part("ask", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
        ],
        tools,
        drop_params=True,
        tool_config_cache_control=ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
        route="converse",
    )

    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=sent_later_mark)]},
        {"role": "user", "content": [_part("ask", cache_control=sent_later_mark)]},
    ]


@pytest.mark.parametrize(
    ("tools", "sent_tools"),
    [
        pytest.param(None, None, id="no-listed-tool"),
        pytest.param(
            [{"type": "web_fetch_20250910", "name": "web_fetch", "cache_control": _ONE_HOUR}],
            [{"type": "web_fetch_20250910", "name": "web_fetch", "cache_control": _ONE_HOUR}],
            id="a-listed-tool-converse-drops",
        ),
        pytest.param(
            [{"type": "bash_20250124", "name": "bash", "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}}],
            [{"type": "bash_20250124", "name": "bash", "cache_control": _EPHEMERAL}],
            id="a-computer-use-tool-converse-sends-outside-the-tool-config",
        ),
    ],
)
def test_later_one_hour_breakpoints_are_shortened_after_the_five_minute_cache_point_beside_the_tool_converse_adds(
    tools: list[object] | None, sent_tools: list[object] | None
):
    result: Final = _resolved(
        [
            {"role": "system", "content": [_part("s", prompt_cache_breakpoint=_EXPLICIT)]},
            {"role": "user", "content": [_part("ask", prompt_cache_breakpoint={"mode": "explicit", "ttl": "1h"})]},
        ],
        tools,
        drop_params=True,
        tool_config_cache_control=ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
        route="converse",
        adds_a_tool=True,
    )

    assert result.tools == (None if sent_tools is None else tuple(sent_tools))
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part("ask", cache_control=_EPHEMERAL)]},
    ]


_LATER_TOOLS: Final = pytest.mark.parametrize(
    "later_tools",
    [
        pytest.param((), id="cache-point-ends-the-tools"),
        pytest.param(({"type": "function", "function": {"name": "u"}},), id="cache-point-between-two-tools"),
    ],
)


@_LATER_TOOLS
@pytest.mark.parametrize(
    "tool_config",
    [None, ChatCompletionCachedContent(type="ephemeral", ttl="5m")],
    ids=["no-tool-config-cache-point", "five-minute-tool-config-cache-point"],
)
@pytest.mark.parametrize(
    ("tool_breakpoint", "cache_point_ttl", "drop_params"),
    [
        pytest.param({"mode": "explicit", "ttl": "1h"}, "1h", False, id="same-ttl"),
        pytest.param({"mode": "explicit", "ttl": "1h"}, "1h", True, id="same-ttl-with-drop-params"),
        pytest.param(_EXPLICIT, "1h", False, id="breakpoint-without-a-ttl-before-an-hour"),
        pytest.param(_EXPLICIT, "5m", False, id="breakpoint-without-a-ttl-before-five-minutes"),
        pytest.param(_EXPLICIT, "5m", True, id="breakpoint-without-a-ttl-with-drop-params"),
        pytest.param({"mode": "explicit", "ttl": "1h"}, "5m", True, id="another-ttl-with-drop-params"),
    ],
)
def test_a_bedrock_cache_point_with_a_ttl_right_after_a_tool_breakpoint_is_the_only_one_sent_there(
    tool_breakpoint: dict[str, object],
    cache_point_ttl: str,
    tool_config: ChatCompletionCachedContent | None,
    drop_params: bool,
    later_tools: tuple[dict[str, object], ...],
):
    cache_point: Final = {"cachePoint": {"type": "default", "ttl": cache_point_ttl}}

    result: Final = _resolved(
        [{"role": "user", "content": "hi"}],
        [
            {"type": "function", "function": {"name": "t"}, "prompt_cache_breakpoint": tool_breakpoint},
            cache_point,
            *later_tools,
        ],
        drop_params=drop_params,
        tool_config_cache_control=tool_config,
        route="converse",
    )

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}}, cache_point, *later_tools]


@_LATER_TOOLS
@pytest.mark.parametrize(
    "tool_config",
    [None, ChatCompletionCachedContent(type="ephemeral", ttl="5m")],
    ids=["no-tool-config-cache-point", "five-minute-tool-config-cache-point"],
)
@pytest.mark.parametrize(
    ("tool_breakpoint", "sent_tool_mark", "cache_point_ttl"),
    [
        pytest.param({"mode": "explicit", "ttl": "1h"}, _ONE_HOUR, "5m", id="hour-before-five-minutes"),
        pytest.param({"mode": "explicit", "ttl": "5m"}, _FIVE_MINUTES, "1h", id="five-minutes-before-an-hour"),
    ],
)
def test_a_tool_breakpoint_and_a_bedrock_cache_point_right_after_it_with_another_ttl_are_both_sent_without_drop_params(
    tool_breakpoint: dict[str, object],
    sent_tool_mark: dict[str, object],
    cache_point_ttl: str,
    tool_config: ChatCompletionCachedContent | None,
    later_tools: tuple[dict[str, object], ...],
):
    cache_point: Final = {"cachePoint": {"type": "default", "ttl": cache_point_ttl}}

    result: Final = _resolved(
        [{"role": "user", "content": "hi"}],
        [
            {"type": "function", "function": {"name": "t"}, "prompt_cache_breakpoint": tool_breakpoint},
            cache_point,
            *later_tools,
        ],
        drop_params=False,
        tool_config_cache_control=tool_config,
        route="converse",
    )

    assert list(result.tools or ()) == [
        {"type": "function", "function": {"name": "t"}, "cache_control": sent_tool_mark},
        cache_point,
        *later_tools,
    ]


@_LATER_TOOLS
def test_a_bedrock_cache_point_with_a_ttl_after_a_tool_breakpoint_keeps_that_breakpoints_slot_of_the_four(
    later_tools: tuple[dict[str, object], ...],
):
    result: Final = _resolved(
        [{"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(4)]}],
        [
            {
                "type": "function",
                "function": {"name": "t"},
                "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
            },
            {"cachePoint": {"type": "default", "ttl": "1h"}},
            *later_tools,
        ],
        drop_params=True,
        route="converse",
    )

    assert list(result.messages) == [
        {
            "role": "user",
            "content": [_part("0"), *(_part(str(index), cache_control=_EPHEMERAL) for index in range(1, 4))],
        }
    ]


@pytest.mark.parametrize(
    ("later_tools", "marked_parts"),
    [
        pytest.param((), 3, id="cache-point-ends-the-tools-and-joins-the-tool-config-cache-point"),
        pytest.param(({"type": "function", "function": {"name": "u"}},), 2, id="cache-point-between-two-tools"),
    ],
)
def test_the_tool_config_cache_point_takes_a_slot_only_when_a_tool_separates_it_from_a_bedrock_cache_point(
    later_tools: tuple[dict[str, object], ...], marked_parts: int
):
    result: Final = _resolved(
        [{"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(4)]}],
        [
            {
                "type": "function",
                "function": {"name": "t"},
                "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
            },
            {"cachePoint": {"type": "default", "ttl": "1h"}},
            *later_tools,
        ],
        drop_params=True,
        tool_config_cache_control=ChatCompletionCachedContent(type="ephemeral", ttl="5m"),
        route="converse",
    )

    assert list(result.messages) == [
        {
            "role": "user",
            "content": [
                *(_part(str(index)) for index in range(4 - marked_parts)),
                *(_part(str(index), cache_control=_EPHEMERAL) for index in range(4 - marked_parts, 4)),
            ],
        }
    ]


@_LATER_TOOLS
def test_a_run_of_bedrock_cache_points_after_a_tool_breakpoint_stands_in_for_it_when_any_of_them_sets_a_ttl(
    later_tools: tuple[dict[str, object], ...],
):
    cache_points: Final = [{"cachePoint": {"type": "default"}}, {"cachePoint": {"type": "default", "ttl": "1h"}}]

    result: Final = _resolved(
        [{"role": "user", "content": "hi"}],
        [
            {
                "type": "function",
                "function": {"name": "t"},
                "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"},
            },
            *cache_points,
            *later_tools,
        ],
        drop_params=False,
        route="converse",
    )

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}}, *cache_points, *later_tools]


@_LATER_TOOLS
def test_a_client_cache_control_on_a_tool_before_a_bedrock_cache_point_with_a_ttl_is_sent_as_written(
    later_tools: tuple[dict[str, object], ...],
):
    tools: Final = [
        {"type": "function", "function": {"name": "t"}, "cache_control": _ONE_HOUR},
        {"cachePoint": {"type": "default", "ttl": "1h"}},
        *later_tools,
    ]

    result: Final = _resolved(
        [{"role": "user", "content": [_part("ask", prompt_cache_breakpoint=_EXPLICIT)]}],
        tools,
        drop_params=True,
        route="converse",
    )

    assert list(result.tools or ()) == tools


@pytest.mark.parametrize(
    ("route", "sent_bash_mark"),
    [("anthropic", _ONE_HOUR), ("converse", _EPHEMERAL)],
    ids=["route_sends_tools_in_order", "route_sends_bash_after_functions"],
)
def test_an_hour_on_a_bash_tool_listed_before_a_five_minute_function_is_dropped_only_where_bash_is_sent_last(
    route: ClaudeRoute, sent_bash_mark: dict[str, object]
):
    bash: Final = {"type": "bash_20250124", "name": "bash"}
    function: Final = {"type": "function", "function": {"name": "t"}, "cache_control": _FIVE_MINUTES}

    result: Final = _resolved(
        [{"role": "user", "content": [_part("ask")]}],
        [{**bash, "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "1h"}}, function],
        drop_params=True,
        route=route,
    )

    assert list(result.tools or ()) == [{**bash, "cache_control": sent_bash_mark}, function]
