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
        {"role": "system", "content": [_part("sys", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "cache_control": _EPHEMERAL}],
        },
    ]
    assert list(result.tools or ()) == [
        {"type": "function", "function": {"name": "a"}, "cache_control": _EPHEMERAL},
        {"type": "function", "function": {"name": "b"}, "cache_control": _EPHEMERAL},
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


def test_drop_params_keeps_the_latest_four_marks_in_the_order_claude_renders_them():
    tools: Final = [{"type": "function", "function": {"name": "t"}, "prompt_cache_breakpoint": _EXPLICIT}]
    messages: Final = [
        {"role": "system", "content": [_part("s", cache_control=_EPHEMERAL)]},
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(3)]},
    ]

    result: Final = _resolved(messages, tools, {"cache_control": _EPHEMERAL}, drop_params=True)

    assert list(result.tools or ()) == [{"type": "function", "function": {"name": "t"}}]
    assert list(result.messages) == [
        {"role": "system", "content": [_part("s")]},
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(3)]},
    ]


def test_without_drop_params_more_than_four_marks_are_sent_for_claude_to_answer():
    messages: Final = [
        {"role": "user", "content": [_part(str(index), prompt_cache_breakpoint=_EXPLICIT) for index in range(5)]}
    ]

    result: Final = _resolved(messages)

    assert list(result.messages) == [
        {"role": "user", "content": [_part(str(index), cache_control=_EPHEMERAL) for index in range(5)]}
    ]
