from __future__ import annotations

from typing import Final

from claude_code.stream_events import describe_run

_INIT: Final = {"type": "system", "subtype": "init", "tools": ["ToolSearch", "WebSearch"], "model": "claude-opus-4-7"}


def _assistant(*blocks: dict[str, object]) -> dict[str, object]:
    return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks), "stop_reason": None}}


def _tool_result(content: object) -> dict[str, object]:
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": content}]}}


def _result(text: str, *, turns: int, stop_reason: str = "end_turn") -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": turns,
        "stop_reason": stop_reason,
        "result": text,
    }


def test_names_every_tool_call_in_order_with_the_final_reply() -> None:
    events: Final = [
        _INIT,
        _assistant({"type": "thinking", "thinking": ""}),
        _assistant({"type": "tool_use", "id": "toolu_1", "name": "ToolSearch", "input": {"query": "select:WebSearch"}}),
        _tool_result([{"type": "tool_reference", "tool_name": "WebSearch"}]),
        _assistant({"type": "tool_use", "id": "toolu_2", "name": "WebSearch", "input": {"query": "Anthropic news"}}),
        _tool_result("Web search error: unavailable"),
        _assistant({"type": "text", "text": "Search was unavailable."}),
        _result("Search was unavailable.", turns=3),
    ]

    assert describe_run(events) == (
        "tool calls: ToolSearch, WebSearch; stop_reason=end_turn; turns=3; reply='Search was unavailable.'"
    )


def test_a_reply_without_any_tool_call_says_none() -> None:
    events: Final = [
        _INIT,
        _assistant({"type": "text", "text": "I can't browse the web."}),
        _result("I can't browse the web.", turns=1),
    ]

    assert describe_run(events) == ('tool calls: none; stop_reason=end_turn; turns=1; reply="I can\'t browse the web."')


def test_a_long_reply_is_cut_to_the_limit() -> None:
    reply: Final = "x" * 50

    description: Final = describe_run([_result(reply, turns=1)], reply_chars=10)

    assert description.endswith(f"reply='{'x' * 10}...'")


def test_a_run_that_ended_before_its_result_event_says_so() -> None:
    events: Final = [_INIT, _assistant({"type": "tool_use", "id": "toolu_1", "name": "ToolSearch", "input": {}})]

    assert describe_run(events) == "tool calls: ToolSearch; no result event"


def test_events_and_blocks_it_does_not_read_are_skipped() -> None:
    events: Final = [
        {"subtype": "no type at all"},
        {"type": "assistant", "message": {"content": "plain string content"}},
        _assistant({"type": "tool_use", "id": "toolu_1"}, {"type": "redacted_thinking"}, {"no": "type"}),
        _assistant({"type": "tool_use", "id": "toolu_2", "name": "WebSearch", "input": {}}),
        {"type": "result", "num_turns": "not a number"},
        _result("done", turns=2, stop_reason="tool_use"),
    ]

    assert describe_run(events) == "tool calls: WebSearch; stop_reason=tool_use; turns=2; reply='done'"
