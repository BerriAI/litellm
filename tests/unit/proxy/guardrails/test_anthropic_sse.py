import json
from collections.abc import Callable
from itertools import chain
from typing import Final

import pytest

from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    assemble_anthropic_sse_stream,
)
from litellm.types.utils import Choices, ModelResponse


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _decoded_event(chunk: bytes) -> tuple[str, dict[str, object]]:
    event_line, data_line = chunk.decode().split("\n")[:2]
    assert event_line.startswith("event: ")
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


def _decode(chunks: tuple[bytes, ...]) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(_decoded_event(chunk) for chunk in chunks)


def _rewritten_delta(
    upstream: tuple[bytes, ...], rewrite: Callable[[ModelResponse], None], keep_explanations: bool = False
) -> object:
    assembled: Final = assemble_anthropic_sse_stream(upstream, restore_identity=True)
    assert assembled is not None
    rewrite(assembled)
    rewritten: Final = anthropic_sse_chunks_from_response(assembled, upstream, keep_explanations=keep_explanations)
    return next(payload["delta"] for event, payload in _decode(rewritten) if event == "message_delta")


def _set_text(text: str) -> Callable[[ModelResponse], None]:
    def rewrite(assembled: ModelResponse) -> None:
        choice: Final = assembled.choices[0]
        assert isinstance(choice, Choices)
        choice.message.content = text

    return rewrite


def _tool_use_frames(index: int, tool_use_id: str, name: str) -> tuple[bytes, ...]:
    return (
        _frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": tool_use_id, "name": name, "input": {}},
            },
        ),
        _frame(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
            },
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": index}),
    )


def _upstream_stream(
    delta: dict[str, object], text: str = "call 555-0100", tool_uses: tuple[tuple[str, str], ...] = ()
) -> tuple[bytes, ...]:
    tool_use_frames = tuple(
        chain.from_iterable(
            _tool_use_frames(index, tool_use_id, name) for index, (tool_use_id, name) in enumerate(tool_uses, start=1)
        )
    )
    return (
        _frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_upstream_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-upstream",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
        _frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *tool_use_frames,
        _frame("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 2}}),
        _frame("message_stop", {"type": "message_stop"}),
    )


_UPSTREAM_ERROR: Final = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


def _masked_through_upstream_error(upstream: tuple[bytes, ...]) -> tuple[tuple[str, dict[str, object]], ...]:
    assembled: Final = assemble_anthropic_sse_stream(upstream, restore_identity=True, through_upstream_error=True)
    assert assembled is not None
    _set_text("call [PHONE]")(assembled)
    return _decode(anthropic_sse_chunks_from_response(assembled, upstream))


def _blocks_open_before(events: tuple[tuple[str, dict[str, object]], ...], position: int) -> set[object]:
    earlier: Final = events[:position]
    started: Final = {payload["index"] for event, payload in earlier if event == "content_block_start"}
    stopped: Final = {payload["index"] for event, payload in earlier if event == "content_block_stop"}
    return started - stopped


def _safeguard_results(tool_uses: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    return [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": tool_uses}}]


SAFEGUARD_RESULTS: Final = _safeguard_results({})
REFUSAL_STOP_DETAILS: Final = {"type": "refusal", "category": "cyber", "explanation": None}


@pytest.mark.parametrize(
    ("stop_reason", "delta_fields"),
    [
        ("end_turn", {"safeguard_results": SAFEGUARD_RESULTS}),
        ("refusal", {"stop_details": REFUSAL_STOP_DETAILS}),
        ("pause_turn", {}),
    ],
    ids=["safeguard_results", "refusal_stop_details", "pause_turn"],
)
def test_rewritten_stream_keeps_the_upstream_message_delta_fields(stop_reason: str, delta_fields: dict[str, object]):
    """A guardrail that rewrites a streamed response assembles it into the OpenAI shape, which has no
    slot for ``safeguard_results`` or ``stop_details``, so the re-emitted ``message_delta`` used to
    drop the classifier verdict a Claude Code auto-mode session reads from it."""
    upstream_delta: dict[str, object] = {"stop_reason": stop_reason, "stop_sequence": None, **delta_fields}
    upstream = _upstream_stream(upstream_delta)
    assembled = assemble_anthropic_sse_stream(upstream, restore_identity=True)
    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    choice.message.content = "call <PHONE>"

    decoded = _decode(anthropic_sse_chunks_from_response(assembled, upstream))
    payloads = {event: payload for event, payload in decoded}

    assert payloads["content_block_delta"]["delta"] == {"type": "text_delta", "text": "call <PHONE>"}
    assert payloads["message_delta"]["delta"] == upstream_delta
    assert not set(delta_fields) & set(payloads["message_start"]["message"])


SSN: Final = "123-45-6789"


@pytest.mark.parametrize(
    ("upstream_delta", "tool_uses", "expected_fields"),
    [
        (
            {
                "stop_reason": "refusal",
                "stop_sequence": None,
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": f"Request involves SSN {SSN}"},
            },
            (),
            {"stop_details": {"type": "refusal", "category": "cyber", "explanation": None}},
        ),
        (
            {
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "safeguard_results": _safeguard_results(
                    {"toolu_bash": {"type": "evaluated", "outcome": "flagged", "explanation": f"leaks SSN {SSN}"}}
                ),
            },
            (("toolu_bash", "Bash"),),
            {"safeguard_results": _safeguard_results({"toolu_bash": {"type": "evaluated", "outcome": "flagged"}})},
        ),
        (
            {
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "safeguard_results": [
                    {
                        "type": "prompt_injection",
                        "status": {"type": "available", "explanation": f"quotes SSN {SSN}"},
                        "explanation": f"saw SSN {SSN}",
                    }
                ],
            },
            (),
            {"safeguard_results": [{"type": "prompt_injection", "status": {"type": "available"}}]},
        ),
    ],
    ids=["stop_details_explanation", "verdict_explanation", "unknown_shape_explanation"],
)
def test_masked_stream_drops_the_unmasked_upstream_explanations(
    upstream_delta: dict[str, object], tool_uses: tuple[tuple[str, str], ...], expected_fields: dict[str, object]
):
    upstream = _upstream_stream(upstream_delta, text=f"my SSN is {SSN}", tool_uses=tool_uses)
    assembled = assemble_anthropic_sse_stream(upstream, restore_identity=True)
    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    choice.message.content = "my SSN is <US_SSN>"

    rewritten = anthropic_sse_chunks_from_response(assembled, upstream)
    message_delta = next(payload for event, payload in _decode(rewritten) if event == "message_delta")

    assert not [chunk for chunk in rewritten if SSN.encode() in chunk]
    assert {key: message_delta["delta"][key] for key in expected_fields} == expected_fields


def test_masked_stream_drops_safeguard_results_it_cannot_read():
    upstream_delta: Final = {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "safeguard_results": {"type": "dangerous_tool_use", "explanation": f"saw SSN {SSN}"},
    }

    assert _rewritten_delta(_upstream_stream(upstream_delta), _set_text("my SSN is <US_SSN>")) == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }


def test_rewritten_stream_keeps_only_the_verdicts_of_tool_uses_it_still_delivers():
    kept_verdict = {"type": "evaluated", "outcome": "not_flagged"}
    upstream = _upstream_stream(
        {
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "safeguard_results": _safeguard_results(
                {"toolu_glob": {"type": "evaluated", "outcome": "flagged"}, "toolu_bash": kept_verdict}
            ),
        },
        tool_uses=(("toolu_glob", "Glob"), ("toolu_bash", "Bash")),
    )
    assembled = assemble_anthropic_sse_stream(upstream)
    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    choice.message.tool_calls = [call for call in choice.message.tool_calls or [] if call.id == "toolu_bash"]
    choice.message.content = "Permission denied: Tool 'Glob' denied"

    decoded = _decode(anthropic_sse_chunks_from_response(assembled, upstream))
    message_delta = next(payload for event, payload in decoded if event == "message_delta")
    delivered_tool_use_ids = [
        payload["content_block"]["id"]
        for event, payload in decoded
        if event == "content_block_start" and payload["content_block"]["type"] == "tool_use"
    ]

    assert delivered_tool_use_ids == ["toolu_bash"]
    assert message_delta["delta"]["safeguard_results"] == _safeguard_results({"toolu_bash": kept_verdict})


def test_rewritten_refusal_gets_no_stop_details_the_upstream_did_not_send():
    """The OpenAI round trip sets ``stop_details`` on every refusal, and the repo reads a refusal
    carrying one as a safeguard refusal, so a plain refusal must not come back as one."""
    upstream_delta: Final = {"stop_reason": "refusal", "stop_sequence": None}

    assert _rewritten_delta(_upstream_stream(upstream_delta), _set_text("call <PHONE>")) == upstream_delta


def test_rewritten_stream_keeps_safeguard_shapes_it_does_not_know():
    upstream_delta: Final = {
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "stop_details": {"type": "pause_turn", "category": None, "explanation": None, "policy_url": "https://x.test"},
        "safeguard_results": [
            {
                "type": "dangerous_tool_use",
                "status": {
                    "type": "available",
                    "tool_uses": {"toolu_bash": {"type": "unavailable", "reason": "overloaded"}},
                },
                "classifier_version": "2026-10",
            },
            {"type": "prompt_injection", "status": {"type": "skipped"}},
        ],
    }
    upstream: Final = _upstream_stream(upstream_delta, tool_uses=(("toolu_bash", "Bash"),))

    assert _rewritten_delta(upstream, _set_text("call <PHONE>")) == upstream_delta


def test_rewrite_that_changes_no_text_keeps_the_upstream_explanations():
    upstream_delta: Final = {
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "safeguard_results": _safeguard_results(
            {"toolu_bash": {"type": "evaluated", "outcome": "flagged", "explanation": "lists the home directory"}}
        ),
    }
    upstream: Final = _upstream_stream(upstream_delta, tool_uses=(("toolu_bash", "Bash"),))

    assert _rewritten_delta(upstream, _set_text("Permission denied"), keep_explanations=True) == upstream_delta


def test_rewritten_stream_drops_the_verdict_of_a_tool_use_whose_input_the_guardrail_changed():
    """A verdict scored on the upstream input must not vouch for the input the client runs."""
    verdict: Final = {"type": "evaluated", "outcome": "not_flagged"}
    upstream: Final = _upstream_stream(
        {
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "safeguard_results": _safeguard_results({"toolu_masked": verdict, "toolu_untouched": verdict}),
        },
        tool_uses=(("toolu_masked", "Bash"), ("toolu_untouched", "Bash")),
    )

    def mask_first_tool_input(assembled: ModelResponse) -> None:
        choice: Final = assembled.choices[0]
        assert isinstance(choice, Choices)
        tool_calls: Final = choice.message.tool_calls or []
        assert [call.id for call in tool_calls] == ["toolu_masked", "toolu_untouched"]
        tool_calls[0].function.arguments = '{"command": "<REDACTED>"}'

    rewritten_delta: Final = _rewritten_delta(upstream, mask_first_tool_input)

    assert isinstance(rewritten_delta, dict)
    assert rewritten_delta["safeguard_results"] == _safeguard_results({"toolu_untouched": verdict})


@pytest.mark.parametrize(
    ("delivered_content", "upstream_stopped_indices"),
    [
        (_upstream_stream({})[:-2], [0]),
        (_upstream_stream({})[:-3], []),
        (_upstream_stream({}, tool_uses=(("toolu_cut_off", "Bash"),))[:-3], [0]),
        (
            (
                *_upstream_stream({}, tool_uses=(("toolu_cut_off", "Bash"),))[:-4],
                _frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"command": "l'},
                    },
                ),
            ),
            [0],
        ),
    ],
    ids=["text-block-stopped", "text-block-cut-off", "tool-use-block-cut-off", "tool-use-input-cut-off"],
)
def test_masked_stream_that_ended_in_an_upstream_error_still_ends_in_that_error(
    delivered_content: tuple[bytes, ...], upstream_stopped_indices: list[int]
):
    rewritten: Final = _masked_through_upstream_error((*delivered_content, _frame("error", _UPSTREAM_ERROR)))

    assert rewritten[-1] == ("error", _UPSTREAM_ERROR)
    assert {"message_delta", "message_stop"}.isdisjoint(event for event, _ in rewritten), (
        "a stream the upstream ended in an error must not be re-emitted as one that completed"
    )
    assert any(payload.get("delta") == {"type": "text_delta", "text": "call [PHONE]"} for _, payload in rewritten)
    assert [payload["index"] for event, payload in rewritten if event == "content_block_stop"] == (
        upstream_stopped_indices
    ), "a block the upstream never finished must not be re-emitted as finished"
    assert not any(payload.get("content_block", {}).get("type") == "tool_use" for _, payload in rewritten), (
        "a tool call the upstream cut off must not be re-emitted as one the client can run"
    )


def test_masked_text_cut_off_after_a_finished_tool_call_is_stopped_before_the_tool_call_starts():
    upstream: Final = (
        _upstream_stream({})[0],
        *_tool_use_frames(0, "toolu_finished", "Bash"),
        _frame(
            "content_block_start",
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "call 555-0100"}},
        ),
        _frame("error", _UPSTREAM_ERROR),
    )

    rewritten: Final = _masked_through_upstream_error(upstream)

    block_starts: Final = [position for position, (event, _) in enumerate(rewritten) if event == "content_block_start"]
    assert [_blocks_open_before(rewritten, position) for position in block_starts] == [set()] * len(block_starts), (
        "a client must see each block stop before the next one starts"
    )
    assert rewritten[-1] == ("error", _UPSTREAM_ERROR)
    assert any(payload.get("content_block", {}).get("type") == "tool_use" for _, payload in rewritten)
    assert any(payload.get("delta") == {"type": "text_delta", "text": "call [PHONE]"} for _, payload in rewritten)


def test_masked_stream_that_failed_after_its_message_delta_still_carries_that_message_delta():
    upstream: Final = (
        *_upstream_stream({"stop_reason": "end_turn", "stop_sequence": None})[:-1],
        _frame("error", _UPSTREAM_ERROR),
    )

    rewritten: Final = _masked_through_upstream_error(upstream)

    assert [event for event, _ in rewritten][-2:] == ["message_delta", "error"]
    assert next(payload for event, payload in rewritten if event == "message_delta")["delta"]["stop_reason"] == "end_turn"


def test_healthy_stream_with_a_tool_input_that_is_not_json_still_raises():
    healthy: Final = _upstream_stream({"stop_reason": "tool_use", "stop_sequence": None})
    tool_block: Final = {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {}}
    tool_input: Final = {"type": "input_json_delta", "partial_json": '{"command" "ls"'}
    upstream: Final = (
        *healthy[:-2],
        _frame("content_block_start", {"type": "content_block_start", "index": 1, "content_block": tool_block}),
        _frame("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": tool_input}),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
        *healthy[-2:],
    )
    assembled: Final = assemble_anthropic_sse_stream(upstream)
    assert assembled is not None

    with pytest.raises(ValueError, match="Failed to parse tool call arguments"):
        anthropic_sse_chunks_from_response(assembled, upstream)


def _events_frames(events: tuple[dict[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(_frame(str(event["type"]), event) for event in events)


def _streamed_block(index: int, content_block: dict[str, object], *deltas: dict[str, object]) -> tuple[dict[str, object], ...]:
    return (
        {"type": "content_block_start", "index": index, "content_block": content_block},
        *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
        {"type": "content_block_stop", "index": index},
    )


def _client_tool_use(index: int, tool_use_id: str, name: str) -> tuple[dict[str, object], ...]:
    return _streamed_block(
        index,
        {"type": "tool_use", "id": tool_use_id, "name": name, "input": {}, "caller": {"type": "direct"}},
        {"type": "input_json_delta", "partial_json": ""},
        {"type": "input_json_delta", "partial_json": '{"city": "Tokyo"}'},
    )


_FULL_USAGE: Final = {
    "input_tokens": 638,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    "output_tokens": 32,
    "service_tier": "standard",
    "inference_geo": "global",
    "speed": "standard",
}
_MESSAGE_START: Final = {
    "type": "message_start",
    "message": {
        "model": "claude-sonnet-5-5",
        "id": "msg_011ChFEWZLP16nZ4aHN6oeRZ",
        "type": "message",
        "role": "assistant",
        "content": [],
        "container": None,
        "stop_reason": None,
        "stop_sequence": None,
        "stop_details": None,
        "usage": _FULL_USAGE,
        "diagnostics": None,
    },
}
_FINAL_USAGE: Final = {
    "input_tokens": 2063,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 164,
    "output_tokens_details": {"thinking_tokens": 0},
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
}


def _message_end(stop_reason: str) -> tuple[dict[str, object], ...]:
    return (
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": None, "container": None},
            "usage": _FINAL_USAGE,
        },
        {"type": "message_stop"},
    )


_MCP_CALL: Final = _streamed_block(
    0,
    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "read_wiki_structure", "input": {}, "server_name": "deepwiki"},
    {"type": "input_json_delta", "partial_json": '{"repoName": "facebook/react"}'},
)
_MCP_RESULT: Final = _streamed_block(
    1,
    {
        "type": "mcp_tool_result",
        "tool_use_id": "mcptoolu_1",
        "is_error": False,
        "content": [{"type": "text", "text": "Available pages for facebook/react"}],
    },
)
_MCP_STREAM_EVENTS: Final = (
    _MESSAGE_START,
    _MCP_CALL[0],
    {"type": "ping"},
    *_MCP_CALL[1:],
    *_MCP_RESULT,
    *_client_tool_use(2, "toolu_time", "get_time"),
    *_client_tool_use(3, "toolu_weather", "get_weather"),
    *_message_end("tool_use"),
)
_DENIAL: Final = "Permission denied: Tool 'get_weather' denied"


def _deny(*denied_tool_use_ids: str) -> Callable[[ModelResponse], None]:
    def rewrite(assembled: ModelResponse) -> None:
        choice: Final = assembled.choices[0]
        assert isinstance(choice, Choices)
        kept_calls: Final = [call for call in choice.message.tool_calls or [] if call.id not in denied_tool_use_ids]
        choice.message.tool_calls = kept_calls or None
        choice.message.content = _DENIAL
        if not kept_calls:
            choice.finish_reason = "stop"

    return rewrite


def _rewritten_events(
    events: tuple[dict[str, object], ...], rewrite: Callable[[ModelResponse], None]
) -> tuple[dict[str, object], ...]:
    upstream: Final = _events_frames(events)
    assembled: Final = assemble_anthropic_sse_stream(upstream, through_upstream_error=True)
    assert assembled is not None
    rewrite(assembled)
    return tuple(payload for _, payload in _decode(anthropic_sse_chunks_from_response(assembled, upstream)))


def _without_block(events: tuple[dict[str, object], ...], index: int) -> tuple[dict[str, object], ...]:
    return tuple(event for event in events if event.get("index") != index)


def _without_pings(events: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(event for event in events if event["type"] != "ping")


def test_denied_tool_call_leaves_every_other_upstream_event_as_the_upstream_sent_it():
    """The rewrite used to rebuild the whole stream from the OpenAI-shaped response, which dropped the
    MCP blocks, each tool's ``caller``, the null keys and most usage fields, and replaced the message id."""
    rewritten: Final = _rewritten_events(_MCP_STREAM_EVENTS, _deny("toolu_weather"))

    kept: Final = _without_pings(_without_block(_MCP_STREAM_EVENTS, 3))
    denial_text: Final = _streamed_block(
        3, {"type": "text", "text": ""}, {"type": "text_delta", "text": _DENIAL}
    )
    assert _without_pings(rewritten) == (*kept[:-2], *denial_text, *kept[-2:])
    assert {"type": "ping"} in rewritten


def test_stream_whose_only_tool_call_is_denied_ends_the_turn_with_the_upstream_usage():
    upstream: Final = (_MESSAGE_START, *_client_tool_use(0, "toolu_weather", "get_weather"), *_message_end("tool_use"))

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    expected_end: Final = _message_end("end_turn")
    assert rewritten == (
        _MESSAGE_START,
        *_streamed_block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": _DENIAL}),
        *expected_end,
    )


def test_denied_tool_call_replaces_the_upstream_text_where_it_was():
    thinking: Final = _streamed_block(
        0,
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "thinking_delta", "thinking": "Both calls are needed."},
        {"type": "signature_delta", "signature": "sig-1"},
    )
    redacted: Final = _streamed_block(1, {"type": "redacted_thinking", "data": "EmwKAhgB"})
    text: Final = _streamed_block(
        2, {"type": "text", "text": ""}, {"type": "text_delta", "text": "I'll check both."}
    )
    upstream: Final = (
        _MESSAGE_START,
        *thinking,
        *redacted,
        *text,
        *_client_tool_use(3, "toolu_time", "get_time"),
        *_client_tool_use(4, "toolu_weather", "get_weather"),
        *_message_end("tool_use"),
    )

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    kept: Final = _without_block(upstream, 4)
    rewritten_text: Final = _streamed_block(
        2, {"type": "text", "text": ""}, {"type": "text_delta", "text": _DENIAL}
    )
    assert rewritten == tuple(
        chain.from_iterable(rewritten_text if event in text else (event,) for event in kept if event not in text[1:])
    )


def test_masked_tool_input_keeps_the_fields_the_masking_did_not_touch():
    upstream: Final = (_MESSAGE_START, *_client_tool_use(0, "toolu_time", "get_time"), *_message_end("tool_use"))

    def mask_input(assembled: ModelResponse) -> None:
        choice: Final = assembled.choices[0]
        assert isinstance(choice, Choices)
        (tool_call,) = choice.message.tool_calls or []
        tool_call.function.arguments = '{"city": "<LOCATION>"}'

    rewritten: Final = _rewritten_events(upstream, mask_input)

    assert rewritten == (
        _MESSAGE_START,
        *_streamed_block(
            0,
            {"type": "tool_use", "id": "toolu_time", "name": "get_time", "input": {}, "caller": {"type": "direct"}},
            {"type": "input_json_delta", "partial_json": '{"city": "<LOCATION>"}'},
        ),
        *_message_end("tool_use"),
    )


_HIDDEN_CALL: Final = {"type": "tool_use", "id": "toolu_hidden", "name": "Bash", "input": {"command": "rm -rf /"}}


def test_rewrite_sends_message_start_with_the_empty_content_anthropic_sends():
    message: Final = _MESSAGE_START["message"]
    assert isinstance(message, dict)
    message_start_with_a_call: Final = {**_MESSAGE_START, "message": {**message, "content": [_HIDDEN_CALL]}}
    upstream: Final = (
        message_start_with_a_call,
        *_client_tool_use(0, "toolu_weather", "get_weather"),
        *_message_end("tool_use"),
    )

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    assert rewritten[0] == _MESSAGE_START
    assert "toolu_hidden" not in json.dumps(rewritten)


@pytest.mark.parametrize(
    "undefined_event",
    [
        {"type": "tool_call", "content_block": _HIDDEN_CALL},
        {"type": "content_block_start", "content_block": _HIDDEN_CALL},
    ],
    ids=["unknown-type", "block-start-without-index"],
)
def test_rewrite_leaves_out_an_event_the_messages_stream_does_not_define(undefined_event: dict[str, object]):
    upstream: Final = (
        _MESSAGE_START,
        undefined_event,
        *_client_tool_use(0, "toolu_weather", "get_weather"),
        *_message_end("tool_use"),
    )

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    assert "toolu_hidden" not in json.dumps(rewritten)
    assert _DENIAL in json.dumps(rewritten)


def _text_block(index: int, text: str) -> tuple[dict[str, object], ...]:
    return _streamed_block(index, {"type": "text", "text": ""}, {"type": "text_delta", "text": text})


def test_masked_text_of_several_text_blocks_is_sent_once_where_the_first_text_block_was():
    search_call: Final = _streamed_block(
        1,
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
        {"type": "input_json_delta", "partial_json": '{"query": "support line"}'},
    )
    search_result: Final = _streamed_block(
        2,
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "url": "https://a.test", "title": "A", "encrypted_content": "e"}],
        },
    )
    upstream: Final = (
        _MESSAGE_START,
        *_text_block(0, "I'll look up a@b.test."),
        *search_call,
        *search_result,
        *_text_block(3, " Call 555-0100."),
        *_message_end("end_turn"),
    )
    masked: Final = "I'll look up [EMAIL]. Call [PHONE]."

    rewritten: Final = _rewritten_events(upstream, _set_text(masked))

    assert rewritten == (_MESSAGE_START, *_text_block(0, masked), *search_call, *search_result, *_message_end("end_turn"))


def test_rewrite_that_removes_a_server_tool_call_also_removes_the_result_that_answers_it():
    upstream: Final = (
        _MESSAGE_START,
        *_streamed_block(
            0,
            {"type": "server_tool_use", "id": "srvtoolu_2", "name": "bash_code_execution", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"command": "cat .env"}'},
        ),
        *_streamed_block(
            1,
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {"type": "bash_code_execution_result", "stdout": "KEY=1", "stderr": "", "return_code": 0},
            },
        ),
        *_text_block(2, "Here is the file."),
        *_message_end("end_turn"),
    )

    def block_the_server_call(assembled: ModelResponse) -> None:
        choice: Final = assembled.choices[0]
        assert isinstance(choice, Choices)
        fields: Final = choice.message.provider_specific_fields or {}
        choice.message.provider_specific_fields = {
            key: value for key, value in fields.items() if key != "server_tool_calls"
        }
        choice.message.content = "Blocked"

    rewritten: Final = _rewritten_events(upstream, block_the_server_call)

    assert rewritten == (_MESSAGE_START, *_text_block(0, "Blocked"), *_message_end("end_turn"))


def test_stream_that_failed_before_its_message_start_is_sent_as_its_error_alone():
    upstream: Final = _events_frames((_MESSAGE_START, *_text_block(0, "call 555-0100"), *_message_end("end_turn")))
    assembled: Final = assemble_anthropic_sse_stream(upstream)
    assert assembled is not None
    _set_text("call [PHONE]")(assembled)

    rewritten: Final = anthropic_sse_chunks_from_response(assembled, _events_frames(({"type": "ping"}, _UPSTREAM_ERROR)))

    assert _decode(rewritten) == (("error", _UPSTREAM_ERROR),)
