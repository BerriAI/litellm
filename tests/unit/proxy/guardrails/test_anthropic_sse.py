"""Assembly contract for raw Anthropic SSE streams.

Four guardrails (bedrock, model_armor, tool_permission, thirdlaw) and the OTel request/response
recorder all scan the body this module assembles, so anything it drops is invisible to every one
of them.
"""

import json
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Final

import pytest

from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_body,
    anthropic_sse_chunks_from_response,
    assemble_anthropic_sse_body,
    assemble_anthropic_sse_stream,
)
from litellm.types.utils import Choices, ModelResponse


def _frames(*events: tuple[str, dict[str, object]]) -> list[bytes]:
    return [f"event: {name}\ndata: {json.dumps(body)}\n\n".encode() for name, body in events]


def _message_start() -> tuple[str, dict[str, object]]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
    )


def _tool_block_start(index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
        },
    )


def _input_delta(partial_json: str, index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def _assembled_tool_input(*events: tuple[str, dict[str, object]]) -> object:
    """The tool block as it lands in the native Anthropic body, which is the shape the /v1/messages
    scan is handed. The ModelResponse conversion keeps the raw partial JSON by another route, so
    asserting through that one would pass either way."""
    body = assemble_anthropic_sse_body(_frames(*events))
    assert body is not None
    content = body["content"]
    assert isinstance(content, Sequence)
    return content[0]["input"]


def test_complete_tool_arguments_assemble_as_the_object_the_model_sent():
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "weather'),
        _input_delta(' in sf"}'),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {"query": "weather in sf"}


def test_truncated_tool_arguments_keep_the_text_instead_of_dropping_it():
    """max_tokens can cut a tool argument mid-JSON. That fragment still reached the client, so
    discarding it here would hand a caller a way to smuggle text past every scanner above."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        _input_delta('{"query": "sk-live-TRUNC'),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    assert "sk-live-TRUNC" in json.dumps(arguments)


def test_a_tool_block_with_no_arguments_assembles_as_an_empty_object():
    """A tool called with no arguments is not truncation, so it must not pick up a raw fragment."""
    arguments = _assembled_tool_input(
        _message_start(),
        _tool_block_start(),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_stop", {"type": "message_stop"}),
    )
    assert arguments == {}


def _text_block_start(index: int = 0) -> tuple[str, dict[str, object]]:
    return (
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
    )


def _block_delta(delta: dict[str, object], index: int = 0) -> tuple[str, dict[str, object]]:
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def _assembled_content(*events: tuple[str, dict[str, object]]) -> Sequence[dict[str, object]]:
    body = assemble_anthropic_sse_body(_frames(*events))
    assert body is not None
    content = body["content"]
    assert isinstance(content, Sequence)
    return content


def test_citation_deltas_land_on_the_text_block_they_annotate():
    citation = {"type": "char_location", "cited_text": "the sky is blue", "document_index": 0}
    content = _assembled_content(
        _message_start(),
        _text_block_start(),
        _block_delta({"type": "text_delta", "text": "Blue."}),
        _block_delta({"type": "citations_delta", "citation": citation}),
        _block_delta({"type": "citations_delta"}),
        _block_delta({"type": "some_future_delta", "text": "ignored"}),
    )
    assert list(content) == [{"type": "text", "text": "Blue.", "citations": (citation,)}]


def test_a_block_type_with_no_deltas_keeps_its_start_payload():
    redacted = {"type": "redacted_thinking", "data": "opaque-blob"}
    content = _assembled_content(
        _message_start(),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": redacted}),
    )
    assert list(content) == [redacted]


def test_malformed_block_events_are_skipped_without_corrupting_the_rest():
    content = _assembled_content(
        _message_start(),
        ("content_block_start", {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
        _text_block_start(),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": "not-a-mapping"}),
        _block_delta({"type": "text_delta", "text": "kept"}),
    )
    assert list(content) == [{"type": "text", "text": "kept"}]


def test_an_undecodable_stream_assembles_to_nothing():
    """Nothing scannable comes out of bytes that are not UTF-8, so callers must treat the stream as
    unassembled rather than scanning a partial body."""
    frames = _frames(_message_start(), _text_block_start())
    assert assemble_anthropic_sse_body([*frames[:1], b"\xff\xfe\xfd", *frames[1:]]) is None


def test_a_replayed_body_keeps_its_citations_and_redacted_thinking():
    """A guardrail rewrite is replayed to the client from the body, so anything the replay drops is
    lost to the caller even though the scan saw it: citations lose their sources, and a
    redacted_thinking block without its data cannot be sent back on the next turn."""
    citation = {"type": "char_location", "cited_text": "the sky is blue", "document_index": 0}
    content = [
        {"type": "redacted_thinking", "data": "opaque-blob"},
        {"type": "text", "text": "The sky is blue.", "citations": [citation]},
        {"type": "text", "text": "No source here."},
    ]
    body: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 7},
    }
    replayed = assemble_anthropic_sse_body(list(anthropic_sse_chunks_from_body(body)))
    assert replayed is not None
    assert json.loads(json.dumps(replayed["content"])) == content


def test_a_tool_input_sent_whole_in_its_block_start_survives_assembly():
    """Server tool blocks reach the stream with their full input in ``content_block_start`` and no
    argument deltas. Blanking that input would replay a query to the client that was never scanned."""
    body: dict[str, object] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "restricted"}}
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 7},
    }
    replayed = assemble_anthropic_sse_body(list(anthropic_sse_chunks_from_body(body)))
    assert replayed is not None
    assert json.loads(json.dumps(replayed["content"])) == body["content"]


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


def _safeguard_results(tool_uses: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    return [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": tool_uses}}]


SAFEGUARD_RESULTS: Final = _safeguard_results({})
REFUSAL_STOP_DETAILS: Final = {"type": "refusal", "category": "cyber", "explanation": None}


@pytest.mark.parametrize(
    ("stop_reason", "delta_fields"),
    [
        ("end_turn", {"safeguard_results": SAFEGUARD_RESULTS}),
        ("refusal", {"stop_details": REFUSAL_STOP_DETAILS}),
    ],
    ids=["safeguard_results", "refusal_stop_details"],
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
