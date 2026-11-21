import json
from collections.abc import Callable
from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import FakeAnthropicMessagesStreamIterator
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    assemble_anthropic_sse_stream,
)
from litellm.types.utils import Choices, ModelResponse


def _frame(data: dict[str, object]) -> bytes:
    event_line: Final = f"event: {data['type']}\n" if "type" in data else ""
    return f"{event_line}data: {json.dumps(data)}\n\n".encode()


def _decoded_event(chunk: bytes) -> dict[str, object]:
    event_line, data_line = chunk.decode().split("\n")[:2]
    payload: Final = json.loads(data_line.removeprefix("data: "))
    assert event_line == f"event: {payload['type']}"
    return payload


def _events_frames(events: tuple[dict[str, object], ...]) -> tuple[bytes, ...]:
    return tuple(_frame(event) for event in events)


def _streamed_block(
    index: int, content_block: dict[str, object], *deltas: dict[str, object]
) -> tuple[dict[str, object], ...]:
    return (
        {"type": "content_block_start", "index": index, "content_block": content_block},
        *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
        {"type": "content_block_stop", "index": index},
    )


def _text_block(index: int, text: str) -> tuple[dict[str, object], ...]:
    return _streamed_block(index, {"type": "text", "text": ""}, {"type": "text_delta", "text": text})


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
_MESSAGE: Final = {
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
}
_MESSAGE_START: Final = {"type": "message_start", "message": _MESSAGE}
_FINAL_USAGE: Final = {
    "input_tokens": 2063,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 164,
    "output_tokens_details": {"thinking_tokens": 0},
    "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
}


def _message_end(stop_reason: str, **delta_fields: object) -> tuple[dict[str, object], ...]:
    return (
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": None,
                "container": None,
                **delta_fields,
            },
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


def _only_choice(assembled: ModelResponse) -> Choices:
    choice: Final = assembled.choices[0]
    assert isinstance(choice, Choices)
    return choice


def _set_text(text: str) -> Callable[[ModelResponse], None]:
    def rewrite(assembled: ModelResponse) -> None:
        _only_choice(assembled).message.content = text

    return rewrite


def _deny(*denied_tool_use_ids: str, text: str = _DENIAL) -> Callable[[ModelResponse], None]:
    def rewrite(assembled: ModelResponse) -> None:
        choice: Final = _only_choice(assembled)
        kept_calls: Final = [call for call in choice.message.tool_calls or [] if call.id not in denied_tool_use_ids]
        choice.message.tool_calls = kept_calls or None
        choice.message.content = text
        if not kept_calls:
            choice.finish_reason = "stop"

    return rewrite


def _rebuilt_from_the_rewrite_alone(assembled: ModelResponse) -> tuple[bytes, ...]:
    anthropic_response: Final = LiteLLMAnthropicMessagesAdapter().translate_openai_response_to_anthropic(
        response=assembled
    )
    return tuple(FakeAnthropicMessagesStreamIterator(response=anthropic_response).chunks)


def _rewritten_events(
    events: tuple[dict[str, object], ...], rewrite: Callable[[ModelResponse], None]
) -> tuple[dict[str, object], ...]:
    upstream: Final = _events_frames(events)
    assembled: Final = assemble_anthropic_sse_stream(upstream)
    assert assembled is not None
    rewrite(assembled)
    return tuple(_decoded_event(chunk) for chunk in anthropic_sse_chunks_from_response(assembled, upstream))


def _without_block(events: tuple[dict[str, object], ...], index: int) -> tuple[dict[str, object], ...]:
    return tuple(event for event in events if event.get("index") != index)


def _without_pings(events: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(event for event in events if event["type"] != "ping")


def test_denied_tool_call_leaves_every_other_upstream_event_as_the_upstream_sent_it():
    """The rewrite used to rebuild the whole stream from the OpenAI-shaped response, which dropped the
    MCP blocks, each tool's ``caller``, the null keys and most usage fields, and replaced the message id."""
    rewritten: Final = _rewritten_events(_MCP_STREAM_EVENTS, _deny("toolu_weather"))

    kept: Final = _without_pings(_without_block(_MCP_STREAM_EVENTS, 3))
    assert _without_pings(rewritten) == (*kept[:-2], *_text_block(3, _DENIAL), *kept[-2:])
    assert {"type": "ping"} in rewritten


def test_stream_whose_only_tool_call_is_denied_ends_the_turn_with_the_upstream_usage():
    upstream: Final = (_MESSAGE_START, *_client_tool_use(0, "toolu_weather", "get_weather"), *_message_end("tool_use"))

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    assert rewritten == (_MESSAGE_START, *_text_block(0, _DENIAL), *_message_end("end_turn"))


def test_denied_tool_call_replaces_the_upstream_text_where_it_was():
    thinking: Final = _streamed_block(
        0,
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "thinking_delta", "thinking": "Both calls are needed."},
        {"type": "signature_delta", "signature": "sig-1"},
    )
    redacted: Final = _streamed_block(1, {"type": "redacted_thinking", "data": "EmwKAhgB"})
    text: Final = _text_block(2, "I'll check both.")
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

    assert rewritten == (
        _MESSAGE_START,
        *thinking,
        *redacted,
        *_text_block(2, _DENIAL),
        *_client_tool_use(3, "toolu_time", "get_time"),
        *_message_end("tool_use"),
    )


def _thinking_block(index: int, thinking: str) -> tuple[dict[str, object], ...]:
    return _streamed_block(
        index,
        {"type": "thinking", "thinking": "", "signature": ""},
        {"type": "thinking_delta", "thinking": thinking},
        {"type": "signature_delta", "signature": "sig-1"},
    )


def test_masked_thinking_is_sent_where_the_upstream_thinking_was():
    upstream: Final = (
        _MESSAGE_START,
        *_thinking_block(0, "The user is a@b.test."),
        *_text_block(1, "Hello."),
        *_message_end("end_turn"),
    )

    def mask_thinking(assembled: ModelResponse) -> None:
        _only_choice(assembled).message.thinking_blocks = [
            {"type": "thinking", "thinking": "The user is [EMAIL].", "signature": "sig-1"}
        ]

    rewritten: Final = _rewritten_events(upstream, mask_thinking)

    assert rewritten == (
        _MESSAGE_START,
        *_thinking_block(0, "The user is [EMAIL]."),
        *_text_block(1, "Hello."),
        *_message_end("end_turn"),
    )


def test_rewrite_that_drops_one_of_two_redacted_thinking_blocks_keeps_the_other_where_it_was():
    kept_block: Final = _streamed_block(0, {"type": "redacted_thinking", "data": "EmwKAhgB"})
    upstream: Final = (
        _MESSAGE_START,
        *kept_block,
        *_streamed_block(1, {"type": "redacted_thinking", "data": "EmwKAhgC"}),
        *_text_block(2, "Hello."),
        *_message_end("end_turn"),
    )

    def drop_second_redacted_block(assembled: ModelResponse) -> None:
        message: Final = _only_choice(assembled).message
        message.thinking_blocks = (message.thinking_blocks or [])[:1]

    rewritten: Final = _rewritten_events(upstream, drop_second_redacted_block)

    assert rewritten == (_MESSAGE_START, *kept_block, *_text_block(1, "Hello."), *_message_end("end_turn"))


def test_masked_tool_input_keeps_the_fields_the_masking_did_not_touch():
    upstream: Final = (_MESSAGE_START, *_client_tool_use(0, "toolu_time", "get_time"), *_message_end("tool_use"))

    def mask_input(assembled: ModelResponse) -> None:
        (tool_call,) = _only_choice(assembled).message.tool_calls or []
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


def test_masked_server_tool_input_stays_a_server_tool_call():
    """The round trip reads a ``server_tool_use`` block back as a client ``tool_use``, which the client would run."""
    search_call: Final = _streamed_block(
        0,
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
        {"type": "input_json_delta", "partial_json": '{"query": "a@b.test"}'},
    )
    upstream: Final = (_MESSAGE_START, *search_call, *_message_end("end_turn"))

    def mask_query(assembled: ModelResponse) -> None:
        (tool_call,) = _only_choice(assembled).message.tool_calls or []
        tool_call.function.arguments = '{"query": "[EMAIL]"}'

    rewritten: Final = _rewritten_events(upstream, mask_query)

    assert rewritten == (
        _MESSAGE_START,
        *_streamed_block(
            0,
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query": "[EMAIL]"}'},
        ),
        *_message_end("end_turn"),
    )


_HIDDEN_CALL: Final = {"type": "tool_use", "id": "toolu_hidden", "name": "Bash", "input": {"command": "rm -rf /"}}


def test_rewrite_sends_message_start_with_the_empty_content_anthropic_sends():
    upstream: Final = (
        {**_MESSAGE_START, "message": {**_MESSAGE, "content": [_HIDDEN_CALL]}},
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
        {"type": "tool_call", "index": 0, "content_block": _HIDDEN_CALL},
        {"type": "content_block_start", "content_block": _HIDDEN_CALL},
    ],
    ids=["unknown-type", "unknown-type-with-a-block-index", "block-start-without-index"],
)
def test_rewrite_leaves_out_an_event_the_messages_stream_does_not_define(undefined_event: dict[str, object]):
    upstream: Final = (
        _MESSAGE_START,
        *_client_tool_use(0, "toolu_time", "get_time"),
        undefined_event,
        *_client_tool_use(1, "toolu_weather", "get_weather"),
        *_message_end("tool_use"),
    )

    rewritten: Final = _rewritten_events(upstream, _deny("toolu_weather"))

    assert "toolu_hidden" not in json.dumps(rewritten)
    assert rewritten == (
        _MESSAGE_START,
        *_client_tool_use(0, "toolu_time", "get_time"),
        *_text_block(1, _DENIAL),
        *_message_end("tool_use"),
    )


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

    assert rewritten == (
        _MESSAGE_START,
        *_text_block(0, masked),
        *search_call,
        *search_result,
        *_message_end("end_turn"),
    )


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

    rewritten: Final = _rewritten_events(upstream, _deny("srvtoolu_2", text="Blocked"))

    assert rewritten == (_MESSAGE_START, *_text_block(0, "Blocked"), *_message_end("end_turn"))


_VERDICTS: Final = {
    "stop_details": {"type": "refusal", "category": "cyber", "explanation": "It asked to call 555-0100"},
    "safeguard_results": [{"type": "tool_use", "tool_use_id": "toolu_x", "explanation": "dials 555-0100"}],
}


def test_masked_stream_never_sends_an_upstream_verdict_that_can_quote_the_unmasked_text():
    upstream: Final = (
        {**_MESSAGE_START, "message": {**_MESSAGE, **_VERDICTS}},
        *_text_block(0, "call 555-0100"),
        *_message_end("end_turn", **_VERDICTS),
    )

    rewritten: Final = _rewritten_events(upstream, _set_text("call [PHONE]"))

    assert "555-0100" not in json.dumps(rewritten)
    message_start, *_, message_delta, _ = rewritten
    assert message_start == {
        **_MESSAGE_START,
        "message": {key: v for key, v in _MESSAGE.items() if key != "stop_details"},
    }
    assert message_delta == {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None, "container": None},
        "usage": _FINAL_USAGE,
    }


def test_masked_stream_keeps_a_null_upstream_verdict():
    upstream: Final = (
        _MESSAGE_START,
        *_text_block(0, "call 555-0100"),
        *_message_end("end_turn", safeguard_results=None),
    )

    rewritten: Final = _rewritten_events(upstream, _set_text("call [PHONE]"))

    assert rewritten == (
        _MESSAGE_START,
        *_text_block(0, "call [PHONE]"),
        *_message_end("end_turn", safeguard_results=None),
    )


_UPSTREAM_ERROR: Final = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
_HEALTHY_PREFIX: Final = (_MESSAGE_START, *_MCP_CALL, *_MCP_RESULT, *_text_block(2, "call 555-0100"))


@pytest.mark.parametrize(
    "source_events",
    [
        (*_HEALTHY_PREFIX, _UPSTREAM_ERROR),
        (*_HEALTHY_PREFIX, {"error": {"message": "upstream gave up"}}),
        _HEALTHY_PREFIX[1:],
        (),
    ],
    ids=["anthropic-error-event", "chat-completions-error-payload", "no-message-start", "no-frames"],
)
def test_stream_that_failed_or_cannot_be_read_is_rebuilt_from_the_rewrite_alone(
    source_events: tuple[dict[str, object], ...],
):
    """No frame of a failed stream reaches the client unless the guardrail read it, as before the splice."""
    assembled: Final = assemble_anthropic_sse_stream(_events_frames((*_HEALTHY_PREFIX, *_message_end("end_turn"))))
    assert assembled is not None
    _set_text("call [PHONE]")(assembled)

    rewritten: Final = anthropic_sse_chunks_from_response(assembled, _events_frames(source_events))

    assert rewritten == _rebuilt_from_the_rewrite_alone(assembled)
    assert b"mcptoolu_1" not in b"".join(rewritten)


def test_denied_tool_call_whose_upstream_input_cannot_be_read_is_rebuilt_from_the_rewrite_alone():
    upstream: Final = _events_frames(
        (
            _MESSAGE_START,
            *_streamed_block(
                0,
                {"type": "tool_use", "id": "toolu_bad", "name": "get_weather", "input": {}},
                {"type": "input_json_delta", "partial_json": "@@ not json @@"},
            ),
            *_message_end("tool_use"),
        )
    )
    assembled: Final = assemble_anthropic_sse_stream(upstream)
    assert assembled is not None
    _deny("toolu_bad")(assembled)

    rewritten: Final = anthropic_sse_chunks_from_response(assembled, upstream)

    assert rewritten == _rebuilt_from_the_rewrite_alone(assembled)
    assert b"toolu_bad" not in b"".join(rewritten)
