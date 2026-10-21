import asyncio
import base64
import json
import os
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame, aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

_MODEL_ID: Final = "global.moonshotai.kimi-k3"
_CONVERSE_MODEL: Final = f"bedrock/converse/{_MODEL_ID}"
_STREAM_TARGET: Final = f"/model/{_MODEL_ID}/converse-stream"
_CONVERSE_TARGET: Final = f"/model/{_MODEL_ID}/converse"
_INVOKE_MODEL_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_INVOKE_MODEL: Final = f"bedrock/invoke/{_INVOKE_MODEL_ID}"
_INVOKE_STREAM_TARGET: Final = f"/model/{_INVOKE_MODEL_ID}/invoke-with-response-stream"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_ANSWER: Final = "bedrock event frame control"
_REJECTION: Final = "structured output schema uses unsupported regex negative look-ahead"
_THROTTLED: Final = "Too many requests, please wait before trying again"
_UNKNOWN_TYPE: Final = "somethingBedrockAddedLater"
_UNKNOWN_ONLY: Final = "none of its 1 events carried a known event type"
_SIGNING_KEY: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_JSON_HEADERS: Final = MappingProxyType({":content-type": "application/json", ":message-type": "event"})
_USAGE: Final[dict[str, JsonValue]] = {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}
_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
        "stopReason": "end_turn",
        **_USAGE,
        "metrics": {"latencyMs": 1},
    }
).encode()
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_PROMPT: Final = "What does the gateway do with this stream?"
_USER_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": _PROMPT}
_CONVERSE_USER_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": [{"text": _PROMPT}]}
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CALL_INDEX: Final = re.compile(r"call-[0-9a-f]{32}-(\d+)")
_PLAIN: Final = "bedrock-event-frames-plain"

Endpoint = Literal["chat", "messages", "responses"]


def _error_body(message: str) -> str:
    return json.dumps({"message": message}, separators=(",", ":"))


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _typed_frame(headers: Mapping[str, str | int], payload: bytes) -> bytes:
    return aws_event_stream_frame({**headers, **_JSON_HEADERS}, payload)


def _exception_message_frame(exception_type: str, message: str) -> bytes:
    return aws_event_stream_frame(
        {":exception-type": exception_type, ":content-type": "application/json", ":message-type": "exception"},
        _error_body(message).encode(),
    )


def _text_frames(text: str) -> tuple[bytes, ...]:
    return (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
    )


_NORMAL: Final = b"".join(
    (*_text_frames(_ANSWER), _frame("messageStop", {"stopReason": "end_turn"}), _frame("metadata", _USAGE))
)
_VALIDATION_FRAME: Final = _frame("validationException", {"message": _REJECTION})
_THROTTLING_FRAME: Final = _frame("throttlingException", {"message": _THROTTLED})
_UNKNOWN_FRAME: Final = _frame(_UNKNOWN_TYPE, {"future": True})
_UNKNOWN_BESIDE_KNOWN: Final = b"".join(
    (
        *_text_frames(_ANSWER),
        _UNKNOWN_FRAME,
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", _USAGE),
    )
)
_THROTTLED_AFTER_TEXT: Final = b"".join((*_text_frames(_ANSWER), _THROTTLING_FRAME))
_EMPTY_DELTA_BESIDE_KNOWN: Final = b"".join(
    (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {}),
        _frame("contentBlockDelta", {"delta": {"text": _ANSWER}, "contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", _USAGE),
    )
)
_EXCEPTION_MID_STREAM: Final = b"".join((*_text_frames(_ANSWER), _VALIDATION_FRAME))
_EXCEPTION_MESSAGE_MID_STREAM: Final = b"".join(
    (*_text_frames(_ANSWER), _exception_message_frame("throttlingException", _THROTTLED))
)


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    return _frame("chunk", {"bytes": base64.b64encode(json.dumps(event).encode()).decode()})


_INVOKE_STREAM: Final = b"".join(
    (
        _invoke_chunk(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_invoke",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": _INVOKE_MODEL_ID,
                    "stop_reason": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            }
        ),
        _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _ANSWER}}),
        _invoke_chunk({"type": "content_block_stop", "index": 0}),
        _invoke_chunk({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        _invoke_chunk({"type": "message_stop"}),
    )
)


@dataclass(frozen=True, slots=True)
class _Streamed:
    status: int
    call_id: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    user: str
    index: int


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    call_id: str
    text: str


def _stream_peer(frames: bytes) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if unquote(request.target) in (_STREAM_TARGET, _INVOKE_STREAM_TARGET):
            return Reply(body=frames, content_type=_EVENT_STREAM)
        return Reply(body=_RESPONSE)

    return respond


def _converse_deployment(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(model=_CONVERSE_MODEL, api_base=wire.url, **_AWS, **extra)


def _auth(gateway: Gateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {gateway.key}"}


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(endpoint: Endpoint, model: str, *, user: str | None = None) -> dict[str, JsonValue]:
    marker: Final[dict[str, JsonValue]] = {} if user is None else {"user": user}
    match endpoint:
        case "chat":
            return {"model": model, "messages": [_USER_TURN], "max_tokens": 16, "stream": True, **_EXTRA, **marker}
        case "messages":
            return {"model": model, "messages": [_USER_TURN], "max_tokens": 16, "stream": True, **_EXTRA}
        case "responses":
            return {"model": model, "input": _PROMPT, "stream": True, **_EXTRA, **marker}


def _stream(
    gateway: Gateway, endpoint: Endpoint, body: Mapping[str, JsonValue], *, key: str | None = None
) -> _Streamed:
    headers: Final = _auth(gateway) if key is None else {"Authorization": f"Bearer {key}"}
    with gateway.client.stream("POST", _path(endpoint), json=body, headers=headers) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
        return _Streamed(response.status_code, response.headers.get("x-litellm-call-id", ""), lines)


def _sse_payloads(lines: Iterable[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]")


def _sse_events(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(line[7:] for line in lines if line.startswith("event: "))


def _first_choice(chunk: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    choices: Final = chunk.get("choices")
    return _JSON.validate_python(choices[0]) if isinstance(choices, list) and choices else None


def _chat_text(chunks: Iterable[dict[str, JsonValue]]) -> str:
    choices: Final = tuple(choice for choice in map(_first_choice, chunks) if choice is not None)
    return "".join(str(_JSON.validate_python(choice["delta"]).get("content") or "") for choice in choices)


def _finish_reasons(chunks: Iterable[dict[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(choice.get("finish_reason") for choice in map(_first_choice, chunks) if choice is not None)


def _chat_id(chunks: Iterable[dict[str, JsonValue]]) -> str:
    (identity,) = {str(chunk["id"]) for chunk in chunks if "id" in chunk}
    return identity


def _message_id(payloads: Iterable[dict[str, JsonValue]]) -> str:
    (started,) = tuple(payload for payload in payloads if payload.get("type") == "message_start")
    return str(_JSON.validate_python(started["message"])["id"])


def _messages_text(payloads: Iterable[dict[str, JsonValue]]) -> str:
    deltas: Final = tuple(payload for payload in payloads if payload.get("type") == "content_block_delta")
    return "".join(str(_JSON.validate_python(delta["delta"]).get("text") or "") for delta in deltas)


def _responses_text(events: Iterable[dict[str, JsonValue]]) -> str:
    return "".join(
        str(event.get("delta") or "") for event in events if event.get("type") == "response.output_text.delta"
    )


def _inner_response_id(identity: str) -> str:
    managed: Final = decrypt_if_encrypted_with(identity.removeprefix("resp_"), _SIGNING_KEY)
    assert managed is not None, identity
    issued: Final = managed.split(";", 1)[0].rsplit("response_id:", 1)[1]
    decoded: Final = base64.b64decode(issued.removeprefix("resp_")).decode()
    return decoded.rsplit("response_id:", 1)[1]


def _completed_response_id(events: Iterable[dict[str, JsonValue]]) -> str:
    (completed,) = tuple(event for event in events if event.get("type") == "response.completed")
    return _inner_response_id(str(_JSON.validate_python(completed["response"])["id"]))


def _only_received(wire: Wire) -> tuple[str, dict[str, JsonValue]]:
    (request,) = wire.drain()
    return unquote(request.target), _JSON.validate_python(json.loads(request.body))


def _assert_stream_request(wire: Wire, target: str = _STREAM_TARGET) -> dict[str, JsonValue]:
    received_target, received = _only_received(wire)
    assert received_target == target, received_target
    return received


def _spend_row(request_id: str) -> dict[str, JsonValue]:
    assert request_id, "No id to look the spend row up by"
    (row,) = eventually(
        lambda: read_rows(
            'SELECT request_id, status, call_type, end_user FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return row


def _failure_row(call_id: str) -> dict[str, JsonValue]:
    row: Final = _spend_row(call_id)
    assert row["status"] == "failure", row
    return row


def _success_row(request_id: str) -> dict[str, JsonValue]:
    row: Final = _spend_row(request_id)
    assert row["status"] == "success", row
    return row


def _rows_for(request_ids: frozenset[str], *, expected: int) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(string_to_array(%s, %s))',
            (",".join(sorted(request_ids)), ","),
        ),
        lambda found: len(found) >= expected,
        seconds=90,
    )
    return tuple(rows)


def _assert_rejected_chat(streamed: _Streamed, status: int, message: str) -> None:
    assert streamed.status == status, (streamed.status, streamed.text)
    error: Final = _JSON.validate_python(json.loads(streamed.text)["error"])
    assert message in str(error["message"]), streamed.text
    assert str(error["code"]) == str(status), streamed.text


def _unescaped(text: str) -> str:
    return text.replace('\\"', '"')


def _assert_rejected_stream_body(streamed: _Streamed, message: str) -> None:
    assert streamed.status == 200, (streamed.status, streamed.text)
    assert message in _unescaped(streamed.text), streamed.text
    assert _ANSWER not in streamed.text, streamed.text


def test_r01_chat_stream_with_a_validation_exception_frame_first_is_a_400_and_a_failure_row(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        _assert_rejected_chat(streamed, 400, f"validationException {_error_body(_REJECTION)}")
        received: Final = _assert_stream_request(wire)
        assert received["messages"] == [_CONVERSE_USER_TURN], received
        _failure_row(streamed.call_id)


async def _consume_openai_chat_stream(client: openai.AsyncOpenAI, model: str) -> None:
    stream = await client.chat.completions.create(
        model=model, messages=[_USER_TURN], max_tokens=16, stream=True, extra_body=_EXTRA
    )
    _ = [chunk async for chunk in stream]


async def test_r02_openai_async_sdk_stream_with_a_validation_exception_frame_first_raises_bad_request(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        async with openai.AsyncOpenAI(
            base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            with pytest.raises(openai.BadRequestError, match=re.escape(_REJECTION)) as raised:
                await _consume_openai_chat_stream(client, model)
        _assert_stream_request(wire)
        _failure_row(raised.value.response.headers.get("x-litellm-call-id", ""))


def test_r03_chat_stream_throttled_after_text_delivers_the_text_then_the_error_and_no_stop_chunk(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_THROTTLED_AFTER_TEXT)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        assert streamed.status == 200, streamed.text
        chunks: Final = _sse_payloads(streamed.lines)
        assert _chat_text(chunks) == _ANSWER, streamed.text
        assert "throttlingException" in streamed.text and _THROTTLED in streamed.text, streamed.text
        assert "stop" not in _finish_reasons(chunks), streamed.text
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


def test_r04_messages_stream_with_a_validation_exception_frame_first_emits_an_error_event_and_no_message_stop(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "messages", _body("messages", model))
        _assert_rejected_stream_body(streamed, f"validationException {_error_body(_REJECTION)}")
        events: Final = _sse_events(streamed.lines)
        assert "error" in events and "message_stop" not in events, streamed.text
        _assert_stream_request(wire)


async def test_r05_anthropic_async_sdk_stream_with_a_validation_exception_frame_first_raises(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        async with anthropic.AsyncAnthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(anthropic.APIError, match=re.escape(_REJECTION)):
                async with client.messages.stream(
                    model=model, max_tokens=16, messages=[_USER_TURN], extra_body=_EXTRA
                ) as stream:
                    _ = [event async for event in stream]
        _assert_stream_request(wire)


def test_r06_responses_stream_with_a_validation_exception_frame_first_fails_the_response(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "responses", _body("responses", model))
        _assert_rejected_stream_body(streamed, f"validationException {_error_body(_REJECTION)}")
        types: Final = tuple(str(event["type"]) for event in _sse_payloads(streamed.lines))
        assert "response.failed" in types and "response.completed" not in types, streamed.text
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


def test_r07_chat_stream_whose_only_frame_has_an_unknown_event_type_is_a_502_naming_the_type(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_UNKNOWN_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        _assert_rejected_chat(streamed, 502, f"{_UNKNOWN_ONLY} (event types=['{_UNKNOWN_TYPE}']")
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


def _assert_text_stream(gateway: Gateway, endpoint: Endpoint, frames: bytes) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    with wire_server(_stream_peer(frames)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, endpoint, _body(endpoint, model, user=marker))
        assert streamed.status == 200, streamed.text
        payloads: Final = _sse_payloads(streamed.lines)
        match endpoint:
            case "chat":
                assert _chat_text(payloads) == _ANSWER, streamed.text
                assert _finish_reasons(payloads)[-1] == "stop", streamed.text
                _success_row(_chat_id(payloads))
            case "messages":
                assert _messages_text(payloads) == _ANSWER, streamed.text
                assert _sse_events(streamed.lines)[-1] == "message_stop", streamed.text
                _success_row(_message_id(payloads))
            case "responses":
                assert _responses_text(payloads) == _ANSWER, streamed.text
                assert str(payloads[-1]["type"]) == "response.completed", streamed.text
                assert _success_row(_completed_response_id(payloads))["end_user"] == marker
        _assert_stream_request(wire)


@pytest.mark.parametrize(
    "endpoint", ("chat", "messages", "responses"), ids=("r08-chat", "r08-messages", "r08-responses")
)
def test_r08_an_unknown_frame_between_known_frames_leaves_the_text_and_the_success_row_intact(
    gateway: Gateway, endpoint: Endpoint
) -> None:
    _assert_text_stream(gateway, endpoint, _UNKNOWN_BESIDE_KNOWN)


@pytest.mark.parametrize(
    "endpoint", ("chat", "messages", "responses"), ids=("r09-chat", "r09-messages", "r09-responses")
)
def test_r09_a_normal_converse_stream_delivers_the_text_and_a_success_row(gateway: Gateway, endpoint: Endpoint) -> None:
    _assert_text_stream(gateway, endpoint, _NORMAL)


def test_r10_a_non_streaming_converse_call_is_untouched(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_NORMAL)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [_USER_TURN], "max_tokens": 16, **_EXTRA}
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == _ANSWER, response.text
        _assert_stream_request(wire, _CONVERSE_TARGET)
        _success_row(str(response.json()["id"]))


def test_r11_an_invoke_framed_anthropic_stream_is_untouched(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_INVOKE_STREAM)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_INVOKE_MODEL, api_key=None, aws_bedrock_runtime_endpoint=wire.url, api_base=wire.url, **_AWS
        )
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        assert streamed.status == 200, streamed.text
        chunks: Final = _sse_payloads(streamed.lines)
        assert _chat_text(chunks) == _ANSWER, streamed.text
        assert _finish_reasons(chunks)[-1] == "stop", streamed.text
        _assert_stream_request(wire, _INVOKE_STREAM_TARGET)
        _success_row(_chat_id(chunks))


@pytest.mark.parametrize(
    ("exception_type", "expected"),
    (("throttlingException", 429), ("somethingNewException", 400)),
    ids=("r12", "r13"),
)
def test_r12_r13_an_exception_message_frame_keeps_its_modeled_status(
    gateway: Gateway, exception_type: str, expected: int
) -> None:
    with (
        wire_server(_stream_peer(_exception_message_frame(exception_type, _THROTTLED))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        _assert_rejected_chat(streamed, expected, _THROTTLED)
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


@pytest.mark.parametrize(
    "frames", (_EXCEPTION_MID_STREAM, _EXCEPTION_MESSAGE_MID_STREAM), ids=("r14-event-frame", "r15-exception-message")
)
def test_r14_r15_passthrough_converse_stream_relays_an_exception_frame_byte_for_byte(
    gateway: Gateway, frames: bytes
) -> None:
    with wire_server(_stream_peer(frames)) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}", api_base=wire.url, aws_bedrock_runtime_endpoint=wire.url, **_AWS
        )
        response: Final = gateway.request(
            "POST", f"/bedrock/model/{deployment}/converse-stream", {"messages": [_CONVERSE_USER_TURN]}
        )
        assert response.status_code == 200, response.text
        assert response.headers.get("content-type") == _EVENT_STREAM, dict(response.headers)
        assert response.content == frames, response.content
        _assert_stream_request(wire)


_HEADERLESS: Final = _typed_frame({}, b'{"future": true}')
_INT_TYPED: Final = _typed_frame({":event-type": 7}, b'{"future": true}')
_EMPTY_TYPED: Final = _typed_frame({":event-type": ""}, b'{"future": true}')
_LONG_TYPE: Final = "x" * 5120
_LONG_TYPED: Final = _typed_frame({":event-type": _LONG_TYPE}, b'{"future": true}')


@pytest.mark.parametrize(
    ("frames", "named"),
    (
        (_HEADERLESS, f"{_UNKNOWN_ONLY} (event types=['<missing>']"),
        (_INT_TYPED, f"{_UNKNOWN_ONLY} (event types=['<missing>']"),
        (_EMPTY_TYPED, f"{_UNKNOWN_ONLY} (event types=['<missing>']"),
        (_LONG_TYPED, f"{_UNKNOWN_ONLY} (event types=['{_LONG_TYPE}']"),
        (
            _UNKNOWN_FRAME + _UNKNOWN_FRAME,
            f"none of its 2 events carried a known event type (event types=['{_UNKNOWN_TYPE}']",
        ),
    ),
    ids=(
        "s01-no-event-type",
        "s02-int-event-type",
        "s03-empty-event-type",
        "s04-5kb-event-type",
        "s05-same-type-twice",
    ),
)
def test_s01_to_s05_odd_event_type_headers_alone_are_a_502_that_names_what_arrived(
    gateway: Gateway, frames: bytes, named: str
) -> None:
    with wire_server(_stream_peer(frames)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        _assert_rejected_chat(streamed, 502, named)
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


def test_s06_a_validation_exception_frame_with_a_non_utf8_body_is_a_400_with_the_bytes_replaced(
    gateway: Gateway,
) -> None:
    frame: Final = _typed_frame({":event-type": "validationException"}, b'{"message": "bad \xff\xfe bytes"}')
    with wire_server(_stream_peer(frame)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        _assert_rejected_chat(streamed, 400, 'validationException {"message": "bad �� bytes"}')
        _assert_stream_request(wire)
        _failure_row(streamed.call_id)


def test_s07_a_validation_exception_frame_with_an_empty_body_is_still_a_400(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_typed_frame({":event-type": "validationException"}, b""))) as wire:
        with gateway.scenario() as scenario:
            model: Final = _converse_deployment(scenario, wire)
            streamed: Final = _stream(gateway, "chat", _body("chat", model))
            _assert_rejected_chat(streamed, 400, "validationException")
            _assert_stream_request(wire)
            _failure_row(streamed.call_id)


def test_s08_a_known_frame_with_an_empty_body_beside_normal_frames_keeps_the_text(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_EMPTY_DELTA_BESIDE_KNOWN)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model))
        assert streamed.status == 200, streamed.text
        chunks: Final = _sse_payloads(streamed.lines)
        assert _chat_text(chunks) == _ANSWER, streamed.text
        assert _finish_reasons(chunks)[-1] == "stop", streamed.text
        _assert_stream_request(wire)
        _success_row(_chat_id(chunks))


def test_s09_an_unauthenticated_stream_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = _stream(gateway, "chat", _body("chat", model), key="sk-integration-bogus")
        assert streamed.status == 401, streamed.text
        assert wire.drain() == (), streamed.text


def test_s10_a_rejected_stream_leaves_a_healthy_deployment_serving(gateway: Gateway) -> None:
    with (
        wire_server(_stream_peer(_VALIDATION_FRAME)) as rejecting,
        wire_server(_stream_peer(_NORMAL)) as healthy,
        gateway.scenario() as scenario,
    ):
        rejected_model: Final = _converse_deployment(scenario, rejecting)
        healthy_model: Final = _converse_deployment(scenario, healthy)
        _assert_rejected_chat(_stream(gateway, "chat", _body("chat", rejected_model)), 400, "validationException")
        streamed: Final = _stream(gateway, "chat", _body("chat", healthy_model))
        assert streamed.status == 200, streamed.text
        assert _chat_text(_sse_payloads(streamed.lines)) == _ANSWER, streamed.text
        assert len(rejecting.drain()) == 1 and len(healthy.drain()) == 1


def test_e01_a_rejected_stream_is_never_served_from_the_response_cache(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        cached: Final = {key: value for key, value in _body("chat", model).items() if key != "cache"}
        first: Final = _stream(gateway, "chat", cached)
        second: Final = _stream(gateway, "chat", cached)
        _assert_rejected_chat(first, 400, "validationException")
        _assert_rejected_chat(second, 400, "validationException")
        assert len(wire.drain()) == 2, (first.text, second.text)


def _sibling_deployment(scenario: Scenario, name: str, wire: Wire, **extra: JsonValue) -> None:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": _CONVERSE_MODEL, "api_base": wire.url, **_AWS, **extra},
            "model_info": {},
        },
    )
    identity: Final = _JSON.validate_python(created["model_info"])["id"]
    assert isinstance(identity, str), created
    scenario.cleanups.callback(scenario.delete_model, identity)


def test_e02_a_throttling_frame_first_is_a_429_after_one_attempt_even_with_retries_and_a_sibling_deployment(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_peer(_THROTTLING_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire, num_retries=2)
        _sibling_deployment(scenario, model, wire, num_retries=2)
        streamed: Final = _stream(gateway, "chat", {**_body("chat", model), "num_retries": 2})
        _assert_rejected_chat(streamed, 429, f"throttlingException {_error_body(_THROTTLED)}")
        attempts: Final = len(wire.drain())
        assert attempts == 1, attempts
        _failure_row(streamed.call_id)


def test_e03_three_rejected_streams_land_one_failure_row_each(gateway: Gateway) -> None:
    with wire_server(_stream_peer(_VALIDATION_FRAME)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        streamed: Final = tuple(_stream(gateway, "chat", _body("chat", model)) for _ in range(3))
        for item in streamed:
            _assert_rejected_chat(item, 400, "validationException")
        call_ids: Final = frozenset(item.call_id for item in streamed)
        assert len(call_ids) == 3, streamed
        rows: Final = _rows_for(call_ids, expected=3)
        assert {str(row["request_id"]) for row in rows} == call_ids, rows
        assert all(row["status"] == "failure" for row in rows), rows
        assert len(wire.drain()) == 3


def _calls(marker: str, endpoint: Endpoint, indexes: range) -> tuple[_Call, ...]:
    return tuple(_Call(endpoint, f"{marker}-{index}", index) for index in indexes)


def _burst_body(model: str, call: _Call) -> dict[str, JsonValue]:
    prompt: Final = f"{_PROMPT} {call.user}"
    match call.endpoint:
        case "chat":
            return {**_body("chat", model, user=call.user), "messages": [{"role": "user", "content": prompt}]}
        case "messages":
            return {**_body("messages", model), "messages": [{"role": "user", "content": prompt}]}
        case "responses":
            return {**_body("responses", model, user=call.user), "input": prompt}


def _call_index(request: Request) -> int:
    found: Final = _CALL_INDEX.search(request.body.decode())
    assert found is not None, request.body
    return int(found.group(1))


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST", _path(call.endpoint), json=_burst_body(model, call), headers={"Authorization": f"Bearer {key}"}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, response.headers.get("x-litellm-call-id", ""), raw.decode())


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _carries_text(served: _Served) -> bool:
    return _ANSWER in served.text


def _is_rejection(served: _Served, message: str) -> bool:
    return (
        message in _unescaped(served.text) and not _carries_text(served) and '"finish_reason":"stop"' not in served.text
    )


def _served_success_id(served: _Served) -> str:
    payloads: Final = _sse_payloads(served.text.splitlines())
    match served.call.endpoint:
        case "chat":
            return _chat_id(payloads)
        case "messages":
            return _message_id(payloads)
        case "responses":
            return _completed_response_id(payloads)


async def test_c01_a_mixed_burst_of_normal_rejected_and_unknown_streams_sorts_every_call_and_row(
    gateway: Gateway,
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = (
        *_calls(marker, "chat", range(0, 10)),
        *_calls(marker, "messages", range(10, 20)),
        *_calls(marker, "responses", range(20, 30)),
    )

    def respond(request: Request) -> Reply:
        match _call_index(request) % 3:
            case 1:
                return Reply(body=_VALIDATION_FRAME, content_type=_EVENT_STREAM)
            case 2:
                return Reply(body=_UNKNOWN_FRAME, content_type=_EVENT_STREAM)
            case _:
                return Reply(body=_NORMAL, content_type=_EVENT_STREAM)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        served: Final = await _burst(_proxy_url(gateway), gateway.key, model, calls)
        assert len(served) == 30
        normal: Final = tuple(item for item in served if item.call.index % 3 == 0)
        rejected: Final = tuple(item for item in served if item.call.index % 3 == 1)
        unknown: Final = tuple(item for item in served if item.call.index % 3 == 2)
        for item in normal:
            assert item.status == 200 and _carries_text(item), (item.call, item.status, item.text)
        for item in rejected:
            assert _is_rejection(item, _REJECTION), (item.call, item.status, item.text)
        for item in unknown:
            assert _is_rejection(item, _UNKNOWN_ONLY), (item.call, item.status, item.text)
        failed_ids: Final = frozenset(
            item.call_id for item in (*rejected, *unknown) if item.call.endpoint != "messages"
        )
        assert len(failed_ids) == 13, failed_ids
        failure_rows: Final = _rows_for(failed_ids, expected=13)
        assert {str(row["request_id"]) for row in failure_rows} == failed_ids, failure_rows
        assert all(row["status"] == "failure" for row in failure_rows), failure_rows
        success_ids: Final = frozenset(_served_success_id(item) for item in normal)
        assert len(success_ids) == 10, success_ids
        success_rows: Final = _rows_for(success_ids, expected=10)
        assert {str(row["request_id"]) for row in success_rows} == success_ids, success_rows
        assert all(row["status"] == "success" for row in success_rows), success_rows
        assert len(wire.drain()) == 30


def _owned_config(wire: Wire, directory: Path) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final[dict[str, JsonValue]] = {
        **base,
        "model_list": [
            {
                "model_name": _PLAIN,
                "litellm_params": {
                    "model": _CONVERSE_MODEL,
                    "api_base": wire.url,
                    "api_key": "integration-provider-key",
                    **_AWS,
                },
            }
        ],
        "router_settings": {**_JSON.validate_python(base["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / "bedrock-event-frames.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_peer_connections(pid: int, peer_url: str) -> int:
    port: Final = urlsplit(peer_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _held_rejection(release: threading.Event, held_indexes: SimpleQueue[int]) -> Callable[[Request], Reply]:
    first: Final = _frame("messageStart", {"role": "assistant"})

    def held(request: Request) -> Reply:
        held_indexes.put(_call_index(request))
        return Reply(content_type=_EVENT_STREAM, chunks=(first, _VALIDATION_FRAME), gate_after_first=release)

    return held


@pytest.mark.timeout(600)
async def test_c02_worker_sigkill_mid_burst_leaves_the_sibling_rejecting_the_held_streams(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    again: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = _calls(marker, "chat", range(20))
    release: Final = threading.Event()
    held_indexes: Final[SimpleQueue[int]] = SimpleQueue()
    with wire_server(_held_rejection(release, held_indexes)) as wire:
        config: Final = _owned_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(_proxy_url(candidate), candidate.key, _PLAIN, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_indexes.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_peer_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            psutil.Process(victim_pid).send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                assert item.status in (200, 400) and _is_rejection(item, _REJECTION), (
                    item.call,
                    item.status,
                    item.text,
                )
            eventually(
                lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda count: count == 3, seconds=60
            )
            follow_up: Final = await _burst(
                _proxy_url(candidate), candidate.key, _PLAIN, _calls(again, "chat", range(6))
            )
            assert len(follow_up) == 6
            for item in follow_up:
                assert item.status in (200, 400) and _is_rejection(item, _REJECTION), (
                    item.call,
                    item.status,
                    item.text,
                )
            assert len(wire.drain()) == 26
            served_ids: Final = frozenset(item.call_id for item in (*served, *follow_up))
            assert len(served_ids) == len(served) + 6, served_ids
            rows: Final = _rows_for(served_ids, expected=len(served_ids))
            assert {str(row["request_id"]) for row in rows} == served_ids, rows
            assert all(row["status"] == "failure" for row in rows), rows


@pytest.mark.timeout(600)
async def test_c03_proxy_terminated_mid_burst_lands_every_served_rejection_at_most_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = _calls(marker, "chat", range(12))
    release: Final = threading.Event()
    held_indexes: Final[SimpleQueue[int]] = SimpleQueue()
    with wire_server(_held_rejection(release, held_indexes)) as wire:
        config: Final = _owned_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=1) as owned:
            candidate: Final = owned.gateway
            burst: Final = asyncio.create_task(
                _burst(_proxy_url(candidate), candidate.key, _PLAIN, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_indexes.qsize, lambda size: size == 12, 60)
            owned.process.terminate()
            release.set()
            served: Final = await burst
            eventually(owned.process.poll, lambda code: code is not None, seconds=60)
            assert len(served) <= 12
            for item in served:
                assert _is_rejection(item, _REJECTION), (item.call, item.status, item.text)
            served_ids: Final = frozenset(item.call_id for item in served if item.call_id)
            landed: Final = tuple(str(row["request_id"]) for row in _rows_for(served_ids, expected=0))
            assert len(landed) == len(set(landed)), landed
            assert set(landed) <= served_ids, (landed, served_ids)
            assert len(wire.drain()) == 12
