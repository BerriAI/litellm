import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import httpx
import openai
import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

_EPHEMERAL: Final[Mapping[str, JsonValue]] = {"type": "ephemeral"}
_EPHEMERAL_1H: Final[Mapping[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_OPTED_IN: Final[Mapping[str, JsonValue]] = {"prompt_cache_options": {"mode": "explicit"}}
_LOOKUP_TOOL: Final[Mapping[str, JsonValue]] = {
    "type": "function",
    "name": "lookup",
    "parameters": {"type": "object", "properties": {}},
}


@dataclass(frozen=True, slots=True)
class _Native:
    gateway: Gateway
    wire: Wire
    reading: str
    stripping: str
    spend: pcb.SpendLogs


@pytest.fixture(scope="module")
def native() -> Iterator[_Native]:
    with (
        wire_server(pcb.respond) as wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
        pcb.spend_logs() as spend,
    ):
        api_base: Final = f"{wire.url}/v1"
        yield _Native(
            gateway,
            wire,
            scenario.model(model=pcb.MODEL, api_base=api_base, **_OPTED_IN),
            scenario.model(model=pcb.MODEL, api_base=api_base),
            spend,
        )


def _v1(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _input_text(text: str) -> dict[str, JsonValue]:
    return {"type": "input_text", "text": text}


def _marked(
    block: Mapping[str, JsonValue], cache_control: Mapping[str, JsonValue] = _EPHEMERAL
) -> dict[str, JsonValue]:
    return {**block, "cache_control": dict(cache_control)}


def _marked_history(marker: str) -> list[JsonValue]:
    return [
        {
            "type": "message",
            "role": "developer",
            "content": [_input_text("Answer briefly"), _marked(_input_text("Stable rules"), _EPHEMERAL_1H)],
        },
        _marked({"role": "user", "content": pcb.prompt(marker)}),
    ]


def _responses(native: _Native, model: str, input: Sequence[JsonValue], *, stream: bool = False) -> httpx.Response:
    return native.gateway.request(
        "POST", "/v1/responses", {"model": model, "input": list(input), "stream": stream, **pcb.NO_CACHE}
    )


def _answered(response: httpx.Response, marker: str) -> str:
    assert response.status_code == 200, response.text
    assert rv.answer(marker) in response.text, response.text
    return response.headers["x-litellm-call-id"]


def _wire_body(request: Request, *, stream: bool = False) -> dict[str, JsonValue]:
    body: Final = pcb.body_of(request)
    assert body["model"] == "gpt-6.1-sol", body
    assert (body.get("stream") is True) is stream, body
    assert b"cache_control" not in request.body, request.body
    return body


def _assert_marked_history_on_wire(body: Mapping[str, JsonValue], marker: str) -> None:
    assert body["prompt_cache_options"] == {"mode": "explicit"}, body
    assert body["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [
                _input_text("Answer briefly"),
                pcb.marked(_input_text("Stable rules"), pcb.EXPLICIT),
            ],
        },
        {"role": "user", "content": [pcb.marked(_input_text(pcb.prompt(marker)), pcb.EXPLICIT)]},
    ], body


def test_openai_sdk_cache_control_reaches_the_wire_as_a_prompt_cache_breakpoint_on_the_same_block(
    native: _Native,
) -> None:
    marker: Final = uuid.uuid4().hex
    with openai.OpenAI(api_key=native.gateway.key, base_url=_v1(native.gateway), max_retries=0) as client:
        raw: Final = client.responses.with_raw_response.create(
            model=native.reading, input=_marked_history(marker), extra_body=dict(pcb.NO_CACHE)
        )
    response: Final = raw.parse()
    assert pcb.answers(response.id, marker), response
    assert response.output_text == rv.answer(marker), response
    _assert_marked_history_on_wire(_wire_body(pcb.posted(native.wire, marker)), marker)
    native.spend.landed(native.reading, raw.headers["x-litellm-call-id"], marker)


def test_stream_carries_the_marks_the_same_way(native: _Native) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _answered(_responses(native, native.reading, _marked_history(marker), stream=True), marker)
    _assert_marked_history_on_wire(_wire_body(pcb.posted(native.wire, marker), stream=True), marker)
    native.spend.landed(native.reading, call_id, marker)


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("found it", [pcb.marked(_input_text("found it"), pcb.EXPLICIT)]),
        (
            [_input_text("first"), _input_text("last")],
            [_input_text("first"), pcb.marked(_input_text("last"), pcb.EXPLICIT)],
        ),
    ),
    ids=("string-output", "list-output"),
)
def test_cache_control_on_a_function_call_output_marks_its_last_part(
    native: _Native, output: JsonValue, expected: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        _marked({"type": "function_call_output", "call_id": "call_1", "output": output}),
    ]
    call_id: Final = _answered(_responses(native, native.reading, history), marker)
    request: Final = pcb.posted(native.wire, marker)
    _wire_body(request)
    assert pcb.function_output(pcb.input_items(request), "call_1") == expected, (
        "OpenAI caches up to a breakpoint on a function_call_output part (prompt caching guide, read 2026-10-12)"
    )
    native.spend.landed(native.reading, call_id, marker)


def test_cache_control_on_an_assistant_turn_and_on_a_tool_is_dropped(native: _Native) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": "first question"},
        {
            "type": "message",
            "role": "assistant",
            "content": [_marked({"type": "output_text", "text": "earlier answer"})],
        },
        {"role": "user", "content": pcb.prompt(marker)},
    ]
    response: Final = native.gateway.request(
        "POST",
        "/v1/responses",
        {"model": native.reading, "input": history, "tools": [_marked(_LOOKUP_TOOL)], **pcb.NO_CACHE},
    )
    call_id: Final = _answered(response, marker)
    request: Final = pcb.posted(native.wire, marker)
    body: Final = _wire_body(request)
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert pcb.content_of(pcb.input_items(request), "assistant") == [
        {"type": "output_text", "text": "earlier answer"}
    ], "OpenAI accepts a breakpoint on output_text but writes no cache for it (live gpt-5.6, 2026-10-12)"
    assert body["tools"] == [dict(_LOOKUP_TOOL)], body
    native.spend.landed(native.reading, call_id, marker)


def test_cache_control_is_stripped_where_the_deployment_does_not_read_breakpoints(native: _Native) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _answered(_responses(native, native.stripping, _marked_history(marker)), marker)
    request: Final = pcb.posted(native.wire, marker)
    body: Final = _wire_body(request)
    assert "prompt_cache_options" not in body, body
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    assert body["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [_input_text("Answer briefly"), _input_text("Stable rules")],
        },
        {"role": "user", "content": pcb.prompt(marker)},
    ], body
    native.spend.landed(native.stripping, call_id, marker)


def test_a_breakpoint_the_client_already_wrote_is_left_alone_beside_a_converted_mark(native: _Native) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                pcb.marked(_input_text(pcb.prompt(marker)), pcb.EXPLICIT_30M),
                _marked(_input_text("and nothing else")),
            ],
        }
    ]
    call_id: Final = _answered(_responses(native, native.reading, history), marker)
    request: Final = pcb.posted(native.wire, marker)
    _wire_body(request)
    (item,) = pcb.input_items(request)
    assert item["content"] == [
        pcb.marked(_input_text(pcb.prompt(marker)), pcb.EXPLICIT_30M),
        pcb.marked(_input_text("and nothing else"), pcb.EXPLICIT),
    ], item
    native.spend.landed(native.reading, call_id, marker)
