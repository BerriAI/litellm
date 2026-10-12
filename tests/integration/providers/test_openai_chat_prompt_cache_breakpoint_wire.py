import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import httpx
import openai
import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, gateway_from_environment, string_value
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(120)

_MODEL: Final = "openai/gpt-5.6"
_EPHEMERAL: Final[Mapping[str, JsonValue]] = {"type": "ephemeral"}
_EPHEMERAL_1H: Final[Mapping[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_OPTED_IN: Final[Mapping[str, JsonValue]] = {"prompt_cache_options": {"mode": "explicit"}}
_LOOKUP_CALL: Final[Mapping[str, JsonValue]] = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": "{}"},
}
_LOOKUP_TOOL: Final[Mapping[str, JsonValue]] = {"type": "function", "function": {"name": "lookup", "parameters": {}}}


def _scripted(request: Request) -> Reply:
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    marker: Final = rv.newest_marker(request.body.decode())
    return chat_reply(
        f"chatcmpl-{marker}", string_value(body["model"]), rv.answer(marker), stream=body.get("stream") is True
    )


_respond: Final = answering_model_discovery(_scripted)


@dataclass(frozen=True, slots=True)
class _Chat:
    gateway: Gateway
    wire: Wire
    reading: str
    verbatim: str
    spend: pcb.SpendLogs

    @property
    def v1(self) -> str:
        return str(self.gateway.client.base_url).rstrip("/") + "/v1"


@pytest.fixture(scope="module")
def chat() -> Iterator[_Chat]:
    with (
        wire_server(_respond) as wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
        pcb.spend_logs() as spend,
    ):
        api_base: Final = f"{wire.url}/v1"
        yield _Chat(
            gateway,
            wire,
            scenario.model(model=_MODEL, api_base=api_base, **_OPTED_IN),
            scenario.model(model=_MODEL, api_base=api_base),
            spend,
        )


def _cached(block: Mapping[str, JsonValue], mark: Mapping[str, JsonValue] = _EPHEMERAL) -> dict[str, JsonValue]:
    return {**block, "cache_control": mark}


def _marked_messages(marker: str) -> list[JsonValue]:
    return [
        {"role": "system", "content": [pcb.text("Answer briefly"), _cached(pcb.text("Stable rules"))]},
        {
            "role": "user",
            "content": [_cached(pcb.text(pcb.prompt(marker)), _EPHEMERAL_1H), pcb.text("and nothing else")],
        },
    ]


def _rewritten_messages(marker: str) -> list[dict[str, JsonValue]]:
    return [
        {
            "role": "system",
            "content": [pcb.text("Answer briefly"), pcb.marked(pcb.text("Stable rules"), pcb.EXPLICIT)],
        },
        {
            "role": "user",
            "content": [pcb.marked(pcb.text(pcb.prompt(marker)), pcb.EXPLICIT), pcb.text("and nothing else")],
        },
    ]


def _post(
    chat: _Chat, model: str, messages: Sequence[JsonValue], *, stream: bool = False, tools: Sequence[JsonValue] = ()
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": list(messages), "stream": stream, **pcb.NO_CACHE}
    return chat.gateway.request("POST", "/v1/chat/completions", {**body, "tools": list(tools)} if tools else body)


def _answered(response: httpx.Response, marker: str, *, stream: bool = False) -> str:
    assert response.status_code == 200, response.text
    if stream:
        assert f'"content": "{rv.answer(marker)}"' in response.text.replace('"content":"', '"content": "'), (
            response.text
        )
    else:
        body: Final = rv.JSON_OBJECT.validate_json(response.text)
        (choice,) = rv.ITEMS.validate_python(body["choices"])
        assert rv.JSON_OBJECT.validate_python(choice["message"])["content"] == rv.answer(marker), body
    return response.headers["x-litellm-call-id"]


def _posted(wire: Wire, marker: str) -> Request:
    (request,) = pcb.with_marker(pcb.drained_posts(wire), marker)
    assert request.target == "/v1/chat/completions", request.target
    return request


def _wire_body(request: Request, *, stream: bool = False) -> dict[str, JsonValue]:
    body: Final = pcb.body_of(request)
    assert body["model"] == "gpt-5.6", body
    assert (body.get("stream") is True) is stream, body
    return body


def _wire_messages(body: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return rv.ITEMS.validate_python(body["messages"])


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_cache_control_reaches_the_wire_as_a_prompt_cache_breakpoint_on_the_same_part(
    chat: _Chat, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _post(chat, chat.reading, _marked_messages(marker), stream=stream)
    call_id: Final = _answered(response, marker, stream=stream)
    request: Final = _posted(chat.wire, marker)
    body: Final = _wire_body(request, stream=stream)
    assert body["prompt_cache_options"] == {"mode": "explicit"}, body
    assert _wire_messages(body) == _rewritten_messages(marker), body
    assert b"cache_control" not in request.body, request.body
    chat.spend.landed(chat.reading, call_id, None)


def test_openai_sdk_marks_are_rewritten_the_same_way(chat: _Chat) -> None:
    marker: Final = uuid.uuid4().hex
    with openai.OpenAI(api_key=chat.gateway.key, base_url=chat.v1, max_retries=0) as client:
        raw: Final = client.chat.completions.with_raw_response.create(
            model=chat.reading,
            messages=[
                {"role": "system", "content": "Answer briefly", "cache_control": _EPHEMERAL},  # pyright: ignore[reportArgumentType]  # Anthropic mark the SDK forwards verbatim
                {"role": "user", "content": pcb.prompt(marker)},
            ],
            extra_body=dict(pcb.NO_CACHE),
        )
    completion: Final = raw.parse()
    assert completion.choices[0].message.content == rv.answer(marker), completion
    request: Final = _posted(chat.wire, marker)
    assert _wire_messages(_wire_body(request)) == [
        {"role": "system", "content": [pcb.marked(pcb.text("Answer briefly"), pcb.EXPLICIT)]},
        {"role": "user", "content": pcb.prompt(marker)},
    ], request.body
    chat.spend.landed(chat.reading, raw.headers["x-litellm-call-id"], None)


def test_tool_message_mark_is_kept_and_tool_definition_mark_is_dropped(chat: _Chat) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": pcb.prompt(marker)},
        {"role": "assistant", "content": None, "tool_calls": [dict(_LOOKUP_CALL)]},
        {"role": "tool", "tool_call_id": "call_1", "content": [_cached(pcb.text("found it"))]},
    ]
    response: Final = _post(chat, chat.reading, messages, tools=[_cached(_LOOKUP_TOOL)])
    call_id: Final = _answered(response, marker)
    request: Final = _posted(chat.wire, marker)
    body: Final = _wire_body(request)
    assert _wire_messages(body)[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [pcb.marked(pcb.text("found it"), pcb.EXPLICIT)],
    }, "a breakpoint on a tool message part writes the cache on live gpt-5.6 Chat Completions (2026-10-12)"
    assert body["tools"] == [_LOOKUP_TOOL], body
    assert b"cache_control" not in request.body, request.body
    chat.spend.landed(chat.reading, call_id, None)


def test_assistant_mark_is_dropped(chat: _Chat) -> None:
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": [_cached(pcb.text("earlier answer"))]},
        {"role": "user", "content": pcb.prompt(marker)},
    ]
    call_id: Final = _answered(_post(chat, chat.reading, messages), marker)
    request: Final = _posted(chat.wire, marker)
    assert _wire_messages(_wire_body(request))[1] == {"role": "assistant", "content": [pcb.text("earlier answer")]}
    assert b"cache_control" not in request.body and b"prompt_cache_breakpoint" not in request.body, request.body
    chat.spend.landed(chat.reading, call_id, None)


def test_a_deployment_that_does_not_read_breakpoints_keeps_cache_control_verbatim(chat: _Chat) -> None:
    marker: Final = uuid.uuid4().hex
    call_id: Final = _answered(_post(chat, chat.verbatim, _marked_messages(marker)), marker)
    request: Final = _posted(chat.wire, marker)
    body: Final = _wire_body(request)
    assert "prompt_cache_options" not in body, body
    assert _wire_messages(body) == _marked_messages(marker), body
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    chat.spend.landed(chat.verbatim, call_id, None)
