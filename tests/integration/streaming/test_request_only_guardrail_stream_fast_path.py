import asyncio
import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import anthropic
import httpx
import openai
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

pytestmark: Final = [
    pytest.mark.timeout(2 * graceful_stop_seconds() + 120),
    pytest.mark.xdist_group("request-only-guardrail-stream-fast-path"),
]

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
Client: TypeAlias = Literal["sync", "async"]

_MODEL: Final = "request-only-guardrail-stream"
_MESSAGES_MODEL: Final = "request-only-guardrail-messages"
_GUARDED_MODEL: Final = "request-only-guardrail-guarded"
_GUARDRAIL: Final = "request-only-check"
_CHUNK_GUARDRAIL: Final = "per-chunk-check"
_OPT_IN: Final = "opt-in-chunk-check"
_INHERITED: Final = "inherited-chunk-check"
_RAISING: Final = "raising-chunk-check"
_CHUNK_ERROR: Final = "Chunk check failed"
_MODULE: Final = "request_only_guardrail"
_LOG_ENV: Final = "REQUEST_ONLY_GUARDRAIL_LOG"
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_PATHS: Final[Mapping[Endpoint, str]] = {
    "chat": "/v1/chat/completions",
    "messages": "/v1/messages",
    "responses": "/v1/responses",
}
_PIECES: Final = tuple(f"word{index} " for index in range(24))
_MARKER: Final = re.compile(rb"marker-([0-9a-f]{32})")
_DROP: Final = re.compile(rb"drop-after-([0-9]+)")
_MODULE_SOURCE: Final = f'''import json
import os
import re

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail

MARKER = re.compile("marker-([0-9a-f]{{32}})")


def _marker_in(value):
    found = MARKER.search(str(value))
    return None if found is None else found.group(1)


def _log(marker, event):
    with open(os.environ["{_LOG_ENV}"], "a") as log:
        log.write(json.dumps({{"marker": marker, "event": event}}) + "\\n")


class RequestOnlyGuardrail(CustomGuardrail):
    def should_run_guardrail(self, data, event_type):
        marker = _marker_in(data.get("messages") or data.get("input"))
        if marker is not None:
            _log(marker, str(getattr(event_type, "value", event_type)))
        return super().should_run_guardrail(data=data, event_type=event_type)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return data


class PerChunkGuardrail(CustomGuardrail):
    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        _log(_marker_in(response), "chunk:" + self.guardrail_name)
        return None


class InheritedPerChunkGuardrail(PerChunkGuardrail):
    pass


class RaisingPerChunkGuardrail(CustomGuardrail):
    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        if "word3 " in response:
            raise HTTPException(status_code=400, detail={{"error": "{_CHUNK_ERROR}"}})
        return None
'''


def _text(marker: str) -> str:
    return f"marker-{marker} " + "".join(_PIECES)


def _pieces(marker: str) -> tuple[str, ...]:
    return (f"marker-{marker} ", *_PIECES)


def _sse(name: str | None, payload: Mapping[str, JsonValue]) -> bytes:
    prefix: Final = f"event: {name}\n" if name is not None else ""
    return f"{prefix}data: {json.dumps(payload)}\n\n".encode()


def _chat_chunk(identity: str, delta: Mapping[str, JsonValue], finish: str | None = None) -> bytes:
    return _sse(
        None,
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
        },
    )


def _chat_stream(marker: str) -> tuple[bytes, ...]:
    identity: Final = f"chatcmpl-{marker}"
    first, *rest = _pieces(marker)
    return (
        _chat_chunk(identity, {"role": "assistant", "content": first}),
        *(_chat_chunk(identity, {"content": piece}) for piece in rest),
        _chat_chunk(identity, {}, finish="stop"),
        b"data: [DONE]\n\n",
    )


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": _text(marker)}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 25},
    }


def _messages_stream(marker: str) -> tuple[bytes, ...]:
    started: Final = {**_message(marker), "content": [], "stop_reason": None}
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "message_start", "message": {**started, "usage": {"input_tokens": 2, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}
            for piece in _pieces(marker)
        ),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 25}},
        {"type": "message_stop"},
    )
    return tuple(_sse(string_value(event["type"]), event) for event in events)


def _response_output(marker: str) -> dict[str, JsonValue]:
    return {
        "type": "message",
        "id": f"msg_{marker}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": _text(marker), "annotations": []}],
    }


def _response(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"resp_{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [_response_output(marker)],
        "usage": {"input_tokens": 2, "output_tokens": 25, "total_tokens": 27},
    }


def _responses_stream(marker: str) -> tuple[bytes, ...]:
    response: Final = _response(marker)
    output: Final = _response_output(marker)
    part: Final[dict[str, JsonValue]] = {"type": "output_text", "text": _text(marker), "annotations": []}
    position: Final[dict[str, JsonValue]] = {"item_id": f"msg_{marker}", "output_index": 0, "content_index": 0}
    started: Final = {**response, "status": "in_progress", "output": []}
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "response": started},
        {"type": "response.in_progress", "response": started},
        {
            "type": "response.output_item.added",
            "item": {**output, "status": "in_progress", "content": []},
            "output_index": 0,
        },
        {"type": "response.content_part.added", **position, "part": {**part, "text": ""}},
        *({"type": "response.output_text.delta", **position, "delta": piece} for piece in _pieces(marker)),
        {"type": "response.output_text.done", **position, "text": _text(marker)},
        {"type": "response.content_part.done", **position, "part": part},
        {"type": "response.output_item.done", "item": output, "output_index": 0},
        {"type": "response.completed", "response": response},
    )
    return tuple(
        _sse(string_value(event["type"]), {**event, "sequence_number": index}) for index, event in enumerate(events)
    )


def _chat_completion(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": _text(marker)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 25, "total_tokens": 27},
    }


def _respond(request: Request) -> Reply:
    found: Final = _MARKER.search(request.body)
    assert found is not None, request.body
    marker: Final = found.group(1).decode()
    streamed: Final = JSON_OBJECT.validate_json(request.body).get("stream") is True
    if request.target.endswith("/messages"):
        if not streamed:
            return Reply(body=json.dumps(_message(marker)).encode())
        return Reply(content_type="text/event-stream", chunks=_messages_stream(marker))
    if request.target.endswith("/responses"):
        if not streamed:
            return Reply(body=json.dumps(_response(marker)).encode())
        return Reply(content_type="text/event-stream", chunks=_responses_stream(marker))
    assert request.target.endswith("/chat/completions"), request.target
    if not streamed:
        return Reply(body=json.dumps(_chat_completion(marker)).encode())
    drop: Final = _DROP.search(request.body)
    if drop is not None:
        return Reply(
            content_type="text/event-stream",
            chunks=_chat_stream(marker),
            abort_after=int(drop.group(1)),
            pause_between_chunks=0.02,
        )
    return Reply(content_type="text/event-stream", chunks=_chat_stream(marker))


def _guardrail(name: str, guardrail_class: str, mode: str, *, default_on: bool = True) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {"guardrail": f"{_MODULE}.{guardrail_class}", "mode": mode, "default_on": default_on},
    }


def _deployment(name: str, model: str, api_base: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": "synthetic-stream-key", **extra},
    }


def _config(upstream_url: str, guardrails: list[JsonValue], settings: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **base,
        "litellm_settings": {**object_value(base["litellm_settings"]), **settings},
        "guardrails": guardrails,
        "model_list": [
            _deployment(_MODEL, "openai/gpt-4o-mini", f"{upstream_url}/v1"),
            _deployment(_MESSAGES_MODEL, "anthropic/claude-sonnet-4-5-20250929", upstream_url),
            _deployment(_GUARDED_MODEL, "openai/gpt-4o-mini", f"{upstream_url}/v1", guardrails=[_OPT_IN]),
        ],
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: OwnedProxy
    log: Path


@contextmanager
def _rig(
    gateway: Gateway,
    directory: Path,
    guardrails: list[JsonValue],
    settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Rig]:
    with wire_server(_respond) as upstream:
        (directory / f"{_MODULE}.py").write_text(_MODULE_SOURCE)
        config_path: Final = directory / "request-only-guardrail-stream-fast-path.yaml"
        config_path.write_text(yaml.safe_dump(_config(upstream.url, guardrails, settings or {})))
        log: Final = directory / "guardrail-asks.jsonl"
        log.touch()
        with owned_proxy_process(gateway, directory, {_LOG_ENV: str(log)}, config=config_path, workers=2) as proxy:
            yield Rig(proxy, log)


def _request_only() -> list[JsonValue]:
    return [_guardrail(_GUARDRAIL, "RequestOnlyGuardrail", "pre_call")]


def _mixed() -> list[JsonValue]:
    return [*_request_only(), _guardrail(_CHUNK_GUARDRAIL, "PerChunkGuardrail", "post_call")]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway, tmp_path_factory.mktemp("request-only-guardrail-stream-fast-path"), _request_only()
    ) as built:
        yield built


@pytest.fixture(scope="module")
def mixed_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway, tmp_path_factory.mktemp("request-only-guardrail-beside-per-chunk-guardrail"), _mixed()
    ) as built:
        yield built


@pytest.fixture(scope="module")
def opt_in_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("request-only-guardrail-beside-opt-in-per-chunk-guardrails"),
        [
            *_request_only(),
            _guardrail(_OPT_IN, "PerChunkGuardrail", "post_call", default_on=False),
            _guardrail(_INHERITED, "InheritedPerChunkGuardrail", "post_call", default_on=False),
            _guardrail(_RAISING, "RaisingPerChunkGuardrail", "post_call", default_on=False),
        ],
        {"include_cost_in_streaming_usage": True},
    ) as built:
        yield built


@pytest.fixture(scope="module")
def chaos_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(session_gateway, tmp_path_factory.mktemp("request-only-guardrail-stream-chaos"), _mixed()) as built:
        yield built


def _events(rig: Rig, marker: str) -> tuple[str, ...]:
    entries: Final = (JSON_OBJECT.validate_json(line) for line in rig.log.read_text().splitlines() if line)
    return tuple(string_value(entry["event"]) for entry in entries if entry["marker"] == marker)


def _prompt(marker: str, directive: str) -> str:
    return f"marker-{marker} {directive}".strip()


def _body(
    endpoint: Endpoint,
    marker: str,
    *,
    stream: bool = True,
    model: str = _MODEL,
    guardrails: Sequence[str] = (),
    directive: str = "",
) -> dict[str, JsonValue]:
    requested: Final[dict[str, JsonValue]] = {"guardrails": list(guardrails)} if guardrails else {}
    if endpoint == "responses":
        return {"model": model, "stream": stream, "input": _prompt(marker, directive), **requested}
    messages: Final[list[JsonValue]] = [{"role": "user", "content": _prompt(marker, directive)}]
    if endpoint == "messages":
        return {"model": _MESSAGES_MODEL, "stream": stream, "max_tokens": 64, "messages": messages, **requested}
    return {"model": model, "stream": stream, "messages": messages, "num_retries": 0, **requested}


def _payloads(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _event_text(event: Mapping[str, JsonValue], endpoint: Endpoint) -> str:
    if endpoint == "responses":
        return string_value(event.get("delta", "")) if event.get("type") == "response.output_text.delta" else ""
    if endpoint == "messages":
        if event.get("type") != "content_block_delta":
            return ""
        return string_value(object_value(event["delta"]).get("text", ""))
    return "".join(
        string_value(object_value(object_value(choice).get("delta", {})).get("content") or "")
        for choice in list_value(event.get("choices") or [])
    )


def _streamed_text(text: str, endpoint: Endpoint) -> str:
    return "".join(_event_text(event, endpoint) for event in _payloads(text))


def _send(
    rig: Rig,
    endpoint: Endpoint,
    marker: str,
    *,
    stream: bool = True,
    model: str = _MODEL,
    guardrails: Sequence[str] = (),
    directive: str = "",
) -> httpx.Response:
    body: Final = _body(endpoint, marker, stream=stream, model=model, guardrails=guardrails, directive=directive)
    return rig.proxy.gateway.request("POST", _PATHS[endpoint], body)


def _stream(rig: Rig, endpoint: Endpoint, marker: str, *, model: str = _MODEL, guardrails: Sequence[str] = ()) -> str:
    response: Final = _send(rig, endpoint, marker, model=model, guardrails=guardrails)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    assert _streamed_text(response.text, endpoint) == _text(marker), response.text
    return response.text


def _delivered_chunks(text: str) -> int:
    return sum(1 for payload in _payloads(text) if "choices" in payload)


def _assert_every_chunk_seen(rig: Rig, marker: str, guardrail: str, text: str) -> None:
    seen: Final = _events(rig, marker).count(f"chunk:{guardrail}")
    assert seen >= _delivered_chunks(text) > 0, (seen, _delivered_chunks(text), _events(rig, marker))


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_a_request_only_guardrail_is_not_asked_about_each_streamed_chunk(rig: Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    _stream(rig, endpoint, marker)
    asks: Final = _events(rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") == 0, f"asked about the response {asks.count('post_call')} times: {asks}"


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_a_request_only_guardrail_answers_non_streaming_calls_on_every_endpoint(rig: Rig, endpoint: Endpoint) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig, endpoint, marker, stream=False)
    assert response.status_code == 200, response.text
    assert _text(marker) in response.text, response.text
    asks: Final = _events(rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") <= 1, asks


def _openai_text(client: openai.OpenAI, endpoint: Endpoint, marker: str) -> str:
    if endpoint == "responses":
        events: Final = client.responses.create(model=_MODEL, input=_prompt(marker, ""), stream=True)
        return "".join(event.delta for event in events if event.type == "response.output_text.delta")
    chunks: Final = client.chat.completions.create(
        model=_MODEL, messages=[{"role": "user", "content": _prompt(marker, "")}], stream=True
    )
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


async def _async_openai_text(client: openai.AsyncOpenAI, endpoint: Endpoint, marker: str) -> str:
    pieces: Final[list[str]] = []
    if endpoint == "responses":
        async for event in await client.responses.create(model=_MODEL, input=_prompt(marker, ""), stream=True):
            if event.type == "response.output_text.delta":
                pieces.append(event.delta)
        return "".join(pieces)
    async for chunk in await client.chat.completions.create(
        model=_MODEL, messages=[{"role": "user", "content": _prompt(marker, "")}], stream=True
    ):
        pieces.extend(choice.delta.content or "" for choice in chunk.choices)
    return "".join(pieces)


def _anthropic_text(client: anthropic.Anthropic, marker: str) -> str:
    events: Final = client.messages.create(
        model=_MESSAGES_MODEL, max_tokens=64, messages=[{"role": "user", "content": _prompt(marker, "")}], stream=True
    )
    return "".join(
        event.delta.text for event in events if event.type == "content_block_delta" and event.delta.type == "text_delta"
    )


async def _async_anthropic_text(client: anthropic.AsyncAnthropic, marker: str) -> str:
    pieces: Final[list[str]] = []
    async for event in await client.messages.create(
        model=_MESSAGES_MODEL, max_tokens=64, messages=[{"role": "user", "content": _prompt(marker, "")}], stream=True
    ):
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            pieces.append(event.delta.text)
    return "".join(pieces)


def _sync_sdk_text(gateway: Gateway, endpoint: Endpoint, marker: str) -> str:
    origin: Final = str(gateway.client.base_url).rstrip("/")
    with httpx.Client(timeout=30, trust_env=False) as transport:
        if endpoint == "messages":
            with anthropic.Anthropic(
                api_key=gateway.key, base_url=origin, max_retries=0, http_client=transport
            ) as client:
                return _anthropic_text(client, marker)
        with openai.OpenAI(
            api_key=gateway.key, base_url=f"{origin}/v1", max_retries=0, http_client=transport
        ) as client:
            return _openai_text(client, endpoint, marker)


def _async_sdk_text(gateway: Gateway, endpoint: Endpoint, marker: str) -> str:
    origin: Final = str(gateway.client.base_url).rstrip("/")

    async def collect() -> str:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as transport:
            if endpoint == "messages":
                async with anthropic.AsyncAnthropic(
                    api_key=gateway.key, base_url=origin, max_retries=0, http_client=transport
                ) as client:
                    return await _async_anthropic_text(client, marker)
            async with openai.AsyncOpenAI(
                api_key=gateway.key, base_url=f"{origin}/v1", max_retries=0, http_client=transport
            ) as client:
                return await _async_openai_text(client, endpoint, marker)

    return asyncio.run(collect())


@pytest.mark.parametrize("client", ["sync", "async"])
@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_sdk_streams_through_a_request_only_guardrail_without_per_chunk_asks(
    rig: Rig, endpoint: Endpoint, client: Client
) -> None:
    marker: Final = uuid.uuid4().hex
    read: Final = _sync_sdk_text if client == "sync" else _async_sdk_text
    assert read(rig.proxy.gateway, endpoint, marker) == _text(marker)
    asks: Final = _events(rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") == 0, asks


def test_a_request_only_guardrail_stream_writes_one_success_spend_row_naming_it(rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(rig, "chat", marker)
    assert response.status_code == 200, response.text
    assert _streamed_text(response.text, "chat") == _text(marker), response.text
    call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: rig.proxy.gateway.rows(
            'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s', (call_id,)
        ),
        lambda found: len(found) >= 1,
        seconds=30,
    )
    assert [row["status"] for row in rows] == ["success"], rows
    assert _GUARDRAIL in list_value(object_value(rows[0]["metadata"]).get("applied_guardrails") or []), rows


def test_a_per_chunk_guardrail_still_sees_every_chunk_without_asking_the_request_only_one(mixed_rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final = _stream(mixed_rig, "chat", marker)
    asks: Final = _events(mixed_rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") <= 1, f"asked about the response {asks.count('post_call')} times: {asks}"
    _assert_every_chunk_seen(mixed_rig, marker, _CHUNK_GUARDRAIL, text)


@pytest.mark.parametrize("endpoint", ["messages", "responses"])
def test_a_per_chunk_guardrail_beside_a_request_only_one_keeps_other_endpoints_whole(
    mixed_rig: Rig, endpoint: Endpoint
) -> None:
    marker: Final = uuid.uuid4().hex
    _stream(mixed_rig, endpoint, marker)
    asks: Final = _events(mixed_rig, marker)
    assert "pre_call" in asks, asks
    assert asks.count("post_call") <= 1, asks


def test_an_opt_in_per_chunk_guardrail_sees_every_chunk_only_when_requested(opt_in_rig: Rig) -> None:
    requested: Final = uuid.uuid4().hex
    text: Final = _stream(opt_in_rig, "chat", requested, guardrails=[_OPT_IN])
    _assert_every_chunk_seen(opt_in_rig, requested, _OPT_IN, text)
    assert _events(opt_in_rig, requested).count("post_call") <= 1, _events(opt_in_rig, requested)

    skipped: Final = uuid.uuid4().hex
    _stream(opt_in_rig, "chat", skipped)
    assert not any(event.startswith("chunk:") for event in _events(opt_in_rig, skipped)), _events(opt_in_rig, skipped)
    assert _events(opt_in_rig, skipped).count("post_call") <= 1, _events(opt_in_rig, skipped)


def test_an_inherited_per_chunk_hook_still_sees_every_chunk(opt_in_rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final = _stream(opt_in_rig, "chat", marker, guardrails=[_INHERITED])
    _assert_every_chunk_seen(opt_in_rig, marker, _INHERITED, text)


def test_a_model_level_per_chunk_guardrail_sees_every_chunk(opt_in_rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    text: Final = _stream(opt_in_rig, "chat", marker, model=_GUARDED_MODEL)
    _assert_every_chunk_seen(opt_in_rig, marker, _OPT_IN, text)


def test_a_raising_per_chunk_guardrail_ends_the_stream_with_its_error(opt_in_rig: Rig) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = _send(opt_in_rig, "chat", marker, guardrails=[_RAISING])
    assert response.status_code == 200, response.text
    lines: Final = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and lines.count("data: [DONE]") == 1, response.text
    payloads: Final = _payloads(response.text)
    assert "error" in payloads[-1] and all("error" not in payload for payload in payloads[:-1]), response.text
    assert object_value(payloads[-1]["error"])["message"] == _CHUNK_ERROR, payloads[-1]
    delivered: Final = "".join(_event_text(payload, "chat") for payload in payloads[:-1])
    assert delivered and _text(marker).startswith(delivered) and "word3 " not in delivered, response.text
    assert _events(opt_in_rig, marker).count("post_call") <= 1, _events(opt_in_rig, marker)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_cost_injection_keeps_streams_whole_beside_a_request_only_guardrail(
    opt_in_rig: Rig, endpoint: Endpoint
) -> None:
    marker: Final = uuid.uuid4().hex
    _stream(opt_in_rig, endpoint, marker)
    assert _events(opt_in_rig, marker).count("post_call") <= 1, _events(opt_in_rig, marker)


@dataclass(frozen=True, slots=True)
class Sent:
    endpoint: Endpoint
    marker: str
    dropped: bool
    response: httpx.Response


def _burst_member(rig: Rig, endpoint: Endpoint, dropped: bool) -> Sent:
    marker: Final = uuid.uuid4().hex
    directive: Final = "drop-after-8" if dropped else ""
    return Sent(endpoint, marker, dropped, _send(rig, endpoint, marker, directive=directive))


def _assert_burst_member(rig: Rig, sent: Sent) -> None:
    assert sent.response.status_code == 200, sent.response.text
    text: Final = _streamed_text(sent.response.text, sent.endpoint)
    if sent.dropped:
        assert text and _text(sent.marker).startswith(text) and text != _text(sent.marker), sent.response.text
    else:
        assert text == _text(sent.marker), sent.response.text
    asks: Final = _events(rig, sent.marker)
    assert "pre_call" in asks and asks.count("post_call") <= 1, asks
    if sent.endpoint == "chat":
        _assert_every_chunk_seen(rig, sent.marker, _CHUNK_GUARDRAIL, sent.response.text)


def test_a_mixed_burst_with_dropped_upstreams_and_a_killed_worker_keeps_every_guardrail_contract(
    chaos_rig: Rig,
) -> None:
    plan: Final[tuple[tuple[Endpoint, bool], ...]] = (
        *(("chat", index % 2 == 0) for index in range(8)),
        *(("messages", False) for _ in range(8)),
        *(("responses", False) for _ in range(8)),
    )
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        burst: Final = tuple(pool.map(lambda member: _burst_member(chaos_rig, *member), plan))
    for sent in burst:
        _assert_burst_member(chaos_rig, sent)

    victim: Final = min(chaos_rig.proxy.worker_pids())
    replacement: Final = chaos_rig.proxy.kill_worker(victim)
    assert replacement in chaos_rig.proxy.worker_pids(), (replacement, chaos_rig.proxy.worker_pids())
    for _ in range(8):
        _assert_burst_member(chaos_rig, _burst_member(chaos_rig, "chat", False))
