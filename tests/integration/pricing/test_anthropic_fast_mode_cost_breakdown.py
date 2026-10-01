from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import threading
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "claude-opus-5-5"
_OPENAI_MODEL: Final = "gpt-4o-mini"
_INPUT_RATE: Final = 1e-6
_OUTPUT_RATE: Final = 2e-6
_CACHE_READ_RATE: Final = 1e-7
_CACHE_WRITE_5M_RATE: Final = 1.25e-6
_CACHE_WRITE_1H_RATE: Final = 2e-6
_FAST_MULTIPLIER: Final = 6.0
_RATES: Final[dict[str, JsonValue]] = {
    "input_cost_per_token": _INPUT_RATE,
    "output_cost_per_token": _OUTPUT_RATE,
    "cache_read_input_token_cost": _CACHE_READ_RATE,
    "cache_creation_input_token_cost": _CACHE_WRITE_5M_RATE,
    "cache_creation_input_token_cost_above_1hr": _CACHE_WRITE_1H_RATE,
}
_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 400,
    "cache_creation_input_tokens": 300,
    "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
}
_NO_CACHE_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}
_CHAT_USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": 800,
    "completion_tokens": 50,
    "total_tokens": 850,
    "prompt_tokens_details": {"cached_tokens": 400},
}
_CONTENT: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "text", "text": "PONG"},)
_STREAMS: Final = (
    pytest.param(False, id="non-streamed"),
    pytest.param(True, id="streamed"),
)
_UPSTREAM_SPEED_VALUES: Final = (
    pytest.param(1, id="integer"),
    pytest.param(["fast"], id="list"),
    pytest.param("", id="empty-string"),
    pytest.param("x" * 5000, id="five-kilobyte-string"),
)
_REQUEST_SPEED_VALUES: Final = (
    pytest.param(1, id="integer"),
    pytest.param(["fast"], id="list"),
    pytest.param("", id="empty-string"),
    pytest.param("x" * 5000, id="five-kilobyte-string"),
)
_INVALID_ENTRY_VALUES: Final = (
    pytest.param("abc", id="string"),
    pytest.param(None, id="null"),
    pytest.param(0, id="zero"),
)


@dataclass(frozen=True, slots=True)
class _Capture:
    row_id: str
    case: str
    status: int
    body: dict[str, JsonValue]
    headers: dict[str, str]
    response_id: str | None
    request_id: str | None
    spend: dict[str, JsonValue] | None
    upstream: tuple[dict[str, JsonValue], ...]


@dataclass(frozen=True, slots=True)
class _SDKResult:
    body: dict[str, JsonValue]
    response_id: str
    headers: dict[str, str]


def _json_object(value: bytes) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(value)


def _response_body(response: httpx.Response, stream: bool) -> dict[str, JsonValue]:
    if stream and (
        response.headers.get("content-type", "").startswith("text/event-stream")
        or response.content.startswith(b"data: ")
    ):
        return {"events": _sse_events(response.content)}
    return _json_object(response.content)


def _sse_events(value: bytes) -> list[dict[str, JsonValue]]:
    return list(
        _json_object(line[6:].encode())
        for line in value.decode(errors="replace").splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _response_id(body: Mapping[str, JsonValue]) -> str | None:
    direct_id: Final = body.get("id")
    if isinstance(direct_id, str):
        return direct_id
    events: Final = body.get("events")
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id: Final = event.get("id")
        if isinstance(event_id, str):
            return event_id
        for key in ("message", "response"):
            nested: Final = event.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]
    return None


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower().startswith("x-litellm-response-cost") or name.lower() == "x-litellm-call-id"
    }


def _upstream_payloads(requests: Sequence[Request]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(_json_object(request.body) for request in requests)


def _row(request_id: str, call_id: str | None = None) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, spend, prompt_tokens, completion_tokens, metadata->'cost_breakdown' AS cost_breakdown "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
            (request_id, call_id or request_id),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _maybe_row(request_id: str | None, call_id: str | None = None) -> dict[str, JsonValue] | None:
    if request_id is None:
        return None
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
            (request_id, call_id or request_id),
        ),
        lambda values: bool(values),
        seconds=5,
        return_last_on_timeout=True,
    )
    return _row(request_id, call_id) if rows else None


def _capture_record(
    row_id: str,
    case: str,
    status: int,
    body: dict[str, JsonValue],
    headers: Mapping[str, str],
    response_id: str | None,
    request_id: str | None,
    spend: dict[str, JsonValue] | None,
    upstream: Sequence[dict[str, JsonValue]],
) -> _Capture:
    return _Capture(row_id, case, status, body, dict(headers), response_id, request_id, spend, tuple(upstream))


def _capture(
    row_id: str,
    case: str,
    response: httpx.Response,
    response_body: dict[str, JsonValue],
    response_id: str | None,
    upstream: Sequence[Request],
    *,
    expect_spend_row: bool = False,
) -> _Capture:
    call_id: Final = response.headers.get("x-litellm-call-id")
    lookup_id: Final = response_id or call_id
    spend: Final = (
        _row(lookup_id, call_id)
        if (expect_spend_row or response.status_code == 200) and lookup_id is not None
        else _maybe_row(lookup_id, call_id)
    )
    request_id: Final = string_value(spend["request_id"]) if spend is not None else lookup_id
    upstream_payloads: Final = _upstream_payloads(upstream)
    return _capture_record(
        row_id,
        case,
        response.status_code,
        response_body,
        _response_headers(response.headers),
        response_id,
        request_id,
        spend,
        upstream_payloads,
    )


def _capture_sdk(
    row_id: str,
    case: str,
    result: _SDKResult,
    upstream: Sequence[Request],
) -> _Capture:
    call_id: Final = result.headers.get("x-litellm-call-id")
    spend: Final = _row(result.response_id, call_id)
    request_id: Final = string_value(spend["request_id"])
    return _capture_record(
        row_id,
        case,
        200,
        result.body,
        result.headers,
        result.response_id,
        request_id,
        spend,
        _upstream_payloads(upstream),
    )


def _capture_transport_error(
    row_id: str,
    case: str,
    error: str,
    upstream: Sequence[Request],
) -> _Capture:
    return _capture_record(
        row_id,
        case,
        0,
        {"transport_error": error},
        {},
        None,
        None,
        None,
        _upstream_payloads(upstream),
    )


def _anthropic_model(
    gateway: Gateway,
    scenario: Scenario,
    wire: Wire,
    model_info: Mapping[str, JsonValue] | None,
    retries: int = 2,
) -> str:
    return scenario.model(
        model=f"anthropic/{_MODEL}",
        api_base=wire.url,
        api_key=cc.ANTHROPIC_API_KEY,
        model_info=model_info,
        num_retries=retries,
        **_RATES,
    )


def _openai_model(gateway: Gateway, scenario: Scenario, wire: Wire, model_info: Mapping[str, JsonValue] | None) -> str:
    return scenario.model(
        model=f"openai/{_OPENAI_MODEL}",
        api_base=f"{wire.url}/v1",
        api_key="synthetic-openai-key",
        model_info=model_info,
        **_RATES,
    )


def _anthropic_reply(identity: str, stream: bool, usage: Mapping[str, JsonValue]) -> Reply:
    return (
        Reply(chunks=cc.message_stream(identity, _MODEL, _CONTENT, dict(usage)), content_type="text/event-stream")
        if stream
        else Reply(body=cc.message_reply(identity, _MODEL, _CONTENT, dict(usage)))
    )


def _chat_reply(identity: str, stream: bool, usage: Mapping[str, JsonValue]) -> Reply:
    if stream:
        chunks: Final = (
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': 'PONG'}, 'finish_reason': None}]})}\n\n".encode(),
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage})}\n\n".encode(),
            b"data: [DONE]\n\n",
        )
        return Reply(chunks=chunks, content_type="text/event-stream")
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": _OPENAI_MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
                "usage": usage,
            }
        ).encode()
    )


def _native_call(
    gateway: Gateway,
    row_id: str,
    case: str,
    request_fields: Mapping[str, JsonValue],
    usage: Mapping[str, JsonValue],
    stream: bool,
    model_info: Mapping[str, JsonValue] | None,
    *,
    retries: int = 2,
    expect_success: bool = True,
    allow_no_upstream: bool = False,
) -> _Capture:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert body["model"] == _MODEL, body
        assert body["stream"] is stream, body
        for name, value in request_fields.items():
            assert body.get(name) == value, body
        if "speed" not in request_fields:
            assert "speed" not in body, body
        return _anthropic_reply(identity, stream, usage)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(gateway, scenario, wire, model_info, retries)
        request_body: Final = {
            **cc.claude_code_request(f"{case}-{uuid.uuid4().hex}"),
            **request_fields,
            "model": model,
            "stream": stream,
        }
        response: Final = gateway.request("POST", "/v1/messages", request_body)
        if expect_success:
            assert response.status_code == 200, response.text
        response_body: Final = _response_body(response, stream)
        response_id: Final = _response_id(response_body)
        upstream: Final = wire.drain()
        assert len(upstream) == 1 or (allow_no_upstream and not upstream), upstream
        capture: Final = _capture(row_id, case, response, response_body, response_id, upstream)
        if expect_success:
            assert capture.request_id == capture.response_id, capture
        return capture


def _sdk_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower().startswith("x-litellm-response-cost") or name.lower() == "x-litellm-call-id"
    }


def _sdk_messages(
    gateway: Gateway,
    key: str,
    model: str,
    stream: bool,
    async_mode: bool,
) -> _SDKResult:
    base_url: Final = str(gateway.client.base_url).rstrip("/")
    prompt: Final = f"pricing audit {uuid.uuid4().hex}"
    if async_mode:
        return asyncio.run(_async_sdk_messages(base_url, key, model, stream, prompt))
    with anthropic.Anthropic(base_url=base_url, api_key=key, max_retries=0) as client:
        if stream:
            with client.messages.with_streaming_response.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body={"speed": "fast"},
            ) as streamed:
                events: Final = _sse_events(streamed.read())
                body: Final = {"events": events}
                response_id: Final = _response_id(body)
                assert response_id is not None, body
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = client.messages.with_raw_response.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"speed": "fast"},
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = string_value(body["id"])
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


async def _async_sdk_messages(base_url: str, key: str, model: str, stream: bool, prompt: str) -> _SDKResult:
    async with anthropic.AsyncAnthropic(base_url=base_url, api_key=key, max_retries=0) as client:
        if stream:
            async with client.messages.with_streaming_response.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body={"speed": "fast"},
            ) as streamed:
                events: Final = _sse_events(await streamed.read())
                body: Final = {"events": events}
                response_id: Final = _response_id(body)
                assert response_id is not None, body
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = await client.messages.with_raw_response.create(
                model=model,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"speed": "fast"},
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = string_value(body["id"])
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


def _sdk_chat(
    gateway: Gateway,
    model: str,
    stream: bool,
    async_mode: bool,
    speed: JsonValue | None,
) -> _SDKResult:
    base_url: Final = f"{str(gateway.client.base_url).rstrip('/')}/v1"
    extra_body: Final = {} if speed is None else {"speed": speed}
    prompt: Final = f"pricing audit {uuid.uuid4().hex}"
    if async_mode:
        return asyncio.run(_async_sdk_chat(base_url, gateway.key, model, stream, extra_body, prompt))
    with openai.OpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
        if stream:
            with client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body,
            ) as streamed:
                chunks: Final = tuple(streamed.parse())
                body: Final = {
                    "chunks": tuple(_JSON_OBJECT.validate_python(chunk.model_dump(mode="json")) for chunk in chunks)
                }
                response_id: Final = chunks[0].id
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body,
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = parsed.id
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


async def _async_sdk_chat(
    base_url: str,
    key: str,
    model: str,
    stream: bool,
    extra_body: Mapping[str, JsonValue],
    prompt: str,
) -> _SDKResult:
    async with openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0) as client:
        if stream:
            async with client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body,
            ) as streamed:
                chunks: Final = [chunk async for chunk in await streamed.parse()]
                body: Final = {
                    "chunks": tuple(_JSON_OBJECT.validate_python(chunk.model_dump(mode="json")) for chunk in chunks)
                }
                response_id: Final = chunks[0].id
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body=extra_body,
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = parsed.id
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


def _sdk_responses(
    gateway: Gateway,
    model: str,
    stream: bool,
    async_mode: bool,
    speed: JsonValue | None,
) -> _SDKResult:
    base_url: Final = f"{str(gateway.client.base_url).rstrip('/')}/v1"
    extra_body: Final = {} if speed is None else {"speed": speed}
    prompt: Final = f"pricing audit {uuid.uuid4().hex}"
    if async_mode:
        return asyncio.run(_async_sdk_responses(base_url, gateway.key, model, stream, extra_body, prompt))
    with openai.OpenAI(base_url=base_url, api_key=gateway.key, max_retries=0) as client:
        if stream:
            with client.responses.with_streaming_response.create(
                model=model,
                input=prompt,
                stream=True,
                extra_body=extra_body,
            ) as streamed:
                events: Final = tuple(streamed.parse())
                body: Final = {
                    "events": tuple(_JSON_OBJECT.validate_python(event.model_dump(mode="json")) for event in events)
                }
                response_id: Final = _response_event_id(body["events"])
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = client.responses.with_raw_response.create(
                model=model,
                input=prompt,
                extra_body=extra_body,
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = string_value(body["id"])
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


async def _async_sdk_responses(
    base_url: str,
    key: str,
    model: str,
    stream: bool,
    extra_body: Mapping[str, JsonValue],
    prompt: str,
) -> _SDKResult:
    async with openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0) as client:
        if stream:
            async with client.responses.with_streaming_response.create(
                model=model,
                input=prompt,
                stream=True,
                extra_body=extra_body,
            ) as streamed:
                events: Final = [event async for event in await streamed.parse()]
                body: Final = {
                    "events": tuple(_JSON_OBJECT.validate_python(event.model_dump(mode="json")) for event in events)
                }
                response_id: Final = _response_event_id(body["events"])
                headers: Final = _sdk_headers(streamed.headers)
        else:
            raw: Final = await client.responses.with_raw_response.create(
                model=model,
                input=prompt,
                extra_body=extra_body,
            )
            parsed: Final = raw.parse()
            body = _JSON_OBJECT.validate_python(parsed.model_dump(mode="json"))
            response_id = string_value(body["id"])
            headers = _sdk_headers(raw.headers)
    return _SDKResult(body, response_id, headers)


def _response_event_id(events: JsonValue) -> str:
    assert isinstance(events, (list, tuple)), events
    for event in events:
        if isinstance(event, dict):
            response: Final = event.get("response")
            if isinstance(response, dict) and isinstance(response.get("id"), str):
                return response["id"]
    raise AssertionError(events)


def _assert_fast(capture: _Capture) -> None:
    assert capture.status == 200, capture
    assert capture.response_id is not None and capture.request_id is not None, capture
    assert capture.body, capture
    assert capture.spend is not None, capture
    assert (
        len(
            read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (capture.request_id,),
            )
        )
        == 1
    ), capture
    assert float(capture.spend["spend"]) == pytest.approx(0.00459), capture
    assert capture.spend["prompt_tokens"] == 800, capture
    assert capture.spend["completion_tokens"] == 50, capture
    breakdown: Final = object_value(capture.spend["cost_breakdown"])
    assert float(breakdown["input_cost"]) == pytest.approx(0.00399), capture
    assert float(breakdown["output_cost"]) == pytest.approx(0.0006), capture
    assert float(breakdown["cache_read_cost"]) == pytest.approx(0.00024), capture
    assert float(breakdown["cache_creation_cost"]) == pytest.approx(0.00315), capture


def _assert_one_x(capture: _Capture) -> None:
    assert capture.status == 200, capture
    assert capture.response_id is not None and capture.request_id == capture.response_id, capture
    assert capture.body, capture
    assert capture.spend is not None, capture
    assert float(capture.spend["spend"]) == pytest.approx(0.000765), capture
    breakdown: Final = object_value(capture.spend["cost_breakdown"])
    assert float(breakdown["input_cost"]) == pytest.approx(0.000665), capture
    assert float(breakdown["output_cost"]) == pytest.approx(0.0001), capture
    assert float(breakdown["cache_read_cost"]) == pytest.approx(0.00004), capture
    assert float(breakdown["cache_creation_cost"]) == pytest.approx(0.000525), capture


def _assert_no_cache_fast(capture: _Capture) -> None:
    assert capture.status == 200, capture
    assert capture.response_id is not None and capture.request_id == capture.response_id, capture
    assert capture.body, capture
    assert capture.spend is not None, capture
    assert float(capture.spend["spend"]) == pytest.approx(0.0012), capture
    breakdown: Final = object_value(capture.spend["cost_breakdown"])
    assert float(breakdown["input_cost"]) == pytest.approx(0.0006), capture
    assert float(breakdown["output_cost"]) == pytest.approx(0.0006), capture
    assert float(breakdown.get("cache_read_cost", 0)) == pytest.approx(0), capture
    assert float(breakdown.get("cache_creation_cost", 0)) == pytest.approx(0), capture


@pytest.mark.parametrize("stream", _STREAMS)
def test_anthropic_sdk_messages_fast_costs_match_multiplied_breakdown(gateway: Gateway, stream: bool) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, stream, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )
        result: Final = _sdk_messages(gateway, scenario.key(), model, stream, False)
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        sent: Final = _upstream_payloads(upstream)[0]
        assert sent["model"] == _MODEL and sent["speed"] == "fast", sent
        capture: Final = _capture_sdk("H2", "streamed" if stream else "non-streamed", result, upstream)
        assert capture.response_id == capture.request_id, capture
        _assert_fast(capture)


@pytest.mark.parametrize("stream", _STREAMS)
def test_async_anthropic_sdk_messages_fast_costs_match_multiplied_breakdown(gateway: Gateway, stream: bool) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, stream, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )
        result: Final = _sdk_messages(gateway, scenario.key(), model, stream, True)
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        sent: Final = _upstream_payloads(upstream)[0]
        assert sent["model"] == _MODEL and sent["speed"] == "fast", sent
        capture: Final = _capture_sdk("H3", "streamed" if stream else "non-streamed", result, upstream)
        assert capture.response_id == capture.request_id, capture
        _assert_fast(capture)


@pytest.mark.parametrize("stream", _STREAMS)
@pytest.mark.parametrize("async_mode", (False, True), ids=("sync", "async"))
def test_openai_sdk_chat_completions_fast_costs_match_multiplied_breakdown(
    gateway: Gateway, stream: bool, async_mode: bool
) -> None:
    identity: Final = f"chatcmpl_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, stream, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )
        result: Final = _sdk_chat(gateway, model, stream, async_mode, "fast")
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        sent: Final = _upstream_payloads(upstream)[0]
        assert sent["model"] == _MODEL and sent["speed"] == "fast", sent
        capture: Final = _capture_sdk("H4", f"{async_mode}-{stream}", result, upstream)
        assert capture.response_id == capture.request_id, capture
        _assert_fast(capture)


@pytest.mark.parametrize("stream", _STREAMS)
@pytest.mark.parametrize("async_mode", (False, True), ids=("sync", "async"))
def test_openai_sdk_responses_fast_costs_match_multiplied_breakdown(
    gateway: Gateway, stream: bool, async_mode: bool
) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, stream, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )
        result: Final = _sdk_responses(gateway, model, stream, async_mode, "fast")
        upstream: Final = wire.drain()
        capture: Final = _capture_sdk("H5", f"{async_mode}-{stream}", result, upstream)
        assert len(upstream) == 1, upstream
        sent: Final = _upstream_payloads(upstream)[0]
        assert sent["model"] == _MODEL, sent
        assert sent.get("speed") == "fast", sent
        _assert_fast(capture)


def test_native_messages_geo_and_fast_multipliers_compound(gateway: Gateway) -> None:
    capture: Final = _native_call(
        gateway,
        "H6",
        "nonstream",
        {"speed": "fast"},
        {**_USAGE, "speed": "fast", "inference_geo": "us"},
        False,
        {"provider_specific_entry": {"us": 1.1, "fast": 6.0}},
    )
    assert capture.spend is not None, capture
    breakdown: Final = object_value(capture.spend["cost_breakdown"])
    assert float(capture.spend["spend"]) == pytest.approx(0.005049), capture
    assert float(breakdown["input_cost"]) == pytest.approx(0.004389), capture
    assert float(breakdown["output_cost"]) == pytest.approx(0.00066), capture
    assert float(breakdown["cache_read_cost"]) == pytest.approx(0.000264), capture
    assert float(breakdown["cache_creation_cost"]) == pytest.approx(0.003465), capture


def test_native_messages_fast_cost_headers_match_breakdown(gateway: Gateway) -> None:
    capture: Final = _native_call(
        gateway,
        "H7",
        "nonstream",
        {"speed": "fast"},
        {**_USAGE, "speed": "fast"},
        False,
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
    )
    _assert_fast(capture)
    assert float(capture.headers.get("x-litellm-response-cost", "nan")) == pytest.approx(0.00459), capture
    assert float(capture.headers.get("x-litellm-response-cost-input", "nan")) == pytest.approx(0.0006), capture
    assert float(capture.headers.get("x-litellm-response-cost-output", "nan")) == pytest.approx(0.0006), capture
    assert float(capture.headers.get("x-litellm-response-cost-cache-read", "nan")) == pytest.approx(0.00024), capture
    assert float(capture.headers.get("x-litellm-response-cost-cache-creation", "nan")) == pytest.approx(0.00315), (
        capture
    )


@pytest.mark.parametrize("stream", _STREAMS)
def test_native_messages_omitted_speed_uses_standard_costs(gateway: Gateway, stream: bool) -> None:
    _assert_one_x(
        _native_call(
            gateway,
            "N1",
            f"N1-{'stream' if stream else 'nonstream'}",
            {},
            _USAGE,
            stream,
            {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
        )
    )


@pytest.mark.parametrize("stream", _STREAMS)
def test_native_messages_standard_speed_uses_standard_costs(gateway: Gateway, stream: bool) -> None:
    _assert_one_x(
        _native_call(
            gateway,
            "N2",
            f"N2-{'stream' if stream else 'nonstream'}",
            {"speed": "standard"},
            {**_USAGE, "speed": "standard"},
            stream,
            {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
        )
    )


@pytest.mark.parametrize("stream", _STREAMS)
def test_native_messages_fast_without_multiplier_uses_standard_costs(gateway: Gateway, stream: bool) -> None:
    _assert_one_x(
        _native_call(
            gateway,
            "N3",
            f"N3-{'stream' if stream else 'nonstream'}",
            {"speed": "fast"},
            {**_USAGE, "speed": "fast"},
            stream,
            None,
        )
    )


@pytest.mark.parametrize("async_mode", (False, True), ids=("sync", "async"))
def test_openai_sdk_chat_cached_prompt_costs_are_stable(gateway: Gateway, async_mode: bool) -> None:
    identity: Final = f"chatcmpl_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _chat_reply(identity, False, _CHAT_USAGE)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _openai_model(gateway, scenario, wire, None)
        result: Final = _sdk_chat(gateway, model, False, async_mode, None)
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        capture: Final = _capture_sdk("N4", "async" if async_mode else "sync", result, upstream)
        assert capture.response_id is not None and capture.request_id == capture.response_id, capture
        assert capture.spend is not None, capture
        breakdown: Final = object_value(capture.spend["cost_breakdown"])
        assert float(capture.spend["spend"]) == pytest.approx(0.00054), capture
        assert float(breakdown["input_cost"]) == pytest.approx(0.00044), capture
        assert float(breakdown["cache_read_cost"]) == pytest.approx(0.00004), capture
        assert float(breakdown["output_cost"]) == pytest.approx(0.0001), capture


def test_cost_estimate_ignores_deployment_fast_multiplier(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_key=cc.ANTHROPIC_API_KEY,
            model_info={"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
            **_RATES,
        )
        response: Final = gateway.request(
            "POST",
            "/cost/estimate",
            {
                "model": model,
                "input_tokens": 800,
                "output_tokens": 50,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 300,
            },
        )
        assert response.status_code == 200, response.text
        body: Final = _json_object(response.content)
        assert body["cost_per_request"] == pytest.approx(0.000615), body
        assert body["input_cost_per_request"] == pytest.approx(0.000515), body
        assert body["output_cost_per_request"] == pytest.approx(0.0001), body
        assert body["cache_read_cost_per_request"] == pytest.approx(0.00004), body
        assert body["cache_creation_cost_per_request"] == pytest.approx(0.000375), body
        _capture_record(
            "N6",
            "cost-estimate",
            response.status_code,
            body,
            _response_headers(response.headers),
            None,
            None,
            None,
            (),
        )


@pytest.mark.parametrize("speed_value", _UPSTREAM_SPEED_VALUES)
def test_native_messages_invalid_upstream_speed_values_use_matching_breakdowns(
    gateway: Gateway, speed_value: JsonValue
) -> None:
    capture: Final = _native_call(
        gateway,
        "S1",
        f"usage-{type(speed_value).__name__}",
        {"speed": "fast"},
        {**_USAGE, "speed": speed_value},
        False,
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
    )
    spend_multiplier: Final = _FAST_MULTIPLIER if speed_value in (1, ["fast"]) else 1.0
    breakdown_multiplier: Final = spend_multiplier
    expected_spend: Final = 0.00459 if spend_multiplier == _FAST_MULTIPLIER else 0.000765
    expected_input_cost: Final = 0.00399 if spend_multiplier == _FAST_MULTIPLIER else 0.000665
    expected_output_cost: Final = 0.0006 if spend_multiplier == _FAST_MULTIPLIER else 0.0001
    expected_cache_read_cost: Final = 0.00024 if breakdown_multiplier == _FAST_MULTIPLIER else 0.00004
    expected_cache_creation_cost: Final = 0.00315 if breakdown_multiplier == _FAST_MULTIPLIER else 0.000525
    assert capture.status == 200, capture
    assert capture.response_id is not None and capture.request_id == capture.response_id, capture
    assert capture.spend is not None, capture
    breakdown: Final = object_value(capture.spend["cost_breakdown"])
    assert float(capture.spend["spend"]) == pytest.approx(expected_spend), capture
    assert float(breakdown["input_cost"]) == pytest.approx(expected_input_cost), capture
    assert float(breakdown["output_cost"]) == pytest.approx(expected_output_cost), capture
    assert float(breakdown["cache_read_cost"]) == pytest.approx(expected_cache_read_cost), capture
    assert float(breakdown["cache_creation_cost"]) == pytest.approx(expected_cache_creation_cost), capture
    if spend_multiplier == _FAST_MULTIPLIER:
        assert float(breakdown["cache_read_cost"]) / 0.00004 == pytest.approx(
            float(capture.spend["spend"]) / 0.000765
        ), capture


@pytest.mark.parametrize("speed_value", _REQUEST_SPEED_VALUES)
def test_native_messages_invalid_request_speed_values_return_client_errors(
    gateway: Gateway, speed_value: JsonValue
) -> None:
    capture: Final = _native_call(
        gateway,
        "S2",
        f"request-{type(speed_value).__name__}",
        {"speed": speed_value},
        {**_USAGE, "speed": "fast"},
        False,
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
        retries=0,
        expect_success=False,
        allow_no_upstream=True,
    )
    assert capture.status in (200, 400, 422), capture
    assert capture.body, capture


@pytest.mark.parametrize("entry", _INVALID_ENTRY_VALUES)
def test_invalid_fast_provider_entries_reject_or_bill_consistently(gateway: Gateway, entry: JsonValue) -> None:
    identity: Final = f"msg_s3_{uuid.uuid4().hex}"
    model_name: Final = f"integration-{uuid.uuid4().hex}"
    model_info: Final = {"provider_specific_entry": {"fast": entry}}

    def respond(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        assert body.get("speed") == "fast", body
        return _anthropic_reply(identity, False, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire:
        registration: Final = gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": f"anthropic/{_MODEL}",
                    "api_base": wire.url,
                    "api_key": cc.ANTHROPIC_API_KEY,
                    **_RATES,
                },
                "model_info": model_info,
            },
        )
        assert registration.status_code in (200, 400, 422), registration.text
        registration_body: Final = _json_object(registration.content)
        _capture_record(
            "S3",
            f"{entry}-registration",
            registration.status_code,
            registration_body,
            _response_headers(registration.headers),
            None,
            None,
            None,
            (),
        )
        if registration.status_code == 200:
            registered: Final = _json_object(registration.content)
            model_id: Final = string_value(object_value(registered["model_info"])["id"])
            try:
                result: Final = gateway.request(
                    "POST",
                    "/v1/messages",
                    {
                        **cc.claude_code_request(f"S3-{type(entry).__name__}"),
                        "model": model_name,
                        "speed": "fast",
                    },
                )
                result_body: Final = (
                    _json_object(result.content) if result.status_code == 200 else {"error": result.text}
                )
                response_id: Final = _response_id(result_body)
                upstream: Final = wire.drain()
                call_id: Final = result.headers.get("x-litellm-call-id")
                lookup_id: Final = response_id or call_id
                spend: Final = _maybe_row(lookup_id, call_id)
                request_id: Final = string_value(spend["request_id"]) if spend is not None else lookup_id
                _capture_record(
                    "S3",
                    str(entry),
                    result.status_code,
                    result_body,
                    _response_headers(result.headers),
                    response_id,
                    request_id,
                    spend,
                    _upstream_payloads(upstream),
                )
                assert result.status_code in (200, 400, 422, 500), result.text
                if result.status_code == 200:
                    assert response_id is not None and len(upstream) == 1, (result.status_code, result_body, upstream)
                    if spend is not None:
                        assert (
                            len(
                                read_rows(
                                    'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                                    (string_value(spend["request_id"]),),
                                )
                            )
                            == 1
                        ), spend
                else:
                    assert spend is None, (result.status_code, result_body, spend)
            finally:
                gateway.request("POST", "/model/delete", {"id": model_id})


def test_null_provider_entry_uses_standard_costs(gateway: Gateway) -> None:
    _assert_one_x(
        _native_call(
            gateway,
            "S4",
            "null-entry",
            {"speed": "fast"},
            {**_USAGE, "speed": "fast"},
            False,
            {"provider_specific_entry": None},
        )
    )


@pytest.mark.parametrize("status", (401, 400), ids=("unauthorized", "bad-request"))
def test_upstream_errors_are_returned_and_logged_with_zero_spend(gateway: Gateway, status: int) -> None:
    identity: Final = f"error_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return Reply(status=status, body=json.dumps({"error": {"type": "upstream", "message": identity}}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )
        call_id: Final = str(uuid.uuid4())
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                **cc.claude_code_request(f"S5-{uuid.uuid4().hex}"),
                "model": model,
                "speed": "fast",
            },
            headers={"x-litellm-call-id": call_id},
        )
        assert response.status_code == status, response.text
        assert identity in response.text, response.text
        upstream: Final = wire.drain()
        assert len(upstream) == 1, upstream
        assert _upstream_payloads(upstream)[0].get("speed") == "fast", upstream
        assert response.headers.get("x-litellm-call-id") == call_id, response.headers
        spend_rows: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, spend, prompt_tokens, completion_tokens, "
                "metadata->'cost_breakdown' AS cost_breakdown FROM \"LiteLLM_SpendLogs\" "
                "WHERE litellm_call_id=%s",
                (call_id,),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        spend: Final = spend_rows[0]
        capture: Final = _capture_record(
            "S5",
            str(status),
            response.status_code,
            _json_object(response.content),
            _response_headers(response.headers),
            None,
            string_value(spend["request_id"]),
            spend,
            _upstream_payloads(upstream),
        )
        assert capture.spend is not None, capture
        assert float(capture.spend["spend"]) == 0.0, capture
        assert capture.spend["cost_breakdown"] is None, capture


def test_unauthenticated_fast_request_does_not_reach_upstream(gateway: Gateway) -> None:
    with wire_server(lambda _: Reply(body=cc.message_reply("never", _MODEL, _CONTENT, _USAGE))) as wire:
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                **cc.claude_code_request("S6"),
                "model": _MODEL,
                "speed": "fast",
            },
            key="sk-unauthenticated-audit-key",
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()
        request_id: Final = response.headers.get("x-litellm-call-id")
        error_body: Final = _json_object(response.content)
        _capture_record(
            "S6",
            "unauthenticated",
            response.status_code,
            error_body,
            _response_headers(response.headers),
            _response_id(error_body),
            request_id,
            _maybe_row(request_id),
            (),
        )
        if request_id is not None:
            assert read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)) == []
        assert gateway.request("GET", "/health/readiness").status_code == 200


def test_repeated_fast_requests_create_one_spend_row_each(gateway: Gateway) -> None:
    identities: Final = iter(f"msg_{uuid.uuid4().hex}" for _ in range(3))

    def respond(request: Request) -> Reply:
        return _anthropic_reply(next(identities), False, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}
        )
        responses: Final = tuple(
            gateway.request(
                "POST",
                "/v1/messages",
                {
                    **cc.claude_code_request("E1-identical"),
                    "model": model,
                    "speed": "fast",
                    "stream": False,
                    "cache": {"no-cache": True},
                },
            )
            for _ in range(3)
        )
        upstream: Final = wire.drain()
        assert len(upstream) == 3, upstream
        captures: Final = tuple(
            _capture(
                "E1",
                str(index),
                response,
                _json_object(response.content),
                string_value(_json_object(response.content)["id"]),
                (upstream[index],),
            )
            for index, response in enumerate(responses)
        )
        assert all(_upstream_payloads((request,))[0]["speed"] == "fast" for request in upstream), upstream
        assert all(capture.status == 200 for capture in captures), captures
        assert len({capture.response_id for capture in captures}) == 3, captures
        assert all(capture.request_id == capture.response_id for capture in captures), captures
        assert all(capture.spend is not None for capture in captures), captures
        assert all(float(capture.spend["spend"]) == pytest.approx(0.00459) for capture in captures), captures
        cost_breakdowns: Final = tuple(
            object_value(capture.spend["cost_breakdown"]) for capture in captures if capture.spend is not None
        )
        assert len(cost_breakdowns) == len(captures), captures
        assert all(breakdown == cost_breakdowns[0] for breakdown in cost_breakdowns), captures


def test_response_cache_hit_twins_have_one_provider_call_and_two_rows(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    run_nonce: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, False, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(
            gateway, scenario, wire, {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}}, retries=0
        )

        def exercise(endpoint: str, case: str, request_body: Mapping[str, JsonValue]) -> None:
            first: Final = gateway.request("POST", endpoint, request_body)
            second: Final = gateway.request("POST", endpoint, request_body)
            assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
            first_body: Final = _json_object(first.content)
            second_body: Final = _json_object(second.content)
            first_id: Final = _response_id(first_body)
            second_id: Final = _response_id(second_body)
            assert first_id is not None and first_id == second_id, (first_body, second_body)
            upstream: Final = wire.drain()
            rows: Final = eventually(
                lambda: read_rows(
                    "SELECT request_id, spend, prompt_tokens, completion_tokens, cache_hit, "
                    "metadata->'cost_breakdown' AS cost_breakdown FROM \"LiteLLM_SpendLogs\" "
                    "WHERE request_id LIKE %s OR litellm_call_id=%s OR litellm_call_id=%s",
                    (
                        f"{first_id}%",
                        first.headers.get("x-litellm-call-id") or first_id,
                        second.headers.get("x-litellm-call-id") or second_id,
                    ),
                ),
                lambda values: len(values) >= 2,
                seconds=30,
                return_last_on_timeout=True,
            )
            _capture_record(
                "E2",
                f"{case}-twins",
                first.status_code,
                {
                    "first": first_body,
                    "second": second_body,
                    "spend_rows": rows,
                    "second_status": second.status_code,
                },
                {
                    **{f"first_{name}": value for name, value in _response_headers(first.headers).items()},
                    **{f"second_{name}": value for name, value in _response_headers(second.headers).items()},
                },
                first_id,
                string_value(rows[0]["request_id"]) if rows else first_id,
                rows[0] if rows else None,
                _upstream_payloads(upstream),
            )
            assert len(upstream) == 1, upstream
            assert _upstream_payloads(upstream)[0].get("speed") == "fast", upstream
            billed: Final = tuple(row for row in rows if row["cache_hit"] not in (True, "True"))
            cached: Final = tuple(row for row in rows if row["cache_hit"] in (True, "True"))
            assert len(billed) == 1 and len(cached) == 1, rows
            assert float(billed[0]["spend"]) == pytest.approx(0.00459), rows
            assert float(cached[0]["spend"]) == pytest.approx(0), rows
            assert cached[0]["cache_hit"] in (True, "True"), rows
            assert string_value(cached[0]["request_id"]).startswith(first_id), rows
            for response, body, row, suffix in (
                (first, first_body, billed[0], "provider"),
                (second, second_body, cached[0], "cache-hit"),
            ):
                _capture_record(
                    "E2",
                    f"{case}-{suffix}",
                    response.status_code,
                    body,
                    _response_headers(response.headers),
                    _response_id(body),
                    string_value(row["request_id"]),
                    row,
                    _upstream_payloads(upstream),
                )

        exercise(
            "/v1/messages",
            "messages",
            {
                **cc.claude_code_request(f"E2-messages-identical-{run_nonce}"),
                "model": model,
                "speed": "fast",
                "stream": False,
            },
        )
        exercise(
            "/v1/chat/completions",
            "chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"E2-chat-identical-{run_nonce}"}],
                "speed": "fast",
            },
        )


def test_fast_requests_without_cache_tokens_have_zero_cache_costs(gateway: Gateway) -> None:
    _assert_no_cache_fast(
        _native_call(
            gateway,
            "E3",
            "zero-cache",
            {"speed": "fast"},
            {**_NO_CACHE_USAGE, "speed": "fast"},
            False,
            {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
        )
    )


def test_fast_multiplier_of_one_keeps_standard_costs(gateway: Gateway) -> None:
    _assert_one_x(
        _native_call(
            gateway,
            "E4",
            "one-multiplier",
            {"speed": "fast"},
            {**_USAGE, "speed": "fast"},
            False,
            {"provider_specific_entry": {"fast": 1.0}},
        )
    )


def test_spend_logs_ui_exposes_the_fast_cost_breakdown(gateway: Gateway) -> None:
    capture: Final = _native_call(
        gateway,
        "E5",
        "ui-log",
        {"speed": "fast"},
        {**_USAGE, "speed": "fast"},
        False,
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
    )
    assert capture.request_id is not None, capture
    now: Final = datetime.now(timezone.utc)
    logs: Final = gateway.request(
        "GET",
        "/spend/logs/ui",
        params={
            "request_id": capture.request_id,
            "start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    assert logs.status_code == 200, logs.text
    page: Final = _json_object(logs.content)
    records: Final = page["data"]
    assert isinstance(records, list) and len(records) == 1, page
    record: Final = object_value(records[0])
    assert float(record["spend"]) == pytest.approx(0.00459), record
    assert float(object_value(record["metadata"])["cost_breakdown"]["cache_read_cost"]) == pytest.approx(0.00024), (
        record
    )
    _capture_record(
        "E5",
        "spend-logs-ui",
        logs.status_code,
        page,
        _response_headers(logs.headers),
        capture.response_id,
        capture.request_id,
        capture.spend,
        capture.upstream,
    )


@pytest.mark.parametrize("stream", _STREAMS)
def test_anthropic_passthrough_fast_request_records_a_spend_row(gateway: Gateway, tmp_path: Path, stream: bool) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _anthropic_reply(identity, stream, {**_USAGE, "speed": "fast"})

    with wire_server(respond) as wire:
        overrides: Final = {
            "ANTHROPIC_API_BASE": wire.url,
            "ANTHROPIC_API_KEY": cc.ANTHROPIC_API_KEY,
        }
        with owned_proxy_process(gateway, tmp_path, overrides, workers=2) as owned:
            with owned.gateway.scenario():
                response: Final = owned.gateway.request(
                    "POST",
                    "/anthropic/v1/messages",
                    {
                        **cc.claude_code_request(f"N5-{uuid.uuid4().hex}"),
                        "model": _MODEL,
                        "speed": "fast",
                        "stream": stream,
                    },
                )
                assert response.status_code == 200, response.text
                response_body: Final = _response_body(response, stream)
                response_id: Final = _response_id(response_body)
                assert response_id is not None, response_body
                upstream: Final = wire.drain()
                assert len(upstream) == 1, upstream
                assert _upstream_payloads(upstream)[0].get("speed") == "fast", upstream
                capture: Final = _capture("N5", str(stream), response, response_body, response_id, upstream)
                assert capture.request_id is not None, capture
                assert capture.spend is not None, capture
                assert (
                    len(
                        read_rows(
                            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                            (capture.request_id,),
                        )
                    )
                    == 1
                ), capture


def test_mixed_fast_requests_record_spend_during_deterministic_outages(gateway: Gateway) -> None:
    run_nonce: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        ordinal_match: Final = re.search(r"C1-(\d+)", json.dumps(body))
        assert ordinal_match is not None, body
        ordinal: Final = int(ordinal_match.group(1))
        if 10 <= ordinal < 15:
            return Reply(status=503, body=b'{"error":{"message":"deterministic outage"}}')
        return _anthropic_reply(
            f"msg_burst_{run_nonce}_{ordinal}",
            bool(body.get("stream")),
            {**_USAGE, "speed": "fast"},
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            model_info={"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
            num_retries=0,
            **_RATES,
        )

        def send(
            ordinal: int,
        ) -> tuple[int, bool, httpx.Response]:
            endpoint: Final = (
                "/v1/messages" if ordinal % 3 == 0 else "/v1/chat/completions" if ordinal % 3 == 1 else "/v1/responses"
            )
            stream: Final = ordinal % 2 == 0
            if endpoint == "/v1/messages":
                body: Final = {
                    **cc.claude_code_request(f"C1-{ordinal}-{run_nonce}"),
                    "metadata": {"audit_ordinal": ordinal},
                    "model": model,
                    "speed": "fast",
                    "stream": stream,
                }
            elif endpoint == "/v1/chat/completions":
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": f"C1-{ordinal}-{run_nonce}"}],
                    "stream": stream,
                    "stream_options": {"include_usage": True},
                    "speed": "fast",
                }
            else:
                body = {"model": model, "input": f"C1-{ordinal}-{run_nonce}", "stream": stream, "speed": "fast"}
            call_id: Final = str(uuid.uuid5(uuid.NAMESPACE_URL, f"litfix-9110-c1-{run_nonce}-{ordinal}"))
            response: Final = gateway.request("POST", endpoint, body, headers={"x-litellm-call-id": call_id})
            assert response.headers.get("x-litellm-call-id") == call_id, response.headers
            return ordinal, stream, response

        with ThreadPoolExecutor(max_workers=30) as executor:
            results: Final = tuple(executor.map(send, range(30)))
        upstream: Final = wire.drain()
        upstream_by_ordinal: Final = {
            int(match.group(1)): request
            for request in upstream
            if (match := re.search(r"C1-(\d+)", json.dumps(_json_object(request.body)))) is not None
        }

        def capture_result(result: tuple[int, bool, httpx.Response]) -> tuple[int, httpx.Response, _Capture]:
            ordinal, stream, response = result
            body: Final = _response_body(response, stream)
            capture: Final = _capture(
                "C1",
                str(ordinal),
                response,
                body,
                _response_id(body),
                (upstream_by_ordinal[ordinal],),
                expect_spend_row=True,
            )
            return ordinal, response, capture

        captures: Final = tuple(capture_result(result) for result in results)
        assert len(upstream) == 30, upstream
        assert len(upstream_by_ordinal) == 30, upstream_by_ordinal
        assert sum(capture.status == 200 for _, _, capture in captures) == 25, captures
        assert sum(capture.status == 503 for _, _, capture in captures) == 5, captures
        for ordinal, response, capture in captures:
            assert capture.body, capture
            assert capture.spend is not None, (ordinal, response.status_code, capture)
            assert capture.request_id is not None, (ordinal, response.status_code, capture)
            call_id: Final = capture.headers["x-litellm-call-id"]
            assert capture.upstream and capture.upstream[0].get("speed") == "fast", capture
            assert (
                len(
                    read_rows(
                        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
                        (call_id,),
                    )
                )
                == 1
            ), capture
            if response.status_code == 200:
                assert capture.response_id is not None, capture
                breakdown: Final = object_value(capture.spend["cost_breakdown"])
                assert float(capture.spend["spend"]) == pytest.approx(0.00459), capture
                assert capture.spend["prompt_tokens"] == 800 and capture.spend["completion_tokens"] == 50, capture
                assert float(breakdown["input_cost"]) == pytest.approx(0.00399), capture
                assert float(breakdown["cache_read_cost"]) == pytest.approx(0.00024), capture
                assert float(breakdown["cache_creation_cost"]) == pytest.approx(0.00315), capture
            else:
                assert response.status_code == 503 and capture.body, capture
                assert capture.headers["x-litellm-call-id"] == capture.request_id, capture
                assert float(capture.spend["spend"]) == 0.0, capture
                assert capture.spend["cost_breakdown"] is None, capture


def test_owned_two_worker_proxy_survives_one_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    gate_once: Final = threading.Semaphore(1)
    run_nonce: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        serialized_body: Final = json.dumps(body)
        ordinal_match: Final = re.search(r"C2-(\d+)", serialized_body)
        assert ordinal_match is not None or "C2-after-kill" in serialized_body, body
        ordinal: Final = int(ordinal_match.group(1)) if ordinal_match is not None else 6
        gate_response: Final = gate_once.acquire(blocking=False)
        return Reply(
            chunks=(
                cc.message_reply(
                    f"msg_c2_{run_nonce}_{ordinal}",
                    _MODEL,
                    _CONTENT,
                    {**_USAGE, "speed": "fast"},
                ),
            ),
            gate_after_first=release if gate_response else None,
        )

    with (
        wire_server(respond) as wire,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"INTEGRATION_UPSTREAM_URL": wire.url},
            workers=2,
        ) as owned,
    ):
        with owned.gateway.scenario() as scenario:
            model: Final = scenario.model(
                model=f"anthropic/{_MODEL}",
                api_base=wire.url,
                api_key=cc.ANTHROPIC_API_KEY,
                model_info={"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
                num_retries=0,
                **_RATES,
            )
            root: Final = owned.process.pid
            children: Final = eventually(
                lambda: tuple(
                    member for member in group_members(os.getpgid(root)) if member.pid != root and member.ppid() == root
                ),
                lambda values: len(values) >= 2,
                seconds=30,
            )

            def send(ordinal: int) -> tuple[int, httpx.Response | None, str | None]:
                try:
                    return (
                        ordinal,
                        owned.gateway.request(
                            "POST",
                            "/v1/messages",
                            {
                                **cc.claude_code_request(f"C2-{ordinal}-{run_nonce}"),
                                "model": model,
                                "speed": "fast",
                                "stream": False,
                            },
                        ),
                        None,
                    )
                except httpx.TransportError as error:
                    return ordinal, None, str(error)

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures: Final = tuple(executor.submit(send, ordinal) for ordinal in range(6))
                try:
                    eventually(lambda: wire.received.qsize(), lambda count: count >= 2, seconds=30)
                    children[0].send_signal(signal.SIGKILL)
                finally:
                    release.set()
                burst_results: Final = tuple(future.result(timeout=45) for future in futures)
            burst_upstream: Final = wire.drain()
            upstream_by_ordinal: Final = {
                int(match.group(1)): request
                for request in burst_upstream
                if (match := re.search(r"C2-(\d+)", json.dumps(_json_object(request.body)))) is not None
            }
            successful_ids: Final = tuple(
                _response_id(_response_body(response, False))
                for _, response, _ in burst_results
                if response is not None and response.status_code == 200
            )
            assert successful_ids and all(isinstance(value, str) for value in successful_ids), burst_results
            assert len(successful_ids) == len(set(successful_ids)), burst_results
            for ordinal, response, error in burst_results:
                request: Final = upstream_by_ordinal.get(ordinal)
                if response is None:
                    assert error is not None, (ordinal, response, error)
                    _capture_transport_error(
                        "C2",
                        str(ordinal),
                        error,
                        (request,) if request is not None else (),
                    )
                    continue
                response_body: Final = _response_body(response, False)
                response_id: Final = _response_id(response_body)
                call_id: Final = response.headers.get("x-litellm-call-id")
                lookup_id: Final = response_id or call_id
                spend: Final = _maybe_row(lookup_id, call_id)
                request_id: Final = string_value(spend["request_id"]) if spend is not None else lookup_id
                _capture_record(
                    "C2",
                    str(ordinal),
                    response.status_code,
                    response_body,
                    _response_headers(response.headers),
                    response_id,
                    request_id,
                    spend,
                    _upstream_payloads((request,)) if request is not None else (),
                )
                if response.status_code == 200:
                    assert response_id is not None and request_id is not None, (ordinal, response_id, request_id)
                    assert spend is not None, (ordinal, response_id)
                    assert (
                        len(
                            read_rows(
                                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                                (request_id,),
                            )
                        )
                        == 1
                    ), (ordinal, response_id)
                else:
                    assert response_body and spend is None, (ordinal, response_body, spend)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/messages",
                {
                    **cc.claude_code_request(f"C2-after-kill-{run_nonce}"),
                    "model": model,
                    "speed": "fast",
                    "stream": False,
                },
            )
            assert response.status_code == 200, response.text
            body: Final = _json_object(response.content)
            response_id: Final = _response_id(body)
            assert response_id is not None, body
            capture: Final = _capture("C2", "after-worker-kill", response, body, response_id, wire.drain())
            assert capture.spend is not None, capture
            assert len(read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,))) == 1
            readiness: Final = owned.gateway.request("GET", "/health/readiness")
            assert readiness.status_code == 200, readiness.text
