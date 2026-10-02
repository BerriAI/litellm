import asyncio
import base64
import json
import os
import re
import signal
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process, ready_deadline_seconds, stop_deadline_seconds
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with

_API_KEY: Final = "synthetic-mantle-bearer"
_SIGNING_KEY: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_GPT56: Final = "openai.gpt-5.6-terra"
_GPT6: Final = "openai.gpt-6-sol"
_GPT54: Final = "openai.gpt-5.4"
_GPT_OSS: Final = "openai.gpt-oss-120b"
_KNOWN_MODELS: Final = frozenset({_GPT56, _GPT6, _GPT54, _GPT_OSS})
_GPT_SERIES: Final = frozenset({_GPT56, _GPT6, _GPT54})
_REJECTS_MAX_TOKENS: Final = frozenset({_GPT56, _GPT6})
_LIMIT_PARAMS: Final = ("max_tokens", "max_completion_tokens")
_CONFIG_MODEL: Final = "bedrock-mantle-gpt-output-limit-chaos"
_OPENAI_BASE: Final = "/openai/v1"
_CHAT_TARGET: Final = "/openai/v1/chat/completions"
_CHAT_TARGETS: Final = frozenset({_CHAT_TARGET, "/v1/chat/completions"})
_RESPONSES_TARGETS: Final = frozenset({"/openai/v1/responses", "/v1/responses"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CHAT_INFO: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"mode": "chat", "supported_endpoints": ["/v1/chat/completions", "/v1/responses"]}
)
_TOOLS: Final[tuple[JsonValue, ...]] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
)
_SPEND_BY_ID: Final = 'SELECT request_id, status, model_group, cache_hit FROM "LiteLLM_SpendLogs" WHERE request_id=%s'
_SPEND_BY_PREFIX: Final = (
    'SELECT request_id, status, model_group, cache_hit FROM "LiteLLM_SpendLogs" WHERE starts_with(request_id, %s)'
)
_SPEND_BY_GROUP: Final = 'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s'

Endpoint = Literal["chat", "messages"]


@dataclass(frozen=True, slots=True)
class _Attempt:
    target: str
    body: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    served: tuple[_Served, ...]
    lost: int


def _question(marker: str) -> str:
    return f"Question marker-{marker}"


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _error(status: int, code: str, message: str, param: str | None, kind: str = "invalid_request_error") -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"code": code, "message": message, "param": param, "type": kind}}).encode(),
    )


def _param_verdict(backend: str, param: str, value: JsonValue) -> Reply | None:
    if value is None:
        return None
    if param == "max_tokens" and backend in _REJECTS_MAX_TOKENS:
        return _error(
            400, "unsupported_parameter", f"Unsupported parameter: '{param}' is not supported with this model.", param
        )
    if isinstance(value, bool) or not isinstance(value, int):
        return _error(
            400,
            "invalid_type",
            f"Invalid type for '{param}': expected an integer, but got {type(value).__name__} instead.",
            param,
        )
    if value < 1:
        return _error(
            400,
            "integer_below_min_value",
            f"Invalid '{param}': integer below minimum value. Expected a value >= 1, but got {value} instead.",
            param,
        )
    return None


def _limit_verdict(backend: str, body: Mapping[str, JsonValue]) -> Reply | None:
    return next(filter(None, (_param_verdict(backend, param, body.get(param)) for param in _LIMIT_PARAMS)), None)


def _chat_reply(backend: str, marker: str, stream: bool, pause: float = 0) -> Reply:
    identity: Final = f"chatcmpl-{marker}-{uuid.uuid4().hex[:8]}"
    usage: Final = {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": backend,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _answer(marker)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": backend}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer "}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": f"marker-{marker}"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
        pause_between_chunks=pause,
    )


def _responses_reply(backend: str, marker: str, stream: bool) -> Reply:
    response: Final = {
        "id": f"resp_upstream_{marker}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": backend,
        "output": [
            {
                "id": f"msg_{marker}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": _answer(marker),
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _peer(*, pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b'{"object": "list", "data": []}')
        if request.headers.get("authorization") != f"Bearer {_API_KEY}":
            return _error(401, "authentication_error", "Invalid bearer token.", None, kind="authentication_error")
        body: Final = _JSON_OBJECT.validate_json(request.body)
        backend: Final = string_value(body["model"])
        if backend not in _KNOWN_MODELS:
            return _error(404, "not_found_error", f"The model '{backend}' does not exist", None)
        marker: Final = _marker_of(request)
        stream: Final = body.get("stream") is True
        if request.target in _RESPONSES_TARGETS:
            return _responses_reply(backend, marker, stream)
        assert request.target in _CHAT_TARGETS, request.target
        verdict: Final = _limit_verdict(backend, body)
        return verdict if verdict is not None else _chat_reply(backend, marker, stream, pause)

    return respond


_mantle: Final = _peer()


def _deployment(
    scenario: Scenario,
    wire: Wire,
    backend: str,
    *,
    base: str = _OPENAI_BASE,
    api_key: str = _API_KEY,
    **extra: JsonValue,
) -> str:
    reasoning: Final[Mapping[str, JsonValue]] = (
        {"reasoning_effort": "none"} if backend.rsplit("/", 1)[-1] in _GPT_SERIES else {}
    )
    return scenario.model(
        model=f"bedrock_mantle/{backend}",
        custom_llm_provider="bedrock_mantle",
        api_base=wire.url + base,
        api_key=api_key,
        model_info=_CHAT_INFO,
        **reasoning,
        **extra,
    )


def _deployment_id(gateway: Gateway, model: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    (entry,) = tuple(object_value(candidate) for candidate in entries if object_value(candidate)["model_name"] == model)
    return string_value(object_value(entry["model_info"])["id"])


def _attempts(wire: Wire) -> tuple[_Attempt, ...]:
    return tuple(
        _Attempt(request.target, _JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
        if request.method == "POST"
    )


def _limits(body: Mapping[str, JsonValue]) -> tuple[JsonValue, JsonValue]:
    return body.get("max_tokens"), body.get("max_completion_tokens")


def _renamed(attempt: _Attempt, value: int) -> None:
    assert "max_tokens" not in attempt.body, attempt.body
    assert attempt.body.get("max_completion_tokens") == value, attempt.body


def _marker_in(attempt: _Attempt) -> str:
    found: Final = _MARKER.search(json.dumps(attempt.body))
    assert found is not None, attempt.body
    return found.group(1)


def _spend_row(identity: str) -> Mapping[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_BY_ID, (identity,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_logged(identity: str, model: str, status: str) -> None:
    row: Final = _spend_row(identity)
    assert (row["status"], row["model_group"]) == (status, model), row


def _assert_failure_logged(response: httpx.Response) -> None:
    assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure", response.text


def _spend_statuses(model: str, expected: int) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(_SPEND_BY_GROUP, (model,)), lambda found: len(found) >= expected, seconds=70
    )
    assert len({row["request_id"] for row in rows}) == len(rows), rows
    return [row["status"] for row in rows]


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _chat_body(model: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": _question(marker)}], "num_retries": 0, **extra}


def _response_id(text: str) -> str:
    return string_value(_JSON_OBJECT.validate_json(text)["id"])


def _stream_text(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> str:
    with gateway.client.stream(
        "POST",
        path,
        json=body,
        headers={"Authorization": f"Bearer {gateway.key}", "anthropic-version": "2023-06-01"},
    ) as response:
        text: Final = response.read().decode()
    assert response.status_code == 200, text
    return text


def _stream_id(text: str) -> str:
    first: Final = next(line for line in text.splitlines() if line.startswith("data: {"))
    return _response_id(first.removeprefix("data: "))


def _responses_stream_id(text: str) -> str:
    first: Final = next(line for line in text.splitlines() if line.startswith("data: {"))
    event: Final = _JSON_OBJECT.validate_json(first.removeprefix("data: "))
    return string_value(object_value(event["response"])["id"])


def _managed_response_id(issued: str) -> str:
    managed: Final = decrypt_if_encrypted_with(issued.removeprefix("resp_"), _SIGNING_KEY)
    assert managed is not None, issued
    return managed.split(";", 1)[0].rsplit("response_id:", 1)[1]


def _upstream_response_id(issued: str) -> str:
    decoded: Final = base64.b64decode(_managed_response_id(issued).removeprefix("resp_")).decode()
    return decoded.rsplit("response_id:", 1)[1]


def _text_blocks(message: anthropic.types.Message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


def test_openai_sdk_max_tokens_reaches_mantle_as_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        with openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": _question(marker)}], max_tokens=1000
            )
        assert completion.choices[0].message.content == _answer(marker), completion
        assert completion.id.startswith(f"chatcmpl-{marker}-"), completion.id
        (attempt,) = _attempts(wire)
        assert attempt.target == _CHAT_TARGET, attempt
        assert attempt.body["model"] == _GPT56, attempt.body
        _renamed(attempt, 1000)
        _assert_logged(completion.id, model, "success")


async def test_async_openai_sdk_stream_keeps_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        async with openai.AsyncOpenAI(
            base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _question(marker)}],
                max_completion_tokens=1000,
                stream=True,
            )
            chunks: Final = [chunk async for chunk in stream]
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        assert text == _answer(marker), text
        (identity,) = {chunk.id for chunk in chunks}
        (attempt,) = _attempts(wire)
        assert attempt.body["stream"] is True, attempt.body
        _renamed(attempt, 1000)
        _assert_logged(identity, model, "success")


def test_stream_with_tools_carries_the_deployment_limit_as_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, max_tokens=1000)
        text: Final = _stream_text(
            gateway,
            "/v1/chat/completions",
            _chat_body(model, marker, stream=True, stream_options={"include_usage": True}, tools=_TOOLS),
        )
        assert f"marker-{marker}" in text and text.rstrip().endswith("data: [DONE]"), text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 1000)
        assert attempt.body["tools"] == list(_TOOLS), attempt.body
        assert attempt.body["stream_options"] == {"include_usage": True}, attempt.body
        _assert_logged(_stream_id(text), model, "success")


def test_anthropic_sdk_messages_max_tokens_reaches_mantle_as_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        with anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            message: Final = client.messages.create(
                model=model, max_tokens=1000, messages=[{"role": "user", "content": _question(marker)}]
            )
        assert _text_blocks(message) == _answer(marker), message
        (attempt,) = _attempts(wire)
        assert attempt.target == _CHAT_TARGET, attempt
        _renamed(attempt, 1000)
        _assert_logged(message.id, model, "success")


async def test_async_anthropic_sdk_messages_stream_reaches_mantle_as_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        async with anthropic.AsyncAnthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            async with client.messages.stream(
                model=model, max_tokens=1000, messages=[{"role": "user", "content": _question(marker)}]
            ) as stream:
                pieces: Final = [piece async for piece in stream.text_stream]
                message: Final = await stream.get_final_message()
        assert "".join(pieces) == _answer(marker), pieces
        (attempt,) = _attempts(wire)
        assert attempt.body["stream"] is True, attempt.body
        _renamed(attempt, 1000)
        _assert_logged(message.id, model, "success")


def test_responses_max_output_tokens_is_forwarded_natively_on_both_legs(gateway: Gateway) -> None:
    unary: Final = uuid.uuid4().hex
    streamed: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": _question(unary), "max_output_tokens": 1000}
        )
        assert response.status_code == 200, response.text
        issued: Final = _response_id(response.text)
        assert _answer(unary) in response.text and _upstream_response_id(issued) == f"resp_upstream_{unary}", (
            response.text
        )
        text: Final = _stream_text(
            gateway,
            "/v1/responses",
            {"model": model, "input": _question(streamed), "max_output_tokens": 1000, "stream": True},
        )
        streamed_issued: Final = _responses_stream_id(text)
        assert _answer(streamed) in text and "response.completed" in text, text
        assert _upstream_response_id(streamed_issued) == f"resp_upstream_{streamed}", text
        attempts: Final = _attempts(wire)
        assert [attempt.target for attempt in attempts] == ["/openai/v1/responses"] * 2, attempts
        for attempt in attempts:
            assert attempt.body["max_output_tokens"] == 1000 and _limits(attempt.body) == (None, None), attempt.body
        _assert_logged(issued, model, "success")
        _assert_logged(_managed_response_id(streamed_issued), model, "success")


def test_region_prefixed_deployment_on_a_v1_base_sends_the_bare_model_with_max_completion_tokens(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, f"us-gov-west-1/{_GPT56}", base="/v1")
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=1000))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        assert (attempt.target, attempt.body["model"]) == ("/v1/chat/completions", _GPT56), attempt
        _renamed(attempt, 1000)
        _assert_logged(_response_id(response.text), model, "success")


def test_gpt6_deployment_sends_max_completion_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT6)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=1000))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        assert attempt.body["model"] == _GPT6, attempt.body
        _renamed(attempt, 1000)
        _assert_logged(_response_id(response.text), model, "success")


@pytest.mark.parametrize("sent", _LIMIT_PARAMS)
def test_gpt_oss_control_keeps_the_openai_compatible_max_tokens(gateway: Gateway, sent: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT_OSS, base="/v1")
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, **{sent: 50}))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        assert attempt.target == "/v1/chat/completions", attempt
        assert _limits(attempt.body) == (50, None) and "max_completion_tokens" not in attempt.body, attempt.body
        _assert_logged(_response_id(response.text), model, "success")


def test_gpt54_control_joins_the_renamed_series(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT54)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 50)
        _assert_logged(_response_id(response.text), model, "success")


def test_explicit_max_completion_tokens_wins_over_max_tokens(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50, max_completion_tokens=1000)
        )
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 1000)
        _assert_logged(_response_id(response.text), model, "success")


def test_request_limit_wins_over_the_deployment_limit(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, max_tokens=1000)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 50)
        _assert_logged(_response_id(response.text), model, "success")


def test_identical_requests_with_no_cache_are_attempted_and_logged_once_each(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        body: Final = _chat_body(model, marker, max_tokens=1000, cache={"no-cache": True})
        first: Final = gateway.request("POST", "/v1/chat/completions", body)
        second: Final = gateway.request("POST", "/v1/chat/completions", body)
        assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
        identities: Final = (_response_id(first.text), _response_id(second.text))
        assert identities[0] != identities[1], identities
        attempts: Final = _attempts(wire)
        assert len(attempts) == 2, attempts
        for attempt in attempts:
            _renamed(attempt, 1000)
        for identity in identities:
            _assert_logged(identity, model, "success")


@pytest.mark.parametrize("hostile", ["abc", [1000], "", "x" * 5120], ids=["string", "list", "empty", "5kb"])
def test_hostile_max_tokens_is_rejected_by_mantle_under_its_new_name(gateway: Gateway, hostile: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=hostile))
        assert response.status_code == 400, response.text
        assert "invalid_type" in response.text and "'max_completion_tokens'" in response.text, response.text
        assert "'max_tokens'" not in response.text, response.text
        (attempt,) = _attempts(wire)
        assert _limits(attempt.body) == (None, hostile) and "max_tokens" not in attempt.body, attempt.body
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        _assert_failure_logged(response)


def test_duplicate_max_tokens_keys_in_raw_json_send_the_last_value_renamed(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        raw: Final = (
            f'{{"model": "{model}", "messages": [{{"role": "user", "content": "{_question(marker)}"}}], '
            '"max_tokens": 50, "max_tokens": 1000}'
        ).encode()
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "content-type": "application/json"},
        )
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 1000)
        _assert_logged(_response_id(response.text), model, "success")


@pytest.mark.parametrize("below", [0, -5])
def test_out_of_range_max_tokens_is_rejected_by_mantle_under_its_new_name(gateway: Gateway, below: int) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=below))
        assert response.status_code == 400, response.text
        assert "integer_below_min_value" in response.text and "'max_completion_tokens'" in response.text, response.text
        assert "'max_tokens'" not in response.text, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, below)
        _assert_failure_logged(response)


def test_wrong_deployment_bearer_reaches_the_caller_as_401_after_one_attempt(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, api_key="synthetic-wrong-bearer")
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 401, response.text
        assert "Invalid bearer token" in response.text, response.text
        assert len(_attempts(wire)) == 1
        _assert_failure_logged(response)


def test_unknown_mantle_model_reaches_the_caller_as_404_after_one_attempt(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, "openai.gpt-5.6-cyber")
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 404, response.text
        assert "The model 'openai.gpt-5.6-cyber' does not exist" in response.text, response.text
        assert len(_attempts(wire)) == 1
        _assert_failure_logged(response)


def test_allowed_openai_params_max_tokens_is_sent_verbatim_beside_the_renamed_limit(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, allowed_openai_params=["max_tokens"])
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 400, response.text
        assert "Unsupported parameter: 'max_tokens'" in response.text, response.text
        (attempt,) = _attempts(wire)
        assert _limits(attempt.body) == (50, 50), attempt.body
        _assert_failure_logged(response)


def test_additional_drop_params_max_completion_tokens_drops_before_the_rename(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, additional_drop_params=["max_completion_tokens"])
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 50)
        _assert_logged(_response_id(response.text), model, "success")


def test_unrelated_key_scoped_to_the_control_deployment_keeps_working(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT_OSS, base="/v1")
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50), key=key
        )
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        assert _limits(attempt.body) == (50, None), attempt.body
        _assert_logged(_response_id(response.text), model, "success")


def test_unauthenticated_request_never_reaches_mantle(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, max_tokens=50), key="sk-synthetic-not-a-key"
        )
        assert response.status_code == 401, response.text
        assert _attempts(wire) == ()


@pytest.mark.parametrize("extra", [{"max_tokens": None}, {}], ids=["null", "missing"])
def test_null_or_missing_max_tokens_sends_no_limit(gateway: Gateway, extra: Mapping[str, JsonValue]) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, **extra))
        assert response.status_code == 200, response.text
        (attempt,) = _attempts(wire)
        assert not ({"max_tokens", "max_completion_tokens"} & attempt.body.keys()), attempt.body
        _assert_logged(_response_id(response.text), model, "success")


def test_identical_requests_without_no_cache_serve_the_second_from_cache(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        body: Final = _chat_body(model, marker, max_tokens=1000)
        first: Final = gateway.request("POST", "/v1/chat/completions", body)
        assert first.status_code == 200, first.text
        identity: Final = _response_id(first.text)
        _assert_logged(identity, model, "success")
        second: Final = gateway.request("POST", "/v1/chat/completions", body)
        assert second.status_code == 200, second.text
        assert _response_id(second.text) == identity, second.text
        (attempt,) = _attempts(wire)
        _renamed(attempt, 1000)
        twin: Final = eventually(
            lambda: read_rows(_SPEND_BY_PREFIX, (f"{identity}_cache_hit",)), lambda rows: len(rows) == 1, seconds=70
        )
        assert (twin[0]["status"], twin[0]["cache_hit"], twin[0]["model_group"]) == ("success", "True", model), twin


@pytest.mark.timeout(150)
def test_model_update_of_the_deployment_limit_stays_renamed_and_converges(gateway: Gateway) -> None:
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56, max_tokens=1000)
        first: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, uuid.uuid4().hex))
        assert first.status_code == 200, first.text
        (initial,) = _attempts(wire)
        _renamed(initial, 1000)
        updated: Final = gateway.request(
            "POST",
            "/model/update",
            {
                "model_name": model,
                "litellm_params": {"model": f"bedrock_mantle/{_GPT56}", "max_tokens": 2000},
                "model_info": {"id": _deployment_id(gateway, model)},
            },
        )
        assert updated.status_code == 200, updated.text

        def observed_limits() -> tuple[JsonValue, ...]:
            with httpx.Client(
                base_url=_proxy_url(gateway),
                timeout=30,
                trust_env=False,
                headers={"Authorization": f"Bearer {gateway.key}", "Connection": "close"},
            ) as fresh_connections:
                responses: Final = tuple(
                    fresh_connections.post("/v1/chat/completions", json=_chat_body(model, uuid.uuid4().hex))
                    for _ in range(8)
                )
            assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
            attempts: Final = _attempts(wire)
            assert all("max_tokens" not in attempt.body for attempt in attempts), attempts
            return tuple(attempt.body.get("max_completion_tokens") for attempt in attempts)

        converged: Final = eventually(observed_limits, lambda limits: limits == (2000,) * 8, seconds=100)
        assert converged == (2000,) * 8, converged


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"


def _burst_body(model: str, call: _Call, limit: int | None) -> dict[str, JsonValue]:
    common: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": call.stream,
        "messages": [{"role": "user", "content": _question(call.marker)}],
    }
    match call.endpoint:
        case "chat":
            return {**common, **({} if limit is None else {"max_tokens": limit})}
        case "messages":
            assert limit is not None
            return {**common, "max_tokens": limit}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call, limit: int | None) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_burst_body(model, call, limit),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str,
    key: str,
    model: str,
    calls: tuple[_Call, ...],
    *,
    limit: int | None,
    tolerate_transport_errors: bool = False,
) -> _Outcome:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call, limit) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    served: Final = tuple(result for result in results if isinstance(result, _Served))
    return _Outcome(served, len(results) - len(served))


def _calls(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _assert_each_attempted_once_renamed(attempts: tuple[_Attempt, ...], calls: tuple[_Call, ...], value: int) -> None:
    assert sorted(_marker_in(attempt) for attempt in attempts) == sorted(call.marker for call in calls), attempts
    for attempt in attempts:
        _renamed(attempt, value)


async def test_concurrent_chat_and_messages_burst_is_renamed_and_logged_once_each(gateway: Gateway) -> None:
    calls: Final = _calls(30, ("chat", "messages"), lambda index: index % 2 == 0)
    with wire_server(_mantle) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        outcome: Final = await _burst(_proxy_url(gateway), gateway.key, model, calls, limit=1000)
        assert (len(outcome.served), outcome.lost) == (30, 0)
        for item in outcome.served:
            _assert_answered_with_its_own_marker(item)
        _assert_each_attempted_once_renamed(_attempts(wire), calls, 1000)
        assert _spend_statuses(model, 30) == ["success"] * 30


async def test_peer_503s_reach_their_callers_once_while_the_rest_are_renamed_and_served(gateway: Gateway) -> None:
    calls: Final = _calls(12, ("chat",), lambda _: False)
    unlucky: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        if request.method == "POST" and _marker_of(request) in unlucky:
            return _error(503, "service_unavailable", "The server is overloaded.", None, kind="server_error")
        return _mantle(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        outcome: Final = await _burst(_proxy_url(gateway), gateway.key, model, calls, limit=1000)
        assert (len(outcome.served), outcome.lost) == (12, 0)
        for item in outcome.served:
            if item.call.marker in unlucky:
                assert item.status == 503 and "The server is overloaded." in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
        recovery: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
        (recovered,) = (await _burst(_proxy_url(gateway), gateway.key, model, (recovery,), limit=1000)).served
        _assert_answered_with_its_own_marker(recovered)
        _assert_each_attempted_once_renamed(_attempts(wire), (*calls, recovery), 1000)


async def test_slow_mantle_streams_are_forwarded_once_renamed(gateway: Gateway) -> None:
    calls: Final = _calls(10, ("chat",), lambda _: True)
    with wire_server(_peer(pause=0.3)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire, _GPT56)
        outcome: Final = await _burst(_proxy_url(gateway), gateway.key, model, calls, limit=1000)
        assert (len(outcome.served), outcome.lost) == (10, 0)
        for item in outcome.served:
            _assert_answered_with_its_own_marker(item)
            assert item.text.rstrip().endswith("data: [DONE]"), item.text
        _assert_each_attempted_once_renamed(_attempts(wire), calls, 1000)
        assert _spend_statuses(model, 10) == ["success"] * 10


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {
                    "model": f"bedrock_mantle/{_GPT56}",
                    "custom_llm_provider": "bedrock_mantle",
                    "api_base": wire.url + _OPENAI_BASE,
                    "api_key": _API_KEY,
                    "reasoning_effort": "none",
                    "max_tokens": 1000,
                },
                "model_info": dict(_CHAT_INFO),
            }
        ],
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
    }
    path: Final = tmp_path / "bedrock-mantle-output-limit-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _holding_peer(held_markers: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return _mantle(request)
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _mantle(request)

    return respond


_CHAOS_TRAFFIC_SECONDS: Final = 90


def _chaos_budget(boots: int, stops: int) -> float:
    return boots * ready_deadline_seconds() + stops * stop_deadline_seconds() + _CHAOS_TRAFFIC_SECONDS


@pytest.mark.timeout(_chaos_budget(boots=2, stops=1))
async def test_worker_sigkill_mid_burst_leaves_the_sibling_renaming_the_yaml_limit(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(_holding_peer(held_markers, release)) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    _proxy_url(candidate),
                    candidate.key,
                    _CONFIG_MODEL,
                    calls,
                    limit=None,
                    tolerate_transport_errors=True,
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            outcome: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert (len(outcome.served), outcome.lost) == (held_by[survivor_pid], held_by[victim_pid]), (
                held_by,
                outcome,
            )
            for item in outcome.served:
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = (
                await _burst(_proxy_url(candidate), candidate.key, _CONFIG_MODEL, (follow_up,), limit=None)
            ).served
            _assert_answered_with_its_own_marker(answered)
            _assert_each_attempted_once_renamed(_attempts(wire), (*calls, follow_up), 1000)
            _assert_logged(_response_id(answered.text), _CONFIG_MODEL, "success")


@pytest.mark.timeout(_chaos_budget(boots=2, stops=2))
async def test_proxy_restart_mid_burst_loses_only_in_flight_requests_and_renames_after_restart(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(_holding_peer(held_markers, release)) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as first:
            burst: Final = asyncio.create_task(
                _burst(
                    _proxy_url(first.gateway),
                    first.gateway.key,
                    _CONFIG_MODEL,
                    calls,
                    limit=None,
                    tolerate_transport_errors=True,
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            first.process.terminate()
            release.set()
            outcome: Final = await burst
        assert len(outcome.served) + outcome.lost == len(calls), outcome
        for item in outcome.served:
            _assert_answered_with_its_own_marker(item)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as second:
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = (
                await _burst(_proxy_url(second.gateway), second.gateway.key, _CONFIG_MODEL, (follow_up,), limit=None)
            ).served
            _assert_answered_with_its_own_marker(answered)
            _assert_each_attempted_once_renamed(_attempts(wire), (*calls, follow_up), 1000)
            _assert_logged(_response_id(answered.text), _CONFIG_MODEL, "success")
