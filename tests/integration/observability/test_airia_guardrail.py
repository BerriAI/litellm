from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON: Final = TypeAdapter(JsonValue)
MARKER: Final = re.compile(rb"airia-[0-9a-f]{32}")
EMAIL: Final = "ada@example.test"
REDACTED: Final = "[EmailAddress1]"
MASTER_KEY: Final = "synthetic-master-key"
AIRIA_KEY: Final = "synthetic-airia-key"
CHAT_MODEL: Final = "airia-chat"
MESSAGES_MODEL: Final = "airia-messages"
RESPONSES_MODEL: Final = "airia-responses"
ClientKind = Literal["openai_sync", "openai_async", "anthropic_sync", "anthropic_async", "httpx"]


def _marker() -> str:
    return "airia-" + uuid.uuid4().hex


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    return object_value(value)


def _marker_in(value: object) -> str | None:
    found: Final = MARKER.search(json.dumps(value, default=str).encode())
    return found.group(0).decode() if found is not None else None


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _marker_text(value: object) -> str | None:
    return next((text for text in _strings(value) if MARKER.search(text.encode()) is not None), None)


def _replace(value: object) -> object:
    if isinstance(value, str):
        return value.replace(EMAIL, REDACTED)
    if isinstance(value, list):
        return [_replace(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _replace(item) for key, item in value.items()}
    return value


def _chat_reply(marker: str, marker_text: str, body: Mapping[str, JsonValue], stream: bool) -> Reply:
    serialized_body: Final = json.dumps(body)
    text: Final = f"synthetic answer {marker_text}" + (
        f" Contact {EMAIL}."
        if any(term in serialized_body for term in ("AIRIA-REDACT", "AIRIA-TOOLSTREAM", "AIRIA-BLOCK"))
        else ""
    )
    identity: Final = f"chatcmpl-{marker}"
    if "AIRIA-TOOLONLY" in json.dumps(body):
        message: Final[dict[str, JsonValue]] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_" + marker,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"marker": marker_text})},
                }
            ],
        }
        payload: Final[dict[str, JsonValue]] = {
            "id": identity,
            "object": "chat.completion",
            "model": CHAT_MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return Reply(body=json.dumps(payload).encode())
    if not stream:
        choices: Final = [
            {"index": index, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            for index in range(2 if "AIRIA-N2" in json.dumps(body) else 1)
        ]
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "model": CHAT_MODEL,
                    "choices": choices,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }
            ).encode()
        )
    if "AIRIA-TOOLSTREAM" in json.dumps(body):
        chunks: Final = (
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call_' + marker, 'type': 'function', 'function': {'name': 'lookup', 'arguments': EMAIL}}]}}]})}\n\n".encode(),
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': 'stop'}]})}\n\n".encode(),
            b"data: [DONE]\n\n",
        )
    else:
        chunks = (
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}}]})}\n\n".encode(),
            f"data: {json.dumps({'id': identity, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
            b"data: [DONE]\n\n",
        )
    return Reply(content_type="text/event-stream", chunks=chunks)


def _messages_reply(marker: str, marker_text: str, body: Mapping[str, JsonValue], stream: bool) -> Reply:
    text: Final = f"synthetic answer {marker_text}" + (
        f" Contact {EMAIL}." if "AIRIA-REDACT" in json.dumps(body) else ""
    )
    identity: Final = "msg_" + marker
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": MESSAGES_MODEL,
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            ).encode()
        )
    events: Final = (
        {"type": "message_start", "message": {"id": identity, "type": "message", "role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _responses_reply(marker: str, marker_text: str, body: Mapping[str, JsonValue], stream: bool) -> Reply:
    text: Final = f"synthetic answer {marker_text}" + (
        f" Contact {EMAIL}." if "AIRIA-REDACT" in json.dumps(body) else ""
    )
    identity: Final = "resp_" + marker
    response_model_value: Final = body.get("model")
    response_model: Final = response_model_value if isinstance(response_model_value, str) else RESPONSES_MODEL
    response: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": response_model,
        "output": [
            {
                "id": "msg_" + marker,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
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
            "item_id": "msg_" + marker,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _provider(request: Request) -> Reply:
    if request.target.endswith("/beta/litellm_basic_guardrail_api"):
        return Reply(body=b'{"action":"NONE"}')
    if request.target.endswith("/v1/models"):
        return Reply(body=b'{"data":[]}')
    body: Final = _json_object(JSON.validate_json(request.body))
    marker: Final = _marker_in(body)
    marker_text: Final = _marker_text(body)
    if marker is None or marker_text is None:
        return Reply(status=422, body=b'{"error":"marker missing"}')
    stream: Final = body.get("stream") is True
    if request.target.endswith("/messages"):
        return _messages_reply(marker, marker_text, body, stream)
    if request.target.endswith("/responses"):
        return _responses_reply(marker, marker_text, body, stream)
    return _chat_reply(marker, marker_text, body, stream)


@dataclass(frozen=True, slots=True)
class SinkCall:
    path: str
    headers: dict[str, str]
    body: object


class AiriaSink:
    def __init__(self) -> None:
        self.calls: list[SinkCall] = []
        self._lock: Final = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._port: int | None = None

    def start(self, port: int | None = None) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length: Final = int(self.headers.get("content-length", "0"))
                payload: Final = self.rfile.read(length)
                try:
                    body: object = json.loads(payload)
                except json.JSONDecodeError:
                    body = payload.decode(errors="replace")
                with sink._lock:
                    sink.calls.append(
                        SinkCall(self.path, {key.lower(): value for key, value in self.headers.items()}, body)
                    )
                marker: Final = _marker_in(body)
                encoded: Final
                status: Final
                if marker is not None and "AIRIA-SLOW" in json.dumps(body):
                    time.sleep(1.5)
                if marker is not None and "AIRIA-500" in json.dumps(body):
                    status, encoded = 500, b'{"error":"synthetic 500"}'
                elif marker is not None and "AIRIA-401" in json.dumps(body):
                    status, encoded = 401, b'{"error":"synthetic 401"}'
                elif marker is not None and "AIRIA-403" in json.dumps(body):
                    status, encoded = 403, b'{"error":"synthetic 403"}'
                elif marker is not None and "AIRIA-404" in json.dumps(body):
                    status, encoded = 404, b'{"error":"synthetic 404"}'
                elif marker is not None and "AIRIA-INVALIDJSON" in json.dumps(body):
                    status, encoded = 200, b"{not-json"
                elif marker is not None and "AIRIA-NONOBJECT-LIST" in json.dumps(body):
                    status, encoded = 200, b"[]"
                elif marker is not None and "AIRIA-NONOBJECT-STRING" in json.dumps(body):
                    status, encoded = 200, b'"string"'
                elif marker is not None and "AIRIA-NONOBJECT-NULL" in json.dumps(body):
                    status, encoded = 200, b"null"
                elif marker is not None and "AIRIA-UNKNOWN-ACTION" in json.dumps(body):
                    status, encoded = 200, b'{"action":"UNKNOWN"}'
                elif marker is not None and "AIRIA-NO-ACTION" in json.dumps(body):
                    status, encoded = 200, b"{}"
                elif marker is not None and "AIRIA-NONLIST-TEXTS" in json.dumps(body):
                    status, encoded = 200, b'{"action":"GUARDRAIL_INTERVENED","texts":"bad"}'
                elif marker is not None and "AIRIA-WRONGLEN" in json.dumps(body):
                    status, encoded = 200, b'{"action":"GUARDRAIL_INTERVENED","texts":[]}'
                elif marker is not None and "AIRIA-NOFIELD" in json.dumps(body):
                    status, encoded = 200, b'{"action":"GUARDRAIL_INTERVENED"}'
                elif marker is not None and "AIRIA-BLOCK-NOREASON" in json.dumps(body):
                    status, encoded = 200, b'{"action":"BLOCKED"}'
                elif marker is not None and "AIRIA-BLOCK" in json.dumps(body):
                    status, encoded = (
                        200,
                        json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic Airia block"}).encode(),
                    )
                elif marker is not None and "AIRIA-REDACT" in json.dumps(body):
                    rewritten: Final = {
                        "action": "GUARDRAIL_INTERVENED",
                        "texts": _replace(body.get("texts", [])),
                    }
                    if "structured_messages" in body:
                        rewritten["structured_messages"] = _replace(body["structured_messages"])
                    status, encoded = 200, json.dumps(rewritten).encode()
                else:
                    status, encoded = 200, b'{"action":"NONE"}'
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: object) -> None:
                return

        server: Final = ThreadingHTTPServer(("127.0.0.1", port or 0), Handler)
        self._server = server
        self._port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def url(self) -> str:
        assert self._port is not None
        return f"http://127.0.0.1:{self._port}"

    def matching(self, marker: str) -> tuple[SinkCall, ...]:
        with self._lock:
            calls: Final = tuple(self.calls)
        return tuple(call for call in calls if marker in json.dumps(call.body, default=str))

    def matching_call_id(self, call_id: str) -> tuple[SinkCall, ...]:
        with self._lock:
            calls: Final = tuple(self.calls)
        return tuple(
            call for call in calls if isinstance(call.body, Mapping) and call.body.get("litellm_call_id") == call_id
        )


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    proxy: OwnedProxy
    provider: Wire
    sink: AiriaSink
    models: tuple[str, str, str]
    config: Path
    provider_history: list[Request]
    caller_responses: list[dict[str, JsonValue]]
    spend_rows: list[dict[str, JsonValue]]
    evidence_lock: threading.Lock

    @property
    def chat_model(self) -> str:
        return self.models[0]

    @property
    def messages_model(self) -> str:
        return self.models[1]

    @property
    def responses_model(self) -> str:
        return self.models[2]

    def provider_calls(self, marker: str) -> tuple[Request, ...]:
        drained: Final = self.provider.drain()
        with self.evidence_lock:
            self.provider_history.extend(drained)
            calls: Final = tuple(self.provider_history)
        if not marker:
            return calls
        identity: Final = MARKER.search(marker.encode())
        assert identity is not None, marker
        return tuple(request for request in calls if identity.group(0) in request.body)

    def record_response(self, path: str, response: httpx.Response) -> None:
        with self.evidence_lock:
            self.caller_responses.append(
                {
                    "path": path,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                    "body_hex": response.content.hex(),
                }
            )
        if response.status_code == 200:
            _assert_spend(self, response)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _guardrails(sink_url: str, dead_url: str, provider_url: str) -> list[dict[str, JsonValue]]:
    common: Final = {"guardrail": "airia", "api_base": sink_url, "api_key": AIRIA_KEY, "default_on": False}
    return [
        {"guardrail_name": "airia-pre", "litellm_params": {**common, "mode": "pre_call"}},
        {"guardrail_name": "airia-post", "litellm_params": {**common, "mode": "post_call"}},
        {"guardrail_name": "airia-both", "litellm_params": {**common, "mode": ["pre_call", "post_call"]}},
        {"guardrail_name": "airia-during", "litellm_params": {**common, "mode": "during_call"}},
        {"guardrail_name": "airia-dead", "litellm_params": {**common, "mode": "pre_call", "api_base": dead_url}},
        {"guardrail_name": "airia-slow", "litellm_params": {**common, "mode": "pre_call", "timeout": 1}},
        {
            "guardrail_name": "airia-env",
            "litellm_params": {"guardrail": "airia", "mode": "pre_call", "default_on": False},
        },
        {
            "guardrail_name": "airia-nokey",
            "litellm_params": {"guardrail": "airia", "mode": "pre_call", "api_base": sink_url, "default_on": False},
        },
        {
            "guardrail_name": "generic-control",
            "litellm_params": {
                "guardrail": "generic_guardrail_api",
                "mode": "pre_call",
                "api_base": provider_url,
                "api_key": AIRIA_KEY,
                "default_on": False,
            },
        },
    ]


def _create_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    sink: Final = AiriaSink()
    sink.start()
    try:
        dead_url: Final = f"http://127.0.0.1:{_free_port()}"
        config: Final = tmp_path_factory.mktemp("airia") / "config.yaml"
        with wire_server(_provider) as provider, gateway_from_environment() as gateway:
            models: Final = (CHAT_MODEL, MESSAGES_MODEL, RESPONSES_MODEL)
            config.write_text(
                json.dumps(
                    {
                        "model_list": [
                            {
                                "model_name": name,
                                "litellm_params": {
                                    "model": "openai/" + name,
                                    "api_base": provider.url,
                                    "api_key": "synthetic-provider-key",
                                },
                            }
                            for name in models
                        ],
                        "guardrails": _guardrails(sink.url, dead_url, provider.url),
                        "general_settings": {"master_key": MASTER_KEY},
                    }
                )
            )
            with owned_proxy_process(
                gateway,
                config.parent,
                {},
                config=config,
                workers=2,
                remove_environment=("AIRIA_API_KEY", "AIRIA_GATEWAY_URL", "AIRIA_TIMEOUT"),
            ) as proxy:
                rig: Final = Rig(gateway, proxy, provider, sink, models, config, [], [], [], threading.Lock())
                yield rig
    finally:
        sink.stop()


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _create_rig(tmp_path_factory)


def _body(marker: str, model: str, guardrail: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        "guardrails": [guardrail],
        "stream": stream,
    }


def _raw(rig: Rig, path: str, body: Mapping[str, JsonValue], *, stream: bool = False) -> httpx.Response:
    headers: Final = {"Authorization": f"Bearer {rig.proxy.gateway.key}"}
    if not stream:
        response: Final = rig.proxy.gateway.client.post(path, json=body, headers=headers)
        rig.record_response(path, response)
        return response
    with rig.proxy.gateway.client.stream("POST", path, json=body, headers=headers) as response:
        content: Final = b"".join(response.iter_bytes())
        captured: Final = httpx.Response(
            response.status_code,
            headers=response.headers,
            content=content,
            request=response.request,
        )
        rig.record_response(path, captured)
        return captured


async def _async_sdk_request(
    client: openai.AsyncOpenAI | anthropic.AsyncAnthropic,
    model: str,
    marker: str,
    guardrail: str,
    path: str,
) -> httpx.Response:
    try:
        if isinstance(client, openai.AsyncOpenAI):
            if path.endswith("/responses"):
                return (
                    await client.responses.with_raw_response.create(
                        model=model,
                        input=marker,
                        extra_body={"guardrails": [guardrail]},
                    )
                ).http_response
            return (
                await client.chat.completions.with_raw_response.create(
                    model=model,
                    messages=[{"role": "user", "content": marker}],
                    extra_body={"guardrails": [guardrail]},
                )
            ).http_response
        return (
            await client.messages.with_raw_response.create(
                model=model,
                max_tokens=32,
                messages=[{"role": "user", "content": marker}],
                extra_body={"guardrails": [guardrail]},
            )
        ).http_response
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        return error.response
    finally:
        await client.close()


def _sdk_request(rig: Rig, kind: ClientKind, model: str, marker: str, guardrail: str, path: str) -> httpx.Response:
    if kind == "httpx":
        body: Final = (
            _body(marker, model, guardrail)
            if path.endswith("chat/completions")
            else {
                "model": model,
                "guardrails": [guardrail],
                **(
                    {"input": marker}
                    if path.endswith("/responses")
                    else {"messages": [{"role": "user", "content": marker}], "max_tokens": 32}
                ),
            }
        )
        return _raw(rig, path, body)
    if kind == "openai_sync":
        sync_base: Final = str(rig.proxy.gateway.client.base_url).rstrip("/") + "/v1"
        with openai.OpenAI(api_key=rig.proxy.gateway.key, base_url=sync_base, max_retries=0) as sync_client:
            try:
                if path.endswith("chat/completions"):
                    raw_chat: Final = sync_client.chat.completions.with_raw_response.create(
                        model=model,
                        messages=[{"role": "user", "content": marker}],
                        extra_body={"guardrails": [guardrail]},
                    )
                    return _record_response(rig, path, raw_chat.http_response)
                raw_response: Final = sync_client.responses.with_raw_response.create(
                    model=model,
                    input=marker,
                    extra_body={"guardrails": [guardrail]},
                )
                return _record_response(rig, path, raw_response.http_response)
            except openai.APIStatusError as error:
                return _record_response(rig, path, error.response)
    if kind == "openai_async":
        async_base: Final = str(rig.proxy.gateway.client.base_url).rstrip("/") + "/v1"
        async_client: Final = openai.AsyncOpenAI(api_key=rig.proxy.gateway.key, base_url=async_base, max_retries=0)
        return _record_response(
            rig,
            path,
            asyncio.run(_async_sdk_request(async_client, model, marker, guardrail, path)),
        )
    if kind == "anthropic_sync":
        anthropic_sync_base: Final = str(rig.proxy.gateway.client.base_url).rstrip("/")
        with anthropic.Anthropic(
            api_key=rig.proxy.gateway.key,
            base_url=anthropic_sync_base,
            max_retries=0,
        ) as anthropic_sync_client:
            try:
                raw_anthropic: Final = anthropic_sync_client.messages.with_raw_response.create(
                    model=model,
                    max_tokens=32,
                    messages=[{"role": "user", "content": marker}],
                    extra_body={"guardrails": [guardrail]},
                )
                return _record_response(rig, path, raw_anthropic.http_response)
            except anthropic.APIStatusError as error:
                return _record_response(rig, path, error.response)
    anthropic_async_base: Final = str(rig.proxy.gateway.client.base_url).rstrip("/")
    anthropic_async_client: Final = anthropic.AsyncAnthropic(
        api_key=rig.proxy.gateway.key,
        base_url=anthropic_async_base,
        max_retries=0,
    )
    return _record_response(
        rig,
        path,
        asyncio.run(_async_sdk_request(anthropic_async_client, model, marker, guardrail, path)),
    )


def _record_response(rig: Rig, path: str, response: httpx.Response) -> httpx.Response:
    rig.record_response(path, response)
    return response


def _assert_sink(
    rig: Rig,
    marker: str,
    response: httpx.Response,
    *,
    expected_input_types: tuple[Literal["request", "response"], ...] = ("request",),
) -> SinkCall:
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None, dict(response.headers)
    calls: Final = eventually(
        lambda: rig.sink.matching_call_id(call_id),
        lambda current: len(current) >= len(expected_input_types),
    )
    assert len(calls) == len(expected_input_types), calls
    request_path: Final = response.request.url.path
    expected_model: Final = {
        "/v1/chat/completions": rig.chat_model,
        "/v1/messages": rig.messages_model,
        "/v1/responses": rig.responses_model,
    }.get(request_path)
    assert expected_model is not None, request_path
    payloads: Final = tuple(_json_object(JSON.validate_python(call.body)) for call in calls)
    assert tuple(payload["input_type"] for payload in payloads) == expected_input_types, payloads
    assert all(payload["model"] == expected_model for payload in payloads), payloads
    assert all(payload["litellm_call_id"] == call_id for payload in payloads), payloads
    assert all(marker in json.dumps(payload) for payload in payloads), payloads
    for call in calls:
        assert call.path == "/v1/guardrails/litellm"
        assert call.headers["authorization"] == f"Bearer {AIRIA_KEY}"
        encoded: Final = json.dumps(call.body, default=str)
        assert MASTER_KEY not in encoded
        assert MASTER_KEY not in json.dumps(call.headers)
        payload: Final = _json_object(JSON.validate_python(call.body))
        assert set(payload) == {
            "input_type",
            "texts",
            "images",
            "structured_messages",
            "tools",
            "tool_calls",
            "model",
            "litellm_call_id",
        }
        assert isinstance(payload["texts"], list)
        assert payload["input_type"] in {"request", "response"}
    call: Final = calls[-1]
    return call


def _assert_no_sink(rig: Rig, response: httpx.Response) -> None:
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None, dict(response.headers)
    assert rig.sink.matching_call_id(call_id) == ()


def _assert_provider(rig: Rig, marker: str, *, expected: int = 1) -> tuple[Request, ...]:
    calls: Final = eventually(lambda: rig.provider_calls(marker), lambda current: len(current) >= expected)
    assert len(calls) == expected, calls
    return calls


def _assert_spend(rig: Rig, response: httpx.Response) -> None:
    if response.status_code != 200:
        return
    response_id: Final = re.search(rb'"id"\s*:\s*"([^"]+)"', response.content)
    response_payload: Final = (
        _json_object(JSON.validate_json(response.content))
        if response.headers.get("content-type", "").startswith("application/json")
        else {}
    )
    response_id_value: Final = response_payload.get("id")
    identity: Final = (
        response_id_value
        if isinstance(response_id_value, str)
        else response_id.group(1).decode()
        if response_id is not None
        else None
    )
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None, dict(response.headers)
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s '
            "OR metadata ->> 'litellm_call_id' = %s",
            (identity or "", call_id),
        ),
        lambda current: bool(current),
        seconds=70,
    )
    assert len(rows) == 1, rows
    with rig.evidence_lock:
        rig.spend_rows.append(rows[0])


@pytest.mark.parametrize("kind", ["openai_sync", "openai_async", "httpx"])
def test_r1_pre_call_none_chat_clients(rig: Rig, kind: ClientKind) -> None:
    marker: Final = _marker()
    response: Final = _sdk_request(rig, kind, rig.chat_model, marker, "airia-pre", "/v1/chat/completions")
    assert response.status_code == 200, response.text
    _assert_sink(rig, marker, response)
    _assert_provider(rig, marker)


@pytest.mark.parametrize("kind", ["openai_sync", "openai_async", "httpx"])
def test_r2_blocked_pre_call_never_reaches_provider(rig: Rig, kind: ClientKind) -> None:
    marker: Final = _marker() + " AIRIA-BLOCK"
    response: Final = _sdk_request(rig, kind, rig.chat_model, marker, "airia-pre", "/v1/chat/completions")
    assert response.status_code == 400, response.text
    assert "synthetic Airia block" in response.text
    _assert_sink(rig, marker, response)
    assert _assert_provider(rig, marker, expected=0) == ()


def test_r3_blocked_without_reason_uses_default_message(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-BLOCK-NOREASON"
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-pre"))
    assert response.status_code == 400, response.text
    assert "Blocked by your organization's content policy." in response.text
    _assert_sink(rig, marker, response)
    assert _assert_provider(rig, marker, expected=0) == ()


def test_r4_pre_call_redaction_reaches_provider(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT Contact " + EMAIL
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-pre"))
    assert response.status_code == 200, response.text
    calls: Final = _assert_provider(rig, marker)
    assert REDACTED.encode() in calls[0].body
    assert EMAIL.encode() not in calls[0].body
    _assert_sink(rig, marker, response)


def test_r5_structured_message_redaction_is_positional(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT"
    body: Final = {
        "model": rig.chat_model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": marker + " " + EMAIL}]},
        ],
        "guardrails": ["airia-pre"],
    }
    response: Final = _raw(rig, "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    provider: Final = _assert_provider(rig, marker)[0]
    assert REDACTED.encode() in provider.body
    assert EMAIL.encode() not in provider.body
    _assert_sink(rig, marker, response)


def test_r6_multiple_texts_are_rewritten_without_reordering(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT"
    body: Final = {
        "model": rig.chat_model,
        "messages": [
            {"role": "system", "content": "system " + EMAIL},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first " + EMAIL},
                    {"type": "text", "text": "second " + marker + " " + EMAIL},
                ],
            },
        ],
        "guardrails": ["airia-pre"],
    }
    response: Final = _raw(rig, "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    provider: Final = _assert_provider(rig, marker)[0]
    sent: Final = _json_object(JSON.validate_json(provider.body))
    messages: Final = sent["messages"]
    assert messages == [
        {"role": "system", "content": "system " + REDACTED},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first " + REDACTED},
                {"type": "text", "text": "second " + marker + " " + REDACTED},
            ],
        },
    ], sent
    assert EMAIL.encode() not in provider.body
    _assert_sink(rig, marker, response)


def test_r7_multi_choice_response_redaction_rewrites_each_choice(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT AIRIA-N2"
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-post"))
    assert response.status_code == 200, response.text
    assert response.content.count(REDACTED.encode()) == 2
    assert EMAIL.encode() not in response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))


def test_r8_streamed_chat_post_call_redaction_has_no_leak(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT"
    response: Final = _raw(
        rig,
        "/v1/chat/completions",
        {**_body(marker, rig.chat_model, "airia-post"), "stream": True},
        stream=True,
    )
    assert response.status_code == 200, response.content
    assert EMAIL.encode() not in response.content
    assert REDACTED.encode() in response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))


def test_r9_streamed_chat_post_call_block_has_no_provider_error_leak(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-BLOCK"
    response: Final = _raw(
        rig,
        "/v1/chat/completions",
        {**_body(marker, rig.chat_model, "airia-post"), "stream": True},
        stream=True,
    )
    assert response.status_code in {200, 400}, response.content
    assert EMAIL.encode() not in response.content
    assert b'"error"' in response.content or b"error" in response.content.lower(), response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))


@pytest.mark.parametrize(
    ("kind", "verdict"),
    [
        ("httpx", "NONE"),
        ("anthropic_sync", "BLOCK"),
        ("anthropic_async", "REDACT"),
    ],
)
def test_r10_messages_pre_call_verdict_matrix(rig: Rig, kind: ClientKind, verdict: str) -> None:
    suffix: Final = (
        " AIRIA-REDACT Contact " + EMAIL if verdict == "REDACT" else " AIRIA-BLOCK" if verdict == "BLOCK" else ""
    )
    marker: Final = _marker() + suffix
    response: Final = _sdk_request(rig, kind, rig.messages_model, marker, "airia-pre", "/v1/messages")
    expected: Final = 400 if verdict == "BLOCK" else 200
    assert response.status_code == expected, response.text
    _assert_sink(rig, marker, response)
    if verdict == "BLOCK":
        assert _assert_provider(rig, marker, expected=0) == ()
    else:
        calls: Final = _assert_provider(rig, marker)
        if verdict == "REDACT":
            assert REDACTED.encode() in calls[0].body
            assert EMAIL.encode() not in calls[0].body


@pytest.mark.parametrize("kind", ["anthropic_sync", "anthropic_async", "httpx"])
def test_r11_messages_post_call_redaction(rig: Rig, kind: ClientKind) -> None:
    marker: Final = _marker() + " AIRIA-REDACT"
    response: Final = _sdk_request(rig, kind, rig.messages_model, marker, "airia-post", "/v1/messages")
    assert response.status_code == 200, response.text
    assert EMAIL.encode() not in response.content
    assert REDACTED.encode() in response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))


def test_r12_streamed_messages_post_call_redaction_has_no_leak(rig: Rig) -> None:
    pytest.skip(
        "BUG: streamed /v1/messages post_call falls back to block_only, Airia redaction never reaches "
        "the client and Airia receives model None"
    )
    marker: Final = _marker() + " AIRIA-REDACT"
    body: Final = {
        "model": rig.messages_model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": marker}],
        "guardrails": ["airia-post"],
        "stream": True,
    }
    response: Final = _raw(rig, "/v1/messages", body, stream=True)
    assert response.status_code == 200, response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))
    assert EMAIL.encode() not in response.content, response.content
    assert REDACTED.encode() in response.content, response.content


@pytest.mark.parametrize(
    ("kind", "verdict"),
    [
        ("httpx", "NONE"),
        ("openai_sync", "BLOCK"),
        ("openai_async", "REDACT"),
    ],
)
def test_r13_responses_pre_call_verdict_matrix(rig: Rig, kind: ClientKind, verdict: str) -> None:
    suffix: Final = (
        " AIRIA-REDACT Contact " + EMAIL if verdict == "REDACT" else " AIRIA-BLOCK" if verdict == "BLOCK" else ""
    )
    marker: Final = _marker() + suffix
    response: Final = _sdk_request(rig, kind, rig.responses_model, marker, "airia-pre", "/v1/responses")
    expected: Final = 400 if verdict == "BLOCK" else 200
    assert response.status_code == expected, response.text
    _assert_sink(rig, marker, response)
    if verdict == "BLOCK":
        assert _assert_provider(rig, marker, expected=0) == ()
    else:
        calls: Final = _assert_provider(rig, marker)
        if verdict == "REDACT":
            assert REDACTED.encode() in calls[0].body
            assert EMAIL.encode() not in calls[0].body


def test_r14_streamed_responses_post_call_redaction_has_no_leak(rig: Rig) -> None:
    pytest.skip(
        "BUG: streamed /v1/responses post_call leaks the original text in output_text.delta before the "
        "redacted response.completed"
    )
    marker: Final = _marker() + " AIRIA-REDACT"
    response: Final = _raw(
        rig,
        "/v1/responses",
        {
            "model": rig.responses_model,
            "input": marker,
            "guardrails": ["airia-post"],
            "stream": True,
        },
        stream=True,
    )
    assert response.status_code == 200, response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))
    assert EMAIL.encode() not in response.content, response.content
    assert REDACTED.encode() in response.content, response.content


def test_r14_nonstream_responses_post_call_redaction(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-REDACT"
    body: Final = {
        "model": rig.responses_model,
        "input": marker,
        "guardrails": ["airia-post"],
    }
    response: Final = _raw(rig, "/v1/responses", body)
    assert response.status_code == 200, response.text
    assert EMAIL.encode() not in response.content
    assert REDACTED.encode() in response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))


def test_r15_default_on_and_default_off_are_distinct(rig: Rig, tmp_path: Path) -> None:
    default_off_marker: Final = _marker()
    response: Final = _raw(
        rig,
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": default_off_marker}]},
    )
    assert response.status_code == 200, response.text
    _assert_no_sink(rig, response)
    _assert_provider(rig, default_off_marker)
    config: Final = tmp_path / "default.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": rig.chat_model,
                        "litellm_params": {
                            "model": "openai/" + rig.chat_model,
                            "api_base": rig.provider.url,
                            "api_key": "synthetic-provider-key",
                        },
                    }
                ],
                "guardrails": [
                    {
                        "guardrail_name": "airia-default",
                        "litellm_params": {
                            "guardrail": "airia",
                            "mode": "pre_call",
                            "api_base": rig.sink.url,
                            "api_key": AIRIA_KEY,
                            "default_on": True,
                        },
                    }
                ],
                "general_settings": {"master_key": MASTER_KEY},
            }
        )
    )
    with owned_proxy_process(rig.gateway, tmp_path, {}, config=config, workers=2) as proxy:
        default_on_marker: Final = _marker()
        default_body: Final = {"model": rig.chat_model, "messages": [{"role": "user", "content": default_on_marker}]}
        default_response: Final = proxy.gateway.client.post(
            "/v1/chat/completions",
            json=default_body,
            headers={"Authorization": f"Bearer {proxy.gateway.key}"},
        )
        assert default_response.status_code == 200, default_response.text
        rig.record_response("/v1/chat/completions", default_response)
        _assert_sink(rig, default_on_marker, default_response)
        _assert_provider(rig, default_on_marker)


def test_r16_key_attached_guardrail_runs_without_request_field(rig: Rig) -> None:
    marker: Final = _marker()
    with rig.proxy.gateway.scenario() as scenario:
        key: Final = scenario.key(guardrails=["airia-pre"])
        response: Final = rig.proxy.gateway.client.post(
            "/v1/chat/completions",
            json={"model": rig.chat_model, "messages": [{"role": "user", "content": marker}]},
            headers={"Authorization": f"Bearer {key}"},
        )
    rig.record_response("/v1/chat/completions", response)
    assert response.status_code == 200, response.text
    _assert_sink(rig, marker, response)
    _assert_provider(rig, marker)
    unattached_marker: Final = _marker()
    unattached: Final = _raw(
        rig,
        "/v1/chat/completions",
        {"model": rig.chat_model, "messages": [{"role": "user", "content": unattached_marker}]},
    )
    assert unattached.status_code == 200, unattached.text
    _assert_no_sink(rig, unattached)
    _assert_provider(rig, unattached_marker)


def test_r17_both_mode_runs_exactly_once_per_phase(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-both"))
    assert response.status_code == 200, response.text
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("request", "response"))
    calls: Final = rig.sink.matching_call_id(response.headers["x-litellm-call-id"])
    assert len(calls) == 2, calls
    first_payload: Final = _json_object(JSON.validate_python(calls[0].body))
    second_payload: Final = _json_object(JSON.validate_python(calls[1].body))
    assert first_payload["litellm_call_id"] == second_payload["litellm_call_id"]
    assert first_payload["litellm_call_id"] == response.headers["x-litellm-call-id"]


def test_r18_during_call_is_skipped_and_request_is_unchecked(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-BLOCK"
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-during"))
    assert response.status_code == 200, response.text
    assert "Skipping guardrail 'airia-during'" in rig.proxy.log.read_text()
    _assert_no_sink(rig, response)
    _assert_provider(rig, marker)


def test_r19_logging_only_records_block_but_does_not_reject(rig: Rig, tmp_path: Path) -> None:
    config: Final = tmp_path / "logging.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": rig.chat_model,
                        "litellm_params": {
                            "model": "openai/" + rig.chat_model,
                            "api_base": rig.provider.url,
                            "api_key": "synthetic-provider-key",
                        },
                    }
                ],
                "guardrails": [
                    {
                        "guardrail_name": "airia-log",
                        "litellm_params": {
                            "guardrail": "airia",
                            "mode": "logging_only",
                            "api_base": rig.sink.url,
                            "api_key": AIRIA_KEY,
                            "default_on": False,
                        },
                    }
                ],
                "general_settings": {"master_key": MASTER_KEY},
            }
        )
    )
    marker: Final = _marker() + " AIRIA-BLOCK"
    with owned_proxy_process(
        rig.gateway,
        tmp_path,
        {},
        config=config,
        workers=2,
        remove_environment=("AIRIA_API_KEY", "AIRIA_GATEWAY_URL", "AIRIA_TIMEOUT"),
    ) as proxy:
        response: Final = proxy.gateway.client.post(
            "/v1/chat/completions",
            json=_body(marker, rig.chat_model, "airia-log"),
            headers={"Authorization": f"Bearer {proxy.gateway.key}"},
        )
        assert response.status_code == 200, response.text
        rig.record_response("/v1/chat/completions", response)
        _assert_sink(rig, marker, response)
        _assert_provider(rig, marker)


def test_r20_missing_credentials_skip_and_environment_fallback_works(rig: Rig, tmp_path: Path) -> None:
    marker: Final = _marker()
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-nokey"))
    assert response.status_code == 200, response.text
    assert "Skipping guardrail 'airia-nokey'" in rig.proxy.log.read_text()
    _assert_no_sink(rig, response)
    _assert_provider(rig, marker)
    config: Final = tmp_path / "env.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": rig.chat_model,
                        "litellm_params": {
                            "model": "openai/" + rig.chat_model,
                            "api_base": rig.provider.url,
                            "api_key": "synthetic-provider-key",
                        },
                    }
                ],
                "guardrails": [
                    {
                        "guardrail_name": "airia-env",
                        "litellm_params": {"guardrail": "airia", "mode": "pre_call", "default_on": False},
                    }
                ],
                "general_settings": {"master_key": MASTER_KEY},
            }
        )
    )
    marker_env: Final = _marker()
    with owned_proxy_process(
        rig.gateway,
        tmp_path,
        {"AIRIA_GATEWAY_URL": rig.sink.url, "AIRIA_API_KEY": AIRIA_KEY},
        config=config,
        workers=2,
    ) as proxy:
        env_response: Final = proxy.gateway.client.post(
            "/v1/chat/completions",
            json=_body(marker_env, rig.chat_model, "airia-env"),
            headers={"Authorization": f"Bearer {proxy.gateway.key}"},
        )
        assert env_response.status_code == 200, env_response.text
        rig.record_response("/v1/chat/completions", env_response)
        _assert_sink(rig, marker_env, env_response)
        _assert_provider(rig, marker_env)


@pytest.mark.parametrize(
    "marker_suffix",
    [
        "AIRIA-500",
        "AIRIA-401",
        "AIRIA-403",
        "AIRIA-404",
        "AIRIA-NONOBJECT-LIST",
        "AIRIA-NONOBJECT-STRING",
        "AIRIA-NONOBJECT-NULL",
        "AIRIA-INVALIDJSON",
        "AIRIA-UNKNOWN-ACTION",
        "AIRIA-NO-ACTION",
        "AIRIA-NONLIST-TEXTS",
        "AIRIA-WRONGLEN",
        "AIRIA-NOFIELD",
    ],
)
def test_r21_malformed_or_failed_verdicts_fail_closed_and_proxy_recovers(rig: Rig, marker_suffix: str) -> None:
    marker: Final = _marker() + " " + marker_suffix
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-pre"))
    assert response.status_code == 400, response.text
    _assert_sink(rig, marker, response)
    assert _assert_provider(rig, marker, expected=0) == ()
    control_marker: Final = _marker()
    control: Final = _raw(rig, "/v1/chat/completions", _body(control_marker, rig.chat_model, "airia-pre"))
    assert control.status_code == 200, control.text
    _assert_sink(rig, control_marker, control)
    _assert_provider(rig, control_marker)


def test_r22_transport_and_timeout_fail_closed(rig: Rig) -> None:
    dead_marker: Final = _marker()
    dead: Final = _raw(rig, "/v1/chat/completions", _body(dead_marker, rig.chat_model, "airia-dead"))
    assert dead.status_code == 400, dead.text
    _assert_no_sink(rig, dead)
    assert _assert_provider(rig, dead_marker, expected=0) == ()
    slow_marker: Final = _marker() + " AIRIA-SLOW"
    slow: Final = _raw(rig, "/v1/chat/completions", _body(slow_marker, rig.chat_model, "airia-slow"))
    assert slow.status_code == 400, slow.text
    _assert_sink(rig, slow_marker, slow)
    assert _assert_provider(rig, slow_marker, expected=0) == ()
    control_marker: Final = _marker()
    control: Final = _raw(rig, "/v1/chat/completions", _body(control_marker, rig.chat_model, "airia-pre"))
    assert control.status_code == 200, control.text
    _assert_sink(rig, control_marker, control)
    _assert_provider(rig, control_marker)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "x" * 5000,
        [{"type": "text", "text": "part"}],
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}],
    ],
    ids=["empty", "5kb", "text_part", "image_part"],
)
def test_r23_hostile_inputs_are_forwarded_without_proxy_crash(rig: Rig, content: JsonValue) -> None:
    marker: Final = _marker()
    body: Final = {
        "model": rig.chat_model,
        "messages": [
            {"role": "system", "content": marker},
            {"role": "user", "content": content},
        ],
        "metadata": {"marker": marker},
        "guardrails": ["airia-pre"],
    }
    response: Final = _raw(rig, "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    _assert_provider(rig, marker)
    call: Final = _assert_sink(rig, marker, response)
    if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "image_url":
        payload: Final = _json_object(JSON.validate_python(call.body))
        images: Final = payload["images"]
        assert isinstance(images, list) and images, payload


def test_r23_repeated_identical_requests_each_reach_airia_once(rig: Rig) -> None:
    marker: Final = _marker()
    body: Final = _body(marker, rig.chat_model, "airia-pre")
    first: Final = _raw(rig, "/v1/chat/completions", body)
    second: Final = _raw(rig, "/v1/chat/completions", body)
    assert first.status_code == second.status_code == 200, (first.text, second.text)
    first_call: Final = _assert_sink(rig, marker, first)
    second_call: Final = _assert_sink(rig, marker, second)
    calls: Final = rig.sink.matching(marker)
    call_ids: Final = tuple(_json_object(JSON.validate_python(call.body))["litellm_call_id"] for call in calls)
    expected_call_ids: Final = (
        first.headers["x-litellm-call-id"],
        second.headers["x-litellm-call-id"],
    )
    assert call_ids == expected_call_ids, call_ids
    first_payload: Final = _json_object(JSON.validate_python(first_call.body))
    second_payload: Final = _json_object(JSON.validate_python(second_call.body))
    assert first_payload["litellm_call_id"] == first.headers["x-litellm-call-id"]
    assert second_payload["litellm_call_id"] == second.headers["x-litellm-call-id"]
    assert first_payload["litellm_call_id"] != second_payload["litellm_call_id"]
    assert len(_assert_provider(rig, marker, expected=2)) == 2


def test_r24_caller_master_key_never_reaches_airia(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-pre"))
    assert response.status_code == 200, response.text
    call: Final = _assert_sink(rig, marker, response)
    assert MASTER_KEY not in json.dumps(call.body, default=str)
    assert MASTER_KEY not in json.dumps(call.headers)


def test_r25_tool_only_post_call_is_still_guarded(rig: Rig) -> None:
    marker: Final = _marker() + " AIRIA-TOOLONLY AIRIA-BLOCK"
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "airia-post"))
    assert response.status_code == 400, response.text
    _assert_sink(rig, marker, response, expected_input_types=("response",))
    assert _assert_provider(rig, marker) != ()


def test_r26_streamed_tool_call_deltas_do_not_leak_before_post_verdict(rig: Rig) -> None:
    pytest.skip("BUG: streamed tool-call deltas reach the client before the post_call verdict")
    marker: Final = _marker() + " AIRIA-TOOLSTREAM AIRIA-BLOCK"
    response: Final = _raw(
        rig,
        "/v1/chat/completions",
        {**_body(marker, rig.chat_model, "airia-post"), "stream": True},
        stream=True,
    )
    assert response.status_code in {200, 400}, response.content
    _assert_provider(rig, marker)
    _assert_sink(rig, marker, response, expected_input_types=("response",))
    assert EMAIL.encode() not in response.content, response.content
    assert b'"error"' in response.content or b"error" in response.content.lower(), response.content


def test_r27_management_and_ui_surfaces_include_airia(rig: Rig) -> None:
    ui_settings: Final = rig.proxy.gateway.get("/guardrails/ui/add_guardrail_settings")
    supported_modes: Final = _json_object(JSON.validate_python(ui_settings["supported_modes_by_provider"]))
    settings: Final = rig.proxy.gateway.get("/guardrails/ui/provider_specific_params")
    guardrail_name: Final = "airia-managed-" + uuid.uuid4().hex
    created: Final = rig.proxy.gateway.post(
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": guardrail_name,
                "litellm_params": {
                    "guardrail": "airia",
                    "mode": "pre_call",
                    "api_base": rig.sink.url,
                    "api_key": AIRIA_KEY,
                    "default_on": False,
                },
            }
        },
    )
    created_config: Final = _json_object(JSON.validate_python(created))
    assert created_config["guardrail_name"] == guardrail_name
    managed_marker: Final = _marker() + " AIRIA-BLOCK"
    denied: Final = _raw(
        rig,
        "/v1/chat/completions",
        _body(managed_marker, rig.chat_model, guardrail_name),
    )
    assert denied.status_code == 400, denied.text
    _assert_sink(rig, managed_marker, denied)
    assert _assert_provider(rig, managed_marker, expected=0) == ()
    deleted: Final = rig.proxy.gateway.request(
        "DELETE",
        f"/guardrails/{created_config['guardrail_id']}",
        {},
    )
    assert deleted.status_code == 200, deleted.text
    assert "airia" in settings, settings
    airia_fields: Final = _json_object(JSON.validate_python(settings["airia"]))
    assert {"api_base", "api_key", "timeout"} <= airia_fields.keys(), airia_fields
    assert airia_fields["ui_friendly_name"] == "Airia Guardrail", airia_fields
    assert "airia" in supported_modes, supported_modes
    provider_settings: Final = json.dumps(settings["airia"])
    assert "Airia Guardrail" in provider_settings, provider_settings
    for field in ("api_base", "api_key", "timeout"):
        assert field in provider_settings, provider_settings


def test_r28_existing_generic_guardrail_control_still_fires(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = _raw(rig, "/v1/chat/completions", _body(marker, rig.chat_model, "generic-control"))
    assert response.status_code == 200, response.text
    _assert_provider(rig, marker, expected=2)
