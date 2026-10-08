"""Akto guardrail in `mode: logging_only`, driven through a real proxy.

The Akto service is the only guardrail double: an owned HTTP sink that answers the `/api/http-proxy`
verdict protocol and records every call. The provider is a second owned sink. The proxy, its
guardrail registry, Postgres and Redis run for real with two workers.
"""

from __future__ import annotations

import json
import socket
import threading
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import anthropic
import openai
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

LOG_KEY: Final = "synthetic-akto-log-key"
INPUT_KEY: Final = "synthetic-akto-input-key"
OUTPUT_KEY: Final = "synthetic-akto-output-key"
DOWN_KEY: Final = "synthetic-akto-down-key"
BLOCK_MARK: Final = "SYNTHETIC-AKTO-BLOCK"
BLOCK_REASON: Final = "Synthetic Akto policy block"
REQUEST_CHECK: Final = {"akto_connector": "litellm", "guardrails": "true", "ingest_data": "true"}
RESPONSE_CHECK: Final = {"akto_connector": "litellm", "response_guardrails": "true", "ingest_data": "true"}
FILE_CHECK: Final = {"akto_connector": "litellm", "file_guardrails": "true"}


@dataclass(frozen=True, slots=True)
class AktoCall:
    path: str
    flags: dict[str, str]
    authorization: str
    payload: dict[str, object]

    def request_text(self) -> str:
        return str(self.payload.get("requestPayload", ""))

    def response_text(self) -> str:
        return str(self.payload.get("responsePayload", ""))


@dataclass(slots=True)
class AktoSink:
    """Owned Akto double that records every verdict request and blocks any payload carrying BLOCK_MARK."""

    calls: list[AktoCall] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> None:
        sink: Final = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                raw: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
                target: Final = urlsplit(self.path)
                call: Final = AktoCall(
                    path=target.path,
                    flags={name: values[0] for name, values in parse_qs(target.query).items()},
                    authorization=self.headers.get("authorization", ""),
                    payload=json.loads(raw),
                )
                with sink.lock:
                    sink.calls.append(call)
                verdict: Final = (
                    {"Allowed": False, "Behaviour": "block", "Reason": BLOCK_REASON}
                    if BLOCK_MARK in raw.decode()
                    else {"Allowed": True}
                )
                body: Final = json.dumps({"data": {"guardrailsResult": verdict}}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()

    def for_key(self, key: str, marker: str) -> tuple[AktoCall, ...]:
        with self.lock:
            return tuple(
                call for call in self.calls if call.authorization == key and marker in json.dumps(call.payload)
            )


def _answer(marker: str) -> str:
    return "synthetic answer " + marker


def _marker_in(body: bytes) -> str:
    text: Final = body.decode()
    start: Final = text.find("mark-")
    assert start >= 0, text
    return text[start : start + 37]


def _sse(events: tuple[dict[str, object], ...], *, named: bool) -> tuple[bytes, ...]:
    return tuple(
        ((f"event: {event['type']}\n" if named else "") + "data: " + json.dumps(event) + "\n\n").encode()
        for event in events
    )


def _chat_reply(marker: str, streaming: bool) -> Reply:
    answer: Final = _answer(marker)
    if not streaming:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + marker,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            ).encode()
        )

    def chunk(delta: dict[str, object], finish: str | None) -> dict[str, object]:
        return {
            "id": "chatcmpl-" + marker,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    events: Final = (chunk({"role": "assistant", "content": answer[:9]}, None), chunk({"content": answer[9:]}, "stop"))
    return Reply(chunks=(*_sse(events, named=False), b"data: [DONE]\n\n"), content_type="text/event-stream")


def _messages_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "msg_" + marker,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5-20250929",
                "content": [{"type": "text", "text": _answer(marker)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode()
    )


def _responses_reply(marker: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "resp_" + marker,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4.1-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msgo_" + marker,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    marker: Final = _marker_in(request.body)
    if request.target.endswith("/v1/messages"):
        return _messages_reply(marker)
    if request.target.endswith("/v1/responses"):
        return _responses_reply(marker)
    assert request.target.endswith("/v1/chat/completions"), request.target
    return _chat_reply(marker, bool(json.loads(request.body).get("stream")))


def _guardrail(name: str, key: str, url: str, *, default_on: bool, **params: object) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "akto",
            "mode": "logging_only",
            "default_on": default_on,
            "akto_base_url": url,
            "akto_api_key": key,
            **params,
        },
    }


def _closed_url() -> str:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{reserve.getsockname()[1]}"


def _rig_config(sink_url: str, root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        _guardrail("akto-log", LOG_KEY, sink_url, default_on=True),
        _guardrail("akto-log-input", INPUT_KEY, sink_url, default_on=False, logging_only_scope="input"),
        _guardrail("akto-log-output", OUTPUT_KEY, sink_url, default_on=False, logging_only_scope="output"),
        _guardrail("akto-log-down", DOWN_KEY, _closed_url(), default_on=False),
    ]
    path: Final = root / "akto-logging-only.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    sink: AktoSink
    provider: Wire
    chat_model: str
    claude_model: str
    responses_model: str

    def base(self) -> str:
        return str(self.proxy.client.base_url).rstrip("/")

    def provider_calls(self, marker: str) -> tuple[Request, ...]:
        return tuple(request for request in self.provider.drain() if marker.encode() in request.body)

    def akto_calls(self, key: str, marker: str, count: int) -> tuple[AktoCall, ...]:
        return eventually(lambda: self.sink.for_key(key, marker), lambda calls: len(calls) >= count, seconds=30)

    def spend_row(self, request_id: str) -> dict[str, object]:
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        return rows[0]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("akto-logging-only")
    sink: Final = AktoSink()
    sink.start()
    try:
        with gateway_from_environment() as gateway, wire_server(_provider) as provider:
            with (
                owned_proxy_process(gateway, root, {}, config=_rig_config(sink.url, root), workers=2) as owned,
                owned.gateway.scenario() as scenario,
            ):
                chat: Final = scenario.model(
                    model="openai/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
                )
                claude: Final = scenario.model(
                    model="anthropic/claude-sonnet-4-5-20250929",
                    api_base=provider.url,
                    api_key="synthetic-anthropic-key",
                )
                responses: Final = scenario.model(
                    model="openai/gpt-4.1-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
                )
                yield Rig(owned.gateway, sink, provider, chat, claude, responses)
    finally:
        sink.stop()


def _marker() -> str:
    return "mark-" + uuid.uuid4().hex


def _assert_request_check(call: AktoCall, marker: str) -> None:
    assert call.path == "/api/http-proxy", call.path
    assert call.flags == REQUEST_CHECK, call.flags
    assert marker in call.request_text(), call.payload


def _assert_response_check(call: AktoCall, marker: str) -> None:
    assert call.path == "/api/http-proxy", call.path
    assert call.flags == RESPONSE_CHECK, call.flags
    assert _answer(marker) in call.response_text(), call.payload


def _split(calls: tuple[AktoCall, ...]) -> tuple[tuple[AktoCall, ...], tuple[AktoCall, ...]]:
    return (
        tuple(call for call in calls if call.flags == REQUEST_CHECK),
        tuple(call for call in calls if call.flags == RESPONSE_CHECK),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("client_kind", ["openai_sync", "openai_async"])
async def test_logging_only_sends_request_and_response_checks_to_akto(rig: Rig, client_kind: str) -> None:
    marker: Final = _marker()
    messages: Final = [{"role": "user", "content": "hello " + marker}]
    if client_kind == "openai_sync":
        client = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
        completion = client.chat.completions.create(model=rig.chat_model, messages=messages)  # pyright: ignore[reportArgumentType, reportCallIssue]
    else:
        async_client = openai.AsyncOpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
        completion = await async_client.chat.completions.create(model=rig.chat_model, messages=messages)  # pyright: ignore[reportArgumentType, reportCallIssue]
    assert completion.choices[0].message.content == _answer(marker)
    assert len(rig.provider_calls(marker)) == 1

    requests, responses = _split(rig.akto_calls(LOG_KEY, marker, 2))
    assert len(requests) == 1 and len(responses) == 1, rig.sink.for_key(LOG_KEY, marker)
    _assert_request_check(requests[0], marker)
    _assert_response_check(responses[0], marker)


def test_logging_only_block_verdict_never_blocks_the_caller(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": [{"role": "user", "content": f"{BLOCK_MARK} {marker}"}]},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _answer(marker)
    assert len(rig.provider_calls(marker)) == 1

    requests, _ = _split(rig.akto_calls(LOG_KEY, marker, 1))
    assert len(requests) == 1, rig.sink.for_key(LOG_KEY, marker)
    _assert_request_check(requests[0], marker)

    entries: Final = object_value(rig.spend_row(response.json()["id"])["metadata"]).get("guardrail_information")
    assert isinstance(entries, list), entries
    akto: Final = [object_value(entry) for entry in entries if object_value(entry).get("guardrail_name") == "akto-log"]
    assert akto, entries
    assert akto[0]["guardrail_mode"] == "logging_only", akto
    assert akto[0]["guardrail_status"] == "guardrail_intervened", akto


def test_logging_only_scope_input_and_output_each_check_one_direction(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": [{"role": "user", "content": "scoped " + marker}]},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text

    input_requests, input_responses = _split(rig.akto_calls(INPUT_KEY, marker, 1))
    output_requests, output_responses = _split(rig.akto_calls(OUTPUT_KEY, marker, 1))
    eventually(lambda: rig.sink.for_key(LOG_KEY, marker), lambda calls: len(calls) >= 2, seconds=30)
    assert (len(input_requests), len(input_responses)) == (1, 0), rig.sink.for_key(INPUT_KEY, marker)
    assert (len(output_requests), len(output_responses)) == (0, 1), rig.sink.for_key(OUTPUT_KEY, marker)
    _assert_request_check(input_requests[0], marker)
    _assert_response_check(output_responses[0], marker)


def test_unreachable_akto_under_logging_only_never_fails_the_caller(rig: Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": [{"role": "user", "content": "outage " + marker}]},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == _answer(marker)
    assert rig.sink.for_key(DOWN_KEY, marker) == ()
    assert len(rig.akto_calls(LOG_KEY, marker, 2)) == 2


def test_logging_only_streaming_chat_sends_the_assembled_answer(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    stream: Final = client.chat.completions.create(
        model=rig.chat_model,
        messages=[{"role": "user", "content": "stream " + marker}],
        stream=True,
    )
    text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    assert text == _answer(marker)

    requests, responses = _split(rig.akto_calls(LOG_KEY, marker, 2))
    assert len(requests) == 1 and len(responses) == 1, rig.sink.for_key(LOG_KEY, marker)
    _assert_request_check(requests[0], marker)
    _assert_response_check(responses[0], marker)


def test_logging_only_anthropic_messages_checks_both_directions(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = anthropic.Anthropic(base_url=rig.base(), api_key=rig.proxy.key, max_retries=0)
    message: Final = client.messages.create(
        model=rig.claude_model, max_tokens=64, messages=[{"role": "user", "content": "claude " + marker}]
    )
    assert message.content[0].type == "text" and message.content[0].text == _answer(marker)

    requests, responses = _split(rig.akto_calls(LOG_KEY, marker, 2))
    assert len(requests) == 1 and len(responses) == 1, rig.sink.for_key(LOG_KEY, marker)
    _assert_request_check(requests[0], marker)
    _assert_response_check(responses[0], marker)


def test_logging_only_responses_api_checks_both_directions(rig: Rig) -> None:
    marker: Final = _marker()
    client: Final = openai.OpenAI(base_url=rig.base() + "/v1", api_key=rig.proxy.key, max_retries=0)
    result: Final = client.responses.create(model=rig.responses_model, input="responses " + marker)
    assert result.output_text == _answer(marker)

    requests, responses = _split(rig.akto_calls(LOG_KEY, marker, 2))
    assert len(requests) == 1 and len(responses) == 1, rig.sink.for_key(LOG_KEY, marker)
    _assert_request_check(requests[0], marker)
    _assert_response_check(responses[0], marker)


def test_logging_only_sends_each_chat_attachment_to_akto_once(rig: Rig) -> None:
    marker: Final = _marker()
    image_url: Final = f"https://example.com/{marker}.png"
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe " + marker},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text

    eventually(lambda: rig.sink.for_key(LOG_KEY, marker), lambda calls: len(calls) >= 3, seconds=30)
    file_checks: Final = tuple(call for call in rig.sink.for_key(LOG_KEY, marker) if call.flags == FILE_CHECK)
    assert len(file_checks) == 1, rig.sink.for_key(LOG_KEY, marker)
    assert file_checks[0].payload["files"] == [{"filename": f"{marker}.png", "type": "image", "url": image_url}]


def test_dashboard_offers_logging_only_for_akto(rig: Rig) -> None:
    response: Final = rig.proxy.client.get(
        "/guardrails/ui/add_guardrail_settings", headers={"Authorization": f"Bearer {rig.proxy.key}"}
    )
    assert response.status_code == 200, response.text
    settings: Final = response.json()
    assert "logging_only" in settings["supported_modes_by_provider"]["akto"], settings["supported_modes_by_provider"]
    assert "akto" not in settings["providers_without_directional_logging_only_scope"]
