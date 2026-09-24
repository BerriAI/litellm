"""Straiker guardrail on both platform APIs, driven through a real proxy.

The Straiker platform is the only double: an owned HTTP sink that speaks the v1 webhook and the v3
detect wire protocols and records every request. The provider is a second owned sink. The proxy,
its guardrail registry, Postgres and Redis run for real with two workers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import signal
import socket
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

V3_KEY: Final = "sk_agt_synthetic_integration_key"
V1_KEY: Final = "synthetic-v1-collection-key"
V3_PATH: Final = "/api/v3/detect"
V1_PATH: Final = "/api/v1/detect/webhook"
BLOCK_MARK: Final = "SYNTHETIC-INJECTION"
KILL_MARK: Final = "SYNTHETIC-KILLSWITCH"
DENY_MARK: Final = "SYNTHETIC-DENY"
SINK_500_MARK: Final = "SYNTHETIC-SINK-500"
SINK_401_MARK: Final = "SYNTHETIC-SINK-401"
SINK_GARBAGE_MARK: Final = "SYNTHETIC-SINK-GARBAGE"
LOG_BLOCK_MARK: Final = "SYNTHETIC-LOG-ONLY-BLOCK"
OPEN_500_MARK: Final = "SYNTHETIC-OPEN-500"
V1_500_MARK: Final = "SYNTHETIC-V1-500"
V1_BLOCK_MARK: Final = "SYNTHETIC-V1-BLOCK"
AUDIT_AGENT: Final = "audit-agent"
POST_AGENT: Final = "post-agent"
LOG_AGENT: Final = "log-agent"
OPEN_AGENT: Final = "open-agent"
BLOCK_MESSAGE: Final = "Straiker blocked this turn: prompt-injection"
DENY_MESSAGE: Final = "Straiker denied this turn"


@dataclass(frozen=True, slots=True)
class Seen:
    target: str
    headers: dict[str, str]
    body: dict[str, object]


@dataclass(slots=True)
class Sink:
    """Owned Straiker platform double on a fixed port so a test can stop and restart it."""

    port: int
    seen: list[Seen] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        sink: Final = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                raw: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
                body: Final = json.loads(raw)
                seen: Final = Seen(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                with sink.lock:
                    sink.seen.append(seen)
                status, payload = _verdict(seen, raw.decode())
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.send_header("connection", "close")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        assert self.server is not None and self.thread is not None
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.server = None
        self.thread = None

    def drain(self) -> tuple[Seen, ...]:
        with self.lock:
            taken: Final = tuple(self.seen)
            self.seen.clear()
        return taken

    def for_marker(self, marker: str) -> tuple[Seen, ...]:
        with self.lock:
            return tuple(s for s in self.seen if marker in json.dumps(s.body))


def _verdict(seen: Seen, text: str) -> tuple[int, bytes]:
    agent: Final = seen.headers.get("x-s6r-agent")
    if (
        SINK_500_MARK in text
        or (OPEN_500_MARK in text and agent == OPEN_AGENT)
        or (V1_500_MARK in text and seen.target == V1_PATH)
    ):
        return 500, b'{"error":"synthetic outage"}'
    if SINK_401_MARK in text:
        return 401, b'{"error":"synthetic bad key"}'
    if SINK_GARBAGE_MARK in text:
        return 200, b"<html>not json</html>"
    if seen.target == V1_PATH:
        if BLOCK_MARK in text or V1_BLOCK_MARK in text:
            return 200, json.dumps({"action": "BLOCKED", "blocked_reason": BLOCK_MESSAGE}).encode()
        return 200, json.dumps({"action": "NONE"}).encode()
    assert seen.target == V3_PATH, seen.target
    turn: Final = "turn-" + hashlib.sha256(text.encode()).hexdigest()[:12]
    if BLOCK_MARK in text or (LOG_BLOCK_MARK in text and agent == LOG_AGENT):
        return 200, json.dumps(
            {
                "hookSpecificOutput": {"permissionDecision": "block"},
                "straiker": {
                    "action": "block",
                    "blocked_by": ["prompt-injection"],
                    "block_message": BLOCK_MESSAGE,
                    "turn_id": turn,
                },
            }
        ).encode()
    if DENY_MARK in text:
        return 200, json.dumps({"action": "deny", "deny_reason": DENY_MESSAGE, "turn_id": turn}).encode()
    if KILL_MARK in text:
        return 200, json.dumps(
            {"straiker": {"action": "block", "block_message": BLOCK_MESSAGE, "turn_id": turn}}
        ).encode()
    return 200, json.dumps(
        {"hookSpecificOutput": {"permissionDecision": "allow"}, "straiker": {"action": "allow", "turn_id": turn}}
    ).encode()


def _marker_in(body: bytes) -> str:
    text: Final = body.decode()
    start: Final = text.find("mark-")
    return text[start : start + 37] if start >= 0 else "mark-" + uuid.uuid4().hex


def _chat_body(marker: str, answer: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-" + marker,
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _chat_chunks(marker: str, answer: str) -> tuple[bytes, ...]:
    def chunk(delta: dict[str, object], finish: str | None) -> bytes:
        return (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-" + marker,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
            + "\n\n"
        ).encode()

    return (
        chunk({"role": "assistant", "content": answer[:3]}, None),
        chunk({"content": answer[3:]}, "stop"),
        b"data: [DONE]\n\n",
    )


def _messages_body(marker: str, answer: str) -> bytes:
    return json.dumps(
        {
            "id": "msg_" + marker,
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    ).encode()


def _messages_chunks(marker: str, answer: str) -> tuple[bytes, ...]:
    def event(name: str, payload: dict[str, object]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()

    return (
        event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_" + marker,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            },
        ),
        event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": answer}},
        ),
        event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 5},
            },
        ),
        event("message_stop", {"type": "message_stop"}),
    )


def _responses_body(marker: str, answer: str) -> bytes:
    return json.dumps(
        {
            "id": "resp_" + marker,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": "msgo_" + marker,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    ).encode()


def _completion_body(marker: str, answer: str) -> bytes:
    return json.dumps(
        {
            "id": "cmpl-" + marker,
            "object": "text_completion",
            "created": 1,
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"index": 0, "text": answer, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    ).encode()


_PROVIDER_CALLS: Final = itertools.count(1)


def _provider(request: Request) -> Reply:
    if not request.body:
        return Reply(status=404, body=b'{"error":"synthetic provider: no body"}')
    marker: Final = _marker_in(request.body)
    ident: Final = f"{marker}-{next(_PROVIDER_CALLS)}"
    body: Final = json.loads(request.body)
    answer: Final = "synthetic answer " + marker + (" " + BLOCK_MARK if "ANSWER-BLOCK" in request.body.decode() else "")
    streaming: Final = bool(body.get("stream"))
    if request.target.endswith("/v1/messages"):
        return (
            Reply(chunks=_messages_chunks(ident, answer), content_type="text/event-stream")
            if streaming
            else Reply(body=_messages_body(ident, answer))
        )
    if request.target.endswith("/v1/responses"):
        return Reply(body=_responses_body(ident, answer))
    if request.target.endswith("/v1/completions"):
        return Reply(body=_completion_body(ident, answer))
    assert request.target.endswith("/v1/chat/completions"), request.target
    return (
        Reply(chunks=_chat_chunks(ident, answer), content_type="text/event-stream")
        if streaming
        else Reply(body=_chat_body(ident, answer))
    )


def _guardrail(name: str, key: str, url: str, mode: str, default_on: bool, **params: object) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "straiker",
            "mode": mode,
            "default_on": default_on,
            "api_key": key,
            "api_base": url,
            "max_retries": 0,
            **params,
        },
    }


def _rig_config(sink_url: str, root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"]["cache"] = False
    config["guardrails"] = [
        _guardrail("straiker-v3", V3_KEY, sink_url, "pre_call", True, agent_ref=AUDIT_AGENT),
        _guardrail("straiker-v3-post", V3_KEY, sink_url, "post_call", False, agent_ref=POST_AGENT),
        _guardrail("straiker-v3-log", V3_KEY, sink_url, "logging_only", False, agent_ref=LOG_AGENT),
        _guardrail("straiker-v3-open", V3_KEY, sink_url, "pre_call", False, fail_on_error=False, agent_ref=OPEN_AGENT),
        _guardrail(
            "straiker-v3-hint",
            V3_KEY,
            sink_url,
            "pre_call",
            False,
            client="named-client",
            format_hint="anthropic.messages",
        ),
        _guardrail("straiker-v3-as-v1", V3_KEY, sink_url, "pre_call", False, api_version="v1"),
        _guardrail("straiker-v1", V1_KEY, sink_url, "pre_call", False),
        _guardrail("straiker-v1-post", V1_KEY, sink_url, "post_call", False),
    ]
    path: Final = root / "straiker.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    owned: OwnedProxy
    sink: Sink
    provider_url: str
    provider_drain: Callable[[], tuple[Request, ...]]
    chat_model: str
    anthropic_model: str
    completion_model: str

    def marker(self) -> str:
        return "mark-" + uuid.uuid4().hex

    def _base(self) -> str:
        return str(self.proxy.client.base_url).rstrip("/")

    def openai(self, key: str | None = None) -> openai.OpenAI:
        return openai.OpenAI(base_url=self._base() + "/v1", api_key=key or self.proxy.key, max_retries=0)

    def async_openai(self, key: str | None = None) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(base_url=self._base() + "/v1", api_key=key or self.proxy.key, max_retries=0)

    def anthropic(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(base_url=self._base(), api_key=self.proxy.key, max_retries=0)

    def async_anthropic(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(base_url=self._base(), api_key=self.proxy.key, max_retries=0)

    def sink_calls(self, marker: str) -> tuple[Seen, ...]:
        return self.sink.for_marker(marker)

    def provider_calls(self, marker: str, requests: tuple[Request, ...]) -> tuple[Request, ...]:
        return tuple(r for r in requests if marker.encode() in r.body)

    def spend_row(self, request_id: str) -> dict[str, object]:
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, model, call_type, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        return rows[0]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("straiker")
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    sink: Final = Sink(port)
    sink.start()
    with gateway_from_environment() as gateway, wire_server(_provider) as provider:
        config: Final = _rig_config(sink.url, root)
        with (
            owned_proxy_process(gateway, root, {}, config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            chat: Final = scenario.model(
                model="openai/gpt-4o-mini", api_base=provider.url + "/v1", api_key="synthetic-openai-key"
            )
            claude: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-anthropic-key"
            )
            completion: Final = scenario.model(
                model="text-completion-openai/gpt-3.5-turbo-instruct",
                api_base=provider.url + "/v1",
                api_key="synthetic-openai-key",
            )
            yield Rig(owned.gateway, owned, sink, provider.url, provider.drain, chat, claude, completion)
    if sink.server is not None:
        sink.stop()


def _messages(text: str, system: str | None = None) -> list[dict[str, object]]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]


def _chat(
    rig: Rig, text: str, *, key: str | None = None, headers: dict[str, str] | None = None, **extra: object
) -> httpx.Response:
    return rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": rig.chat_model, "messages": _messages(text), **extra},
        headers={"Authorization": f"Bearer {key or rig.proxy.key}", **(headers or {})},
    )


def _v3_request_calls(rig: Rig, marker: str, agent: str | None = AUDIT_AGENT) -> tuple[Seen, ...]:
    return tuple(
        s
        for s in rig.sink_calls(marker)
        if s.target == V3_PATH and "straiker_phase" not in s.body and s.headers.get("x-s6r-agent") == agent
    )


def _v3_response_calls(rig: Rig, marker: str, agent: str | None = POST_AGENT) -> tuple[Seen, ...]:
    return tuple(
        s
        for s in rig.sink_calls(marker)
        if s.target == V3_PATH
        and s.body.get("straiker_phase") == "response-sync"
        and s.headers.get("x-s6r-agent") == agent
    )


def _v1_calls(rig: Rig, marker: str, key: str) -> tuple[Seen, ...]:
    return tuple(
        s for s in rig.sink_calls(marker) if s.target == V1_PATH and s.headers.get("authorization") == "Bearer " + key
    )


# H1: default_on v3 pre_call, OpenAI SDK sync, non-streaming
def test_v3_pre_call_allow_relays_provider_body_and_key_identity(rig: Rig) -> None:
    marker: Final = rig.marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias="alias-" + marker, metadata={"user_api_key_user_email": "n/a"})
        response: Final = rig.openai(key).chat.completions.create(
            model=rig.chat_model,
            messages=[{"role": "user", "content": "hello " + marker}],
            temperature=0.2,
            user="end-" + marker,
        )
    assert response.id.startswith("chatcmpl-" + marker), response.id
    assert response.choices[0].message.content == "synthetic answer " + marker
    calls: Final = _v3_request_calls(rig, marker)
    assert len(calls) == 1, calls
    sent: Final = calls[0]
    assert sent.headers["authorization"] == "Bearer " + V3_KEY
    assert "x-straiker-webhook-format" not in sent.headers
    assert sent.headers["x-s6r-agent"] == "audit-agent"
    assert sent.body["messages"] == [{"role": "user", "content": "hello " + marker}]
    assert sent.body["temperature"] == 0.2
    assert sent.body["model"] == rig.chat_model
    assert "api_key" not in sent.body and "synthetic-openai-key" not in json.dumps(sent.body)
    assert object_value(sent.body["metadata"])["user_api_key_alias"] == "alias-" + marker
    assert sent.body.get("session_id", "").startswith("litellm-")
    upstream: Final = rig.provider_calls(marker, rig.provider_drain())
    assert len(upstream) == 1 and upstream[0].target == "/v1/chat/completions"
    row: Final = rig.spend_row(response.id)
    assert row["model"] == "openai/gpt-4o-mini", row


# H2: v3 block verdict on the request phase blocks with the platform's message
def test_v3_block_verdict_returns_400_with_block_message_and_no_provider_call(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{BLOCK_MARK} {marker}")
    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == BLOCK_MESSAGE, response.text
    assert len(_v3_request_calls(rig, marker)) == 1
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# H3: a resend of a blocked conversation is blocked by the process that saw the block, without a second detect call
def test_v3_blocked_conversation_replays_block_without_asking_again(rig: Rig) -> None:
    marker: Final = rig.marker()
    session: Final = {"x-claude-code-session-id": "session-" + marker}
    first: Final = _chat(rig, f"{BLOCK_MARK} {marker}", headers=session)
    assert first.status_code == 400, first.text
    baseline: Final = len(_v3_request_calls(rig, marker))
    assert baseline == 1
    outcomes: Final = tuple(_chat(rig, f"{BLOCK_MARK} {marker}", headers=session) for _ in range(6))
    assert all(r.status_code == 400 and r.json()["error"]["message"] == BLOCK_MESSAGE for r in outcomes), [
        r.text for r in outcomes
    ]
    later: Final = len(_v3_request_calls(rig, marker))
    # Two workers: only the worker that saw the block replays from memory, the other asks Straiker once
    assert baseline <= later <= 2, later
    grown: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={
            "model": rig.chat_model,
            "messages": _messages(f"{BLOCK_MARK} {marker}")
            + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "more"}],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}", **session},
    )
    assert grown.status_code == 400, grown.text
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# H4: a kill-switch block (no blocked_by) blocks but is not remembered, so Straiker is asked every time
def test_v3_killswitch_block_is_not_remembered(rig: Rig) -> None:
    marker: Final = rig.marker()
    session: Final = {"x-claude-code-session-id": "session-" + marker}
    outcomes: Final = tuple(_chat(rig, f"{KILL_MARK} {marker}", headers=session) for _ in range(3))
    assert all(r.status_code == 400 and r.json()["error"]["message"] == BLOCK_MESSAGE for r in outcomes)
    assert len(_v3_request_calls(rig, marker)) == 3


# H4b: a deny decision on the flat envelope also blocks, with the deny_reason
def test_v3_flat_deny_decision_blocks_with_deny_reason(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{DENY_MARK} {marker}")
    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == DENY_MESSAGE


# H5: post_call non-streaming, selected per request, async OpenAI SDK
@pytest.mark.asyncio
async def test_v3_post_call_sends_response_phase_with_answer(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = await rig.async_openai().chat.completions.create(
        model=rig.chat_model,
        messages=[{"role": "user", "content": "post " + marker}],
        extra_body={"guardrails": ["straiker-v3-post"]},
    )
    assert response.id.startswith("chatcmpl-" + marker), response.id
    calls: Final = eventually(lambda: _v3_response_calls(rig, marker), lambda c: len(c) == 1)
    phase: Final = calls[0].body
    assert phase["model"] == "gpt-4o-mini", "the deployment's model, not the alias"
    assert object_value(phase["request"])["messages"] == [{"role": "user", "content": "post " + marker}]
    assert json.loads(str(phase["sse"]))["id"].startswith("chatcmpl-" + marker)
    assert json.loads(str(phase["sse"]))["choices"][0]["message"]["content"] == "synthetic answer " + marker
    assert len(_v3_request_calls(rig, marker)) == 1, "the default_on pre_call route still runs beside the selected one"
    assert rig.spend_row(response.id)["request_id"] == response.id


# H5b: post_call block replaces the answer with the block message as a 200
def test_v3_post_call_block_replaces_answer(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, "ANSWER-BLOCK " + marker, guardrails=["straiker-v3-post"])
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == BLOCK_MESSAGE, response.text
    assert len(_v3_response_calls(rig, marker)) == 1


# H6: post_call streaming, OpenAI SDK sync; the stream is consumed to the end before the phase is sent
def test_v3_post_call_streaming_sends_assembled_answer(rig: Rig) -> None:
    marker: Final = rig.marker()
    stream: Final = rig.openai().chat.completions.create(
        model=rig.chat_model,
        messages=[{"role": "user", "content": "stream " + marker}],
        stream=True,
        extra_body={"guardrails": ["straiker-v3-post"]},
    )
    chunks: Final = list(stream)
    assert chunks and all(c.id.startswith("chatcmpl-" + marker) for c in chunks)
    text: Final = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "synthetic answer " + marker
    calls: Final = eventually(lambda: _v3_response_calls(rig, marker), lambda c: len(c) == 1)
    sse: Final = json.loads(str(calls[0].body["sse"]))
    assert "synthetic answer " + marker in json.dumps(sse)
    assert object_value(calls[0].body["request"])["stream"] is True


# H7: Anthropic Messages sync, pre_call, session header and recognised client
def test_v3_anthropic_messages_relays_system_and_routing_headers(rig: Rig) -> None:
    marker: Final = rig.marker()
    client: Final = rig.anthropic().with_options(
        default_headers={"x-claude-code-session-id": "cc-" + marker, "User-Agent": "claude-cli/2.0.0 (external, cli)"}
    )
    response: Final = client.messages.create(
        model=rig.anthropic_model,
        max_tokens=16,
        system="synthetic system " + marker,
        messages=[{"role": "user", "content": "anthropic " + marker}],
    )
    assert response.id.startswith("msg_" + marker), response.id
    assert response.content[0].text == "synthetic answer " + marker
    calls: Final = _v3_request_calls(rig, marker)
    assert len(calls) == 1, calls
    sent: Final = calls[0]
    assert sent.headers["x-claude-code-session-id"] == "cc-" + marker
    assert sent.headers["x-s6r-client"] == "claude"
    assert sent.headers["x-s6r-agent"] == "audit-agent", "YAML agent_ref wins over the User-Agent derived agent"
    assert sent.body["session_id"] == "cc-" + marker
    assert sent.body["system"] == "synthetic system " + marker
    assert sent.body["messages"] == [{"role": "user", "content": "anthropic " + marker}]
    assert sent.body["max_tokens"] == 16
    upstream: Final = rig.provider_calls(marker, rig.provider_drain())
    assert len(upstream) == 1 and upstream[0].target == "/v1/messages"
    assert rig.spend_row(response.id)["call_type"] == "anthropic_messages"


# H8: Anthropic Messages streaming, async SDK, post_call: the answer is scored in Messages shape
@pytest.mark.asyncio
async def test_v3_anthropic_streaming_post_call_scores_messages_shaped_answer(rig: Rig) -> None:
    marker: Final = rig.marker()
    client: Final = rig.async_anthropic()
    async with client.messages.stream(
        model=rig.anthropic_model,
        max_tokens=16,
        messages=[{"role": "user", "content": "astream " + marker}],
        extra_body={"guardrails": ["straiker-v3-post"]},
    ) as stream:
        final: Final = await stream.get_final_message()
    assert final.id.startswith("msg_" + marker), final.id
    assert final.content[0].text == "synthetic answer " + marker
    calls: Final = eventually(lambda: _v3_response_calls(rig, marker), lambda c: len(c) == 1)
    sse: Final = json.loads(str(calls[0].body["sse"]))
    assert sse.get("type") == "message", sse
    assert sse["content"][0]["text"] == "synthetic answer " + marker


# H9: Responses API, raw httpx, pre_call relays `input`, `instructions`, and the answer on post_call
def test_v3_responses_api_relays_input_and_answer(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = rig.proxy.client.post(
        "/v1/responses",
        json={
            "model": rig.chat_model,
            "input": "responses " + marker,
            "instructions": "be brief",
            "guardrails": ["straiker-v3", "straiker-v3-post"],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"].startswith("resp_"), response.text
    assert "synthetic answer " + marker in response.text
    pre: Final = _v3_request_calls(rig, marker)
    assert len(pre) == 1 and pre[0].body["input"] == "responses " + marker and pre[0].body["instructions"] == "be brief"
    post: Final = eventually(lambda: _v3_response_calls(rig, marker), lambda c: len(c) == 1)
    assert "synthetic answer " + marker in str(post[0].body["sse"])
    upstream: Final = rig.provider_calls(marker, rig.provider_drain())
    assert len(upstream) == 1 and upstream[0].target == "/v1/responses"


# H10/H11: Completions API prompt becomes messages on the request phase; the answer is sent as a chat completion
def test_v3_completions_prompt_is_relayed_as_messages_and_answer_as_chat(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = rig.proxy.client.post(
        "/v1/completions",
        json={
            "model": rig.completion_model,
            "prompt": "complete " + marker,
            "guardrails": ["straiker-v3", "straiker-v3-post"],
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"].startswith("cmpl-" + marker), response.json()["id"]
    pre: Final = _v3_request_calls(rig, marker)
    assert len(pre) == 1, pre
    assert pre[0].body["messages"] == [{"role": "user", "content": "complete " + marker}]
    assert "prompt" not in pre[0].body
    post: Final = eventually(lambda: _v3_response_calls(rig, marker), lambda c: len(c) == 1)
    sse: Final = json.loads(str(post[0].body["sse"]))
    assert sse["object"] == "chat.completion", sse
    assert sse["choices"][0]["message"]["content"] == "synthetic answer " + marker


# H12: tool and MCP server credentials are redacted one level deep; a schema property named headers is kept
def test_v3_redacts_tool_credentials_but_keeps_schema_properties(rig: Rig) -> None:
    marker: Final = rig.marker()
    tools: Final = [
        {
            "type": "function",
            "authorization": "Bearer synthetic-tool-secret",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"headers": {"type": "string"}}},
            },
        }
    ]
    response: Final = _chat(
        rig,
        "tools " + marker,
        tools=tools,
        mcp_servers=[{"url": "http://mcp", "authorization_token": "synthetic-mcp-secret"}],
    )
    assert response.status_code == 200, response.text
    sent: Final = _v3_request_calls(rig, marker)[0].body
    assert sent["tools"][0]["authorization"] == "[redacted]"  # pyright: ignore[reportIndexIssue]  # sink body is loose JSON
    assert sent["tools"][0]["function"]["parameters"]["properties"]["headers"] == {"type": "string"}  # pyright: ignore[reportIndexIssue]  # sink body is loose JSON
    assert sent["mcp_servers"][0]["authorization_token"] == "[redacted]"  # pyright: ignore[reportIndexIssue]  # sink body is loose JSON
    assert "synthetic-tool-secret" not in json.dumps(sent) and "synthetic-mcp-secret" not in json.dumps(sent)


# U1: a v1 collection key still speaks the v1 webhook with the litellm envelope
def test_v1_key_keeps_webhook_envelope_and_format_header(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, "v1 " + marker, guardrails=["straiker-v1"])
    assert response.status_code == 200, response.text
    calls: Final = _v1_calls(rig, marker, V1_KEY)
    assert len(calls) == 1, rig.sink_calls(marker)
    assert calls[0].headers["x-straiker-webhook-format"] == "litellm"
    assert calls[0].body["schema_version"] and object_value(calls[0].body["event"])["type"]
    assert "v1 " + marker in json.dumps(object_value(calls[0].body["request"]))
    assert len(_v3_request_calls(rig, marker)) == 1, "the default_on v3 route runs beside it"


# U2: v1 block verdict still blocks
def test_v1_block_verdict_still_blocks(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{V1_BLOCK_MARK} {marker}", guardrails=["straiker-v1"])
    assert response.status_code == 400, response.text
    assert response.json()["error"]["message"] == BLOCK_MESSAGE
    assert len(_v1_calls(rig, marker, V1_KEY)) == 1, rig.sink_calls(marker)
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# U3: v1 post_call still receives the response envelope
def test_v1_post_call_sends_response_envelope(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, "v1post " + marker, guardrails=["straiker-v1-post"])
    assert response.status_code == 200, response.text
    calls: Final = eventually(lambda: _v1_calls(rig, marker, V1_KEY), lambda c: len(c) == 1)
    assert "synthetic answer " + marker in json.dumps(calls[0].body.get("response"))


# E: explicit api_version v1 with a v3-shaped key follows the configuration, not the key
def test_explicit_api_version_v1_overrides_key_prefix(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, "explicit " + marker, guardrails=["straiker-v3-as-v1"])
    assert response.status_code == 200, response.text
    calls: Final = _v1_calls(rig, marker, V3_KEY)
    assert len(calls) == 1, rig.sink_calls(marker)
    assert calls[0].headers["x-straiker-webhook-format"] == "litellm"


# E: configured client and format_hint ride as headers; request header for agent fills in when YAML has none
def test_v3_client_and_format_hint_headers_and_request_agent_header(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(
        rig, "hint " + marker, guardrails=["straiker-v3-hint"], headers={"x-s6r-agent": "caller-agent"}
    )
    assert response.status_code == 200, response.text
    hinted: Final = tuple(s for s in _v3_request_calls(rig, marker, agent="caller-agent"))
    assert len(hinted) == 1, rig.sink_calls(marker)
    sent: Final = hinted[0]
    assert sent.headers["x-s6r-client"] == "named-client"
    assert sent.headers["x-s6r-format"] == "anthropic.messages"
    assert sent.headers["x-s6r-agent"] == "caller-agent"


# E: identity precedence: the key's user email wins over an end user in the body
def test_v3_user_prefers_key_email_over_body_user(rig: Rig) -> None:
    marker: Final = rig.marker()
    with rig.proxy.scenario() as scenario:
        user: Final = scenario.user(user_email=f"{marker}@example.test")
        key: Final = scenario.key(user_id=user)
        response: Final = _chat(rig, "identity " + marker, key=key, user="body-user-" + marker)
    assert response.status_code == 200, response.text
    sent: Final = _v3_request_calls(rig, marker)[0].body
    meta: Final = object_value(sent["original"])
    assert object_value(object_value(object_value(meta["processed"])["Meta"]))["user"] == f"{marker}@example.test"
    assert object_value(sent["metadata"])["user_api_key_user_email"] == f"{marker}@example.test"
    assert sent["user"] == "body-user-" + marker


# E: logging_only observes the turn but never blocks
def test_v3_logging_only_observes_block_verdict_without_blocking(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{LOG_BLOCK_MARK} {marker}")
    assert response.status_code == 200, response.text
    assert response.json()["id"].startswith("chatcmpl-" + marker), response.json()["id"]
    calls: Final = eventually(lambda: _v3_request_calls(rig, marker, agent=LOG_AGENT), lambda c: len(c) >= 1)
    assert calls[0].headers["x-s6r-agent"] == LOG_AGENT
    assert len(rig.provider_calls(marker, rig.provider_drain())) == 1
    row: Final = rig.spend_row(response.json()["id"])
    assert row["request_id"] == response.json()["id"]


# E: the same identical allowed request three times yields three detect calls and three spend rows
def test_v3_repeated_allowed_request_is_scored_and_logged_each_time(rig: Rig) -> None:
    marker: Final = rig.marker()
    responses: Final = tuple(_chat(rig, "repeat " + marker) for _ in range(3))
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    ids: Final = {r.json()["id"] for r in responses}
    assert len(ids) == 3 and all(i.startswith("chatcmpl-" + marker) for i in ids), ids
    assert len(_v3_request_calls(rig, marker)) == 3
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', ("chatcmpl-" + marker + "%",)
        ),
        lambda values: len(values) == 3,
        seconds=70,
    )
    assert {str(r["request_id"]) for r in rows} == ids


# S1: platform answers 500: fail closed with the reason in the body, no provider call
def test_v3_sink_500_fails_closed_with_reason(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{SINK_500_MARK} {marker}")
    assert response.status_code == 400, response.text
    assert "Straiker detection unavailable" in response.json()["error"]["message"], response.text
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# S1a: the v1 webhook route fails the same way when the platform answers 500
def test_v1_sink_500_fails_closed_with_reason(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{V1_500_MARK} {marker}", guardrails=["straiker-v1"])
    assert response.status_code == 400, response.text
    assert "Straiker detection unavailable" in response.json()["error"]["message"], response.text
    assert len(_v1_calls(rig, marker, V1_KEY)) == 1
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# S1b: fail_on_error false lets the request through on a 500
def test_v3_fail_open_guardrail_passes_on_sink_500(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{OPEN_500_MARK} {marker}", guardrails=["straiker-v3-open"])
    assert response.status_code == 200, response.text
    assert response.json()["id"].startswith("chatcmpl-" + marker), response.json()["id"]
    assert len(_v3_request_calls(rig, marker, agent=OPEN_AGENT)) == 1


# S2: platform rejects the key: 401 is not retried and fails closed
def test_v3_sink_401_fails_closed_once(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{SINK_401_MARK} {marker}")
    assert response.status_code == 400, response.text
    assert "401" in response.json()["error"]["message"], response.text
    assert len(_v3_request_calls(rig, marker)) == 1


# S3: platform answers non JSON: fail closed, caller sees the parse failure
def test_v3_sink_garbage_fails_closed(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, f"{SINK_GARBAGE_MARK} {marker}")
    assert response.status_code == 400, response.text
    assert "Straiker detection unavailable" in response.json()["error"]["message"]


# S4: unauthenticated request never reaches the platform
def test_unauthenticated_request_does_not_reach_platform(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = _chat(rig, "anon " + marker, key="sk-not-a-real-key")
    assert response.status_code == 401, response.text
    assert rig.sink_calls(marker) == ()


# S5: unknown model: the guardrail still runs, then the router error reaches the caller
def test_unknown_model_error_reaches_caller_after_detect(rig: Rig) -> None:
    marker: Final = rig.marker()
    response: Final = rig.proxy.client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model-" + marker, "messages": _messages("unknown " + marker)},
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code in (400, 401, 404), response.text
    assert "no-such-model-" + marker in response.text
    assert len(_v3_request_calls(rig, marker)) == 1, rig.sink_calls(marker)
    assert rig.provider_calls(marker, rig.provider_drain()) == ()


# S6: odd shapes in the routing header and a 5 KB prompt are relayed verbatim, not crashed on
def test_v3_oversized_prompt_and_odd_header_values_are_relayed(rig: Rig) -> None:
    marker: Final = rig.marker()
    big: Final = "x" * 5000 + " " + marker
    response: Final = _chat(rig, big, headers={"x-claude-code-session-id": "", "x-s6r-agent": "1"})
    assert response.status_code == 200, response.text
    sent: Final = _v3_request_calls(rig, marker)[0]
    assert sent.body["messages"] == [{"role": "user", "content": big}]
    assert sent.headers["x-s6r-agent"] == "audit-agent"
    assert "x-claude-code-session-id" not in sent.headers
    assert sent.body.get("session_id", "").startswith("litellm-")


# S7: a guardrail with a malformed format_hint is rejected at /guardrails/apply_guardrail time, not at boot
def test_malformed_format_hint_config_is_rejected_by_guardrail_management(rig: Rig) -> None:
    response: Final = rig.proxy.client.post(
        "/guardrails",
        json={
            "guardrail": {
                "guardrail_name": "straiker-bad-" + uuid.uuid4().hex,
                "litellm_params": {
                    "guardrail": "straiker",
                    "mode": "pre_call",
                    "api_key": V3_KEY,
                    "api_base": rig.sink.url,
                    "format_hint": "bogus",
                },
            }
        },
        headers={"Authorization": f"Bearer {rig.proxy.key}"},
    )
    assert response.status_code in (400, 422, 500), response.text
    assert "format_hint" in response.text or "bogus" in response.text, response.text
    healthy: Final = _chat(rig, "still-fine " + uuid.uuid4().hex)
    assert healthy.status_code == 200, healthy.text


# S8: /key/health reports the key without touching the platform
def test_key_health_does_not_call_platform(rig: Rig) -> None:
    marker: Final = rig.marker()
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias="health-" + marker)
        response: Final = rig.proxy.client.post("/key/health", headers={"Authorization": f"Bearer {key}"})
    assert response.status_code == 200, response.text
    assert response.json()["key"] == "healthy"
    assert rig.sink_calls(marker) == ()


# C1: 30 request mixed burst while the platform sink is down mid burst, then recovers; every allowed id lands once
def test_burst_with_platform_outage_recovers_without_duplicate_spend(rig: Rig) -> None:
    burst: Final = 30
    markers: Final = tuple(rig.marker() for _ in range(burst))
    down: Final = threading.Event()
    up: Final = threading.Event()

    def call(index: int) -> tuple[int, int, str]:
        if index == 8:
            rig.sink.stop()
            down.set()
        if index == 20:
            assert down.wait(10)
            rig.sink.start()
            up.set()
        marker: Final = markers[index]
        if index % 3 == 0:
            response: Final = rig.proxy.client.post(
                "/v1/messages",
                json={
                    "model": rig.anthropic_model,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "burst " + marker}],
                },
                headers={"Authorization": f"Bearer {rig.proxy.key}"},
            )
            return index, response.status_code, response.text
        streaming: Final = index % 2 == 1
        response = _chat(rig, "burst " + marker, stream=streaming)
        return index, response.status_code, response.text

    with ThreadPoolExecutor(max_workers=6) as pool:
        results: Final = sorted(pool.map(call, range(burst)))
    assert up.is_set()
    statuses: Final = {index: status for index, status, _ in results}
    assert all(status in (200, 400) for status in statuses.values()), results
    failed: Final = tuple(index for index, status, text in results if status == 400)
    assert failed, "the outage must be visible to at least one caller"
    assert all("Straiker detection unavailable" in text for index, status, text in results if status == 400), results
    for index, status, text in results:
        if status != 200 or (index % 3 != 0 and index % 2 == 1):
            continue
        marker = markers[index]
        expected: Final = ("msg_" if index % 3 == 0 else "chatcmpl-") + marker + "%"
        rows: Final = eventually(
            lambda like=expected: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (like,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert len(rows) == 1, rows
    provider_seen: Final = rig.provider_drain()
    for index, status, _ in results:
        if status == 400:
            assert rig.provider_calls(markers[index], provider_seen) == (), (
                "a failed-closed turn must not reach the provider"
            )
    recovered: Final = _chat(rig, "after-outage " + rig.marker())
    assert recovered.status_code == 200, recovered.text


# C2: one proxy worker is killed during a burst; the other keeps serving and detect still runs for each call
def _uvicorn_workers(parent: psutil.Process, *, exclude: int = 0) -> tuple[psutil.Process, ...]:
    return tuple(
        c for c in parent.children() if c.is_running() and c.pid != exclude and "spawn_main" in " ".join(c.cmdline())
    )


def test_burst_survives_one_worker_kill(rig: Rig) -> None:
    parent: Final = psutil.Process(rig.owned.process.pid)
    workers: Final = eventually(lambda: _uvicorn_workers(parent), lambda c: len(c) >= 2)
    victim: Final = workers[0].pid
    markers: Final = tuple(rig.marker() for _ in range(24))

    def fresh_chat(text: str) -> tuple[int, str]:
        with httpx.Client(base_url=rig._base(), timeout=15, trust_env=False) as fresh:
            try:
                response: Final = fresh.post(
                    "/v1/chat/completions",
                    json={"model": rig.chat_model, "messages": _messages(text)},
                    headers={"Authorization": f"Bearer {rig.proxy.key}"},
                )
            except httpx.TransportError as error:
                return 0, repr(error)
        return response.status_code, response.text

    def call(index: int) -> tuple[int, str]:
        if index == 6:
            os.kill(victim, signal.SIGKILL)
        return fresh_chat("kill " + markers[index])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results: Final = tuple(pool.map(call, range(24)))
    ok: Final = tuple(i for i, (status, _) in enumerate(results) if status == 200)
    assert len(ok) >= 20, results
    for index in ok:
        assert len(_v3_request_calls(rig, markers[index])) >= 1, markers[index]
    eventually(lambda: _uvicorn_workers(parent, exclude=victim), lambda c: len(c) >= 2)
    after: Final = fresh_chat("after-kill " + rig.marker())
    assert after[0] == 200, after


# C3: proxy restart between a blocked turn and its replay: the memory is per process and empties, so Straiker is asked again
def test_proxy_restart_forgets_blocked_turns_and_asks_platform_again(tmp_path: Path, rig: Rig) -> None:
    with gateway_from_environment() as gateway:
        config: Final = _rig_config(rig.sink.url, tmp_path)
        marker: Final = rig.marker()
        session: Final = {"x-claude-code-session-id": "restart-" + marker}
        body: Final = {"model": rig.chat_model, "messages": _messages(f"{BLOCK_MARK} {marker}")}
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as first:
            blocked: Final = first.client.post(
                "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {first.key}", **session}
            )
            assert blocked.status_code == 400, blocked.text
            replayed: Final = first.client.post(
                "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {first.key}", **session}
            )
            assert replayed.status_code == 400, replayed.text
            assert len(_v3_request_calls(rig, marker)) == 1, "one worker replays from memory"
        with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as second:
            again: Final = second.client.post(
                "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {second.key}", **session}
            )
            assert again.status_code == 400, again.text
        assert len(_v3_request_calls(rig, marker)) == 2, "a restarted process has no memory and asks once more"
