"""A deployment's ``provider_affinity_header`` names the header that carries the caller's stable session id to
the provider. /v1/chat/completions already sends it; /v1/messages must send the same header, from the
``x-litellm-session-id`` header or ``litellm_metadata.session_id``, on native Anthropic-format providers and on
the Responses bridge, and must reject a session id that would split the header (LIT-8405).

Each test is labelled: ``regression`` fails on the LIT-8405 base on the bug's own symptom (the header is
missing, or a 200 where a 400 belongs), ``guard`` passes on that base and pins behavior the fix must keep.

The OpenCode cases reach ``https://opencode.ai`` through an owned CONNECT tunnel that terminates TLS at a
scripted peer, so the deployment's real OpenCode api_base is used without any request leaving the box."""

import contextlib
import json
import socket
import socketserver
import ssl
import threading
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.ports import allocate_port
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)

AFFINITY_HEADER: Final = "x-session-affinity"
OPENCODE_HEADER: Final = "x-opencode-session"
OPENCODE_AUTHORITY: Final = "opencode.ai:443"
OPENCODE_BASE: Final = "https://opencode.ai/inference/go/anthropic"

ANTHROPIC_MODEL: Final = "affinity-anthropic"
AZURE_AI_MODEL: Final = "affinity-azure-ai-claude"
RESPONSES_MODEL: Final = "affinity-responses"
FORWARDING_MODEL: Final = "affinity-anthropic-forwarding-client-headers"
OPENCODE_AFFINITY_MODEL: Final = "affinity-opencode-anthropic"
OPENCODE_DEFAULT_MODEL: Final = "opencode-anthropic-default"

ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
AZURE_AI_KEY: Final = "synthetic-azure-ai-key"
OPENAI_KEY: Final = "synthetic-openai-key"
OPENCODE_KEY: Final = "synthetic-opencode-key"


def _message(text: str) -> dict[str, JsonValue]:
    return {
        "id": "msg_affinity",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }


def _message_stream(text: str) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "message_start", "message": {**_message(""), "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def _responses_reply(text: str) -> dict[str, JsonValue]:
    return {
        "id": "resp_affinity",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "msg_affinity_output",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    }


def _respond(request: Request) -> Reply:
    body: Final = json.loads(request.body) if request.body else {}
    if request.target.endswith("/responses"):
        return Reply(body=json.dumps(_responses_reply("ok")).encode())
    if isinstance(body, dict) and body.get("stream") is True:
        return Reply(chunks=_message_stream("ok"), content_type="text/event-stream")
    return Reply(body=json.dumps(_message("ok")).encode())


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@dataclass(frozen=True, slots=True)
class _Tunnel:
    url: str
    authorities: SimpleQueue[str]

    def drain(self) -> tuple[str, ...]:
        drained: Final[list[str]] = []
        with contextlib.suppress(Empty):
            while True:
                drained.append(self.authorities.get_nowait())
        return tuple(drained)


@contextmanager
def _opencode_tunnel(destination: Wire) -> Generator[_Tunnel, None, None]:
    """CONNECT proxy that sends only ``opencode.ai:443`` on to the TLS peer and refuses every other host."""
    authorities: Final[SimpleQueue[str]] = SimpleQueue()
    destination_port: Final = int(destination.url.rsplit(":", 1)[1])

    class Tunnel(socketserver.StreamRequestHandler):
        rbufsize = 0
        request: socket.socket

        def handle(self) -> None:
            authority: Final = self.rfile.readline().decode().split()[1]
            while self.rfile.readline() not in (b"\r\n", b""):
                pass
            authorities.put(authority)
            if authority != OPENCODE_AUTHORITY:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.request.settimeout(10)
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=12)

    port: Final = allocate_port()
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), Tunnel) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield _Tunnel(f"http://127.0.0.1:{port}", authorities)
        finally:
            server.shutdown()
            thread.join(timeout=6)


def _opencode_tls(directory: Path) -> ssl.SSLContext:
    certificate, key = write_self_signed_cert(directory, ("opencode.ai",))
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    return context


def _deployment(name: str, model: str, api_base: str, api_key: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, "api_base": api_base, "api_key": api_key, **extra}}


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    provider: Wire
    opencode: Wire
    tunnel: _Tunnel


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("messages-provider-affinity")
    with (
        wire_server(_respond) as provider,
        wire_server(_respond, tls=_opencode_tls(directory)) as opencode,
        _opencode_tunnel(opencode) as tunnel,
    ):
        affinity: Final[dict[str, JsonValue]] = {"provider_affinity_header": AFFINITY_HEADER}
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            _deployment(ANTHROPIC_MODEL, "anthropic/claude-sonnet-4-6", provider.url, ANTHROPIC_KEY, **affinity),
            _deployment(AZURE_AI_MODEL, "azure_ai/claude-sonnet-4-6", provider.url, AZURE_AI_KEY, **affinity),
            _deployment(RESPONSES_MODEL, "openai/responses/gpt-6.1-sol", provider.url, OPENAI_KEY, **affinity),
            _deployment(FORWARDING_MODEL, "anthropic/claude-sonnet-4-6", provider.url, ANTHROPIC_KEY, **affinity),
            _deployment(
                OPENCODE_AFFINITY_MODEL, "anthropic/claude-sonnet-4-6", OPENCODE_BASE, OPENCODE_KEY, **affinity
            ),
            _deployment(OPENCODE_DEFAULT_MODEL, "anthropic/claude-sonnet-4-6", OPENCODE_BASE, OPENCODE_KEY),
        ]
        settings: Final = {
            key: value for key, value in config["litellm_settings"].items() if key not in ("cache", "cache_params")
        }
        config["litellm_settings"] = {
            **settings,
            "model_group_settings": {"forward_client_headers_to_llm_api": [FORWARDING_MODEL]},
        }
        path: Final = directory / "messages-provider-affinity.yaml"
        path.write_text(yaml.safe_dump(config))
        environment: Final = {
            "HTTPS_PROXY": tunnel.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "AIOHTTP_TRUST_ENV": "True",
            "SSL_VERIFY": "False",
        }
        with (
            gateway_from_environment() as base,
            owned_proxy(base, directory, environment, config=path) as proxy,
        ):
            yield _Rig(proxy, provider, opencode, tunnel)


@pytest.fixture(autouse=True)
def _drained(rig: _Rig) -> None:
    rig.provider.drain()
    rig.opencode.drain()
    rig.tunnel.drain()


def _messages_body(model: str, prompt: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": prompt}], **extra}


def _prompt(label: str) -> str:
    return f"{label} {uuid.uuid4().hex}"


def _only(received: tuple[Request, ...], target: str) -> Request:
    assert [(request.method, request.target) for request in received] == [("POST", target)], [
        (request.method, request.target) for request in received
    ]
    return received[0]


def _header_values(request: Request, name: str) -> list[str]:
    return [value for header, value in request.header_lines if header.lower() == name]


def _assert_messages_reply(response_status: int, text: str, streamed: bool) -> None:
    assert response_status == 200, text
    if streamed:
        assert "message_stop" in text, text
    else:
        assert json.loads(text)["content"] == [{"type": "text", "text": "ok"}], text


# regression (LIT-8405): native Anthropic /v1/messages drops the configured affinity header
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_native_anthropic_messages_sends_the_configured_affinity_header(rig: _Rig, stream: bool) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(ANTHROPIC_MODEL, _prompt("anthropic affinity"), **({"stream": True} if stream else {})),
        headers={"x-litellm-session-id": session},
    )
    _assert_messages_reply(response.status_code, response.text, stream)
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert sent.headers.get("x-api-key") == ANTHROPIC_KEY, sent.headers
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines


# regression (LIT-8405): a non-Anthropic native Messages provider (Azure AI Claude) drops it too
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_azure_ai_claude_messages_sends_the_configured_affinity_header(rig: _Rig, stream: bool) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(AZURE_AI_MODEL, _prompt("azure ai affinity"), **({"stream": True} if stream else {})),
        headers={"x-litellm-session-id": session},
    )
    _assert_messages_reply(response.status_code, response.text, stream)
    sent: Final = _only(rig.provider.drain(), "/anthropic/v1/messages")
    assert sent.headers.get("x-api-key") == AZURE_AI_KEY, sent.headers
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines


# regression (LIT-8405): on an OpenCode base the configured header replaces the OpenCode default
def test_opencode_base_messages_sends_the_configured_header_instead_of_the_opencode_default(rig: _Rig) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(OPENCODE_AFFINITY_MODEL, _prompt("opencode configured affinity")),
        headers={"x-litellm-session-id": session},
    )
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    assert rig.tunnel.drain() == (OPENCODE_AUTHORITY,)
    sent: Final = _only(rig.opencode.drain(), "/inference/go/anthropic/v1/messages")
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines
    assert _header_values(sent, OPENCODE_HEADER) == [], sent.header_lines


# guard: a caller that already sends the configured header keeps its own value, sent exactly once
def test_messages_keeps_a_caller_supplied_affinity_header_once(rig: _Rig) -> None:
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(FORWARDING_MODEL, _prompt("caller affinity")),
        headers={"x-litellm-session-id": f"sess-{uuid.uuid4().hex}", "X-Session-Affinity": "mine"},
    )
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert _header_values(sent, AFFINITY_HEADER) == ["mine"], sent.header_lines


# guard: with no caller session id, a session id the proxy generates internally is never forwarded
def test_messages_without_a_caller_session_sends_no_affinity_header(rig: _Rig) -> None:
    response: Final = rig.proxy.request("POST", "/v1/messages", _messages_body(ANTHROPIC_MODEL, _prompt("no session")))
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert _header_values(sent, AFFINITY_HEADER) == [], sent.header_lines


# regression (LIT-8405): the session id may also come from litellm_metadata in the request body
def test_messages_sends_the_affinity_header_from_body_metadata_session_id(rig: _Rig) -> None:
    session: Final = f"sess-body-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(ANTHROPIC_MODEL, _prompt("body session"), litellm_metadata={"session_id": session}),
    )
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines


# regression (LIT-8405): a session id that would split the header is a 400 and never reaches the provider
@pytest.mark.parametrize("control", ["\r\nx-injected: 1", "\nx-injected: 1"], ids=["crlf", "lf"])
def test_messages_rejects_a_session_id_with_header_control_characters(rig: _Rig, control: str) -> None:
    rejected: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(ANTHROPIC_MODEL, _prompt("control session"), litellm_metadata={"session_id": f"sess{control}"}),
    )
    assert rejected.status_code == 400, rejected.text
    follow_up: Final = _prompt("after control session")
    accepted: Final = rig.proxy.request("POST", "/v1/messages", _messages_body(ANTHROPIC_MODEL, follow_up))
    _assert_messages_reply(accepted.status_code, accepted.text, streamed=False)
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert json.loads(sent.body)["messages"] == [{"role": "user", "content": follow_up}], sent.body


# guard: /v1/chat/completions on the same Anthropic deployment already sends the header
def test_chat_completions_on_anthropic_sends_the_configured_affinity_header(rig: _Rig) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": ANTHROPIC_MODEL, "max_tokens": 32, "messages": [{"role": "user", "content": _prompt("chat")}]},
        headers={"x-litellm-session-id": session},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok", response.text
    sent: Final = _only(rig.provider.drain(), "/v1/messages")
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines


# guard: /v1/messages through the Responses bridge already sends the header
def test_messages_on_the_responses_bridge_sends_the_configured_affinity_header(rig: _Rig) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(RESPONSES_MODEL, _prompt("responses bridge")),
        headers={"x-litellm-session-id": session},
    )
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    sent: Final = _only(rig.provider.drain(), "/responses")
    assert sent.headers.get("authorization") == f"Bearer {OPENAI_KEY}", sent.headers
    assert _header_values(sent, AFFINITY_HEADER) == [session], sent.header_lines


# guard: an OpenCode base with no configured header keeps sending the OpenCode default session header
def test_opencode_base_messages_without_configuration_sends_the_opencode_session_header(rig: _Rig) -> None:
    session: Final = f"sess-{uuid.uuid4().hex}"
    response: Final = rig.proxy.request(
        "POST",
        "/v1/messages",
        _messages_body(OPENCODE_DEFAULT_MODEL, _prompt("opencode default affinity")),
        headers={"x-litellm-session-id": session},
    )
    _assert_messages_reply(response.status_code, response.text, streamed=False)
    assert rig.tunnel.drain() == (OPENCODE_AUTHORITY,)
    sent: Final = _only(rig.opencode.drain(), "/inference/go/anthropic/v1/messages")
    assert _header_values(sent, OPENCODE_HEADER) == [session], sent.header_lines
    assert _header_values(sent, AFFINITY_HEADER) == [], sent.header_lines
