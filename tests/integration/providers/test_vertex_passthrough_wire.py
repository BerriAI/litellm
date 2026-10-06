import contextlib
import datetime
import json
import socket
import socketserver
import ssl
import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Final

import anthropic
import httpx
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(240)

_PROJECT: Final = "scripted-project"
_TOKEN: Final = "scripted-vertex-token"
_CALLER_GOOGLE_TOKEN: Final = "caller-google-token"
_GEMINI_ROUTER_NAME: Final = "vertex-gemini-router"
_REGIONAL: Final = "us-central1-aiplatform.googleapis.com:443"
_GLOBAL: Final = "aiplatform.googleapis.com:443"
_BYO: Final = "us-east5-aiplatform.googleapis.com:443"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_GEMINI_REPLY: Final[dict[str, JsonValue]] = {
    "candidates": [{"content": {"parts": [{"text": "scripted vertex"}], "role": "model"}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
}
_GEMINI_SSE: Final = (f"data: {json.dumps(_GEMINI_REPLY)}\r\n\r\n".encode(),)
_GEMINI_BODY: Final[dict[str, JsonValue]] = {
    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    "systemInstruction": {"parts": [{"text": "be terse"}]},
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 32},
}
_GEMINI_SDK_BODY: Final[dict[str, JsonValue]] = {
    **_GEMINI_BODY,
    "systemInstruction": {"parts": [{"text": "be terse"}], "role": "user"},
}
_CLAUDE_REPLY: Final[dict[str, JsonValue]] = {
    "id": "msg_scripted",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "scripted claude"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 2},
}
_CLAUDE_EVENTS: Final[tuple[dict[str, JsonValue], ...]] = (
    {"type": "message_start", "message": {**_CLAUDE_REPLY, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "scripted claude"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
)
_CLAUDE_SSE: Final = tuple(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in _CLAUDE_EVENTS
)


@dataclass(frozen=True, slots=True)
class _Tunnel:
    url: str
    authorities: SimpleQueue[str]

    def drain(self) -> list[str]:
        drained: Final[list[str]] = []
        with contextlib.suppress(Empty):
            while True:
                drained.append(self.authorities.get_nowait())
        return drained


def _tls_context(directory: Path) -> ssl.SSLContext:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    now: Final = datetime.datetime.now(datetime.UTC)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "aiplatform.googleapis.com")])
    certificate: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    certificate_file: Final = directory / "vertex.pem"
    key_file: Final = directory / "vertex.key"
    certificate_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate_file, key_file)
    return context


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@contextmanager
def _vertex_tunnel(destination: Wire) -> Generator[_Tunnel, None, None]:
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
            if authority not in (_REGIONAL, _GLOBAL, _BYO):
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.request.settimeout(10)
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=12)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Tunnel) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield _Tunnel(f"http://127.0.0.1:{server.server_address[1]}", authorities)
        finally:
            server.shutdown()
            thread.join(timeout=6)


def _vertex_peer(request: Request) -> Reply:
    if "enerateContent" in request.target:
        if ":streamGenerateContent" in request.target:
            return Reply(chunks=_GEMINI_SSE, content_type="text/event-stream")
        return Reply(body=json.dumps(_GEMINI_REPLY).encode())
    if ":streamRawPredict" in request.target:
        return Reply(chunks=_CLAUDE_SSE, content_type="text/event-stream")
    return Reply(body=json.dumps(_CLAUDE_REPLY).encode())


def _token_peer(request: Request) -> Reply:
    assert request.target == "/_oauth/token", request.target
    return Reply(body=json.dumps({"access_token": _TOKEN, "expires_in": 3600, "token_type": "Bearer"}).encode())


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: Gateway
    vertex: Wire
    tokens: Wire
    tunnel: _Tunnel


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("vertex-passthrough")
    with (
        wire_server(_vertex_peer, tls=_tls_context(directory)) as vertex,
        wire_server(_token_peer) as tokens,
        _vertex_tunnel(vertex) as tunnel,
    ):
        credentials: Final = service_account_json(_PROJECT, tokens.url)
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            *config.get("model_list", []),
            {
                "model_name": _GEMINI_ROUTER_NAME,
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-flash",
                    "vertex_project": _PROJECT,
                    "vertex_location": "us-central1",
                    "vertex_credentials": credentials,
                    "use_in_pass_through": True,
                },
            },
            {
                "model_name": "claude-sonnet-4-5",
                "litellm_params": {
                    "model": "vertex_ai/claude-sonnet-4-5",
                    "vertex_project": _PROJECT,
                    "vertex_location": "global",
                    "vertex_credentials": credentials,
                    "use_in_pass_through": True,
                },
            },
        ]
        path: Final = directory / "vertex-passthrough.yaml"
        path.write_text(yaml.safe_dump(config))
        environment: Final = {
            "HTTPS_PROXY": tunnel.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "AIOHTTP_TRUST_ENV": "True",
            "SSL_VERIFY": "False",
        }
        with (
            gateway_from_environment() as base,
            owned_proxy(base, directory, environment, config=path, workers=2) as proxy,
        ):
            yield _Rig(proxy, vertex, tokens, tunnel)


@pytest.fixture(autouse=True)
def _drained(rig: _Rig) -> None:
    rig.vertex.drain()
    rig.tunnel.drain()


def _assert_minted_token_and_no_virtual_key(upstream: Request, key: str) -> None:
    assert upstream.headers["authorization"] == f"Bearer {_TOKEN}", upstream.headers
    assert {name: value for name, value in upstream.headers.items() if key in value} == {}, upstream.headers
    assert key not in upstream.target, upstream.target


@pytest.mark.parametrize("alias", ["/vertex_ai", "/vertex-ai"])
def test_google_genai_vertex_calls_reach_the_router_deployment_with_a_minted_token(rig: _Rig, alias: str) -> None:
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        sent: Final[list[httpx.Request]] = []
        client: Final = genai.Client(
            vertexai=True,
            project=_PROJECT,
            location="us-central1",
            credentials=Credentials(token=key),
            http_options=types.HttpOptions(
                base_url=f"{str(rig.proxy.client.base_url).rstrip('/')}{alias}",
                client_args={"event_hooks": {"request": [sent.append]}, "timeout": 30, "trust_env": False},
            ),
        )
        generated: Final = client.models.generate_content(
            model=_GEMINI_ROUTER_NAME,
            contents="hi",
            config=types.GenerateContentConfig(
                system_instruction="be terse",
                temperature=0.2,
                max_output_tokens=32,
            ),
        )
        streamed: Final = "".join(
            chunk.text or ""
            for chunk in client.models.generate_content_stream(
                model=_GEMINI_ROUTER_NAME,
                contents="hi",
                config=types.GenerateContentConfig(
                    system_instruction="be terse",
                    temperature=0.2,
                    max_output_tokens=32,
                ),
            )
        )
        assert generated.text == "scripted vertex", generated
        assert streamed == "scripted vertex", streamed
        received: Final = rig.vertex.drain()
        model_path: Final = f"/v1beta1/projects/{_PROJECT}/locations/us-central1/publishers/google/models"
        assert [(request.method, request.target) for request in received] == [
            ("POST", f"{model_path}/gemini-2.5-flash:generateContent"),
            ("POST", f"{model_path}/gemini-2.5-flash:streamGenerateContent?alt=sse"),
        ], received
        assert rig.tunnel.drain() == [_REGIONAL, _REGIONAL]
        assert len(sent) == len(received), sent
        for upstream, sdk_request in zip(received, sent, strict=True):
            sdk_body: Final = _JSON_OBJECT.validate_json(sdk_request.content)
            assert _JSON_OBJECT.validate_json(upstream.body) == sdk_body, upstream.body
            assert sdk_body == _GEMINI_SDK_BODY, upstream.body
            _assert_minted_token_and_no_virtual_key(upstream, key)


def test_vertex_abbreviated_beta_route_uses_router_project_and_location(rig: _Rig) -> None:
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = rig.proxy.client.post(
            f"/vertex-ai/v1beta1/publishers/google/models/{_GEMINI_ROUTER_NAME}:generateContent",
            json=_GEMINI_BODY,
            headers={"Authorization": f"Bearer {key}"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _GEMINI_REPLY, response.text
        received: Final = rig.vertex.drain()
        assert [(request.method, request.target) for request in received] == [
            (
                "POST",
                f"/v1beta1/projects/{_PROJECT}/locations/us-central1/publishers/google/models/"
                "gemini-2.5-flash:generateContent",
            )
        ], received
        assert rig.tunnel.drain() == [_REGIONAL]
        assert _JSON_OBJECT.validate_json(received[0].body) == _GEMINI_BODY, received[0].body
        _assert_minted_token_and_no_virtual_key(received[0], key)


@pytest.mark.parametrize("alias", ["/vertex_ai", "/vertex-ai"])
def test_vertex_rest_spellings_mint_token_and_add_sse_query_upstream(rig: _Rig, alias: str) -> None:
    model_path: Final = f"/v1/projects/{_PROJECT}/locations/us-central1/publishers/google/models"
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        headers: Final = {"Authorization": f"Bearer {key}"}
        generated: Final = rig.proxy.client.post(
            f"{alias}{model_path}/{_GEMINI_ROUTER_NAME}:generateContent", json=_GEMINI_BODY, headers=headers
        )
        assert generated.status_code == 200, generated.text
        assert generated.json() == _GEMINI_REPLY, generated.text
        with rig.proxy.client.stream(
            "POST",
            f"{alias}{model_path}/{_GEMINI_ROUTER_NAME}:streamGenerateContent",
            json=_GEMINI_BODY,
            headers=headers,
        ) as streamed_response:
            streamed: Final = streamed_response.read()
            assert streamed_response.status_code == 200, streamed
        assert streamed == b"".join(_GEMINI_SSE), streamed
        received: Final = rig.vertex.drain()
        assert [(request.method, request.target) for request in received] == [
            ("POST", f"{model_path}/gemini-2.5-flash:generateContent"),
            ("POST", f"{model_path}/gemini-2.5-flash:streamGenerateContent?alt=sse"),
        ], received
        assert rig.tunnel.drain() == [_REGIONAL, _REGIONAL]
        for upstream in received:
            assert _JSON_OBJECT.validate_json(upstream.body) == _GEMINI_BODY, upstream.body
            _assert_minted_token_and_no_virtual_key(upstream, key)


@pytest.mark.parametrize("alias", ["/vertex_ai", "/vertex-ai"])
@pytest.mark.parametrize("stream", [False, True], ids=["rawPredict", "streamRawPredict"])
def test_anthropic_vertex_sdk_partner_call_on_the_global_location_reaches_upstream_intact(
    rig: _Rig, alias: str, stream: bool
) -> None:
    sent: Final[list[httpx.Request]] = []
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        client: Final = anthropic.AnthropicVertex(
            base_url=f"{str(rig.proxy.client.base_url).rstrip('/')}{alias}/v1",
            region="global",
            project_id=_PROJECT,
            access_token=key,
            max_retries=0,
            http_client=httpx.Client(timeout=30, trust_env=False, event_hooks={"request": [sent.append]}),
        )
        arguments: Final[dict[str, JsonValue]] = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        }
        if stream:
            with client.messages.stream(**arguments) as events:  # type: ignore[arg-type]
                text = events.get_final_text()
        else:
            text = "".join(
                block.text
                for block in client.messages.create(**arguments).content  # type: ignore[arg-type]
                if block.type == "text"
            )
        client.close()
        assert text == "scripted claude"
        action: Final = "streamRawPredict" if stream else "rawPredict"
        model_path: Final = f"/v1/projects/{_PROJECT}/locations/global/publishers/anthropic/models/claude-sonnet-4-5"
        received: Final = rig.vertex.drain()
        assert [(request.method, request.target) for request in received] == [
            ("POST", f"{model_path}:{action}" + ("?alt=sse" if stream else ""))
        ], received
        assert rig.tunnel.drain() == [_GLOBAL]
        upstream: Final = received[0]
        assert _JSON_OBJECT.validate_json(upstream.body) == _JSON_OBJECT.validate_json(sent[0].content), upstream.body
        _assert_minted_token_and_no_virtual_key(upstream, key)


def test_credential_less_vertex_project_forwards_the_caller_google_token_and_strips_the_virtual_key(
    rig: _Rig,
) -> None:
    model_path: Final = "/v1/projects/byo-project/locations/us-east5/publishers/anthropic/models/claude-byo"
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final[dict[str, JsonValue]] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        response: Final = rig.proxy.client.post(
            f"/vertex_ai{model_path}:rawPredict",
            json=body,
            headers={"Authorization": f"Bearer {key}", "x-goog-api-key": _CALLER_GOOGLE_TOKEN},
        )
        assert response.status_code == 200, response.text
        assert response.json() == _CLAUDE_REPLY, response.text
        received: Final = rig.vertex.drain()
        assert [(request.method, request.target) for request in received] == [("POST", f"{model_path}:rawPredict")], (
            received
        )
        assert rig.tunnel.drain() == [_BYO]
        upstream: Final = received[0]
        assert upstream.headers["x-goog-api-key"] == _CALLER_GOOGLE_TOKEN, upstream.headers
        assert {name: value for name, value in upstream.headers.items() if key in value} == {}, upstream.headers
        assert "authorization" not in upstream.headers, upstream.headers
        assert "x-litellm-api-key" not in upstream.headers, upstream.headers
        assert _JSON_OBJECT.validate_json(upstream.body) == body, upstream.body


@pytest.mark.parametrize("alias", ["/vertex_ai", "/vertex-ai"])
def test_credential_less_vertex_project_forwards_the_caller_bearer_token_when_the_virtual_key_rides_in_x_litellm_api_key(
    rig: _Rig, alias: str
) -> None:
    model_path: Final = "/v1/projects/byo-project/locations/us-east5/publishers/anthropic/models/claude-byo"
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final[dict[str, JsonValue]] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        }
        response: Final = rig.proxy.client.post(
            f"{alias}{model_path}:rawPredict",
            json=body,
            headers={
                "x-litellm-api-key": f"Bearer {key}",
                "Authorization": f"Bearer {_CALLER_GOOGLE_TOKEN}",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == _CLAUDE_REPLY, response.text
        received: Final = rig.vertex.drain()
        assert [(request.method, request.target) for request in received] == [
            ("POST", f"{model_path}:rawPredict")
        ], received
        assert rig.tunnel.drain() == [_BYO]
        upstream: Final = received[0]
        assert upstream.headers["authorization"] == f"Bearer {_CALLER_GOOGLE_TOKEN}", upstream.headers
        assert {name: value for name, value in upstream.headers.items() if key in value} == {}, upstream.headers
        assert "x-litellm-api-key" not in upstream.headers, upstream.headers
        assert "x-goog-api-key" not in upstream.headers, upstream.headers
        assert key not in upstream.target, upstream.target
        assert _JSON_OBJECT.validate_json(upstream.body) == body, upstream.body
