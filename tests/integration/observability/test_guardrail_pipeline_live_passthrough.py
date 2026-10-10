import base64
import contextlib
import json
import re
import socket
import socketserver
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held, object_value
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.upstream import aws_event_stream_frame
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

SECRET_WORD: Final = "pipelinesecretword"
REWRITTEN: Final = "[PIPELINE_REDACTED]"
TOKEN: Final = re.compile(rb"plive-[0-9a-f]{32}-\d+")
REGION: Final = "us-west-2"
VERTEX_HOST: Final = "us-central1-aiplatform.googleapis.com"
AGENT_RUNTIME_HOST: Final = f"bedrock-agent-runtime.{REGION}.amazonaws.com"
TUNNEL_HOSTS: Final = frozenset((VERTEX_HOST, AGENT_RUNTIME_HOST))
BEDROCK_MODEL: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
VERTEX_PROJECT: Final = "plive-pass-through"
SSE: Final = "text/event-stream"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"


def _sse(payload: Mapping[str, JsonValue], event: str | None = None) -> bytes:
    prefix: Final = b"" if event is None else f"event: {event}\n".encode()
    return prefix + b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chat_frames(text: str) -> tuple[bytes, ...]:
    def frame(delta: Mapping[str, JsonValue], finish: str | None) -> bytes:
        return _sse(
            {
                "id": "chatcmpl-plive",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish}],
            }
        )

    return (
        frame({"role": "assistant", "content": text}, None),
        frame({"content": " tail"}, "stop"),
        b"data: [DONE]\n\n",
    )


def _anthropic_frames(text: str) -> tuple[bytes, ...]:
    message: Final = {
        "id": "msg_plive",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [],
        "stop_reason": None,
        "usage": {"input_tokens": 5, "output_tokens": 0},
    }
    return (
        _sse({"type": "message_start", "message": message}, "message_start"),
        _sse(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            "content_block_start",
        ),
        _sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            "content_block_delta",
        ),
        _sse(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " tail"}},
            "content_block_delta",
        ),
        _sse({"type": "content_block_stop", "index": 0}, "content_block_stop"),
        _sse(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
            "message_delta",
        ),
        _sse({"type": "message_stop"}, "message_stop"),
    )


def _gemini_frames(text: str) -> tuple[bytes, ...]:
    def frame(piece: str, finish: str | None) -> bytes:
        candidate: Final = {"content": {"role": "model", "parts": [{"text": piece}]}, "index": 0}
        return _sse(
            {
                "candidates": [candidate if finish is None else {**candidate, "finishReason": finish}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9},
                "modelVersion": "gemini-2.5-flash",
            }
        )

    return (frame(text, None), frame(" tail", "STOP"))


def _watsonx_frames(text: str) -> tuple[bytes, ...]:
    def frame(piece: str, reason: str) -> bytes:
        return _sse(
            {
                "model_id": "ibm/granite-3-8b-instruct",
                "results": [
                    {
                        "generated_text": piece,
                        "generated_token_count": 2,
                        "input_token_count": 5,
                        "stop_reason": reason,
                    }
                ],
            },
            "message",
        )

    return (frame(text, "not_finished"), frame(" tail", "eos_token"))


def _event(kind: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": kind, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def _converse_frames(text: str) -> tuple[bytes, ...]:
    return (
        _event("messageStart", {"role": "assistant"}),
        _event("contentBlockDelta", {"delta": {"text": text}, "contentBlockIndex": 0}),
        _event("contentBlockDelta", {"delta": {"text": " tail"}, "contentBlockIndex": 0}),
        _event("messageStop", {"stopReason": "end_turn"}),
        _event("metadata", {"usage": {"inputTokens": 5, "outputTokens": 4, "totalTokens": 9}}),
    )


def _agent_frames(text: str) -> tuple[bytes, ...]:
    return tuple(_event("chunk", {"bytes": base64.b64encode(piece.encode()).decode()}) for piece in (text, " tail"))


def _frames(target: str, text: str) -> tuple[tuple[bytes, ...], str]:
    path: Final = target.split("?", 1)[0]
    if path.endswith("/converse-stream"):
        return _converse_frames(text), EVENT_STREAM
    if path.startswith("/agents/"):
        return _agent_frames(text), EVENT_STREAM
    if path.endswith(":streamGenerateContent"):
        return _gemini_frames(text), SSE
    if path.endswith("/v1/messages"):
        return _anthropic_frames(text), SSE
    if path.endswith("/generation_stream"):
        return _watsonx_frames(text), SSE
    return _chat_frames(text), SSE


def _token_of(body: bytes) -> str:
    found: Final = TOKEN.search(body)
    assert found is not None, body
    return found.group().decode()


def _provider_text(token: str) -> str:
    return f"{SECRET_WORD} {token}"


def _expected(target: str, token: str) -> bytes:
    chunks, _content_type = _frames(target, _provider_text(token))
    return b"".join(chunks)


class _Gates:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.held: Mapping[str, threading.Event] = MappingProxyType({})

    def hold(self, token: str) -> threading.Event:
        gate: Final = threading.Event()
        with self._lock:
            self.held = MappingProxyType({**self.held, token: gate})
        return gate


class _Received:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.pieces: tuple[bytes, ...] = ()

    def add(self, piece: bytes) -> None:
        with self._lock:
            self.pieces = (*self.pieces, piece)

    @property
    def content(self) -> bytes:
        return b"".join(self.pieces)


def _replying(gates: _Gates) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        if request.target.startswith("/_oauth/token"):
            return Reply(
                body=b'{"access_token": "synthetic-vertex-access-token", "expires_in": 3600, "token_type": "Bearer"}'
            )
        token: Final = _token_of(request.body)
        chunks, content_type = _frames(request.target, _provider_text(token))
        return Reply(chunks=chunks, content_type=content_type, gate_after_first=gates.held.get(token))

    return reply


def _rewrite(request: Request) -> Reply:
    texts: Final = object_value(json.loads(request.body)).get("texts") or []
    assert isinstance(texts, list), request.body
    rewritten: Final = [str(text).replace(SECRET_WORD, REWRITTEN) for text in texts]
    return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": rewritten}).encode())


class _Log:
    def __init__(self, wire: Wire) -> None:
        self.wire: Final = wire
        self._lock: Final = threading.Lock()
        self.seen: tuple[Request, ...] = ()

    def matching(self, token: str) -> tuple[Request, ...]:
        with self._lock:
            self.seen = (*self.seen, *self.wire.drain())
            return tuple(request for request in self.seen if token.encode() in request.body)


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@contextlib.contextmanager
def _connect_tunnel(port: int) -> Iterator[str]:
    class Handler(socketserver.StreamRequestHandler):
        rbufsize = 0
        request: socket.socket

        def handle(self) -> None:
            line: Final = self.rfile.readline().decode().split()
            authority: Final = line[1] if len(line) > 1 else ""
            while self.rfile.readline() not in (b"\r\n", b""):
                pass
            if authority.rsplit(":", 1)[0] not in TUNNEL_HOSTS:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.request.settimeout(15)
            with socket.create_connection(("127.0.0.1", port), timeout=15) as upstream:
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=16)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=6)


@dataclass(frozen=True, slots=True)
class _Models:
    azure: str
    bedrock: str
    nim: str
    vllm: str
    gigachat: str


@dataclass(frozen=True, slots=True)
class _PassRig:
    owned: OwnedProxy
    upstream: _Log
    tunneled: _Log
    peer: _Log
    rail: str
    models: _Models
    gates: _Gates


def _model_list(models: _Models, upstream: str) -> list[JsonValue]:
    return [
        {
            "model_name": models.azure,
            "litellm_params": {
                "model": f"azure/{models.azure}",
                "api_base": upstream,
                "api_key": "synthetic-azure-key",
                "api_version": "2024-10-21",
            },
        },
        {
            "model_name": models.bedrock,
            "litellm_params": {
                "model": f"bedrock/{BEDROCK_MODEL}",
                "aws_bedrock_runtime_endpoint": upstream,
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": REGION,
            },
        },
        {
            "model_name": models.nim,
            "litellm_params": {
                "model": "nvidia_nim/meta/llama-3.1-8b-instruct",
                "api_base": f"{upstream}/v1",
                "api_key": "synthetic-nim-key",
            },
        },
        {
            "model_name": models.vllm,
            "litellm_params": {"model": "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct", "api_base": upstream},
        },
        {
            "model_name": models.gigachat,
            "litellm_params": {"model": "gigachat/GigaChat-2-Max", "api_base": upstream, "api_key": "synthetic"},
        },
    ]


def _environment(upstream: str, credentials: Path) -> dict[str, JsonValue]:
    return {
        "ANTHROPIC_API_BASE": upstream,
        "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
        "GEMINI_API_BASE": upstream,
        "GEMINI_API_KEY": "synthetic-gemini-key",
        "OPENAI_API_BASE": upstream,
        "OPENAI_API_KEY": "synthetic-openai-key",
        "AZURE_API_BASE": upstream,
        "AZURE_API_KEY": "synthetic-azure-key",
        "AWS_BEDROCK_RUNTIME_ENDPOINT": upstream,
        "AWS_ACCESS_KEY_ID": "AKIASCRIPTEDPROVIDER",
        "AWS_SECRET_ACCESS_KEY": "scripted-secret",
        "AWS_REGION_NAME": REGION,
        "COHERE_API_BASE": upstream,
        "COHERE_API_KEY": "synthetic-cohere-key",
        "MISTRAL_API_BASE": upstream,
        "MISTRAL_API_KEY": "synthetic-mistral-key",
        "OPENROUTER_API_BASE": f"{upstream}/api/v1",
        "OPENROUTER_API_KEY": "synthetic-openrouter-key",
        "VLLM_API_BASE": upstream,
        "WATSONX_API_BASE": upstream,
        "WATSONX_TOKEN": "synthetic-watsonx-token",
        "GIGACHAT_API_BASE": upstream,
        "GIGACHAT_ACCESS_TOKEN": "synthetic-gigachat-token",
        "TINYFISH_AGENT_API_BASE": upstream,
        "TINYFISH_API_KEY": "synthetic-tinyfish-key",
        "DEFAULT_VERTEXAI_PROJECT": VERTEX_PROJECT,
        "DEFAULT_VERTEXAI_LOCATION": "us-central1",
        "DEFAULT_GOOGLE_APPLICATION_CREDENTIALS": str(credentials),
    }


def _config(rail: str, peer: str, upstream: str, models: _Models, credentials: Path) -> dict[str, JsonValue]:
    base: Final = object_value(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    return {
        **base,
        "environment_variables": {
            **object_value(base.get("environment_variables") or {}),
            **_environment(upstream, credentials),
        },
        "general_settings": {
            **object_value(base["general_settings"]),
            "pass_through_endpoints": [
                {"path": "/plive-config-route", "target": f"{upstream}/configured", "auth": True}
            ],
        },
        "guardrails": [
            {
                "guardrail_name": rail,
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "post_call",
                    "default_on": False,
                    "api_base": f"{peer}/{rail}",
                    "api_key": "synthetic-guardrail-key",
                    "streaming_buffer_until_moderated": False,
                },
            }
        ],
        "model_list": _model_list(models, upstream),
        "policies": {
            f"{rail}-policy": {
                "guardrails": {"add": [rail]},
                "pipeline": {
                    "mode": "post_call",
                    "steps": [{"guardrail": rail, "on_pass": "allow", "on_fail": "next"}],
                },
            }
        },
        "policy_attachments": [{"policy": f"{rail}-policy", "scope": "*"}],
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_PassRig]:
    directory: Final = tmp_path_factory.mktemp("pipeline-live-passthrough")
    suffix: Final = uuid.uuid4().hex[:8]
    rail: Final = f"plive-pt-{suffix}"
    models: Final = _Models(*(f"plive-pt-{kind}-{suffix}" for kind in ("azure", "bedrock", "nim", "vllm", "gigachat")))
    certificate, key = write_self_signed_cert(directory, tuple(TUNNEL_HOSTS))
    gates: Final = _Gates()
    with (
        gateway_from_environment() as root,
        wire_server(_replying(gates)) as upstream,
        wire_server(_replying(gates), tls=server_context(certificate, key)) as tunneled,
        wire_server(_rewrite) as peer,
        _connect_tunnel(int(tunneled.url.rsplit(":", 1)[1])) as tunnel,
    ):
        credentials: Final = directory / "vertex-service-account.json"
        credentials.write_text(service_account_json(VERTEX_PROJECT, upstream.url))
        config: Final = directory / "pipeline-live-passthrough.yaml"
        config.write_text(yaml.safe_dump(_config(rail, peer.url, upstream.url, models, credentials)))
        overrides: Final = {
            "HTTPS_PROXY": tunnel,
            "https_proxy": tunnel,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "SSL_CERT_FILE": str(certificate),
        }
        with owned_proxy_process(root, directory, overrides, config=config, workers=2) as owned:
            yield _PassRig(owned, _Log(upstream), _Log(tunneled), _Log(peer), rail, models, gates)


@dataclass(frozen=True, slots=True)
class _Route:
    path: str
    body: Mapping[str, JsonValue]
    upstream_target: str
    tunneled: bool = False
    headers: Mapping[str, str] = MappingProxyType({})


def _chat_body(prompt: str, model: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}


def _gemini_body(prompt: str) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}


VERTEX_TARGET: Final = (
    f"/v1/projects/{VERTEX_PROJECT}/locations/us-central1"
    "/publishers/google/models/gemini-2.5-flash:streamGenerateContent"
)


def _route(row: str, prompt: str, models: _Models, key: str) -> _Route:
    converse: Final = {"messages": [{"role": "user", "content": [{"text": prompt}]}]}
    azure_path: Final = "/openai/deployments/plive-direct/chat/completions"
    azure_target: Final = f"{azure_path}?api-version=2024-10-21"
    match row:
        case "G1-anthropic":
            body: Final = {**_chat_body(prompt, "claude-sonnet-4-5"), "max_tokens": 32}
            return _Route("/anthropic/v1/messages", body, "/v1/messages")
        case "G2-gemini":
            return _Route(
                "/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
                _gemini_body(prompt),
                "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
                headers=MappingProxyType({"x-goog-api-key": key}),
            )
        case "G3-openai":
            return _Route("/openai/v1/chat/completions", _chat_body(prompt, "gpt-4o-mini"), "/v1/chat/completions")
        case "G4-openai-passthrough":
            return _Route(
                "/openai_passthrough/v1/chat/completions", _chat_body(prompt, "gpt-4o-mini"), "/v1/chat/completions"
            )
        case "G5-azure-direct":
            return _Route(f"/azure{azure_target}", _chat_body(prompt, "plive-direct"), azure_path)
        case "G6-azure-router-model":
            return _Route(
                f"/azure/openai/deployments/{models.azure}/chat/completions?api-version=2024-10-21",
                {"messages": [{"role": "user", "content": prompt}], "stream": True},
                f"/openai/deployments/{models.azure}/chat/completions",
            )
        case "G7-azure-ai-direct":
            return _Route(f"/azure_ai{azure_target}", _chat_body(prompt, "plive-direct"), azure_path)
        case "G9-bedrock-router-model":
            return _Route(
                f"/bedrock/model/{models.bedrock}/converse-stream", converse, f"/model/{BEDROCK_MODEL}/converse-stream"
            )
        case "G10-bedrock-agent-runtime":
            agent_path: Final = f"/agents/PLIVEAGENT/agentAliases/PLIVEALIAS/sessions/{uuid.uuid4().hex}/text"
            return _Route(f"/bedrock{agent_path}", {"inputText": prompt}, agent_path, tunneled=True)
        case "G11-vertex-ai":
            return _Route(f"/vertex_ai{VERTEX_TARGET}?alt=sse", _gemini_body(prompt), VERTEX_TARGET, tunneled=True)
        case "G12-vertex-ai-dash-alias":
            return _Route(f"/vertex-ai{VERTEX_TARGET}?alt=sse", _gemini_body(prompt), VERTEX_TARGET, tunneled=True)
        case "G13-cohere":
            return _Route("/cohere/v2/chat", _chat_body(prompt, "command-a-03-2025"), "/v2/chat")
        case "G14-mistral":
            return _Route(
                "/mistral/v1/chat/completions", _chat_body(prompt, "mistral-large-latest"), "/v1/chat/completions"
            )
        case "G15-nvidia-nim-router-model":
            return _Route(
                f"/nvidia_nim/{models.nim}/v1/chat/completions",
                _chat_body(prompt, "meta/llama-3.1-8b-instruct"),
                "/v1/chat/completions",
            )
        case "G16-openrouter":
            return _Route(
                "/openrouter/v1/chat/completions", _chat_body(prompt, "openai/gpt-4o-mini"), "/api/v1/chat/completions"
            )
        case "G17-vllm-router-model":
            return _Route("/vllm/v1/chat/completions", _chat_body(prompt, models.vllm), "/v1/chat/completions")
        case "G18-vllm-direct":
            return _Route(
                "/vllm/v1/chat/completions",
                _chat_body(prompt, "meta-llama/Llama-3.1-8B-Instruct"),
                "/v1/chat/completions",
            )
        case "G19-watsonx":
            return _Route(
                "/watsonx/ml/v1/text/generation_stream?version=2024-03-13",
                {"input": prompt, "model_id": "ibm/granite-3-8b-instruct", "project_id": "plive-project"},
                "/ml/v1/text/generation_stream",
            )
        case "G20-gigachat-direct":
            return _Route("/gigachat/chat/completions", _chat_body(prompt, "GigaChat-2-Max"), "/chat/completions")
        case "G21-gigachat-router-model":
            return _Route("/gigachat/chat/completions", _chat_body(prompt, models.gigachat), "/chat/completions")
        case "G22-tinyfish-run-sse":
            return _Route(
                "/tinyfish/v1/automation/run-sse",
                {"url": "https://example.com", "goal": prompt},
                "/v1/automation/run-sse",
            )
        case "G23-config-pass-through-endpoint":
            return _Route("/plive-config-route", _chat_body(prompt, "gpt-4o-mini"), "/configured")
    raise AssertionError(row)


ROWS: Final = (
    "G1-anthropic",
    "G2-gemini",
    "G3-openai",
    "G4-openai-passthrough",
    "G5-azure-direct",
    "G6-azure-router-model",
    "G7-azure-ai-direct",
    "G10-bedrock-agent-runtime",
    "G11-vertex-ai",
    "G12-vertex-ai-dash-alias",
    "G13-cohere",
    "G14-mistral",
    "G15-nvidia-nim-router-model",
    "G16-openrouter",
    "G17-vllm-router-model",
    "G18-vllm-direct",
    "G19-watsonx",
    "G20-gigachat-direct",
    "G21-gigachat-router-model",
    "G22-tinyfish-run-sse",
    "G23-config-pass-through-endpoint",
)


@dataclass(frozen=True, slots=True)
class _Sent:
    token: str
    route: _Route
    released_while_held: bytes
    content: bytes


def _stream(gateway: Gateway, route: _Route, received: _Received) -> int:
    headers: Final = {"Authorization": f"Bearer {gateway.key}", **route.headers}
    with gateway.client.stream("POST", route.path, json=dict(route.body), headers=headers) as response:
        for piece in response.iter_bytes():
            received.add(piece)
        return response.status_code


def _prepared(rig: _PassRig, row: str) -> tuple[str, _Route]:
    token: Final = f"plive-{uuid.uuid4().hex}-0"
    return token, _route(row, f"synthetic prompt {token}", rig.models, rig.owned.gateway.key)


def _assert_reached_the_vendor(rig: _PassRig, row: str, sent: tuple[str, _Route], status: int, body: bytes) -> None:
    token, route = sent
    log: Final = rig.tunneled if route.tunneled else rig.upstream
    targets: Final = tuple(request.target.split("?", 1)[0] for request in log.matching(token))
    assert status == 200, (row, status, body, targets)
    assert targets == (route.upstream_target,), targets


def _send(rig: _PassRig, row: str) -> _Sent:
    token, route = _prepared(rig, row)
    first_chunk: Final = _frames(route.upstream_target, _provider_text(token))[0][0]
    gate: Final = rig.gates.hold(token)
    received: Final = _Received()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Final = pool.submit(_stream, rig.owned.gateway, route, received)
        try:
            early: Final = eventually(
                lambda: received.content,
                lambda content: content.startswith(first_chunk),
                seconds=8,
                return_last_on_timeout=True,
            )
        finally:
            gate.set()
        status: Final = future.result(timeout=60)
    _assert_reached_the_vendor(rig, row, (token, route), status, received.content)
    return _Sent(token, route, early, received.content)


@pytest.mark.parametrize("row", ROWS)
def test_g_a_live_pipeline_leaves_every_pass_through_stream_live_unscanned_and_byte_identical(
    rig: _PassRig, row: str
) -> None:
    sent: Final = _send(rig, row)
    first_chunk: Final = _frames(sent.route.upstream_target, _provider_text(sent.token))[0][0]
    observed: Final = (
        sent.released_while_held.startswith(first_chunk),
        sent.content == _expected(sent.route.upstream_target, sent.token),
    )
    assert observed == (True, True), (row, sent)
    held(lambda: rig.peer.matching(sent.token), lambda scans: not scans, holding=3)


def test_g9_a_bedrock_router_model_converse_stream_keeps_buffering_for_the_pipeline_rewrite(rig: _PassRig) -> None:
    token, route = _prepared(rig, "G9-bedrock-router-model")
    response: Final = rig.owned.gateway.request("POST", route.path, route.body, headers=route.headers)
    _assert_reached_the_vendor(rig, "G9-bedrock-router-model", (token, route), response.status_code, response.content)
    content: Final = response.content
    assert f"{REWRITTEN} {token}".encode() in content and SECRET_WORD.encode() not in content, content
    held(lambda: len(rig.peer.matching(token)), lambda scans: scans == 1, holding=3)
