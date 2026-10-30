import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_KEY: Final = "synthetic-passthrough-key"
_GOOGLE_TOKEN: Final = "synthetic-google-access-token"
_VERTEX_HOST: Final = "us-central1-aiplatform.googleapis.com"
_VERTEX_MODEL: Final = "gemini-2.5-flash"
_FRAMES: Final = (
    b'data: {"id":"pt-1","choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
    b'data: {"id":"pt-1","choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
)
_GEMINI_FRAMES: Final = tuple(
    f"data: {json.dumps(frame)}\r\n\r\n".encode()
    for frame in (
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}, "index": 0}]},
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "lo"}]}, "index": 0, "finishReason": "STOP"}
            ],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
            "modelVersion": _VERTEX_MODEL,
        },
    )
)
_ROUTES: Final = MappingProxyType(
    {
        "openai": ("/openai/chat/completions", "/chat/completions"),
        "mistral": ("/mistral/v1/chat/completions", "/v1/chat/completions"),
        "cohere": ("/cohere/v2/chat", "/v2/chat"),
    }
)


@dataclass(frozen=True, slots=True)
class _Relay:
    wire: Wire
    vertex: Wire
    proxy: Gateway


def _sse(request: Request) -> Reply:
    return Reply(content_type="text/event-stream", chunks=_FRAMES)


def _gemini_sse(request: Request) -> Reply:
    return Reply(content_type="text/event-stream", chunks=_GEMINI_FRAMES)


@pytest.fixture(scope="module")
def relay(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Relay]:
    directory: Final = tmp_path_factory.mktemp("passthrough-stream")
    certificate, key = write_self_signed_cert(directory, (_VERTEX_HOST,))
    with (
        wire_server(_sse) as wire,
        wire_server(_gemini_sse, tls=server_context(certificate, key)) as vertex,
        tunnelling_forward_proxy(int(vertex.url.rsplit(":", 1)[1])) as tunnel,
        gateway_from_environment() as rig,
    ):
        bases: Final = {f"{name}_API_BASE": wire.url for name in ("OPENAI", "MISTRAL", "COHERE")}
        keys: Final = {f"{name}_API_KEY": _KEY for name in ("OPENAI", "MISTRAL", "COHERE")}
        tunnelled: Final = {
            "HTTPS_PROXY": tunnel.url,
            "https_proxy": tunnel.url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "SSL_CERT_FILE": str(certificate),
        }
        with owned_proxy(rig, directory, {**bases, **keys, **tunnelled}, workers=2) as proxy:
            yield _Relay(wire, vertex, proxy)


def _relayed(
    proxy: Gateway, path: str, body: dict[str, JsonValue], headers: Mapping[str, str] | None = None
) -> tuple[int, bytes]:
    sent: Final = headers or {"Authorization": f"Bearer {proxy.key}"}
    with proxy.client.stream("POST", path, json=body, headers=dict(sent)) as response:
        return response.status_code, b"".join(response.iter_raw())


def _only_post(wire: Wire) -> Request:
    requests: Final = wire.drain()
    assert [request.method for request in requests] == ["POST"], [request.target for request in requests]
    return requests[0]


@pytest.mark.parametrize("route", tuple(_ROUTES))
def test_a_provider_passthrough_stream_is_relayed_frame_for_frame(relay: _Relay, route: str) -> None:
    proxy_path, upstream_path = _ROUTES[route]
    marker: Final = f"passthrough-stream-{uuid.uuid4().hex}"
    relayed: Final = _relayed(
        relay.proxy,
        proxy_path,
        {"model": "relay-model", "stream": True, "messages": [{"role": "user", "content": marker}]},
    )
    assert relayed == (200, b"".join(_FRAMES)), relayed
    forwarded: Final = _only_post(relay.wire)
    assert urlsplit(forwarded.target).path == upstream_path, forwarded.target
    assert marker.encode() in forwarded.body, forwarded.body[:400]


def test_a_vertex_passthrough_stream_is_relayed_frame_for_frame(relay: _Relay) -> None:
    project: Final = f"relay-{uuid.uuid4().hex[:12]}"
    model_path: Final = f"/v1/projects/{project}/locations/us-central1/publishers/google/models/{_VERTEX_MODEL}"
    marker: Final = f"vertex-passthrough-stream-{uuid.uuid4().hex}"
    relayed: Final = _relayed(
        relay.proxy,
        f"/vertex_ai{model_path}:streamGenerateContent",
        {"contents": [{"role": "user", "parts": [{"text": marker}]}]},
        {"x-litellm-api-key": relay.proxy.key, "Authorization": f"Bearer {_GOOGLE_TOKEN}"},
    )
    assert relayed == (200, b"".join(_GEMINI_FRAMES)), relayed
    forwarded: Final = _only_post(relay.vertex)
    assert forwarded.target == f"{model_path}:streamGenerateContent?alt=sse", forwarded.target
    assert forwarded.headers["authorization"] == f"Bearer {_GOOGLE_TOKEN}", forwarded.headers
    assert marker.encode() in forwarded.body, forwarded.body[:400]


def test_an_azure_passthrough_stream_to_a_router_model_is_relayed_frame_for_frame(gateway: Gateway) -> None:
    with wire_server(_sse) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="azure/relay-deployment", api_base=wire.url, api_version="2024-10-21")
        marker: Final = f"azure-passthrough-stream-{uuid.uuid4().hex}"
        relayed: Final = _relayed(
            gateway,
            f"/azure/openai/deployments/{model}/chat/completions?api-version=2024-10-21",
            {"stream": True, "messages": [{"role": "user", "content": marker}]},
        )
        assert relayed == (200, b"".join(_FRAMES)), relayed
        forwarded: Final = _only_post(wire)
        assert urlsplit(forwarded.target).path.endswith("/chat/completions"), forwarded.target
        assert marker.encode() in forwarded.body, forwarded.body[:400]
