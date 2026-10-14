import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server

_LOCATION: Final = "us-central1"
_HOST: Final = f"{_LOCATION}-aiplatform.googleapis.com"
_PROJECT: Final = "audit-vertex-pass-through"
_STREAMING_MODEL: Final = "gemini-3.8-flash"
_UNAVAILABLE_MODEL: Final = "gemini-3.8-flash-unavailable"
_VENDOR_TOKEN: Final = "Bearer synthetic-vertex-token"
_CHUNKS: Final = (
    b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}]}\r\n\r\n',
    b'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}, "finishReason": "STOP"}],'
    b' "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}}\r\n\r\n',
)
_UNAVAILABLE_MESSAGE: Final = "The service is currently unavailable."
_UNAVAILABLE_BODY: Final = json.dumps(
    {"error": {"code": 503, "message": _UNAVAILABLE_MESSAGE, "status": "UNAVAILABLE"}}
).encode()
_SCRUBBED: Final = frozenset(
    {
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "DEFAULT_VERTEXAI_PROJECT",
        "DEFAULT_VERTEXAI_LOCATION",
        "DEFAULT_GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)


@dataclass(frozen=True, slots=True)
class _Vertex:
    wire: Wire
    tunnel: ForwardProxy
    proxy: Gateway

    def stream(self, model: str, marker: str) -> httpx.Response:
        return self.proxy.request(
            "POST",
            f"/vertex_ai/v1/{_model_path(model)}:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": marker}]}]},
            headers={"Authorization": _VENDOR_TOKEN, "x-litellm-api-key": f"Bearer {self.proxy.key}"},
        )


def _model_path(model: str) -> str:
    return f"projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{model}"


def _respond(request: Request) -> Reply:
    if request.target.startswith(f"/v1/{_model_path(_UNAVAILABLE_MODEL)}:"):
        return Reply(status=503, body=_UNAVAILABLE_BODY)
    if request.target.startswith(f"/v1/{_model_path(_STREAMING_MODEL)}:"):
        return Reply(content_type="text/event-stream", chunks=_CHUNKS)
    return Reply(status=404, body=b'{"error": "no scripted Vertex model"}')


@pytest.fixture(scope="module")
def vertex(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Vertex]:
    directory: Final = tmp_path_factory.mktemp("vertex-pass-through-stream")
    cert, key = write_self_signed_cert(directory, (_HOST,))
    with (
        wire_server(_respond, tls=server_context(cert, key)) as wire,
        tunnelling_forward_proxy(int(wire.url.rsplit(":", 1)[1])) as tunnel,
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            directory,
            {"HTTPS_PROXY": tunnel.url, "NO_PROXY": "127.0.0.1,localhost", "SSL_CERT_FILE": str(cert)},
            remove_environment=tuple(name for name in os.environ if name.upper() in _SCRUBBED),
            workers=2,
        ) as proxy,
    ):
        yield _Vertex(wire, tunnel, proxy)


@pytest.fixture(autouse=True)
def fresh_edges(vertex: _Vertex) -> None:
    vertex.wire.drain()
    vertex.tunnel.drain()


def test_vertex_pass_through_stream_relays_every_sse_chunk_byte_for_byte(vertex: _Vertex) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = vertex.stream(_STREAMING_MODEL, marker)
    sent: Final = vertex.wire.drain()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    assert response.content == b"".join(_CHUNKS), response.text
    assert [request.target for request in sent] == [
        f"/v1/{_model_path(_STREAMING_MODEL)}:streamGenerateContent?alt=sse"
    ], sent
    assert sent[0].headers["authorization"] == _VENDOR_TOKEN
    assert json.loads(sent[0].body) == {"contents": [{"role": "user", "parts": [{"text": marker}]}]}
    assert set(vertex.tunnel.targets()) == {f"{_HOST}:443"}


def test_vertex_pass_through_stream_relays_an_upstream_503_as_503(vertex: _Vertex) -> None:
    response: Final = vertex.stream(_UNAVAILABLE_MODEL, uuid.uuid4().hex)

    assert response.status_code == 503, response.text
    assert _UNAVAILABLE_MESSAGE in response.text, response.text
    assert "overloaded_error" not in response.text, response.text
    assert [request.target for request in vertex.wire.drain()] == [
        f"/v1/{_model_path(_UNAVAILABLE_MODEL)}:streamGenerateContent?alt=sse"
    ]
