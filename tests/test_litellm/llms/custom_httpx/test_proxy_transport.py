from collections.abc import Generator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
import ssl
from socketserver import ThreadingMixIn
import threading
from typing import Final

import httpx
import pytest

from litellm.llms.custom_httpx.proxy_transport import (
    AsyncProxyTransport,
    ProxyTransport,
    _get_http_proxy,
)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    forwarded_proto: str | None


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


@pytest.fixture
def recording_proxy_server() -> Generator[tuple[str, list[RecordedRequest]], None, None]:
    recorded_requests: Final[list[RecordedRequest]] = []

    class RecordingProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._record_and_respond()

        def do_POST(self) -> None:
            self._record_and_respond()

        def _record_and_respond(self) -> None:
            recorded_requests.append(
                RecordedRequest(
                    method=self.command,
                    path=self.path,
                    forwarded_proto=self.headers.get("X-Forwarded-Proto"),
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadedServer(("127.0.0.1", 0), RecordingProxyHandler)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", recorded_requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_proxy_transport_uses_configured_proxy_and_downgrades_https(
    recording_proxy_server: tuple[str, list[RecordedRequest]],
) -> None:
    proxy_url, recorded_requests = recording_proxy_server
    transport: Final = ProxyTransport(proxy_url=proxy_url)
    client: Final = httpx.Client(transport=transport)
    try:
        response: Final = client.post(
            "https://upstream.invalid/v1/chat/completions",
            json={"model": "test"},
        )
    finally:
        client.close()

    assert response.status_code == 200
    assert len(recorded_requests) == 1
    assert recorded_requests[0].method == "POST"
    assert recorded_requests[0].path == "http://upstream.invalid/v1/chat/completions"
    assert recorded_requests[0].forwarded_proto == "https"


@pytest.mark.asyncio
async def test_async_proxy_transport_uses_configured_proxy_and_downgrades_https(
    recording_proxy_server: tuple[str, list[RecordedRequest]],
) -> None:
    proxy_url, recorded_requests = recording_proxy_server
    transport: Final = AsyncProxyTransport(proxy_url=proxy_url)
    client: Final = httpx.AsyncClient(transport=transport)
    try:
        response: Final = await client.post(
            "https://upstream.invalid/v1/chat/completions",
            json={"model": "test"},
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert len(recorded_requests) == 1
    assert recorded_requests[0].method == "POST"
    assert recorded_requests[0].path == "http://upstream.invalid/v1/chat/completions"
    assert recorded_requests[0].forwarded_proto == "https"


def test_proxy_transport_preserves_plain_http_scheme(
    recording_proxy_server: tuple[str, list[RecordedRequest]],
) -> None:
    proxy_url, recorded_requests = recording_proxy_server
    transport: Final = ProxyTransport(proxy_url=proxy_url)
    client: Final = httpx.Client(transport=transport)
    try:
        response: Final = client.get("http://upstream.invalid/v1/models")
    finally:
        client.close()

    assert response.status_code == 200
    assert len(recorded_requests) == 1
    assert recorded_requests[0].method == "GET"
    assert recorded_requests[0].path == "http://upstream.invalid/v1/models"
    assert recorded_requests[0].forwarded_proto == "http"


@pytest.mark.parametrize(
    ("disable_verify", "expect_ssl_context"),
    [
        ("False", False),
        ("True", True),
    ],
)
def test_get_http_proxy_tls_verification(
    monkeypatch: pytest.MonkeyPatch, disable_verify: str, expect_ssl_context: bool
) -> None:
    monkeypatch.setenv("DISABLE_OUTBOUND_PROXY_TLS_VERIFICATION", disable_verify)
    proxy: Final = _get_http_proxy("http://proxy.test:8080")

    if expect_ssl_context:
        assert isinstance(proxy.ssl_context, ssl.SSLContext)
        assert proxy.ssl_context.check_hostname is False
        assert proxy.ssl_context.verify_mode == ssl.CERT_NONE
    else:
        assert proxy.ssl_context is None

