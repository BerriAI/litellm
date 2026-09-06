import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import litellm


class _ImageVariationsUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.dumps({"created": 1, "data": [{"url": "https://example.invalid/variation.png"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture
def image_variations_upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ImageVariationsUpstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def shared_sessions(monkeypatch: pytest.MonkeyPatch):
    sync_session = httpx.Client()
    monkeypatch.setattr(litellm, "client_session", sync_session)
    monkeypatch.setattr(litellm, "aclient_session", httpx.AsyncClient())
    try:
        yield
    finally:
        sync_session.close()


def _square_png() -> io.BytesIO:
    image = io.BytesIO(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image.name = "square.png"
    return image


def test_sync_image_variation_uses_the_sync_shared_session(image_variations_upstream, shared_sessions):
    response = litellm.image_variation(
        image=_square_png(),
        model="dall-e-2",
        api_key="sk-test",
        api_base=image_variations_upstream,
    )

    assert response.data is not None
    assert response.data[0].url == "https://example.invalid/variation.png"


@pytest.mark.asyncio
async def test_async_image_variation_uses_the_async_shared_session(image_variations_upstream, shared_sessions):
    response = await litellm.aimage_variation(
        image=_square_png(),
        model="dall-e-2",
        api_key="sk-test",
        api_base=image_variations_upstream,
    )

    assert response.data is not None
    assert response.data[0].url == "https://example.invalid/variation.png"
