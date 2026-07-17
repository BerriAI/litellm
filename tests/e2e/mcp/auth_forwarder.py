"""In-process reverse proxy that stamps the LiteLLM key onto every request.

The official conformance suite's client offers no way to send custom headers,
but the gateway's MCP routes require the virtual key. This forwarder plays the
role of an MCP host's HTTP layer configured with the key header: it listens on
a loopback port, injects `x-litellm-api-key: Bearer <key>`, and relays
everything (including SSE streams, unbuffered) to the proxy under test.

Runs inside the pytest process on a background uvicorn thread; tests get the
local base URL from `serve()` and hand `{base}/{alias}/mcp` to the conformance
CLI as the server under test.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

from e2e_config import PROXY_BASE_URL, REQUEST_TIMEOUT

_HOP_BY_HOP_REQUEST_HEADERS = frozenset({"host", "content-length"})
_HOP_BY_HOP_RESPONSE_HEADERS = frozenset({"content-length", "transfer-encoding", "content-encoding"})


def _build_app(upstream: httpx.AsyncClient, litellm_key: str) -> Starlette:
    async def forward(request: Request) -> StreamingResponse:
        url = httpx.URL(path=request.url.path, query=request.url.query.encode())
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP_REQUEST_HEADERS
        }
        headers["x-litellm-api-key"] = f"Bearer {litellm_key}"
        proxied = upstream.build_request(request.method, url, headers=headers, content=await request.body())
        response = await upstream.send(proxied, stream=True)
        response_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS
        }
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    methods = ["GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS"]
    return Starlette(routes=[Route("/{path:path}", forward, methods=methods)])


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return cast("int", sock.getsockname()[1])


@contextmanager
def serve(litellm_key: str) -> Generator[str]:
    """Serve the forwarder for the block's duration; yields its base URL."""
    upstream = httpx.AsyncClient(base_url=PROXY_BASE_URL, timeout=httpx.Timeout(REQUEST_TIMEOUT))
    port = _free_loopback_port()
    server = uvicorn.Server(
        uvicorn.Config(_build_app(upstream, litellm_key), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "auth forwarder never started"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
