from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

CONFIGURED_KEEPALIVE_SECONDS: Final = 1
IDLE_SECONDS: Final = 2
RESPONSES: Final = TypeAdapter(list[dict[str, JsonValue]])

REBUILT_SESSION_EXCHANGE: Final = textwrap.dedent(
    """
    import asyncio, json, sys
    from aiohttp import ClientSession
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    async def main(base_url: str, idle_seconds: float) -> None:
        shared = ClientSession()
        handler = AsyncHTTPHandler(shared_session=shared)
        await shared.close()
        first = await handler.post(f"{base_url}/embeddings", json={"input": "warm-up"})
        await asyncio.sleep(idle_seconds)
        second = await handler.post(f"{base_url}/embeddings", json={"input": "warm-up"})
        print(json.dumps([first.json(), second.json()]))
        await handler.close()

    asyncio.run(main(sys.argv[1], float(sys.argv[2])))
    """
)


class _ConnectionCountingPeer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _ConnectionHandler)
        self.lock = threading.Lock()
        self.connections = 0

    def next_connection(self) -> int:
        with self.lock:
            self.connections += 1
            return self.connections


class _ConnectionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _ConnectionCountingPeer

    def setup(self) -> None:
        super().setup()
        self.connection_number = self.server.next_connection()

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body: Final = json.dumps({"connection": self.connection_number}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def connection_counting_peer() -> Iterator[str]:
    server: Final = _ConnectionCountingPeer(("127.0.0.1", 0))
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


def _rebuilt_session_exchange(base_url: str) -> list[dict[str, JsonValue]]:
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", REBUILT_SESSION_EXCHANGE, base_url, str(IDLE_SECONDS)],
        env={
            **os.environ,
            "AIOHTTP_KEEPALIVE_TIMEOUT": str(CONFIGURED_KEEPALIVE_SECONDS),
            "AIOHTTP_SO_KEEPALIVE": "true",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return RESPONSES.validate_json(completed.stdout)


@pytest.mark.covers("sdk.aiohttp_transport.rebuilt_shared_session_keeps_configured_keepalive_timeout")
def test_rebuilt_shared_session_drops_idle_connection_after_configured_keepalive_timeout(
    connection_counting_peer: str,
) -> None:
    observed: Final = _rebuilt_session_exchange(connection_counting_peer)
    assert observed == [{"connection": 1}, {"connection": 2}], observed
