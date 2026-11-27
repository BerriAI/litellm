import asyncio
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final, Literal

import pytest
from prisma import Prisma
from prisma.engine.errors import EngineConnectionError

from litellm.proxy.db.prisma_client import PrismaWrapper
from tests._process_helpers import process_is_gone

_FAKE_QUERY_ENGINE: Final = """
import http.server
import json
import os
import pathlib
import signal
import sys
import time

signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
if sys.argv[1:] == ["--version"]:
    print("query-engine fake")
    sys.exit(0)
pathlib.Path(os.environ["FAKE_ENGINE_PID_FILE"]).write_text(str(os.getpid()))
behavior = os.environ["FAKE_ENGINE_BEHAVIOR"]
if behavior == "exit":
    sys.exit(1)
while behavior == "silent":
    time.sleep(60)
requests = pathlib.Path(os.environ["FAKE_ENGINE_REQUESTS"])


class Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        with requests.open("a") as log:
            print(self.command, self.path, file=log)
        body = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", 0), Status)
port = server.server_address[1]
message = f"Started query engine http server on http://127.0.0.1:{port}"
print("query-engine banner that is not JSON", flush=True)
print(
    json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00.000000Z",
            "level": "INFO",
            "fields": {"message": message, "ip": "127.0.0.1", "port": str(port)},
            "target": "query_engine::server",
        }
    ),
    flush=True,
)
server.serve_forever()
"""


@dataclass(frozen=True, slots=True)
class _FakeQueryEngine:
    pid_file: Path
    requests_file: Path

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self.requests_file.read_text().splitlines()) if self.requests_file.exists() else ()

    def is_gone(self) -> bool:
        return process_is_gone(int(self.pid_file.read_text()), within_seconds=5)


def _install_fake_query_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: Literal["publish", "silent", "exit"]
) -> _FakeQueryEngine:
    binary: Final = tmp_path / "query-engine"
    binary.write_text(f"#!{sys.executable}\n{_FAKE_QUERY_ENGINE}")
    binary.chmod(0o755)
    engine: Final = _FakeQueryEngine(pid_file=tmp_path / "engine.pid", requests_file=tmp_path / "requests.log")
    monkeypatch.setenv("PRISMA_QUERY_ENGINE_BINARY", str(binary))
    monkeypatch.setenv("FAKE_ENGINE_BEHAVIOR", behavior)
    monkeypatch.setenv("FAKE_ENGINE_PID_FILE", str(engine.pid_file))
    monkeypatch.setenv("FAKE_ENGINE_REQUESTS", str(engine.requests_file))
    return engine


@pytest.mark.asyncio
async def test_the_client_connects_to_the_address_the_query_engine_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: Final = _install_fake_query_engine(tmp_path, monkeypatch, "publish")
    client: Final = PrismaWrapper(Prisma(datasource={"url": "postgresql://unused@localhost/unused"}))
    try:
        await client.connect(timedelta(seconds=30))
        assert "GET /status" in engine.requests
    finally:
        await client.disconnect()
    assert engine.is_gone()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("behavior", "engine_timeout"),
    (("silent", timedelta(seconds=1)), ("exit", timedelta(minutes=10))),
    ids=("never prints", "exits first"),
)
async def test_a_query_engine_that_publishes_no_address_fails_the_connect_and_is_stopped(
    behavior: Literal["silent", "exit"], engine_timeout: timedelta, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: Final = _install_fake_query_engine(tmp_path, monkeypatch, behavior)
    client: Final = PrismaWrapper(Prisma(datasource={"url": "postgresql://unused@localhost/unused"}))
    try:
        with pytest.raises(EngineConnectionError):
            await asyncio.wait_for(client.connect(engine_timeout), timeout=60)
    finally:
        await client.disconnect()
    assert engine.is_gone()
