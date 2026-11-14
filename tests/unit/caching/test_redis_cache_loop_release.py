"""Sync calls run on a throwaway loop each, and must not leave a Redis connection behind (#48589)."""

import asyncio
import socketserver
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.caching.redis_cache import RedisCache
from litellm.litellm_core_utils.asyncify import run_async_function


class _RespStub(socketserver.ThreadingUnixStreamServer):
    """Answers +OK to every command and counts the connections still open."""

    daemon_threads = True

    def __init__(self, path: str) -> None:
        super().__init__(path, _OkHandler)
        self.open_connections = 0
        self.lock = threading.Lock()

    def track(self, delta: int) -> None:
        with self.lock:
            self.open_connections += delta


class _OkHandler(socketserver.StreamRequestHandler):
    server: _RespStub  # pyright: ignore[reportIncompatibleVariableOverride]  # narrowed to the stub

    def handle(self) -> None:
        self.server.track(1)
        try:
            while self._skip_command():
                self.wfile.write(b"+OK\r\n")
        finally:
            self.server.track(-1)

    def _skip_command(self) -> bool:
        header: Final = self.rfile.readline()
        if not header.startswith(b"*"):
            return False
        for _ in range(int(header[1:])):
            size = int(self.rfile.readline()[1:])
            _ = self.rfile.read(size + 2)
        return True


@pytest.fixture
def redis_stub(socket_dir: Path) -> Iterator[_RespStub]:
    stub: Final = _RespStub(str(socket_dir / "redis.sock"))
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    yield stub
    stub.shutdown()
    stub.server_close()


@pytest.fixture
def client_cache(monkeypatch: pytest.MonkeyPatch) -> LLMClientCache:
    fresh: Final = LLMClientCache(evicted_client_closer=EvictedClientCloser())
    monkeypatch.setattr("litellm.in_memory_llm_clients_cache", fresh)
    return fresh


def _redis_cache(stub: _RespStub) -> RedisCache:
    cache: Final = RedisCache(url=f"unix://{stub.server_address}")
    cache.set_cache("warm", "value")  # the stub counts the sync client's connection before it replies
    return cache


def _settled_open_connections(stub: _RespStub, expected: int) -> int:
    for _ in range(100):
        if stub.open_connections == expected:
            break
        threading.Event().wait(0.01)  # a closed connection reaches the stub's thread a moment later
    return stub.open_connections


def test_sync_calls_close_their_redis_connections(redis_stub: _RespStub, client_cache: LLMClientCache) -> None:
    cache: Final = _redis_cache(redis_stub)
    before: Final = redis_stub.open_connections

    for _ in range(5):
        run_async_function(cache.async_set_cache, "key", "value")

    assert _settled_open_connections(redis_stub, before) == before
    assert client_cache.cache_dict == {}


def test_long_lived_loop_reuses_its_redis_client(redis_stub: _RespStub, client_cache: LLMClientCache) -> None:
    cache: Final = _redis_cache(redis_stub)
    before: Final = redis_stub.open_connections

    async def two_writes() -> tuple[object, object, int]:
        await cache.async_set_cache("key", "value")
        first: Final = cache.init_async_client()
        await cache.async_set_cache("key", "value")
        return first, cache.init_async_client(), redis_stub.open_connections

    first, second, open_connections = asyncio.run(two_writes())

    assert first is second
    assert open_connections == before + 1
