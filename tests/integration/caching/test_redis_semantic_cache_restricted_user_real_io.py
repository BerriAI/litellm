from __future__ import annotations

import json
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from queue import SimpleQueue
from typing import BinaryIO, Final

import pytest
from integration._support.wire import Reply, Request, wire_server

from litellm.caching.redis_semantic_cache import RedisSemanticCache

_Command = tuple[bytes, ...]

# Redis Cloud's predefined Read-Write role is `+@all -@dangerous`, and `MODULE` and `INFO` are the
# @dangerous commands vector libraries send to probe modules and versions (`ACL CAT dangerous` on
# Redis 8.2, https://redis.io/docs/latest/operate/oss_and_stack/management/security/acl/, read 2026-11-22)
_REFUSED_FOR_READ_WRITE_ROLE: Final = frozenset({b"MODULE", b"INFO"})


def _read_bulk(stream: BinaryIO) -> bytes:
    size: Final = int(stream.readline()[1:])
    return stream.read(size + 2)[:-2]


def _read_command(stream: BinaryIO) -> _Command | None:
    header: Final = stream.readline()
    if not header:
        return None
    return tuple(_read_bulk(stream) for _ in range(int(header[1:])))


def _reply(command: _Command) -> bytes:
    name: Final = command[0].upper()
    if name in _REFUSED_FOR_READ_WRITE_ROLE:
        refused: Final = b"|".join(part.lower() for part in command[:2])
        return b"-NOPERM User readwrite has no permissions to run the '" + refused + b"' command\r\n"
    if name == b"FT._LIST":
        return b"*0\r\n"
    if name == b"FT.SEARCH":
        return b"*1\r\n:0\r\n"
    return b"+OK\r\n"


class _ThreadingServer(socketserver.ThreadingTCPServer):
    daemon_threads = True


@contextmanager
def _read_write_role_redis() -> Iterator[tuple[str, SimpleQueue[_Command]]]:
    received: Final[SimpleQueue[_Command]] = SimpleQueue()

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for command in iter(lambda: _read_command(self.rfile), None):
                received.put(command)
                _ = self.wfile.write(_reply(command))

    with _ThreadingServer(("127.0.0.1", 0), _Handler) as server:
        serving: Final = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        try:
            yield f"redis://readwrite:readwrite-password@127.0.0.1:{server.server_address[1]}", received
        finally:
            server.shutdown()


def _embed(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/v1/embeddings")
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
    )


async def test_semantic_cache_stores_and_looks_up_for_a_redis_user_without_dangerous_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index: Final = "litellm_semantic_cache_read_write_role"
    messages: Final = [{"role": "user", "content": "What is the capital of France?"}]
    with wire_server(_embed) as embeddings, _read_write_role_redis() as (redis_url, received):
        monkeypatch.setenv("OPENAI_API_KEY", "scripted-embedding-key")
        monkeypatch.setenv("OPENAI_BASE_URL", embeddings.url + "/v1")
        cache: Final = RedisSemanticCache(
            redis_url=redis_url,
            similarity_threshold=0.8,
            embedding_model="openai/text-embedding-3-small",
            index_name=index,
        )

        await cache.async_set_cache("cache-key", "Paris", messages=messages)
        assert await cache.async_get_cache("cache-key", messages=messages) is None
        commands: Final = tuple(received.get_nowait() for _ in range(received.qsize()))

    assert any(command[:2] == (b"FT.CREATE", index.encode()) for command in commands), (
        "the semantic-cache index was never created"
    )
    assert any(command[0] == b"HSET" and command[1].startswith(index.encode() + b":") for command in commands), (
        "the cache entry was never stored"
    )
    assert any(command[:2] == (b"FT.SEARCH", index.encode()) for command in commands), (
        "the cache was never searched"
    )
