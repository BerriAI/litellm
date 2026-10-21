import os
import socket
import ssl
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
from integration._support.tls import server_context, write_self_signed_cert
from litellm import Router
from redis import Redis

PAYLOAD: Final = {"transport": "tls"}


@dataclass(frozen=True, slots=True)
class TlsRelay:
    url: str
    handshakes: SimpleQueue[str]


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while chunk := source.recv(65536):
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        sink.close()


def _serve(listener: socket.socket, context: ssl.SSLContext, handshakes: SimpleQueue[str]) -> None:
    while True:
        try:
            raw, _ = listener.accept()
        except OSError:
            return
        try:
            secured = context.wrap_socket(raw, server_side=True)
        except (ssl.SSLError, OSError):
            raw.close()
            continue
        handshakes.put(str(secured.version()))
        backend = socket.create_connection((os.environ["REDIS_HOST"], int(os.environ["REDIS_PORT"])))
        threading.Thread(target=_pipe, args=(secured, backend), daemon=True).start()
        threading.Thread(target=_pipe, args=(backend, secured), daemon=True).start()


@contextmanager
def tls_relay(directory: Path) -> Iterator[TlsRelay]:
    cert: Final = write_self_signed_cert(directory)
    handshakes: Final = SimpleQueue[str]()
    with socket.create_server(("127.0.0.1", 0)) as listener:
        thread: Final = threading.Thread(target=_serve, args=(listener, server_context(*cert), handshakes), daemon=True)
        thread.start()
        port: Final = listener.getsockname()[1]
        yield TlsRelay(f"rediss://127.0.0.1:{port}/0?ssl_ca_certs={cert[0]}", handshakes)
        listener.close()
        thread.join(timeout=5)


def _router(redis_url: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "tls-cache",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "synthetic-tls-key"},
            }
        ],
        redis_url=redis_url,
    )


def _plain_redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True)


@pytest.mark.asyncio
async def test_async_router_cache_built_from_a_rediss_url_talks_tls_to_redis(tmp_path: Path) -> None:
    with tls_relay(tmp_path) as relay, _plain_redis() as plain:
        cache: Final = _router(relay.url).cache.redis_cache
        assert cache is not None
        assert await cache.ping() is True
        key: Final = f"tls-async-{uuid.uuid4().hex}"
        await cache.async_set_cache(key, PAYLOAD, ttl=60)
        assert plain.exists(key) == 1
        assert await cache.async_get_cache(key) == PAYLOAD
        assert relay.handshakes.qsize() >= 1
        assert relay.handshakes.get_nowait().startswith("TLS")


def test_sync_router_cache_built_from_a_rediss_url_talks_tls_to_redis(tmp_path: Path) -> None:
    with tls_relay(tmp_path) as relay, _plain_redis() as plain:
        cache: Final = _router(relay.url).cache.redis_cache
        assert cache is not None
        assert cache.sync_ping() is True
        key: Final = f"tls-sync-{uuid.uuid4().hex}"
        cache.set_cache(key, PAYLOAD, ttl=60)
        assert plain.exists(key) == 1
        assert cache.get_cache(key) == PAYLOAD
        assert relay.handshakes.qsize() >= 1
        assert relay.handshakes.get_nowait().startswith("TLS")
