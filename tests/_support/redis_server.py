import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import redis


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def redis_server() -> Iterator[int]:
    port: Final = _free_port()
    server: Final = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        client: Final = redis.Redis(port=port)
        deadline: Final = time.monotonic() + 10
        while True:
            try:
                client.ping()
                break
            except redis.ConnectionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        yield port
    finally:
        server.terminate()
        server.wait()
