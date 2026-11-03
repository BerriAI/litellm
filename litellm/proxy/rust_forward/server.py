from dataclasses import dataclass
from enum import Enum


class Server(Enum):
    UVICORN = "uvicorn"
    GUNICORN = "gunicorn"
    HYPERCORN = "Hypercorn"
    GRANIAN = "Granian"


class Listener(Enum):
    TCP = "plain TCP"
    TLS = "TLS"
    UNIX_SOCKET = "a Unix socket"
    INHERITED_FD = "an inherited file descriptor"


@dataclass(frozen=True, slots=True)
class PythonServer:
    """What an entry point would serve with on the public port if Rust were not in front of it."""

    server: Server
    listener: Listener
    host: str
    port: int
    keepalive_seconds: int
    graceful_timeout_seconds: float
    proxy_headers: bool
    trusted_proxies: str | None
