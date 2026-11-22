import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.server_listener import NO_TRUSTED_PROXIES

_TRUSTED_PROXIES: Final = TypeAdapter(str)


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
    trusted_proxies: str
    config_path: str | None


def local_config_path(cli_config: str | None, environ: Mapping[str, str]) -> str | None:
    """The local YAML file the proxy loads its config from, by the precedence of ``proxy_server.initialize``.

    ``CONFIG_FILE_PATH`` decides alone once it is set, even when it names no file, and ``--config`` is read otherwise.
    ``None`` when neither names a YAML file on disk or when ``LITELLM_CONFIG_BUCKET_NAME`` moves the config to a bucket.
    """
    path: Final = environ.get("CONFIG_FILE_PATH", cli_config)
    if path is None or environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None:
        return None
    if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in (".yaml", ".yml"):
        return None
    return os.path.abspath(path)


class UvicornCliParams(BaseModel):
    """The options uvicorn's own CLI parsed that decide how it serves the public port."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    host: str
    port: int
    uds: str | None
    fd: int | None
    ssl_keyfile: str | None
    ssl_certfile: str | None
    timeout_keep_alive: int
    timeout_graceful_shutdown: int | None
    proxy_headers: bool
    forwarded_allow_ips: str | None

    @property
    def listener(self) -> Listener:
        if self.ssl_keyfile is not None or self.ssl_certfile is not None:
            return Listener.TLS
        if self.uds is not None:
            return Listener.UNIX_SOCKET
        if self.fd is not None:
            return Listener.INHERITED_FD
        return Listener.TCP

    @property
    def trusted_proxies(self) -> str:
        return uvicorn_trusted_proxies(self.forwarded_allow_ips) if self.proxy_headers else NO_TRUSTED_PROXIES


def uvicorn_trusted_proxies(configured: str | None) -> str:
    import uvicorn

    return _TRUSTED_PROXIES.validate_python(
        uvicorn.Config(app="", log_config=None, forwarded_allow_ips=configured).forwarded_allow_ips
    )
