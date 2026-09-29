"""An owned, source-built gateway whose Postgres sits behind a cuttable TCP relay.

The relay listens on 127.0.0.1:<port> and forwards to the real database parsed
from DATABASE_URL; the gateway's own database_url names the relay instead.
`relay.cut()` closes every live forwarded socket and makes new connections fail,
so the gateway observes a genuine outage rather than a refused dial, and
`relay.restore()` resumes forwarding. Modelled on mcp/oauth_gateway.py: same
Popen with start_new_session, same /health/liveliness readiness loop, same
stop_process_group teardown, config written under tmp_path.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from e2e_http import NoBody, is_ok
from idp import Keycloak, stop_process_group
from models import LiteLLMParamsBody, ModelDeleteBody
from proxy_client import ProxyClient, build_proxy_client
from pydantic import TypeAdapter

INHERITED_ENV_PREFIXES: Final = ("REDIS_", "MICROSOFT_", "GOOGLE_", "GENERIC_", "PROXY_")
RELAY_DATABASE_URL_ENV: Final = "E2E_RELAY_DATABASE_URL"


@dataclass(frozen=True, slots=True)
class DatabaseTarget:
    host: str
    port: int

    @staticmethod
    def from_url(database_url: str) -> DatabaseTarget:
        parsed: Final = urllib.parse.urlparse(database_url)
        assert parsed.hostname, "DATABASE_URL has no host"
        return DatabaseTarget(host=parsed.hostname, port=parsed.port or 5432)


def relay_url(database_url: str, relay_port: int) -> str:
    parsed: Final = urllib.parse.urlparse(database_url)
    credentials: Final = f"{parsed.username}:{parsed.password}@" if parsed.username is not None else ""
    return urllib.parse.urlunparse(parsed._replace(netloc=f"{credentials}127.0.0.1:{relay_port}"))


def bound_port(listener: socket.socket) -> int:
    return TypeAdapter(tuple[str, int]).validate_python(listener.getsockname())[1]


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return bound_port(listener)


@dataclass(slots=True)
class PostgresRelay:
    """A TCP relay the test can sever mid-run. While cut, accepted sockets are
    closed at once, so a pooled client sees its connection die and a fresh dial
    fails rather than hanging."""

    target: DatabaseTarget
    host: str = "127.0.0.1"
    port: int = 0
    _listener: socket.socket | None = field(default=None, init=False, repr=False)
    _connections: set[socket.socket] = field(default_factory=set, init=False, repr=False)
    _cut: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        listener: Final = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, 0))
        listener.listen()
        self.port = bound_port(listener)
        self._listener = listener
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                accepted, _addr = self._listener.accept() if self._listener else (None, None)
            except OSError:
                return
            if accepted is None:
                return
            with self._lock:
                if self._cut:
                    accepted.close()
                    continue
                self._connections.add(accepted)
            try:
                upstream: socket.socket | None = socket.create_connection(
                    (self.target.host, self.target.port), timeout=10
                )
            except OSError:
                self._drop(accepted)
                continue
            with self._lock:
                self._connections.add(upstream)
            threading.Thread(target=self._pump, args=(accepted, upstream), daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, accepted), daemon=True).start()

    def _pump(self, source: socket.socket, sink: socket.socket) -> None:
        try:
            while chunk := source.recv(65536):
                sink.sendall(chunk)
        except OSError:
            pass
        finally:
            self._drop(source)
            self._drop(sink)

    def _drop(self, conn: socket.socket) -> None:
        with self._lock:
            self._connections.discard(conn)
        try:
            conn.close()
        except OSError:
            pass

    def cut(self) -> None:
        with self._lock:
            self._cut = True
            live: Final = tuple(self._connections)
        for conn in live:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._drop(conn)

    def restore(self) -> None:
        with self._lock:
            self._cut = False

    def stop(self) -> None:
        self.cut()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass


@dataclass(slots=True)
class DbOutageGateway:
    base_url: str
    proxy: ProxyClient
    relay: PostgresRelay
    _environment: dict[str, str] = field(repr=False)
    _command: tuple[str, ...] = field(repr=False)
    _log_path: Path
    _child: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        with self._log_path.open("ab") as log:
            self._child = subprocess.Popen(
                self._command,
                env=self._environment,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        deadline: Final = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert self._child.poll() is None, "owned db-outage gateway exited; inspect its private log"
            result = self.proxy.transport.probe("/health/liveliness", params=NoBody())
            if result.status_code == 200:
                return
            time.sleep(0.5)
        raise AssertionError("owned db-outage gateway did not become ready")

    def stop(self) -> None:
        if self._child is not None:
            stop_process_group(self._child)
            assert self._child.poll() is not None, "owned gateway process is still alive"


def owned_db_outage_gateway(idp: Keycloak, model: str, directory: Path, cleanup: ExitStack) -> DbOutageGateway:
    for name in ("DATABASE_URL", "LITELLM_LICENSE", "LITELLM_SALT_KEY", "LITELLM_MASTER_KEY"):
        assert os.environ.get(name), f"{name} is required for the owned db-outage gateway"
    port: Final = available_port()
    base_url: Final = f"http://127.0.0.1:{port}"

    relay: Final = PostgresRelay(target=DatabaseTarget.from_url(os.environ["DATABASE_URL"]))
    relay.start()
    cleanup.callback(relay.stop)

    config: Final = directory / "db-outage-gateway.yaml"
    config.write_text(
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        f"  database_url: os.environ/{RELAY_DATABASE_URL_ENV}\n"
        "  user_api_key_cache_ttl: 2\n"
        "  enable_jwt_auth: true\n"
        "  litellm_jwtauth:\n"
        "    user_id_jwt_field: sub\n"
        "    user_email_jwt_field: email\n"
        "    team_ids_jwt_field: groups\n"
        "    user_id_upsert: true\n"
    )
    environment: Final = {
        **{key: value for key, value in os.environ.items() if not key.startswith(INHERITED_ENV_PREFIXES)},
        RELAY_DATABASE_URL_ENV: relay_url(os.environ["DATABASE_URL"], relay.port),
        "PROXY_BASE_URL": base_url,
        "JWT_PUBLIC_KEY_URL": idp.jwks_url,
        "JWT_ISSUER": idp.issuer,
        "JWT_AUDIENCE": "litellm-e2e",
        "DISABLE_SCHEMA_UPDATE": "true",
        "STORE_MODEL_IN_DB": "True",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    gateway: Final = DbOutageGateway(
        base_url=base_url,
        proxy=build_proxy_client(
            base_url=base_url,
            control_plane_base_url=base_url,
            replica_urls=(base_url,),
            master_key=os.environ["LITELLM_MASTER_KEY"],
        ),
        relay=relay,
        _environment=environment,
        _command=(sys.executable, "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(port)),
        _log_path=directory / "db-outage-gateway.log",
    )
    cleanup.callback(gateway.stop)
    gateway.start()
    model_id: Final = gateway.proxy.create_model(
        model_name=model,
        litellm_params=LiteLLMParamsBody(model=f"openai/{model}", api_key="os.environ/OPENAI_API_KEY"),
    )

    def _delete_model_when_db_back() -> None:
        relay.restore()
        deadline: Final = time.monotonic() + 30
        while time.monotonic() < deadline:
            if is_ok(
                gateway.proxy.transport.post(
                    "/model/delete",
                    headers=gateway.proxy.management_headers(),
                    json=ModelDeleteBody(id=model_id),
                    response_type=NoBody,
                )
            ):
                return
            time.sleep(1)

    cleanup.callback(_delete_model_when_db_back)
    return gateway
