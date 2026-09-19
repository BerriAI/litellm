"""An owned, source-built OAuth gateway with cold restarts and credential observations.

Only this child process is restarted. Its database and SSO client survive while
its process-local caches do not; Redis is deliberately absent from its config.
The optional live edge measures headers without recording credentials or bodies.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import psycopg
from e2e_http import NoBody
from idp import Keycloak, stop_process_group
from proxy_client import ProxyClient, build_proxy_client
from psycopg.rows import class_row
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

INHERITED_ENV_PREFIXES: Final = ("REDIS_", "MICROSOFT_", "GOOGLE_", "GENERIC_", "PROXY_")


class StoredOAuth(BaseModel):
    type: str
    access_token: SecretStr


@dataclass(frozen=True, slots=True)
class CredentialRow:
    credential_b64: str = field(repr=False)


def stored_oauth(user_id: str, server_id: str) -> StoredOAuth:
    """Read the encrypted credential because management APIs omit the plaintext token."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    with psycopg.Connection[CredentialRow].connect(
        os.environ["DATABASE_URL"], row_factory=class_row(CredentialRow)
    ) as conn:
        row: Final = conn.execute(
            'SELECT credential_b64 FROM "LiteLLM_MCPUserCredentials" WHERE user_id = %s AND server_id = %s',
            (user_id, server_id),
        ).fetchone()
    assert row is not None, "canonical user/server has no persisted credential"
    plaintext: Final = decrypt_value_helper(
        row.credential_b64, "e2e_mcp_oauth", exception_type="debug", return_original_value=False
    )
    assert plaintext is not None, "persisted credential must decrypt with the gateway salt"
    assert plaintext != row.credential_b64, "persisted credential must be encrypted"
    try:
        credential: Final = StoredOAuth.model_validate_json(plaintext)
    except ValidationError:
        raise AssertionError("decrypted credential is not an OAuth payload") from None
    assert credential.type == "oauth2"
    assert bool(credential.access_token.get_secret_value()), "stored upstream token is empty"
    return credential


class RpcMethod(BaseModel):
    method: str = ""


@dataclass(slots=True)
class OAuthObservation:
    gateway_token: str = field(default="", repr=False)
    _seen: tuple[tuple[str, str, bool], ...] = field(default=(), init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def observe(self, url: str, headers: Mapping[str, str], body: bytes | None) -> None:
        if body is None or not url.endswith("/mcp"):
            return
        try:
            operation: Final = RpcMethod.model_validate_json(body).method
        except ValidationError:
            return
        if operation not in ("tools/list", "tools/call"):
            return
        received: Final = headers.get("authorization", "")
        gateway_leaked: Final = any(
            value in (self.gateway_token, f"Bearer {self.gateway_token}") for value in headers.values()
        )
        with self._lock:
            self._seen = (*self._seen, (operation, received, gateway_leaked))

    def assert_forwarded(self, expected: StoredOAuth) -> None:
        with self._lock:
            snapshot: Final = self._seen
            self._seen = ()
        assert {item[0] for item in snapshot} == {"tools/list", "tools/call"}, "missing upstream observations"
        expected_header: Final = f"Bearer {expected.access_token.get_secret_value()}"
        assert all(item[1] == expected_header for item in snapshot), "upstream bearer did not match the stored token"
        assert all(not item[2] for item in snapshot), "gateway bearer leaked to the upstream"


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return TypeAdapter(tuple[str, int]).validate_python(listener.getsockname())[1]


@dataclass(slots=True)
class OAuthGateway:
    base_url: str
    proxy: ProxyClient
    _environment: Mapping[str, str] = field(repr=False)
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
            assert self._child.poll() is None, "owned OAuth gateway exited; inspect its private log"
            result = self.proxy.transport.probe("/health/liveliness", params=NoBody())
            if result.status_code == 200:
                return
            time.sleep(0.5)
        raise AssertionError("owned OAuth gateway did not become ready")

    def stop(self) -> None:
        if self._child is not None:
            stop_process_group(self._child)
            assert self._child.poll() is not None, "old gateway process is still alive"

    def restart(self) -> None:
        assert self._child is not None
        previous: Final = self._child.pid
        self.stop()
        self.start()
        assert self._child.pid != previous, "gateway restart did not create a new process"


def owned_gateway(idp: Keycloak, directory: Path, cleanup: ExitStack) -> OAuthGateway:
    for name in ("DATABASE_URL", "LITELLM_LICENSE", "LITELLM_SALT_KEY", "LITELLM_MASTER_KEY"):
        assert os.environ.get(name), f"{name} is required for the owned OAuth gateway"
    port: Final = available_port()
    base_url: Final = f"http://127.0.0.1:{port}"

    def defer(callback: Callable[[], object]) -> None:
        cleanup.callback(callback)

    browser: Final = idp.browser_client(callback_url=f"{base_url}/sso/callback", defer=defer)
    config: Final = directory / "oauth-gateway.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  enable_jwt_auth: true\n"
        "  litellm_jwtauth:\n"
        "    user_id_jwt_field: sub\n"
        "    user_email_jwt_field: email\n"
        "    team_ids_jwt_field: groups\n"
        "    user_id_upsert: true\n"
    )
    environment: Final = {
        **{key: value for key, value in os.environ.items() if not key.startswith(INHERITED_ENV_PREFIXES)},
        **browser.environment(idp.discovery()),
        "PROXY_BASE_URL": base_url,
        "JWT_PUBLIC_KEY_URL": idp.jwks_url,
        "JWT_ISSUER": idp.issuer,
        "JWT_AUDIENCE": "litellm-e2e",
        "DISABLE_SCHEMA_UPDATE": "true",
        "STORE_MODEL_IN_DB": "True",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    gateway: Final = OAuthGateway(
        base_url=base_url,
        proxy=build_proxy_client(
            base_url=base_url,
            control_plane_base_url=base_url,
            replica_urls=(base_url,),
            master_key=os.environ["LITELLM_MASTER_KEY"],
        ),
        _environment=environment,
        _command=(sys.executable, "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(port)),
        _log_path=directory / "oauth-gateway.log",
    )
    cleanup.callback(gateway.stop)
    gateway.start()
    return gateway
