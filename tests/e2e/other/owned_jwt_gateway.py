"""An owned, source-built proxy whose `litellm_jwtauth` block a test controls.

The shared proxy on :4000 runs the CONTRIBUTING.md JWT block, so a test that
needs a different `litellm_jwtauth` config boots its own gateway on a free port
against the same database and the same Keycloak realm. The caller supplies the
`litellm_jwtauth` mapping verbatim, which is exactly what makes a config an
unfixed proxy rejects observable as a boot failure in this gateway's own log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from e2e_config import INHERITED_ENV_PREFIXES, available_port
from e2e_http import NoBody
from idp import Keycloak, stop_process_group
from proxy_client import ProxyClient, build_proxy_client

MODEL_NAME: Final = "gemini-3.8-flash"


@dataclass(slots=True)
class OwnedJwtGateway:
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
            assert self._child.poll() is None, "owned JWT gateway exited; inspect its private log"
            result = self.proxy.transport.probe("/health/liveliness", params=NoBody())
            if result.status_code == 200:
                return
            time.sleep(0.5)
        raise AssertionError("owned JWT gateway did not become ready")

    def stop(self) -> None:
        if self._child is not None:
            stop_process_group(self._child)
            assert self._child.poll() is not None, "old gateway process is still alive"


def owned_jwt_gateway(
    idp: Keycloak, directory: Path, cleanup: ExitStack, *, litellm_jwtauth: str, name: str
) -> OwnedJwtGateway:
    for env_name in ("DATABASE_URL", "LITELLM_LICENSE", "LITELLM_MASTER_KEY"):
        assert os.environ.get(env_name), f"{env_name} is required for the owned JWT gateway"
    port: Final = available_port()
    base_url: Final = f"http://127.0.0.1:{port}"
    config: Final = directory / f"{name}.yaml"
    config.write_text(
        "model_list:\n"
        f"  - model_name: {MODEL_NAME}\n"
        "    litellm_params:\n"
        f"      model: gemini/{MODEL_NAME}\n"
        "      api_key: os.environ/GEMINI_API_KEY\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  proxy_batch_write_at: 5\n"
        "  enable_jwt_auth: true\n"
        "  litellm_jwtauth:\n" + "".join(f"    {line}\n" for line in litellm_jwtauth.strip().splitlines())
    )
    environment: Final = {
        **{key: value for key, value in os.environ.items() if not key.startswith(INHERITED_ENV_PREFIXES)},
        "JWT_PUBLIC_KEY_URL": idp.jwks_url,
        "JWT_ISSUER": idp.issuer,
        "JWT_AUDIENCE": "litellm-e2e",
        "LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY": "true",
        "DISABLE_SCHEMA_UPDATE": "true",
        "STORE_MODEL_IN_DB": "True",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    gateway: Final = OwnedJwtGateway(
        base_url=base_url,
        proxy=build_proxy_client(
            base_url=base_url,
            control_plane_base_url=base_url,
            replica_urls=(base_url,),
            master_key=os.environ["LITELLM_MASTER_KEY"],
        ),
        _environment=environment,
        _command=(sys.executable, "-m", "litellm.proxy.proxy_cli", "--config", str(config), "--port", str(port)),
        _log_path=directory / f"{name}.log",
    )
    cleanup.callback(gateway.stop)
    gateway.start()
    return gateway
