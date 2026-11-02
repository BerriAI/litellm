import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import stop_root_process


def ready(client: httpx.Client) -> bool:
    try:
        return client.get("/health/liveliness").status_code == 200
    except httpx.TransportError:
        return False


@pytest.mark.parametrize("server_arguments", [(), ("--run_gunicorn",)])
def test_gateway_preserves_metrics_bind_and_mcp_client_ip(
    gateway: Gateway, tmp_path: Path, server_arguments: tuple[str, ...]
) -> None:
    config: Final = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [],
                "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "use_x_forwarded_for": False},
            }
        )
    )
    metrics_port: Final = allocate_port()
    public_port: Final = allocate_port()
    environment: Final = {
        **{key: value for key, value in os.environ.items() if key not in ("DATABASE_URL", "DATABASE_URL_READ_REPLICA")},
        "LITELLM_MASTER_KEY": gateway.key,
        "STORE_MODEL_IN_DB": "False",
        "PYTHON_DOTENV_DISABLED": "1",
    }
    with (tmp_path / "proxy.log").open("wb") as output:
        with subprocess.Popen(
            (
                sys.executable,
                "-I",
                "-m",
                "litellm.proxy.proxy_cli",
                "--rust_gateway",
                "--config",
                str(config),
                "--host",
                "0.0.0.0",
                "--port",
                str(public_port),
                "--num_workers",
                "2",
                "--timeout_worker_healthcheck",
                "120",
                "--prometheus_metrics_port",
                str(metrics_port),
                *server_arguments,
            ),
            env=environment,
            stdout=output,
            stderr=output,
        ) as supervisor:
            try:
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{public_port}",
                    transport=httpx.HTTPTransport(local_address="127.0.0.2"),
                    trust_env=False,
                    follow_redirects=True,
                    timeout=5,
                ) as external:
                    assert eventually(lambda: ready(external), bool, seconds=180), (tmp_path / "proxy.log").read_text()
                    response: Final = external.get(f"http://127.0.0.2:{metrics_port}/metrics")
                    assert response.status_code == 200, response.text
                    client_ip: Final = external.get(
                        "/v1/mcp/network/client-ip",
                        headers={"Authorization": f"Bearer {gateway.key}", "X-Forwarded-For": "1.2.3.4"},
                    )
                    assert client_ip.status_code == 200, client_ip.text
                    assert client_ip.json() == {"ip": "127.0.0.2"}
            finally:
                assert stop_root_process(supervisor), (tmp_path / "proxy.log").read_text()
            assert supervisor.returncode == 0, (tmp_path / "proxy.log").read_text()
