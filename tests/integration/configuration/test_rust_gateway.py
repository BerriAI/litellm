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

TRUSTED_BALANCER: Final = "127.0.0.3"


def ready(client: httpx.Client) -> bool:
    try:
        return client.get("/health/liveliness").status_code == 200
    except httpx.TransportError:
        return False


@pytest.mark.parametrize("server_arguments", [(), ("--run_gunicorn",)])
def client_ip(base_url: str, local_address: str, key: str, forwarded_for: str) -> httpx.Response:
    with httpx.Client(
        base_url=base_url,
        transport=httpx.HTTPTransport(local_address=local_address),
        trust_env=False,
        timeout=5,
    ) as client:
        return client.get(
            "/v1/mcp/network/client-ip",
            headers={"Authorization": f"Bearer {key}", "X-Forwarded-For": forwarded_for},
        )


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
        "LITELLM_RUST_GATEWAY": "true",
        "FORWARDED_ALLOW_IPS": TRUSTED_BALANCER,
    }
    with (tmp_path / "proxy.log").open("wb") as output:
        with subprocess.Popen(
            (
                sys.executable,
                "-I",
                "-m",
                "litellm.proxy.proxy_cli",
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
                public: Final = f"http://127.0.0.1:{public_port}"
                spoofed: Final = client_ip(public, "127.0.0.2", gateway.key, "1.2.3.4")
                assert spoofed.status_code == 200, spoofed.text
                assert spoofed.json() == {"ip": "127.0.0.2"}
                behind_balancer: Final = client_ip(public, TRUSTED_BALANCER, gateway.key, "203.0.113.9")
                assert behind_balancer.status_code == 200, behind_balancer.text
                assert behind_balancer.json() == {"ip": "203.0.113.9"}
                borrowed: Final = client_ip(public, "127.0.0.2", gateway.key, f"203.0.113.9, {TRUSTED_BALANCER}")
                assert borrowed.status_code == 200, borrowed.text
                assert borrowed.json() == {"ip": "127.0.0.2"}
            finally:
                assert stop_root_process(supervisor), (tmp_path / "proxy.log").read_text()
            assert supervisor.returncode == 0, (tmp_path / "proxy.log").read_text()
