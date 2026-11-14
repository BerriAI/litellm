from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest

from tests.integration._support.client import eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.rust_forward import MASTER_KEY, EntryPoint, forwarded_proxy

TRUSTED_BALANCER: Final = "127.0.0.3"
CLIENT: Final = "127.0.0.2"


@dataclass(frozen=True, slots=True)
class Caller:
    connects_from: str
    forwarded_for: str
    resolved_ip: str


CALLERS: Final = (
    Caller(CLIENT, "1.2.3.4", resolved_ip=CLIENT),
    Caller(TRUSTED_BALANCER, "203.0.113.9", resolved_ip="203.0.113.9"),
    Caller(CLIENT, f"203.0.113.9, {TRUSTED_BALANCER}", resolved_ip=CLIENT),
)


def _client(url: str, local_address: str) -> httpx.Client:
    return httpx.Client(
        base_url=url, transport=httpx.HTTPTransport(local_address=local_address), trust_env=False, timeout=5
    )


def _resolved_ip(url: str, caller: Caller) -> str:
    with _client(url, caller.connects_from) as client:
        response: Final = client.get(
            "/v1/mcp/network/client-ip",
            headers={"Authorization": f"Bearer {MASTER_KEY}", "X-Forwarded-For": caller.forwarded_for},
        )
    assert response.status_code == 200, response.text
    return response.json()["ip"]


@pytest.mark.parametrize("server_arguments", [(), ("--run_gunicorn",)])
def test_only_the_trusted_balancer_can_name_the_client_ip_behind_rust(
    tmp_path: Path, server_arguments: tuple[str, ...]
) -> None:
    metrics_port: Final = allocate_port()
    with forwarded_proxy(
        EntryPoint.LITELLM,
        {
            "model_list": [],
            "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "use_x_forwarded_for": False},
        },
        tmp_path,
        ("--prometheus_metrics_port", str(metrics_port), *server_arguments),
        {"FORWARDED_ALLOW_IPS": TRUSTED_BALANCER},
    ) as proxy:
        with _client(f"http://{CLIENT}:{metrics_port}", CLIENT) as external:
            metrics: Final = external.get("/metrics", follow_redirects=True)
        assert metrics.status_code == 200, metrics.text
        assert tuple(_resolved_ip(proxy.url, caller) for caller in CALLERS) == tuple(
            caller.resolved_ip for caller in CALLERS
        )


@pytest.mark.parametrize("server_arguments", [("--no-proxy-headers",), ("--forwarded-allow-ips", "127.0.0.1")])
def test_gateway_launcher_logs_the_real_client_address_behind_rust(
    tmp_path: Path, server_arguments: tuple[str, ...]
) -> None:
    with forwarded_proxy(
        EntryPoint.GATEWAY,
        {"model_list": [], "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"}},
        tmp_path,
        server_arguments,
    ) as proxy:
        with _client(proxy.url, CLIENT) as external:
            response: Final = external.get(
                "/health/liveliness?client-ip-proof", headers={"X-Forwarded-For": "198.51.100.10"}
            )
        assert response.status_code == 200, response.text
        logged: Final = eventually(
            lambda: tuple(
                line for line in proxy.log.read_text().splitlines() if "GET /health/liveliness?client-ip-proof " in line
            ),
            bool,
            seconds=10,
        )
    assert len(logged) == 1 and f"{CLIENT}:" in logged[0], logged


@pytest.mark.parametrize("entry_point", list(EntryPoint), ids=[entry_point.name for entry_point in EntryPoint])
def test_the_python_child_is_started_with_the_supervisors_own_command(tmp_path: Path, entry_point: EntryPoint) -> None:
    with forwarded_proxy(
        entry_point,
        {"model_list": [], "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"}},
        tmp_path,
    ):
        supervisor: Final = next(
            started for started in psutil.Process().children() if entry_point.value in started.cmdline()
        )
        own_command: Final = supervisor.cmdline()[1:]
        python_children: Final = [
            child.cmdline()[1:] for child in supervisor.children() if entry_point.value in child.cmdline()
        ]
    assert "-I" in own_command
    assert [child[: len(own_command)] for child in python_children] == [own_command]
