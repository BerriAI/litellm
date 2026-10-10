import os
import select
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, patch

import pytest
import redis

from litellm._redis import _get_redis_env_kwarg_mapping, get_redis_client
from litellm._redis_credential_provider import _token_cache


class _RedisLogReader:
    def __init__(self, process: subprocess.Popen[bytes], log_path: Path) -> None:
        if process.stdout is None:
            pytest.fail("Redis stdout is unavailable")
        self.stdout = process.stdout
        self.log_path = log_path
        self.buffer = b""

    def wait_for(self, marker: str, timeout: float) -> None:
        deadline: Final = time.monotonic() + timeout
        marker_bytes: Final = marker.encode()
        while time.monotonic() < deadline:
            line_end: Final = self.buffer.find(b"\n")
            if line_end >= 0:
                line: Final = self.buffer[: line_end + 1]
                self.buffer = self.buffer[line_end + 1 :]
                with self.log_path.open("ab") as log:
                    log.write(line)
                if marker_bytes in line:
                    return
                continue
            remaining: Final = deadline - time.monotonic()
            readable: Final = select.select((self.stdout,), (), (), remaining)[0]
            if not readable:
                break
            chunk: Final = os.read(self.stdout.fileno(), 4096)
            if not chunk:
                break
            self.buffer += chunk
        self.log_path.write_bytes(self.log_path.read_bytes() + self.buffer)
        pytest.fail(f"Redis did not report {marker!r}: {self.log_path.read_text()}")


@pytest.fixture
def clean_cluster_iam_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in ("REDIS_URL", "REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES", *_get_redis_env_kwarg_mapping()):
        monkeypatch.delenv(var, raising=False)
    _token_cache.clear()
    yield
    _token_cache.clear()


@pytest.fixture
def authenticated_redis_cluster(tmp_path: Path, unused_tcp_port_factory: Callable[[], int]) -> Iterator[int]:
    server: Final = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server is required for the cluster authentication regression tests")
    port: Final = unused_tcp_port_factory()
    bus_port: Final = unused_tcp_port_factory()
    log_path: Final = tmp_path / "redis.log"
    config: Final = tmp_path / "redis.conf"
    config.write_text(
        f"bind 127.0.0.1\nport {port}\ncluster-port {bus_port}\n"
        f'cluster-enabled yes\ncluster-config-file "{tmp_path / "nodes.conf"}"\n'
        f'dir "{tmp_path}"\nsave ""\nappendonly no\n'
    )
    log_path.write_text("")
    process: Final = subprocess.Popen(
        (server, str(config)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    reader: Final = _RedisLogReader(process, log_path)
    try:
        reader.wait_for("Ready to accept connections", timeout=10)
        with redis.Redis(host="127.0.0.1", port=port, socket_timeout=1, socket_connect_timeout=1) as admin:
            admin.ping()
            admin.execute_command("CLUSTER", "ADDSLOTS", *range(16384))
            reader.wait_for("Cluster state changed: ok", timeout=10)
            if admin.cluster("INFO")["cluster_state"] != "ok":
                pytest.fail(f"Redis cluster did not become ready: {log_path.read_text()}")
            admin.execute_command(
                "ACL", "SETUSER", "identity-object-id", "on", ">local-fixture-token", "allcommands", "allkeys"
            )
            admin.execute_command("ACL", "SETUSER", "default", "resetpass", ">local-fixture-token")
        yield port
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


def test_sync_cluster_authenticates_with_azure_credentials(
    clean_cluster_iam_environment: None, monkeypatch: pytest.MonkeyPatch, authenticated_redis_cluster: int
) -> None:
    monkeypatch.setenv("REDIS_USERNAME", "identity-object-id")
    credential: Final = MagicMock()
    credential.get_token.return_value = SimpleNamespace(token="local-fixture-token")

    with patch("azure.identity.DefaultAzureCredential", return_value=credential):
        with get_redis_client(
            startup_nodes=[{"host": "127.0.0.1", "port": authenticated_redis_cluster}],
            azure_redis_ad_token=True,
            password="stale-password",
            socket_timeout=1,
            socket_connect_timeout=1,
        ) as client:
            assert client.ping() is True
            assert client.set("iam-regression", "success") is True
            assert client.get("iam-regression") == b"success"


def test_sync_cluster_authenticates_with_gcp_credentials(
    clean_cluster_iam_environment: None, authenticated_redis_cluster: int
) -> None:
    iam_client: Final = MagicMock()
    iam_client.generate_access_token.return_value = SimpleNamespace(access_token="local-fixture-token")

    with patch("google.cloud.iam_credentials_v1.IAMCredentialsClient", return_value=iam_client):
        with get_redis_client(
            startup_nodes=[{"host": "127.0.0.1", "port": authenticated_redis_cluster}],
            gcp_service_account="projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com",
            username="stale-user",
            password="stale-password",
            socket_timeout=1,
            socket_connect_timeout=1,
        ) as client:
            assert client.ping() is True
            assert client.set("iam-regression", "success") is True
            assert client.get("iam-regression") == b"success"
