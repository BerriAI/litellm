import os
import socket
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import psutil

from integration._support.client import Gateway


def in_group(process: psutil.Process, group: int) -> bool:
    try:
        return os.getpgid(process.pid) == group
    except ProcessLookupError:
        return False


def group_members(group: int) -> tuple[psutil.Process, ...]:
    return tuple(process for process in psutil.process_iter() if in_group(process, group))


def signal_group(group: int, action: int) -> None:
    try:
        os.killpg(group, action)
    except ProcessLookupError:
        pass


def stop_root_process(process: subprocess.Popen[bytes]) -> bool:
    if process.poll() is not None:
        return True
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return True


@contextmanager
def owned_proxy(gateway: Gateway, directory: Path, overrides: Mapping[str, str], *, config: Path | None = None, remove_environment: tuple[str, ...] = ()) -> Iterator[Gateway]:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = reserve.getsockname()[1]
    root: Final = Path(__file__).resolve().parents[3]
    environment: Final = {
        **{name: value for name, value in os.environ.items() if name not in remove_environment},
        "LITELLM_MASTER_KEY": gateway.key,
        "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
        "STORE_MODEL_IN_DB": "True",
        **overrides,
    }
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(directory)))
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"owned-proxy-{uuid.uuid4().hex}.log").open("w") as log:
        process: Final = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "integration._support.proxy",
                "--config",
                str(config or "tests/integration/proxy_config.yaml"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--num_workers",
                "1",
                "--telemetry",
                "False",
                "--use_prisma_db_push",
                "--enforce_prisma_migration_check",
            ],
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                deadline: Final = time.monotonic() + 70
                while True:
                    assert process.poll() is None, "Owned proxy exited before readiness"
                    try:
                        if client.get("/health/readiness", timeout=2).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert time.monotonic() < deadline, "Owned proxy readiness deadline exceeded"
                    time.sleep(0.1)
                yield Gateway(client, gateway.key, gateway.upstream_url)
        finally:
            root_stopped: Final = stop_root_process(process)
            residual: Final = group_members(process.pid)
            if residual:
                signal_group(process.pid, signal.SIGTERM)
                psutil.wait_procs(residual, timeout=5)
            remaining: Final = group_members(process.pid)
            if remaining:
                signal_group(process.pid, signal.SIGKILL)
                psutil.wait_procs(remaining, timeout=3)
            process.wait(timeout=3)
            survivors: Final = group_members(process.pid)
            assert not survivors, "Owned proxy child survived cleanup"
            assert root_stopped and not remaining, "Owned proxy required forced cleanup"
