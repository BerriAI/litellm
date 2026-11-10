import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import stop_root_process

MASTER_KEY: Final = "sk-rust-forward-integration-key-not-a-secret"
WORKERS: Final = "2"
HEALTHCHECK_SECONDS: Final = "120"


class EntryPoint(Enum):
    LITELLM = "litellm.proxy.proxy_cli"
    GATEWAY = "gateway.launch"


@dataclass(frozen=True, slots=True)
class ForwardedProxy:
    port: int
    log: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _serving(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health/liveliness", timeout=5, trust_env=False).status_code == 200
    except httpx.TransportError:
        return False


def _launch(entry_point: EntryPoint, config: Path) -> tuple[tuple[str, ...], Mapping[str, str]]:
    match entry_point:
        case EntryPoint.LITELLM:
            return (
                (
                    "--config",
                    str(config),
                    "--num_workers",
                    WORKERS,
                    "--timeout_worker_healthcheck",
                    HEALTHCHECK_SECONDS,
                ),
                MappingProxyType({}),
            )
        case EntryPoint.GATEWAY:
            return (
                ("--workers", WORKERS, "--timeout-worker-healthcheck", HEALTHCHECK_SECONDS),
                MappingProxyType({"CONFIG_FILE_PATH": str(config), "LITELLM_PGBOUNCER_ENABLED": "false"}),
            )


@contextmanager
def forwarded_proxy(
    entry_point: EntryPoint,
    config: Mapping[str, JsonValue],
    directory: Path,
    arguments: tuple[str, ...] = (),
    environment: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[ForwardedProxy]:
    """``entry_point`` on 0.0.0.0 with two workers, ``LITELLM_RUST=1`` and no database. It must exit 0 on SIGTERM."""
    proxy: Final = ForwardedProxy(allocate_port(), directory / f"{entry_point.name.lower()}.log")
    config_file: Final = directory / "config.yaml"
    config_file.write_text(json.dumps(config))
    launch_arguments, launch_environment = _launch(entry_point, config_file)
    inherited: Final = {
        name: value for name, value in os.environ.items() if not name.startswith("DATABASE_") and name != "DIRECT_URL"
    }
    with (
        proxy.log.open("wb") as output,
        subprocess.Popen(
            (
                sys.executable,
                "-I",
                "-m",
                entry_point.value,
                "--host",
                "0.0.0.0",
                "--port",
                str(proxy.port),
                *launch_arguments,
                *arguments,
            ),
            env={
                **inherited,
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "LITELLM_RUST": "true",
                "STORE_MODEL_IN_DB": "False",
                "PYTHON_DOTENV_DISABLED": "1",
                **launch_environment,
                **environment,
            },
            stdout=output,
            stderr=output,
        ) as process,
    ):
        try:
            assert eventually(lambda: _serving(proxy.url), bool, seconds=180), proxy.log.read_text()
            yield proxy
        finally:
            assert stop_root_process(process), proxy.log.read_text()
        assert process.returncode == 0, proxy.log.read_text()
