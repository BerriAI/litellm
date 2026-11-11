import subprocess
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Final

import httpx
import pytest
from tenacity import Retrying, retry_if_exception_type, stop_after_delay, wait_fixed

CLICKHOUSE_IMAGE: Final = (
    "clickhouse/clickhouse-server:26.9.6.6@sha256:eb4870e7ca7ed70c259eebfcfbee6cf797017f6b5436c2926bbbfe3d4d28486e"
)


@contextmanager
def clickhouse_service() -> Generator[str]:
    container: Final = subprocess.run(
        (
            "docker",
            "run",
            "--rm",
            "--detach",
            "--env",
            "CLICKHOUSE_SKIP_USER_SETUP=1",
            "--publish",
            "127.0.0.1::8123",
            CLICKHOUSE_IMAGE,
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    try:
        address: Final = subprocess.run(
            ("docker", "port", container, "8123/tcp"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        url: Final = f"http://{address}"
        with httpx.Client(timeout=1, trust_env=False) as client:
            for attempt in Retrying(
                retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
                stop=stop_after_delay(30),
                wait=wait_fixed(0.1),
                reraise=True,
            ):
                with attempt:
                    client.get(f"{url}/ping").raise_for_status()
        yield url
    finally:
        subprocess.run(("docker", "rm", "--force", container), check=True, capture_output=True, timeout=30)


@pytest.fixture
def clickhouse_url() -> Iterator[str]:
    with clickhouse_service() as url:
        yield url
