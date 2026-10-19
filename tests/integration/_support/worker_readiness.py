from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from itertools import accumulate
from types import MappingProxyType
from typing import Final

import httpx

from integration._support.client import Gateway, JSON_OBJECT, object_value
from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS


def _record_latest_readback(
    latest_by_worker: Mapping[int, int],
    observation: tuple[int, int],
) -> Mapping[int, int]:
    worker_pid, count = observation
    return MappingProxyType({**latest_by_worker, worker_pid: count})


def wait_for_worker_readiness(
    gateway: Gateway,
    group: str,
    deployments: int,
    workers: int,
) -> None:
    deadline: Final = time.monotonic() + (30 if workers == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS * 2 + 30)
    headers: Final = {"Authorization": f"Bearer {gateway.key}"}

    def readbacks() -> Iterator[tuple[int, int]]:
        while time.monotonic() < deadline:
            timeout: Final = deadline - time.monotonic()
            with httpx.Client(
                base_url=str(gateway.client.base_url),
                timeout=timeout,
                trust_env=False,
                verify=False,
            ) as client:
                debug_response: Final = client.get("/debug/memory/summary", headers=headers)
                response: Final = client.get(
                    "/model/info",
                    headers=headers,
                )
            assert debug_response.status_code == 200, debug_response.text
            assert response.status_code == 200, response.text
            assert debug_response.extensions.get("network_stream") is not None
            assert debug_response.extensions.get("network_stream") is response.extensions.get("network_stream")
            debug_body: Final = JSON_OBJECT.validate_json(debug_response.content)
            worker_pid: Final = debug_body.get("worker_pid")
            assert isinstance(worker_pid, int), debug_body
            body: Final = JSON_OBJECT.validate_json(response.content)
            entries: Final = body.get("data")
            assert isinstance(entries, list), body
            count: Final = sum(object_value(entry).get("model_name") == group for entry in entries)
            yield worker_pid, count

    initial_counts: Final[Mapping[int, int]] = MappingProxyType({})
    latest_counts: Final = accumulate(readbacks(), _record_latest_readback, initial=initial_counts)
    for latest_by_worker in latest_counts:
        if len(latest_by_worker) >= workers and all(count == deployments for count in latest_by_worker.values()):
            return
    raise AssertionError(f"Expected {deployments} deployments on {workers} workers before the readiness deadline")
