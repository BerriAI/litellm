from __future__ import annotations

import time
from itertools import groupby, islice
from typing import Final, Iterator

import httpx

from integration._support.client import Gateway, JSON_OBJECT, object_value
from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS


def wait_for_worker_readiness(
    gateway: Gateway,
    group: str,
    deployments: int,
    workers: int,
) -> None:
    required_polls: Final = workers * 5
    deadline: Final = time.monotonic() + (30 if workers == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS * 2 + 30)

    def readbacks() -> Iterator[bool]:
        while time.monotonic() < deadline:
            timeout: Final = deadline - time.monotonic()
            with httpx.Client(
                base_url=str(gateway.client.base_url),
                timeout=timeout,
                trust_env=False,
                verify=False,
            ) as client:
                response: Final = client.get(
                    "/model/info",
                    headers={"Authorization": f"Bearer {gateway.key}"},
                )
            assert response.status_code == 200, response.text
            body: Final = JSON_OBJECT.validate_json(response.content)
            entries: Final = body.get("data")
            assert isinstance(entries, list), body
            count: Final = sum(object_value(entry).get("model_name") == group for entry in entries)
            yield count == deployments

    for matches, polls in groupby(readbacks()):
        if matches and len(tuple(islice(polls, required_polls))) == required_polls:
            return
    raise AssertionError(
        f"Expected {deployments} deployments on {workers} workers for {required_polls} consecutive readbacks"
    )
