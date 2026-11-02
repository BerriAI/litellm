from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Final

import httpx

from tests.integration._support.client import Gateway, Scenario

DEPLOYMENT: Final = "deployment-under-test"
CREATING_WORKER: Final = "101"
LAGGING_WORKER: Final = "202"


@dataclass(frozen=True, slots=True)
class LaggingProxy:
    reachable_at: float
    synced_at: float
    arrivals: Iterator[int] = field(default_factory=itertools.count)
    created: threading.Event = field(default_factory=threading.Event)

    def _worker(self) -> str:
        arrival: Final = next(self.arrivals)
        reachable: Final = time.monotonic() >= self.reachable_at
        return LAGGING_WORKER if reachable and arrival % 2 else CREATING_WORKER

    def respond(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/model/new":
            self.created.set()
            return httpx.Response(200, json={"model_info": {"id": DEPLOYMENT}})
        worker: Final = self._worker()
        serves: Final = self.created.is_set() and (worker == CREATING_WORKER or time.monotonic() >= self.synced_at)
        return httpx.Response(
            200,
            json={"model_ids": [DEPLOYMENT] if serves else []},
            headers={"x-integration-worker-pid": worker},
        )


def _created_model(proxy: LaggingProxy, workers: int) -> str:
    with (
        httpx.Client(base_url="http://127.0.0.1:4000", transport=httpx.MockTransport(proxy.respond)) as client,
        ExitStack() as cleanups,
    ):
        name: Final = Scenario(Gateway(client, "sk-test", "http://127.0.0.1:4001", workers=workers), cleanups).model()
        cleanups.pop_all()
        return name


def test_model_on_multi_worker_proxy_returns_only_once_every_worker_serves_it() -> None:
    started: Final = time.monotonic()
    proxy: Final = LaggingProxy(reachable_at=started + 0.3, synced_at=started + 0.8)

    _created_model(proxy, workers=2)

    assert time.monotonic() >= proxy.synced_at


def test_model_on_single_worker_proxy_returns_without_probing_workers() -> None:
    started: Final = time.monotonic()
    proxy: Final = LaggingProxy(reachable_at=started, synced_at=started + 60)

    _created_model(proxy, workers=1)

    assert time.monotonic() < proxy.synced_at
