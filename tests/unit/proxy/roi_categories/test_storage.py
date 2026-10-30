import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.proxy.roi_categories.models import Category, Settings
from litellm.proxy.roi_categories.service import process_batch
from litellm.proxy.roi_categories.storage import CategoryStorage
from litellm.router_strategy.complexity_router.jev_classifier import JevSystemOneRequest, JevSystemOneResponse
from litellm.rust_bridge.clickhouse import SpendStorageConfig

NOW: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)
SETTINGS: Final = Settings(
    version="version",
    started_at=NOW,
    activated_at=NOW,
    discovery_model="discover",
    classifier_model="classify",
    categories=(Category(id="build", name="Build", description="CI tasks"),),
)


class TrafficServer:
    def __init__(self) -> None:
        self.assignments: tuple[object, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        sql: Final = request.content.decode()
        if sql.startswith("INSERT"):
            self.assignments = (*self.assignments, json.loads(sql.split("\n", 1)[1]))
            return httpx.Response(200)
        return httpx.Response(200, text='{"request_id":"one","excerpt":"Build failed","attempts":2}\n')


class UnavailableClassifier:
    async def evaluate(
        self, request: JevSystemOneRequest, timeout_s: float, request_kwargs: Mapping[str, object] | None = None
    ) -> JevSystemOneResponse:
        raise httpx.ConnectError("Provider unavailable")


@pytest.mark.asyncio
async def test_provider_failure_persists_unclassified_attempt_instead_of_losing_request() -> None:
    server: Final = TrafficServer()
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        await process_batch(
            SETTINGS, CategoryStorage(SpendStorageConfig("http://clickhouse"), client), UnavailableClassifier(), NOW
        )
    assert server.assignments == (
        {"request_id": "one", "version": "version", "category_id": "unclassified", "attempts": 3, "completed": False},
    )


@pytest.mark.asyncio
async def test_report_keeps_unknown_cost_counts_and_parameterizes_user_input() -> None:
    injected: Final = "'; DROP TABLE spend_logs; --"

    def serve(request: httpx.Request) -> httpx.Response:
        assert injected not in request.content.decode()
        assert request.url.params["param_model"] == injected
        assert request.url.params["param_user"] == "user-1"
        assert request.url.params["password"] == "fixture-password"
        return httpx.Response(
            200,
            text='{"day":"2026-01-02","category_id":"unclassified","model":"unknown","spend":0,"requests":1,"unknown_cost_requests":1}\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        rows: Final = await CategoryStorage(
            SpendStorageConfig("http://clickhouse?password=fixture-password"), client
        ).report(SETTINGS, NOW, NOW, injected, "user-1")
    assert rows[0].unknown_cost_requests == 1
    assert rows[0].spend == 0
    assert rows[0].category_id == "unclassified"
