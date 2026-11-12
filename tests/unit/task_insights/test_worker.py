from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import DiscoveryRun, TaskInsightsState
from litellm.task_insights.database import connected_database
from litellm.task_insights.worker import DiscoveryPipeline, run_once

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)
_RUN = DiscoveryRun(window=(_AT, _AT), max_requests=50)


class _Store:
    def __init__(self, status: str = "idle") -> None:
        self.status = status
        self.failures: list[str] = []
        self.claimed = 0

    async def get_state(self):
        return TaskInsightsState(discovery_status=self.status)

    async def claim_discovery(self):
        self.claimed += 1
        return _RUN if self.status == "requested" else None

    async def fail_discovery(self, error, finished_at):
        self.failures.append(error)

    async def discovery_samples(self, run):
        return ()

    async def taxonomy_version(self):
        return _AT

    async def frozen_definitions(self):
        return ()


@pytest.mark.asyncio
async def test_without_gateway_access_a_queued_discovery_fails_with_the_reason():
    store = _Store("requested")

    assert await run_once(store, pipeline=None, decider=None, batch_size=10, now=lambda: _AT) == 0

    assert store.claimed == 1
    assert store.failures and "LITELLM_TASK_INSIGHTS_GATEWAY_KEY" in store.failures[0]


@pytest.mark.asyncio
async def test_without_gateway_access_nothing_happens_when_no_discovery_is_queued():
    store = _Store("idle")

    await run_once(store, pipeline=None, decider=None, batch_size=10, now=lambda: _AT)

    assert (store.claimed, store.failures) == (0, [])


@pytest.mark.asyncio
async def test_with_a_pipeline_the_queued_discovery_runs_through_it():
    store = _Store("requested")
    pipeline = DiscoveryPipeline(embedder=object(), clusterer=object(), synthesizer=object())

    await run_once(store, pipeline=pipeline, decider=None, batch_size=10, now=lambda: _AT)

    assert store.claimed == 1
    assert store.failures and "at least" in store.failures[0]


class _Connection:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def connect(self):
        self.events.append("connect")

    async def disconnect(self):
        self.events.append("disconnect")


@pytest.mark.asyncio
async def test_the_owned_connection_closes_even_when_the_work_fails():
    connection = _Connection()

    async def seed() -> None:
        async with connected_database(lambda: connection) as database:
            connection.events.append("used" if database is connection else "wrong handle")
            raise RuntimeError("seed failed")

    with pytest.raises(RuntimeError):
        await seed()

    assert connection.events == ["connect", "used", "disconnect"]
