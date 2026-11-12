from datetime import datetime, timezone

import pytest

from litellm.models.task_insights import Decision, DiscoveryRun, PendingRequest, TaskDefinition, TaskInsightsState
from litellm.task_insights.database import connected_database
from litellm.task_insights.worker import DiscoveryPipeline, run_once, serve

_AT = datetime(2026, 11, 11, tzinfo=timezone.utc)
_RUN = DiscoveryRun(window=(_AT, _AT), max_requests=50)


class _Store:
    def __init__(self, status: str = "idle", claimable: bool = True) -> None:
        self.status = status
        self.claimable = claimable
        self.failures: list[str] = []
        self.claimed = 0
        self.recovered = 0

    async def recover_discovery(self):
        self.recovered += 1
        return 0

    async def get_state(self):
        return TaskInsightsState(discovery_status=self.status)

    async def claim_discovery(self):
        self.claimed += 1
        return _RUN if self.status == "requested" and self.claimable else None

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


@pytest.mark.asyncio
async def test_a_discovery_another_worker_claimed_is_left_to_it():
    store = _Store("requested", claimable=False)

    await run_once(store, pipeline=None, decider=None, batch_size=10, now=lambda: _AT)

    assert (store.claimed, store.failures) == (1, [])


class _Stop(Exception):
    pass


@pytest.mark.asyncio
async def test_serve_recovers_first_and_runs_a_single_pass_when_asked():
    store = _Store()
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    await serve(
        store, pipeline=None, decider=None, batch_size=10, poll_seconds=5, now=lambda: _AT, sleep=sleep, once=True
    )

    assert (store.recovered, sleeps) == (1, [])


class _BucketingStore(_Store):
    def __init__(self, batches: list[int]) -> None:
        super().__init__()
        self.batches = batches

    async def frozen_definitions(self):
        return (
            TaskDefinition(
                task_id="t",
                name="Task",
                description="",
                inclusion_criteria="",
                exclusion_criteria="",
                status="frozen",
                source="discovered",
                map_x=0,
                map_y=0,
                created_at=_AT,
            ),
        )

    async def pending_requests(self, limit):
        size = self.batches.pop(0) if self.batches else 0
        return tuple(
            PendingRequest(request_id=f"r{index}", text_hash=f"h{index}", request_text="t", context_text=None)
            for index in range(size)
        )

    async def decisions_by_hash(self, text_hashes):
        return {}

    async def save_decisions(self, decisions, bucketed_at, taxonomy_version):
        return len(decisions)


class _Decider:
    decision_model = "jev"

    async def decide(self, tasks, request):
        return Decision(outcome="assigned", task_id="t", decision_model="jev")


@pytest.mark.asyncio
async def test_serve_runs_full_batches_back_to_back_and_waits_after_a_partial_one():
    store = _BucketingStore([4, 4, 1, 0])
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise _Stop

    with pytest.raises(_Stop):
        await serve(
            store, pipeline=None, decider=_Decider(), batch_size=4, poll_seconds=5, now=lambda: _AT, sleep=sleep, once=False
        )

    assert (store.batches, sleeps, store.recovered) == ([], [5, 5], 1)
