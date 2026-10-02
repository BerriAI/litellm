import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_strategy.adaptive_router.update_queue import (
    AdaptiveRouterUpdateQueue,
)


@pytest.fixture
def queue():
    return AdaptiveRouterUpdateQueue()


@pytest.fixture
def mock_prisma():
    """Prisma client with both adaptive router models stubbed as AsyncMocks."""
    p = MagicMock()
    p.db.litellm_adaptiverouterstate.find_unique = AsyncMock(return_value=None)
    p.db.litellm_adaptiverouterstate.upsert = AsyncMock()
    p.db.litellm_adaptiveroutersession.upsert = AsyncMock()
    return p


@pytest.mark.asyncio
async def test_add_state_delta_aggregates_same_key(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "general", "gpt-4", 0.0, 1.0)
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 1


@pytest.mark.asyncio
async def test_add_state_delta_separate_keys(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 1.0, 0.0)
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 2


@pytest.mark.asyncio
async def test_add_session_state_last_write_wins(queue):
    await queue.add_session_state("s1", "r1", "gpt-4", {"misalignment_count": 1})
    await queue.add_session_state("s1", "r1", "gpt-4", {"misalignment_count": 5})
    sizes = await queue.queue_size()
    assert sizes["session_pending"] == 1

    flushed = []
    p = MagicMock()

    async def upsert(**kwargs):
        flushed.append(kwargs)

    p.db.litellm_adaptiveroutersession.upsert = upsert
    await queue.flush_session_to_db(p)
    assert len(flushed) == 1
    assert flushed[0]["data"]["update"]["misalignment_count"] == 5


@pytest.mark.asyncio
async def test_flush_state_drains_aggregator(queue, mock_prisma):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 0.0, 1.0)
    n = await queue.flush_state_to_db(mock_prisma)
    assert n == 2
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 0


@pytest.mark.asyncio
async def test_flush_state_sums_correctly(queue, mock_prisma):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "general", "gpt-4", 2.0, 1.0)
    await queue.flush_state_to_db(mock_prisma)
    # find_unique returned None (cold start), so alpha = 1+2 = 3, beta = 0+1 = 1
    call = mock_prisma.db.litellm_adaptiverouterstate.upsert.call_args
    assert call.kwargs["data"]["create"]["alpha"] == 3.0
    assert call.kwargs["data"]["create"]["beta"] == 1.0
    assert call.kwargs["data"]["create"]["total_samples"] == 2


@pytest.mark.asyncio
async def test_flush_session_drains_aggregator(queue, mock_prisma):
    await queue.add_session_state("s1", "r1", "gpt-4", {"classified_type": "general"})
    n = await queue.flush_session_to_db(mock_prisma)
    assert n == 1
    sizes = await queue.queue_size()
    assert sizes["session_pending"] == 0


@pytest.mark.asyncio
async def test_flush_empty_queue_returns_zero(queue, mock_prisma):
    assert await queue.flush_state_to_db(mock_prisma) == 0
    assert await queue.flush_session_to_db(mock_prisma) == 0


@pytest.mark.asyncio
async def test_flush_state_isolation_from_concurrent_adds(queue, mock_prisma):
    """Adds during a flush should land in the NEW aggregator, not the drained batch."""
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    flush_task = asyncio.create_task(queue.flush_state_to_db(mock_prisma))
    # Yield control so the flush task can swap the aggregator before we add again.
    await asyncio.sleep(0)
    await queue.add_state_delta("r1", "general", "gpt-5", 2.0, 0.0)
    await flush_task
    sizes = await queue.queue_size()
    assert sizes["state_pending"] == 1


@pytest.mark.asyncio
async def test_max_size_observability(queue):
    await queue.add_state_delta("r1", "general", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "writing", "gpt-4", 1.0, 0.0)
    await queue.add_state_delta("r1", "code_generation", "gpt-4", 1.0, 0.0)
    sizes = await queue.queue_size()
    assert sizes["max_state_seen"] >= 3


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["state", "session"])
async def test_failed_flush_retains_only_failed_rows_and_reports_successes(queue, mock_prisma, kind):
    for model in ("a", "b", "c"):
        if kind == "state":
            await queue.add_state_delta("router", "general", model, 1.0, 0.5)
        else:
            await queue.add_session_state("session", "router", model, {"turn_count": 1})
    table = getattr(mock_prisma.db, f"litellm_adaptiverouter{kind}")
    table.upsert.side_effect = [None, RuntimeError("database unavailable"), None, None]
    flush = getattr(queue, f"flush_{kind}_to_db")

    assert await flush(mock_prisma) == 2
    assert (await queue.queue_size())[f"{kind}_pending"] == 1
    assert await flush(mock_prisma) == 1
    assert (await queue.queue_size())[f"{kind}_pending"] == 0
    assert [call.kwargs["data"]["create"]["model_name"] for call in table.upsert.call_args_list] == ["a", "b", "c", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["state", "session"])
async def test_failed_flush_merges_concurrent_updates_without_replaying_successes(queue, mock_prisma, kind):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_write(**kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("database unavailable")

    table = getattr(mock_prisma.db, f"litellm_adaptiverouter{kind}")
    table.upsert.side_effect = fail_write
    flush = getattr(queue, f"flush_{kind}_to_db")
    if kind == "state":
        await queue.add_state_delta("router", "general", "model", 1.0, 0.5)
    else:
        await queue.add_session_state("session", "router", "model", {"turn_count": 1})
    task = asyncio.create_task(flush(mock_prisma))
    await started.wait()
    if kind == "state":
        await queue.add_state_delta("router", "general", "model", 2.0, 1.0)
    else:
        await queue.add_session_state("session", "router", "model", {"turn_count": 2})
    release.set()
    assert await task == 0

    table.upsert.side_effect = None
    assert await flush(mock_prisma) == 1
    update = table.upsert.call_args.kwargs["data"]["update"]
    assert update == (
        {"alpha": {"increment": 3.0}, "beta": {"increment": 1.5}, "total_samples": {"increment": 2}}
        if kind == "state"
        else {"turn_count": 2}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["state", "session"])
async def test_cancelled_flush_retains_unfinished_rows_without_replaying_acknowledged_rows(queue, mock_prisma, kind):
    started = asyncio.Event()
    blocked = asyncio.Event()

    async def write(**kwargs):
        if kwargs["data"]["create"]["model_name"] == "b":
            started.set()
            await blocked.wait()

    for model in ("a", "b", "c"):
        if kind == "state":
            await queue.add_state_delta("router", "general", model, 1.0, 0.5)
        else:
            await queue.add_session_state("session", "router", model, {"turn_count": 1})
    table = getattr(mock_prisma.db, f"litellm_adaptiverouter{kind}")
    table.upsert.side_effect = write
    flush = getattr(queue, f"flush_{kind}_to_db")
    task = asyncio.create_task(flush(mock_prisma))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (await queue.queue_size())[f"{kind}_pending"] == 2
    table.upsert.side_effect = None
    assert await flush(mock_prisma) == 2
    assert [call.kwargs["data"]["create"]["model_name"] for call in table.upsert.call_args_list] == ["a", "b", "b", "c"]


@pytest.mark.asyncio
async def test_overlapping_session_flushes_cannot_persist_an_older_snapshot_last(queue, mock_prisma):
    started = asyncio.Event()
    release = asyncio.Event()
    stored = []

    async def write(**kwargs):
        turn_count = kwargs["data"]["update"]["turn_count"]
        if turn_count == 1:
            started.set()
            await release.wait()
        stored.append(turn_count)

    mock_prisma.db.litellm_adaptiveroutersession.upsert.side_effect = write
    await queue.add_session_state("session", "router", "model", {"turn_count": 1})
    first = asyncio.create_task(queue.flush_session_to_db(mock_prisma))
    await started.wait()
    await queue.add_session_state("session", "router", "model", {"turn_count": 2})
    second = asyncio.create_task(queue.flush_session_to_db(mock_prisma))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [1, 1]
    assert stored == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["state", "session"])
async def test_repeated_write_failures_keep_the_queue_until_recovery(queue, mock_prisma, kind):
    if kind == "state":
        await queue.add_state_delta("router", "general", "model", 1.0, 0.5)
    else:
        await queue.add_session_state("session", "router", "model", {"turn_count": 1})
    table = getattr(mock_prisma.db, f"litellm_adaptiverouter{kind}")
    table.upsert.side_effect = RuntimeError("database unavailable")
    flush = getattr(queue, f"flush_{kind}_to_db")
    for _ in range(3):
        assert await flush(mock_prisma) == 0
        assert (await queue.queue_size())[f"{kind}_pending"] == 1
    table.upsert.side_effect = None

    assert await flush(mock_prisma) == 1
    assert await flush(mock_prisma) == 0
    update = table.upsert.call_args.kwargs["data"]["update"]
    assert update == (
        {"alpha": {"increment": 1.0}, "beta": {"increment": 0.5}, "total_samples": {"increment": 1}}
        if kind == "state"
        else {"turn_count": 1}
    )


@pytest.mark.asyncio
async def test_unacknowledged_state_commit_is_retried_with_at_least_once_semantics(queue, mock_prisma):
    persisted = []

    async def commit_then_lose_response(**kwargs):
        persisted.append(kwargs["data"]["update"])
        if len(persisted) == 1:
            raise ConnectionError("commit response lost")

    mock_prisma.db.litellm_adaptiverouterstate.upsert.side_effect = commit_then_lose_response
    await queue.add_state_delta("router", "general", "model", 1.0, 0.5)

    assert await queue.flush_state_to_db(mock_prisma) == 0
    assert (await queue.queue_size())["state_pending"] == 1
    assert await queue.flush_state_to_db(mock_prisma) == 1
    assert (
        persisted == [{"alpha": {"increment": 1.0}, "beta": {"increment": 0.5}, "total_samples": {"increment": 1}}] * 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["state", "session"])
async def test_queue_high_water_mark_includes_restored_rows_and_new_keys(queue, mock_prisma, kind):
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_write(**kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("database unavailable")

    table = getattr(mock_prisma.db, f"litellm_adaptiverouter{kind}")
    table.upsert.side_effect = fail_write
    if kind == "state":
        await queue.add_state_delta("router", "general", "a", 1.0, 0.5)
    else:
        await queue.add_session_state("session", "router", "a", {"turn_count": 1})
    task = asyncio.create_task(getattr(queue, f"flush_{kind}_to_db")(mock_prisma))
    await started.wait()
    for model in ("b", "c"):
        if kind == "state":
            await queue.add_state_delta("router", "general", model, 1.0, 0.5)
        else:
            await queue.add_session_state("session", "router", model, {"turn_count": 1})
    release.set()

    assert await task == 0
    sizes = await queue.queue_size()
    assert sizes[f"{kind}_pending"] == 3
    assert sizes[f"max_{kind}_seen"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_field", ["session_id", "router_name", "model_name"])
async def test_session_keys_that_postgres_cannot_store_are_not_queued(queue, mock_prisma, invalid_field):
    key = {"session_id": "session", "router_name": "router", "model_name": "model"}
    key[invalid_field] += "\0"

    await queue.add_session_state(**key, state_dict={"turn_count": 1})

    assert (await queue.queue_size())["session_pending"] == 0
    assert await queue.flush_session_to_db(mock_prisma) == 0
    mock_prisma.db.litellm_adaptiveroutersession.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_session_retention_keeps_only_the_newest_rows(queue, mock_prisma, monkeypatch):
    monkeypatch.setattr(
        "litellm.router_strategy.adaptive_router.update_queue._MAX_SESSION_RETRY_ENTRIES", 3, raising=False
    )
    table = mock_prisma.db.litellm_adaptiveroutersession
    table.upsert.side_effect = RuntimeError("database unavailable")
    for index in range(6):
        await queue.add_session_state(f"session-{index}", "router", "model", {"turn_count": index})
    await queue.add_session_state("session-0", "router", "model", {"turn_count": 9})

    assert await queue.flush_session_to_db(mock_prisma) == 0
    assert (await queue.queue_size())["session_pending"] == 3
    table.reset_mock()
    table.upsert.side_effect = None

    assert await queue.flush_session_to_db(mock_prisma) == 3
    assert [
        (call.kwargs["data"]["create"]["session_id"], call.kwargs["data"]["update"]["turn_count"])
        for call in table.upsert.await_args_list
    ] == [("session-0", 9), ("session-4", 4), ("session-5", 5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_session_retention_prioritizes_new_snapshots_during_a_failed_flush(
    queue, mock_prisma, monkeypatch, cancelled
):
    monkeypatch.setattr(
        "litellm.router_strategy.adaptive_router.update_queue._MAX_SESSION_RETRY_ENTRIES", 2, raising=False
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_write(**kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("database unavailable")

    table = mock_prisma.db.litellm_adaptiveroutersession
    table.upsert.side_effect = fail_write
    for session in ("a", "b", "c"):
        await queue.add_session_state(session, "router", "model", {"turn_count": 1})
    task = asyncio.create_task(queue.flush_session_to_db(mock_prisma))
    await started.wait()
    await queue.add_session_state("a", "router", "model", {"turn_count": 2})
    await queue.add_session_state("d", "router", "model", {"turn_count": 3})
    if cancelled:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        assert await task == 0
    assert (await queue.queue_size())["session_pending"] == 2
    table.reset_mock()
    table.upsert.side_effect = None
    assert await queue.flush_session_to_db(mock_prisma) == 2
    assert [
        (call.kwargs["data"]["create"]["session_id"], call.kwargs["data"]["update"]["turn_count"])
        for call in table.upsert.await_args_list
    ] == [("a", 2), ("d", 3)]


@pytest.mark.asyncio
async def test_successful_flush_does_not_cap_newly_queued_sessions(queue, mock_prisma, monkeypatch):
    monkeypatch.setattr(
        "litellm.router_strategy.adaptive_router.update_queue._MAX_SESSION_RETRY_ENTRIES", 1, raising=False
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def write(**kwargs):
        started.set()
        await release.wait()

    table = mock_prisma.db.litellm_adaptiveroutersession
    table.upsert.side_effect = write
    await queue.add_session_state("old", "router", "model", {"turn_count": 1})
    task = asyncio.create_task(queue.flush_session_to_db(mock_prisma))
    await started.wait()
    for session in ("a", "b"):
        await queue.add_session_state(session, "router", "model", {"turn_count": 1})
    release.set()

    assert await task == 1
    assert (await queue.queue_size())["session_pending"] == 2
    table.upsert.side_effect = None
    assert await queue.flush_session_to_db(mock_prisma) == 2
