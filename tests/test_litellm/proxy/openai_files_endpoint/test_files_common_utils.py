import os
import sys
from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.openai_files_endpoints.common_utils import (
    apply_unified_file_ids,
    map_raw_file_ids_to_unified,
)
from litellm.types.utils import LiteLLMBatch


def _batch(input_file_id, output_file_id, error_file_id) -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch-1",
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id=input_file_id,
        object="batch",
        status="cancelled",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
    )


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_empty_ids_skips_db():
    prisma_client = MagicMock()

    assert await map_raw_file_ids_to_unified(frozenset(), prisma_client) == {}

    prisma_client.db.litellm_managedfiletable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_no_prisma_client_returns_empty():
    assert await map_raw_file_ids_to_unified(frozenset({"file-raw-1"}), None) == {}


@pytest.mark.asyncio
async def test_map_raw_file_ids_to_unified_bulk_queries_and_filters_to_requested_ids():
    row_a = MagicMock(
        unified_file_id="unified-a",
        flat_model_file_ids=["file-raw-a", "file-raw-other"],
    )
    row_b = MagicMock(unified_file_id="unified-b", flat_model_file_ids=["file-raw-b"])
    prisma_client = MagicMock()
    prisma_client.db.litellm_managedfiletable.find_many = AsyncMock(return_value=[row_a, row_b])

    mapping = await map_raw_file_ids_to_unified(
        frozenset({"file-raw-b", "file-raw-a", "file-raw-missing"}), prisma_client
    )

    prisma_client.db.litellm_managedfiletable.find_many.assert_awaited_once_with(
        where={"flat_model_file_ids": {"hasSome": ["file-raw-a", "file-raw-b", "file-raw-missing"]}}
    )
    assert dict(mapping) == {"file-raw-a": "unified-a", "file-raw-b": "unified-b"}


def test_apply_unified_file_ids_swaps_only_mapped_ids():
    batch = _batch(input_file_id="file-raw-in", output_file_id="file-raw-out", error_file_id=None)

    apply_unified_file_ids(batch, MappingProxyType({"file-raw-out": "unified-out"}))

    assert batch.input_file_id == "file-raw-in"
    assert batch.output_file_id == "unified-out"
    assert batch.error_file_id is None


def test_apply_unified_file_ids_swaps_all_three_ids():
    batch = _batch(
        input_file_id="file-raw-in",
        output_file_id="file-raw-out",
        error_file_id="file-raw-err",
    )

    apply_unified_file_ids(
        batch,
        MappingProxyType(
            {
                "file-raw-in": "unified-in",
                "file-raw-out": "unified-out",
                "file-raw-err": "unified-err",
            }
        ),
    )

    assert (batch.input_file_id, batch.output_file_id, batch.error_file_id) == (
        "unified-in",
        "unified-out",
        "unified-err",
    )


class _FakeScheduler:
    def __init__(self, job):
        self._job = job

    def get_job(self, job_id):
        assert job_id == "check_batch_cost_job"
        return self._job


class _FakePoller:
    def __init__(self, confirmed):
        self.batch_processed_support_confirmed = confirmed

    def check_batch_cost(self):
        return None


def _job_for(poller):
    if poller is None:
        return None
    job = MagicMock()
    job.func = poller.check_batch_cost
    return job


@pytest.mark.parametrize(
    "polling_enabled, job, expected",
    [
        (True, _job_for(_FakePoller(confirmed=True)), True),
        (True, _job_for(_FakePoller(confirmed=False)), False),
        (True, None, False),
        (False, _job_for(_FakePoller(confirmed=True)), False),
    ],
    ids=[
        "poller-running-and-column-confirmed",
        "poller-running-but-column-unconfirmed",
        "job-absent-enterprise-import-failed",
        "polling-disabled-by-config",
    ],
)
def test_batch_cost_poller_is_active(monkeypatch, polling_enabled, job, expected):
    """The predicate must only claim the poller when it can actually be relied on, so a
    proxy with polling switched off, without the enterprise job, or whose poller has not
    confirmed batch_processed support keeps accounting for batch cost on the retrieve
    path. The unconfirmed case is the one that matters for legacy schemas: without the
    column the poller falls back to a query excluding terminal statuses, so a batch the
    retrieve path already marked complete would never be accounted by anyone."""
    import litellm.constants
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy.openai_files_endpoints.common_utils import (
        batch_cost_poller_is_active,
    )

    monkeypatch.setattr(litellm.constants, "PROXY_BATCH_POLLING_ENABLED", polling_enabled, raising=False)
    monkeypatch.setattr(proxy_server_module, "scheduler", _FakeScheduler(job), raising=False)

    assert batch_cost_poller_is_active() is expected


def test_batch_cost_poller_is_active_is_false_when_no_scheduler_exists(monkeypatch):
    import litellm.constants
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy.openai_files_endpoints.common_utils import (
        batch_cost_poller_is_active,
    )

    monkeypatch.setattr(litellm.constants, "PROXY_BATCH_POLLING_ENABLED", True, raising=False)
    monkeypatch.setattr(proxy_server_module, "scheduler", None, raising=False)

    assert batch_cost_poller_is_active() is False


def _completed_batch() -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch-done",
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id="file-in",
        object="batch",
        status="completed",
        output_file_id="file-out",
    )


async def _run_update(monkeypatch, poller_active: bool) -> dict:
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    monkeypatch.setattr(cu, "batch_cost_poller_is_active", lambda: poller_active)
    monkeypatch.setattr(cu, "ensure_batch_response_managed_file_ids", AsyncMock())

    prisma_client = MagicMock()
    update_mock = AsyncMock()
    prisma_client.db.litellm_managedobjecttable.update = update_mock

    db_batch_object = MagicMock()
    db_batch_object.status = "in_progress"

    await cu.update_batch_in_database(
        batch_id="unified-batch-id",
        unified_batch_id="unified-batch-id",
        response=_completed_batch(),
        managed_files_obj=MagicMock(),
        prisma_client=prisma_client,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
        operation="retrieve",
    )

    assert update_mock.await_count == 1
    return update_mock.await_args.kwargs["data"]


@pytest.mark.asyncio
async def test_retrieving_a_completed_batch_leaves_batch_processed_to_the_cost_poller(monkeypatch):
    """batch_processed is what removes a batch from CheckBatchCost's queue, which selects
    batch_processed=False. Retrieving a batch records no cost when the poller is active, so
    setting the flag here retired the poller on behalf of work nobody had done: a cost
    callback that then failed lost the batch's cost permanently with no retry left. The
    status update must still happen so callers see the terminal state."""
    data = await _run_update(monkeypatch, poller_active=True)

    assert "batch_processed" not in data
    assert data["status"] == "complete"


@pytest.mark.asyncio
async def test_retrieving_a_completed_batch_still_marks_processed_without_a_cost_poller(monkeypatch):
    """With no poller to hand off to, this path is the only accountant, so it keeps setting
    the flag. Otherwise a proxy with polling disabled would never unblock file deletion."""
    data = await _run_update(monkeypatch, poller_active=False)

    assert data["batch_processed"] is True
    assert data["status"] == "complete"


def test_batch_cost_poller_is_active_is_false_when_the_job_has_no_bound_poller(monkeypatch):
    """A scheduler that hands back a plain function rather than a bound method leaves no
    poller to interrogate, so the predicate stays conservative instead of assuming the
    column is supported."""
    import litellm.constants
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy.openai_files_endpoints.common_utils import (
        batch_cost_poller_is_active,
    )

    def unbound_check_batch_cost():
        return None

    job = MagicMock()
    job.func = unbound_check_batch_cost

    monkeypatch.setattr(litellm.constants, "PROXY_BATCH_POLLING_ENABLED", True, raising=False)
    monkeypatch.setattr(proxy_server_module, "scheduler", _FakeScheduler(job), raising=False)

    assert batch_cost_poller_is_active() is False


def test_batch_cost_poller_is_active_is_false_when_get_job_raises(monkeypatch):
    """Scheduler backends raise varied types; an unreadable scheduler must not be read as
    a working poller."""
    import litellm.constants
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy.openai_files_endpoints.common_utils import (
        batch_cost_poller_is_active,
    )

    class _ExplodingScheduler:
        def get_job(self, job_id):
            raise RuntimeError("scheduler not started")

    monkeypatch.setattr(litellm.constants, "PROXY_BATCH_POLLING_ENABLED", True, raising=False)
    monkeypatch.setattr(proxy_server_module, "scheduler", _ExplodingScheduler(), raising=False)

    assert batch_cost_poller_is_active() is False



@pytest.mark.asyncio
async def test_retrieving_a_batch_whose_status_is_unchanged_writes_nothing(monkeypatch):
    """A caller polling an already-complete batch must not write at all, so repeated polls
    cannot flip batch_processed or disturb whichever component owns accounting."""
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    monkeypatch.setattr(cu, "batch_cost_poller_is_active", lambda: False)
    monkeypatch.setattr(cu, "ensure_batch_response_managed_file_ids", AsyncMock())

    prisma_client = MagicMock()
    update_mock = AsyncMock()
    prisma_client.db.litellm_managedobjecttable.update = update_mock

    db_batch_object = MagicMock()
    db_batch_object.status = "completed"

    await cu.update_batch_in_database(
        batch_id="unified-batch-id",
        unified_batch_id="unified-batch-id",
        response=_completed_batch(),
        managed_files_obj=MagicMock(),
        prisma_client=prisma_client,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
        operation="retrieve",
    )

    update_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_batch_in_database_is_a_noop_for_unmanaged_batches(monkeypatch):
    """Batches with no managed object row have neither the flag nor a poller queue entry, so
    this path must leave them alone entirely."""
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    prisma_client = MagicMock()
    update_mock = AsyncMock()
    prisma_client.db.litellm_managedobjecttable.update = update_mock

    await cu.update_batch_in_database(
        batch_id="batch-raw-xyz",
        unified_batch_id=False,
        response=_completed_batch(),
        managed_files_obj=MagicMock(),
        prisma_client=prisma_client,
        verbose_proxy_logger=MagicMock(),
        operation="retrieve",
    )

    update_mock.assert_not_awaited()
