from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import pytest


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
    data = await _run_update(monkeypatch, poller_active=True)

    assert "batch_processed" not in data
    assert data["status"] == "complete"


@pytest.mark.asyncio
async def test_retrieving_a_completed_batch_still_marks_processed_without_a_cost_poller(monkeypatch):
    data = await _run_update(monkeypatch, poller_active=False)

    assert data["batch_processed"] is True
    assert data["status"] == "complete"


def test_batch_cost_poller_is_active_is_false_when_the_job_has_no_bound_poller(monkeypatch):
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


@pytest.mark.asyncio
async def test_the_caller_s_accounting_decision_wins_over_a_later_poller_transition(monkeypatch):
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    monkeypatch.setattr(cu, "batch_cost_poller_is_active", lambda: True)
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
        poller_owns_accounting=False,
    )

    data = update_mock.await_args.kwargs["data"]
    assert data["batch_processed"] is True
    assert data["status"] == "complete"


@pytest.mark.asyncio
async def test_a_caller_that_handed_off_accounting_still_leaves_the_marker_alone(monkeypatch):
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    monkeypatch.setattr(cu, "batch_cost_poller_is_active", lambda: False)
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
        poller_owns_accounting=True,
    )

    data = update_mock.await_args.kwargs["data"]
    assert "batch_processed" not in data
    assert data["status"] == "complete"


# =========================================================================== #
# add_internal_model_credentials - the snapshot that lets a completed
# batch's output file be read, and therefore its cost be recorded
# =========================================================================== #


def test_add_internal_model_credentials_attaches_an_immutable_snapshot():
    """Cost accounting for a completed batch reads its output file, and Bedrock resolves
    that bucket only from this snapshot. It must be immutable so nothing downstream can
    redirect the bucket that managed file ids are validated against."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(
        return_value={"s3_bucket_name": "configured-bucket", "aws_region_name": "us-east-1"}
    )
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials(data=data, llm_router=router, model_id="deployment-1")

    snapshot = data["_litellm_internal_model_credentials"]
    assert snapshot["s3_bucket_name"] == "configured-bucket"
    assert isinstance(snapshot, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot["s3_bucket_name"] = "attacker-bucket"
    router.get_deployment_credentials_with_provider.assert_called_once_with(model_id="deployment-1")


@pytest.mark.parametrize(
    "model_id, credentials",
    [(None, {"s3_bucket_name": "b"}), ("deployment-1", None)],
    ids=["no-model-id", "deployment-has-no-credentials"],
)
def test_add_internal_model_credentials_is_a_noop_without_a_resolvable_deployment(model_id, credentials):
    """An unroutable batch must be left alone rather than given an empty snapshot, which
    would look like a configured bucket of nothing."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(return_value=credentials)
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials(data=data, llm_router=router, model_id=model_id)

    assert "_litellm_internal_model_credentials" not in data


def test_add_internal_model_credentials_survives_a_failing_deployment_lookup():
    """The snapshot only enables cost accounting, so a batch whose deployment no longer
    resolves, which happens when a model group is removed while batches are in flight,
    must still be retrievable rather than failing the request on the lookup."""
    from litellm.proxy.openai_files_endpoints.common_utils import (
        add_internal_model_credentials,
    )

    router = MagicMock()
    router.get_deployment_credentials_with_provider = MagicMock(side_effect=KeyError("deployment-gone"))
    data = {"batch_id": "unified-batch-id"}

    add_internal_model_credentials(data=data, llm_router=router, model_id="deployment-gone")

    assert data == {"batch_id": "unified-batch-id"}


from litellm.proxy.openai_files_endpoints.common_utils import (
    _completed_batch_safe_to_retire,
)


def _completed_batch_for_retire(
    output_file_id: str | None, completed: int | None = None
) -> LiteLLMBatch:
    kwargs = dict(
        id="batch-1",
        completion_window="24h",
        created_at=1234567890,
        endpoint="/v1/chat/completions",
        input_file_id="file-in",
        object="batch",
        status="completed",
        output_file_id=output_file_id,
        error_file_id=None,
    )
    if completed is not None:
        kwargs["request_counts"] = {"total": completed, "completed": completed, "failed": 0}
    return LiteLLMBatch(**kwargs)


class TestCompletedBatchSafeToRetire:
    """A completed batch is only safe to retire from cost recovery once its output
    file has arrived or the provider proves no successful lines (#37713)."""

    def test_output_file_present_is_safe(self):
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire("file-out")) is True

    def test_no_output_and_no_successful_lines_is_safe(self):
        # Every request line errored -> nothing left to recover.
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None, completed=0)) is True

    def test_no_output_but_successful_lines_is_not_safe(self):
        # The bug: output_file_id is lagging; retiring here loses the spend record.
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None, completed=5)) is False

    def test_no_output_and_unknown_counts_is_not_safe(self):
        # Counts unknown -> stay eligible so the next poller pass revisits it.
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None)) is False
