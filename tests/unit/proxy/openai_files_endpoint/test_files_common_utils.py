from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastapi import HTTPException
from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

import litellm
from litellm import Router
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    set_fallback_generalizations,
)
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    apply_team_provider_credentials,
    apply_unified_file_ids,
    canonical_provider_file_id,
    encode_file_id_with_model,
    get_credentials_for_model,
    is_litellm_executed_batch,
    map_raw_file_ids_to_unified,
    model_less_provider_credentials,
    model_less_team_deployment,
    prepare_data_with_credentials,
    provider_file_id_spellings,
    team_deployment_the_caller_may_use,
)
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.proxy.utils import InternalUsageCache, handle_exception_on_proxy
from litellm.types.utils import LiteLLMBatch
from tests.unit.proxy.router_entry_point_stubs import route_entry_points_to_stubs

_RAW_MODEL_WITH_PROMPT: Final = "opus-4.6 Please summarize my medical records\nPatient has diabetes"


def test_get_credentials_for_model_rejects_an_unknown_model_without_persisting_the_raw_model():
    llm_router: Final = route_entry_points_to_stubs(MagicMock())
    llm_router.get_deployment_credentials_with_provider.return_value = None

    with pytest.raises(ProxyModelNotFoundError) as raised:
        get_credentials_for_model(
            llm_router=llm_router, model_id=_RAW_MODEL_WITH_PROMPT, operation_context="file upload"
        )

    assert (raised.value.status_code, handle_exception_on_proxy(raised.value).code) == (400, "400")
    assert _RAW_MODEL_WITH_PROMPT in raised.value.detail["error"]
    assert raised.value.retryable_with_model_read_through is False
    assert raised.value.spend_log_error_message.startswith("file upload: ")
    assert "medical records" not in raised.value.spend_log_error_message


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


from openai.types.batch import BatchRequestCounts

from litellm.proxy.openai_files_endpoints.common_utils import (
    _completed_batch_safe_to_retire,
)


def _completed_batch_for_retire(output_file_id: str | None, counts: BatchRequestCounts | None = None) -> LiteLLMBatch:
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
    if counts is not None:
        kwargs["request_counts"] = counts
    return LiteLLMBatch(**kwargs)


class TestCompletedBatchSafeToRetire:
    """A completed batch is only safe to retire from cost recovery once its output
    file has arrived or the provider proves it enumerated a positive total of
    request lines and none succeeded (#37713, LIT-6360)."""

    def test_output_file_present_is_safe(self):
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire("file-out")) is True

    def test_no_output_and_synthesized_zero_counts_is_not_safe(self):
        counts = BatchRequestCounts(total=0, completed=0, failed=0)
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None, counts)) is False

    def test_no_output_but_successful_lines_is_not_safe(self):
        counts = BatchRequestCounts(total=100, completed=100, failed=0)
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None, counts)) is False

    def test_no_output_and_all_lines_failed_is_safe(self):
        counts = BatchRequestCounts(total=100, completed=0, failed=100)
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None, counts)) is True

    def test_no_output_and_unknown_counts_is_not_safe(self):
        assert _completed_batch_safe_to_retire(_completed_batch_for_retire(None)) is False


@pytest.mark.parametrize(
    "decoded_unified_batch_id, executed",
    [
        ("litellm_proxy;model_id:my-vllm;llm_batch_id:litellm_batch_0123abcd", True),
        ("litellm_proxy;model_id:my-vllm;llm_batch_id:batch_0123abcd", False),
        ("litellm_proxy;model_id:my-vllm;generic_response_id:resp_0123abcd", False),
        ("litellm_proxy;model_id:my-vllm;llm_output_file_id:file-0123abcd", False),
        ("batch_0123abcd", False),
    ],
)
def test_is_litellm_executed_batch_reads_the_llm_batch_id_prefix(decoded_unified_batch_id: str, executed: bool):
    assert is_litellm_executed_batch(decoded_unified_batch_id) is executed


def _managed_files_hook(prisma_client: MagicMock) -> _PROXY_LiteLLMManagedFiles:
    return _PROXY_LiteLLMManagedFiles(internal_usage_cache=InternalUsageCache(DualCache()), prisma_client=prisma_client)


def _managed_object_prisma(db_batch_object: MagicMock | None) -> MagicMock:
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=db_batch_object)
    prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
    return prisma_client


@pytest.mark.asyncio
async def test_batch_status_sync_from_provider_to_database(caplog: pytest.LogCaptureFixture):
    import json
    import logging

    from litellm.proxy.openai_files_endpoints.common_utils import (
        get_batch_from_database,
        update_batch_in_database,
    )

    batch_id: Final = "batch_test123"
    unified_batch_id: Final = "litellm_proxy:test_unified_batch"
    stored_row: Final = MagicMock(
        unified_object_id=batch_id,
        status="validating",
        file_object=json.dumps(
            {
                "id": batch_id,
                "object": "batch",
                "status": "validating",
                "endpoint": "/v1/chat/completions",
                "input_file_id": "file-test123",
                "completion_window": "24h",
                "created_at": 1234567890,
            }
        ),
    )
    prisma_client: Final = _managed_object_prisma(stored_row)
    managed_files: Final = _managed_files_hook(prisma_client)
    logger: Final = logging.getLogger("test_batch_status_sync")

    db_batch_object, response_batch = await get_batch_from_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        managed_files_obj=managed_files,
        prisma_client=prisma_client,
        verbose_proxy_logger=logger,
    )

    prisma_client.db.litellm_managedobjecttable.find_first.assert_awaited_once_with(
        where={"unified_object_id": batch_id}
    )
    assert db_batch_object is stored_row
    assert isinstance(response_batch, LiteLLMBatch)
    assert response_batch.id == batch_id
    assert response_batch.status == "validating"
    assert response_batch.input_file_id == "file-test123"

    completed: Final = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="completed",
        endpoint="/v1/chat/completions",
        input_file_id="file-test123",
        completion_window="24h",
        created_at=1234567890,
        output_file_id="file-output123",
    )
    with caplog.at_level(logging.INFO, logger=logger.name):
        await update_batch_in_database(
            batch_id=batch_id,
            unified_batch_id=unified_batch_id,
            response=completed,
            managed_files_obj=managed_files,
            prisma_client=prisma_client,
            verbose_proxy_logger=logger,
            db_batch_object=db_batch_object,
            operation="retrieve",
            poller_owns_accounting=False,
        )

    update: Final = prisma_client.db.litellm_managedobjecttable.update
    update.assert_awaited_once()
    assert update.await_args.kwargs["where"] == {"unified_object_id": batch_id}
    written: Final = update.await_args.kwargs["data"]
    assert written["status"] == "complete"
    assert written["batch_processed"] is True
    assert written["updated_at"] is not None
    assert json.loads(written["file_object"])["status"] == "completed"
    assert json.loads(written["file_object"])["output_file_id"] == "file-output123"
    assert f"Updating batch {batch_id} status from validating to completed" in caplog.messages


@pytest.mark.asyncio
async def test_batch_cancel_updates_database(caplog: pytest.LogCaptureFixture):
    import json
    import logging

    from litellm.proxy.openai_files_endpoints.common_utils import update_batch_in_database

    batch_id: Final = "batch_cancel_test"
    cancelled: Final = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="cancelled",
        endpoint="/v1/chat/completions",
        input_file_id="file-test123",
        completion_window="24h",
        created_at=1234567890,
        cancelled_at=1234567999,
    )
    prisma_client: Final = _managed_object_prisma(None)
    logger: Final = logging.getLogger("test_batch_cancel_updates_database")

    with caplog.at_level(logging.INFO, logger=logger.name):
        await update_batch_in_database(
            batch_id=batch_id,
            unified_batch_id="litellm_proxy:cancel_test",
            response=cancelled,
            managed_files_obj=_managed_files_hook(prisma_client),
            prisma_client=prisma_client,
            verbose_proxy_logger=logger,
            operation="cancel",
        )

    update: Final = prisma_client.db.litellm_managedobjecttable.update
    update.assert_awaited_once()
    assert update.await_args.kwargs["where"] == {"unified_object_id": batch_id}
    written: Final = update.await_args.kwargs["data"]
    assert written["status"] == "cancelled"
    assert "batch_processed" not in written
    assert json.loads(written["file_object"])["cancelled_at"] == 1234567999
    assert f"Updating batch {batch_id} status to cancelled after cancel" in caplog.messages


def _team_openai_router(team_public_model_name: str, deployment_model: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": team_public_model_name,
                "litellm_params": {"model": deployment_model, "api_key": "team-openai-key"},
                "model_info": {
                    "id": "team-a-openai",
                    "team_id": "team-a",
                    "team_public_model_name": team_public_model_name,
                },
            },
            {
                "model_name": "claude-opus-4-6",
                "litellm_params": {"model": "anthropic/claude-opus-4-6", "api_key": "anthropic-key"},
            },
        ]
    )


async def _team_deployment_credentials(
    llm_router: Router, caller: UserAPIKeyAuth, custom_llm_provider: str
) -> Mapping[str, object] | None:
    deployment: Final = await model_less_team_deployment(llm_router, caller, custom_llm_provider)
    return None if deployment is None else deployment.credentials


async def _team_only_openai_credentials(
    team_public_model_name: str, deployment_model: str, key_models: list[str]
) -> Mapping[str, object] | None:
    return await _team_deployment_credentials(
        _team_openai_router(team_public_model_name, deployment_model),
        UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="team-a",
            team_models=[team_public_model_name, "claude-opus-4-6"],
            models=key_models,
        ),
        "openai",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_public_model_name, deployment_model, key_models",
    [
        ("openai/*", "openai/*", ["gpt-4o"]),
        ("openai/*", "openai/*", ["openai/gpt-4o"]),
        ("gpt-4o", "openai/gpt-4o", ["openai/*"]),
    ],
)
async def test_a_restricted_key_resolves_the_team_deployment_its_wildcard_or_model_covers(
    team_public_model_name: str, deployment_model: str, key_models: list[str]
):
    credentials = await _team_only_openai_credentials(team_public_model_name, deployment_model, key_models)

    assert credentials is not None
    assert credentials["api_key"] == "team-openai-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("key_models", [["claude-opus-4-6"], ["anthropic/*"]])
async def test_a_restricted_key_outside_the_team_wildcard_never_resolves_its_credentials(key_models: list[str]):
    assert await _team_only_openai_credentials("openai/*", "openai/*", key_models) is None


def _family_wildcard_of_a_model_in(known_models: set[str]) -> str:
    family, _, _ = min(model for model in known_models if "-" in model and "/" not in model).partition("-")
    return f"{family}-*"


def _team_wildcard_router(custom_llm_provider: str) -> Router:
    team_wildcard: Final = f"{custom_llm_provider}/*"
    return Router(
        model_list=[
            {
                "model_name": team_wildcard,
                "litellm_params": {"model": team_wildcard, "api_key": f"team-{custom_llm_provider}-key"},
                "model_info": {"id": "team-a-wildcard", "team_id": "team-a", "team_public_model_name": team_wildcard},
            }
        ]
    )


def _team_wildcard_caller(custom_llm_provider: str, key_wildcard: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="team-a",
        team_models=[f"{custom_llm_provider}/*"],
        models=[key_wildcard],
    )


async def _team_wildcard_credentials(custom_llm_provider: str, key_wildcard: str) -> Mapping[str, object] | None:
    return await _team_deployment_credentials(
        _team_wildcard_router(custom_llm_provider),
        _team_wildcard_caller(custom_llm_provider, key_wildcard),
        custom_llm_provider,
    )


@pytest.mark.asyncio
async def test_a_wildcard_key_resolves_the_team_wildcard_deployment_serving_a_model_both_cover():
    key_wildcard: Final = _family_wildcard_of_a_model_in(litellm.open_ai_chat_completion_models)

    credentials: Final = await _team_wildcard_credentials("openai", key_wildcard)

    assert credentials is not None
    assert credentials["api_key"] == "team-openai-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_provider_models, team_provider",
    [(litellm.anthropic_models, "openai"), (litellm.open_ai_chat_completion_models, "anthropic")],
)
async def test_a_wildcard_key_never_resolves_a_team_wildcard_deployment_of_another_provider(
    key_provider_models: set[str], team_provider: str
):
    assert (
        await _team_wildcard_credentials(team_provider, _family_wildcard_of_a_model_in(key_provider_models)) is None
    )


@pytest.fixture
def family_only_a_routing_rule_maps_to_anthropic() -> Iterator[str]:
    previous_rules: Final = get_fallback_generalization_rules()
    family: Final = "lit-unlisted-family"
    set_fallback_generalizations(
        [*previous_rules, {"name": family, "pattern": rf"^{family}-\d+$", "model_info": {"litellm_provider": "anthropic"}}]
    )
    yield family
    set_fallback_generalizations(previous_rules)


@pytest.mark.asyncio
async def test_a_wildcard_key_resolves_the_team_wildcard_deployment_for_a_family_the_cost_map_does_not_list(
    family_only_a_routing_rule_maps_to_anthropic: str,
):
    credentials: Final = await _team_wildcard_credentials("anthropic", f"{family_only_a_routing_rule_maps_to_anthropic}-9*")

    assert credentials is not None
    assert credentials["api_key"] == "team-anthropic-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_provider_models, may_use",
    [(litellm.open_ai_chat_completion_models, True), (litellm.anthropic_models, False)],
)
async def test_the_team_wildcard_deployment_is_usable_only_by_a_key_wildcard_reaching_a_model_it_serves(
    key_provider_models: set[str], may_use: bool
):
    llm_router: Final = _team_wildcard_router("openai")
    team_member: Final = _team_wildcard_caller("openai", "openai/*")
    deployment: Final = await team_deployment_the_caller_may_use(llm_router, team_member, "openai")
    caller: Final = _team_wildcard_caller("openai", _family_wildcard_of_a_model_in(key_provider_models))

    usable: Final = await team_deployment_the_caller_may_use(llm_router, caller, "openai")

    assert deployment is not None
    assert (usable == deployment) is may_use


@pytest.mark.parametrize(
    "request_credentials, sent_api_key, sent_api_base",
    [
        ({}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": ""}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": "   "}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": "\n"}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": "sk-forwarded"}, "sk-forwarded", "https://team.example/v1"),
        ({"api_base": "https://caller.example/v1"}, "team-openai-key", "https://team.example/v1"),
        (
            {"api_key": "sk-forwarded", "api_base": "https://caller.example/v1"},
            "sk-forwarded",
            "https://caller.example/v1",
        ),
        ({"api_key": 123}, "team-openai-key", "https://team.example/v1"),
        (
            {"api_key": "sk-forwarded", "api_base": ["https://caller.example/v1"]},
            "sk-forwarded",
            "https://team.example/v1",
        ),
        ({"api_key": "sk-forwarded", "api_base": "  "}, "sk-forwarded", "https://team.example/v1"),
        ({"litellm_credential_name": "no-such-credential"}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": "sk-forwarded\n", "drop_params": True}, "team-openai-key", "https://team.example/v1"),
        ({"api_key": "sk-for warded"}, "sk-for warded", "https://team.example/v1"),
    ],
)
@pytest.mark.asyncio
async def test_the_team_key_only_goes_to_the_team_host_unless_the_request_brings_its_own_key(
    request_credentials: dict[str, object], sent_api_key: str, sent_api_base: str
):
    data: Final[dict[str, object]] = dict(request_credentials)

    await _apply_team_a_openai_credentials(data, _team_a_caller(["gpt-4o"], ["openai/*"]))

    assert (data.get("api_key"), data.get("api_base")) == (sent_api_key, sent_api_base)


@pytest.mark.parametrize("api_key", ["sk-forwarded\n", " sk-forwarded", "\u200b", "sk-forwarded\u00a0"])
@pytest.mark.asyncio
async def test_a_request_key_that_cannot_be_an_http_header_is_a_bad_request(api_key: str):
    data: Final[dict[str, object]] = {"api_key": api_key}

    with pytest.raises(HTTPException) as refused:
        await _apply_team_a_openai_credentials(data, _team_a_caller(["gpt-4o"], ["openai/*"]))

    assert refused.value.status_code == 400
    assert api_key not in str(refused.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("team_models, gets_the_team_key", [(["openai/*"], True), (["claude-opus-4-6"], False)])
async def test_a_wildcard_key_gets_the_team_key_only_when_its_team_allows_the_model_it_reaches(
    team_models: list[str], gets_the_team_key: bool
):
    key_wildcard: Final = _family_wildcard_of_a_model_in(litellm.open_ai_chat_completion_models)
    data: Final[dict[str, object]] = {}

    await _apply_team_a_openai_credentials(data, _team_a_caller([key_wildcard], team_models))

    assert (data.get("api_key") == "team-openai-key") is gets_the_team_key


def _a_family_with_several_models(known_models: set[str]) -> tuple[str, tuple[str, ...]]:
    families: Final = sorted(
        {model.partition("-")[0] for model in known_models if "-" in model and "/" not in model}
    )
    family_models: Final = (
        (family, tuple(sorted(model for model in known_models if model.startswith(f"{family}-") and "/" not in model)))
        for family in families
    )
    family, models = next((family, models) for family, models in family_models if len(models) > 1)
    return f"{family}-*", models


@pytest.mark.asyncio
async def test_a_wildcard_key_gets_the_team_key_through_the_one_model_its_team_allows():
    key_wildcard, family_models = _a_family_with_several_models(litellm.open_ai_chat_completion_models)
    data: Final[dict[str, object]] = {}

    await _apply_team_a_openai_credentials(data, _team_a_caller([key_wildcard], [family_models[-1]]))

    assert data.get("api_key") == "team-openai-key"


@pytest.mark.asyncio
async def test_a_wildcard_key_gets_the_team_key_through_the_one_model_its_project_allows():
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj, LiteLLM_TeamTableCachedObj
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, project_cache_key

    key_wildcard, family_models = _a_family_with_several_models(litellm.open_ai_chat_completion_models)
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="team_id:team-a", value=LiteLLM_TeamTableCachedObj(team_id="team-a", models=["openai/*"])
    )
    await cache.async_set_cache(
        key=project_cache_key("proj-1"),
        value=LiteLLM_ProjectTableCachedObj(project_id="proj-1", models=[family_models[-1]]),
    )
    data: Final[dict[str, object]] = {}

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),  # test-quality-ok: proxy_server global, no seam
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),  # test-quality-ok: proxy_server global, no seam
    ):
        await _apply_team_a_openai_credentials(
            data,
            UserAPIKeyAuth(
                api_key="test-key", team_id="team-a", team_models=["openai/*"], models=[key_wildcard], project_id="proj-1"
            ),
        )

    assert data.get("api_key") == "team-openai-key"


@pytest.mark.asyncio
async def test_an_exact_name_key_gets_the_team_key_when_its_team_allows_any_of_its_models():
    _, family_models = _a_family_with_several_models(litellm.open_ai_chat_completion_models)
    data: Final[dict[str, object]] = {}

    await _apply_team_a_openai_credentials(data, _team_a_caller(list(family_models[:2]), [family_models[1]]))

    assert data.get("api_key") == "team-openai-key"


@pytest.mark.asyncio
async def test_a_wildcard_key_keeps_the_team_deployment_it_names_over_an_earlier_team_wildcard():
    key_wildcard, family_models = _a_family_with_several_models(litellm.open_ai_chat_completion_models)
    named_model: Final = family_models[0]
    data: Final[dict[str, object]] = {}

    await apply_team_provider_credentials(
        data=data,
        llm_router=Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {"model": "openai/*", "api_key": "team-wildcard-key"},
                    "model_info": {"id": "team-a-wildcard", "team_id": "team-a"},
                },
                {
                    "model_name": named_model,
                    "litellm_params": {"model": f"openai/{named_model}", "api_key": "team-named-model-key"},
                    "model_info": {"id": "team-a-named-model", "team_id": "team-a"},
                },
            ]
        ),
        user_api_key_dict=_team_a_caller([key_wildcard], ["openai/*"]),
        custom_llm_provider="openai",
    )

    assert data.get("api_key") == "team-named-model-key"


def _team_a_caller(key_models: list[str], team_models: list[str]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="test-key", user_id="test-user", team_id="team-a", team_models=team_models, models=key_models
    )


async def _apply_team_a_openai_credentials(data: dict[str, object], caller: UserAPIKeyAuth) -> None:
    await apply_team_provider_credentials(
        data=data,
        llm_router=Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {
                        "model": "openai/*",
                        "api_key": "team-openai-key",
                        "api_base": "https://team.example/v1",
                    },
                    "model_info": {"id": "team-a-openai", "team_id": "team-a"},
                }
            ]
        ),
        user_api_key_dict=caller,
        custom_llm_provider="openai",
    )


def _team_openai_deployment(
    deployment_id: str, api_key: str, created_at: datetime | None = None, db_model: bool = True
) -> dict[str, object]:
    return {
        "model_name": f"team-gpt-{deployment_id}",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": api_key},
        "model_info": {"id": deployment_id, "team_id": "team-a", "db_model": db_model}
        | ({} if created_at is None else {"created_at": created_at}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_ids", [("deployment-a", "deployment-b"), ("deployment-b", "deployment-a")])
async def test_a_team_with_two_openai_deployments_resolves_the_same_one_in_any_router_order(
    deployment_ids: tuple[str, str],
):
    router: Final = Router(
        model_list=[_team_openai_deployment(deployment_id, f"{deployment_id}-key") for deployment_id in deployment_ids]
    )

    credentials: Final = await _team_deployment_credentials(
        router, UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="team-a"), "openai"
    )

    assert credentials is not None
    assert credentials["api_key"] == "deployment-a-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_first", [True, False])
async def test_a_team_keeps_its_older_openai_deployment_when_a_newer_one_has_a_lower_id(newer_first: bool):
    older: Final = _team_openai_deployment("deployment-z", "older-key", datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer: Final = _team_openai_deployment("deployment-a", "newer-key", datetime(2026, 6, 1, tzinfo=timezone.utc))
    router: Final = Router(model_list=[newer, older] if newer_first else [older, newer])

    credentials: Final = await _team_deployment_credentials(
        router, UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="team-a"), "openai"
    )

    assert credentials is not None
    assert credentials["api_key"] == "older-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_first", [True, False])
async def test_the_team_provider_deployment_is_the_one_whose_credentials_the_team_resolves(newer_first: bool):
    older: Final = _team_openai_deployment("deployment-z", "older-key", datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer: Final = _team_openai_deployment("deployment-a", "newer-key", datetime(2026, 6, 1, tzinfo=timezone.utc))
    router: Final = Router(model_list=[newer, older] if newer_first else [older, newer])
    team_member: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="team-a")

    deployment: Final = await model_less_team_deployment(router, team_member, "openai")

    assert deployment is not None
    assert (deployment.deployment_id, deployment.public_model_name) == ("deployment-z", "team-gpt-deployment-z")
    assert deployment.credentials == await model_less_provider_credentials(router, team_member, "openai")
    assert deployment.credentials["api_key"] == "older-key"


@pytest.mark.asyncio
async def test_the_team_provider_deployment_carries_the_team_public_model_name():
    deployment: Final = await model_less_team_deployment(
        _team_openai_router("openai/*", "openai/*"),
        UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="team-a"),
        "openai",
    )

    assert deployment is not None
    assert (deployment.deployment_id, deployment.public_model_name) == ("team-a-openai", "openai/*")


@pytest.mark.asyncio
async def test_a_key_with_only_an_accessible_non_team_deployment_has_no_team_provider_deployment():
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai/gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "openai-deployment-key"},
            }
        ]
    )
    caller: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", models=["openai/*"])

    accessible_credentials: Final = await model_less_provider_credentials(router, caller, "openai")

    assert accessible_credentials is not None
    assert accessible_credentials["api_key"] == "openai-deployment-key"
    assert await model_less_team_deployment(router, caller, "openai") is None


@pytest.mark.asyncio
async def test_a_team_keeps_its_first_config_file_openai_deployment_over_older_db_ones():
    db_deployment: Final = _team_openai_deployment("deployment-a", "db-key", datetime(2020, 1, 1, tzinfo=timezone.utc))
    config_deployments: Final = tuple(
        _team_openai_deployment(deployment_id, f"{deployment_id}-key", db_model=False)
        for deployment_id in ("deployment-z", "deployment-b")
    )
    router: Final = Router(model_list=[db_deployment, *config_deployments])

    credentials: Final = await _team_deployment_credentials(
        router, UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="team-a"), "openai"
    )

    assert credentials is not None
    assert credentials["api_key"] == "deployment-z-key"


@pytest.mark.parametrize(
    ("custom_llm_provider", "openrouter_key", "api_key"),
    [("openai", "sk-or-files-openai", "openai-key"), ("openrouter", "sk-or-files-openrouter", "sk-or-files-openrouter")],
)
@pytest.mark.asyncio
async def test_a_provider_credential_lookup_never_lists_an_openrouter_wildcards_live_catalog(
    monkeypatch: pytest.MonkeyPatch, custom_llm_provider: str, openrouter_key: str, api_key: str
):
    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    router: Final = Router(
        model_list=[
            {
                "model_name": "openrouter/*",
                "litellm_params": {
                    "model": "openrouter/*",
                    "api_key": openrouter_key,
                    "api_base": "https://openrouter.ai/api/v1",
                },
            },
            {"model_name": "gpt-5.4-mini", "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "openai-key"}},
        ]
    )

    with respx.mock(assert_all_called=False) as respx_mock:
        catalog_route: Final = respx_mock.get("https://openrouter.ai/api/v1/models/user").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "openai/gpt-5.4-mini"}]})
        )
        credentials: Final = await model_less_provider_credentials(
            router, UserAPIKeyAuth(api_key="test-key", user_id="test-user"), custom_llm_provider
        )

    assert credentials is not None
    assert (credentials["custom_llm_provider"], credentials["api_key"]) == (custom_llm_provider, api_key)
    assert catalog_route.call_count == 0


@pytest.mark.parametrize(
    ("key_models", "deployments", "custom_llm_provider", "api_key"),
    [
        (
            ["anthropic/*"],
            (("anthropic/claude-*", "anthropic/claude-*", "anthropic-narrow-key"),),
            "anthropic",
            "anthropic-narrow-key",
        ),
        (["anthropic/claude-3*"], (("anthropic/*", "anthropic/*", "anthropic-wide-key"),), "anthropic", "anthropic-wide-key"),
        (["openrouter/*"], (("openrouter/openai/*", "openrouter/openai/*", "sk-or-narrow"),), "openrouter", "sk-or-narrow"),
        (
            ["team/*"],
            (("team/*", "openrouter/*", "sk-or-team"), ("team/openai/*", "openai/*", "openai-team-key")),
            "openai",
            "openai-team-key",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_key_wildcard_finds_the_deployment_it_grants_without_listing_any_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
    key_models: list[str],
    deployments: tuple[tuple[str, str, str], ...],
    custom_llm_provider: str,
    api_key: str,
):
    monkeypatch.setattr(litellm, "check_provider_endpoint", True)
    router: Final = Router(
        model_list=[
            {"model_name": model_name, "litellm_params": {"model": model, "api_key": deployment_key}}
            for model_name, model, deployment_key in deployments
        ]
    )

    with respx.mock(assert_all_called=False) as respx_mock:
        catalog_requests: Final = respx_mock.route().mock(return_value=httpx.Response(200, json={"data": []}))
        credentials: Final = await model_less_provider_credentials(
            router, UserAPIKeyAuth(api_key="test-key", user_id="test-user", models=key_models), custom_llm_provider
        )

    assert credentials is not None
    assert (credentials["custom_llm_provider"], credentials["api_key"]) == (custom_llm_provider, api_key)
    assert catalog_requests.call_count == 0


@pytest.mark.parametrize(
    ("deployed_name", "api_key"), [("gpt-4o-mini", None), ("openai/gpt-4o-mini", "openai-deployment-key")]
)
@pytest.mark.asyncio
async def test_a_provider_wildcard_grant_finds_only_deployments_named_under_it(deployed_name: str, api_key: str | None):
    router: Final = Router(
        model_list=[
            {
                "model_name": deployed_name,
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "openai-deployment-key"},
            }
        ]
    )

    credentials: Final = await model_less_provider_credentials(
        router, UserAPIKeyAuth(api_key="test-key", user_id="test-user", models=["openai/*"]), "openai"
    )

    assert (None if credentials is None else credentials["api_key"]) == api_key


@pytest.mark.parametrize("field", ["s3_bucket_name", "s3_output_bucket_name"])
def test_create_paths_keep_a_request_bucket_and_reads_keep_the_deployment_bucket(field: str) -> None:
    deployment: Final = {"custom_llm_provider": "bedrock", "api_key": "sk-deployment", field: "model-bucket"}
    create_with_bucket: Final[dict[str, object]] = {field: "team-a-bucket"}
    create_without_bucket: Final[dict[str, object]] = {}
    read_with_bucket: Final[dict[str, object]] = {field: "team-a-bucket"}

    prepare_data_with_credentials(data=create_with_bucket, credentials=dict(deployment), keep_request_buckets=True)
    prepare_data_with_credentials(data=create_without_bucket, credentials=dict(deployment), keep_request_buckets=True)
    prepare_data_with_credentials(data=read_with_bucket, credentials=dict(deployment))

    assert create_with_bucket == {field: "team-a-bucket", "api_key": "sk-deployment"}
    assert create_without_bucket == {field: "model-bucket", "api_key": "sk-deployment"}
    assert read_with_bucket == {field: "model-bucket", "api_key": "sk-deployment"}


@pytest.mark.parametrize(
    ("file_id", "canonical"),
    (
        ("batch-out1", "batch-out1"),
        ("files/batch-out1", "batch-out1"),
        ("files%2Fbatch-out1", "batch-out1"),
        ("batch-out1/", "batch-out1"),
        ("/files/batch-out1/", "batch-out1"),
        ("https://generativelanguage.googleapis.com/v1beta/files/batch-out1", "batch-out1"),
        ("file-abc123", "file-abc123"),
        ("s3://bucket/path/files/output.jsonl", "s3://bucket/path/files/output.jsonl"),
        ("gs://bucket/prefix/predictions.jsonl", "gs://bucket/prefix/predictions.jsonl"),
        (
            "https://acct.blob.core.windows.net/uploads/0c4e.files%2Ffile-victim",
            "https://acct.blob.core.windows.net/uploads/0c4e.files%2Ffile-victim",
        ),
        ("x://h/files/batch-out1", "x://h/files/batch-out1"),
    ),
)
def test_canonical_provider_file_id_collapses_every_spelling_to_the_stored_id(file_id: str, canonical: str):
    assert canonical_provider_file_id(file_id) == canonical


def test_provider_file_id_spellings_unwraps_a_model_encoded_id_and_covers_the_stored_forms():
    wrapped: Final = encode_file_id_with_model(file_id="batch-out1", model="gemini-deployment")

    assert provider_file_id_spellings(wrapped) == (
        wrapped,
        "batch-out1",
        "files/batch-out1",
        "https://generativelanguage.googleapis.com/v1beta/files/batch-out1",
    )
    for spelling in ("batch-out1", "files%2Fbatch-out1", "batch-out1/"):
        assert set(provider_file_id_spellings(spelling)) >= set(provider_file_id_spellings(wrapped)[1:]), spelling


@pytest.mark.parametrize(
    "respelling",
    (
        "HTTPS://generativelanguage.googleapis.com/v1beta/files/batch-out1",
        "x://h/files/batch-out1",
        "S3://b/files/batch-out1",
        "x%3A%2F%2Fh%2Ffiles%2Fbatch-out1",
        "https://example.invalid/batch-out1",
        "http://example.invalid/batch-out1?alt=media#download",
        "https%3A%2F%2Fexample.invalid%2Fbatch-out1",
        "https://example.invalid/%23/files/batch-out1",
        "https://example.invalid/%3F/files/batch-out1",
        "https://generativelanguage.googleapis.com/%23/files/batch-out1",
        "https://generativelanguage.googleapis.com/%3F/files/batch-out1",
        "https://x%23/batch-out1",
        "https://x%3F/batch-out1",
        "https://x%2Ffiles%2F/batch-out1",
    ),
)
def test_provider_file_id_spellings_reduce_any_scheme_the_way_gemini_does(respelling: str):
    assert "batch-out1" in provider_file_id_spellings(respelling)
