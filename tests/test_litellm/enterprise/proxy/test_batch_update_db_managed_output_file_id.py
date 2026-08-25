"""Regression: update_batch_in_database must not persist raw provider output_file_id."""

import json
from types import SimpleNamespace
from typing import Optional
import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    ensure_batch_response_managed_file_ids,
    update_batch_in_database,
)
from litellm.types.utils import LiteLLMBatch


def _build_batch_response(
    *,
    batch_id: str = "batch_managed_ids_test",
    status: str = "completed",
    output_file_id: Optional[str] = "file-rawoutput789",
    error_file_id: Optional[str] = None,
    hidden_params: Optional[dict] = None,
) -> LiteLLMBatch:
    batch = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status=status,
        endpoint="/v1/chat/completions",
        input_file_id="file-input123",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        completion_window="24h",
        created_at=1234567890,
    )
    if hidden_params is not None:
        batch._hidden_params = hidden_params  # type: ignore[attr-defined]
    return batch


def _build_managed_files_mock(unified_id: str = "file-bWFuYWdlZF9vdXRwdXRfaWQ="):
    mock = MagicMock()
    mock.get_unified_output_file_id = MagicMock(return_value=unified_id)
    mock.store_unified_file_id = AsyncMock()
    return mock


def _build_prisma_mock(db_batch_object=None):
    mock = MagicMock()
    mock.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    mock.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=db_batch_object)
    mock.db.litellm_managedobjecttable.update = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_update_batch_in_database_stores_unified_output_file_id():
    raw_output_file_id = "file-rawoutput789"
    unified_output_file_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    batch_id = "batch_managed_ids_test"
    unified_batch_id = (
        "litellm_proxy;model_id:my-model;llm_batch_id:batch_managed_ids_test"
    )

    response = _build_batch_response(
        batch_id=batch_id,
        output_file_id=raw_output_file_id,
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = _build_managed_files_mock(unified_id=unified_output_file_id)
    mock_prisma = _build_prisma_mock()

    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    stored = json.loads(
        mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"][
            "file_object"
        ]
    )
    assert stored["output_file_id"] == unified_output_file_id
    assert stored["output_file_id"] != raw_output_file_id


@pytest.mark.asyncio
async def test_cancel_path_registers_output_file_under_batch_owner():
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    db_batch_object = SimpleNamespace(
        created_by="batch-owner", team_id="batch-team", status="in_progress"
    )
    response = _build_batch_response(
        status="cancelling",
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )
    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    mock_prisma = _build_prisma_mock(db_batch_object=db_batch_object)

    await update_batch_in_database(
        batch_id="batch_managed_ids_test",
        unified_batch_id="litellm_proxy;model_id:my-model;llm_batch_id:batch_managed_ids_test",
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        operation="cancel",
    )

    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs[
        "user_api_key_dict"
    ]
    assert forwarded_auth.user_id == "batch-owner"
    assert forwarded_auth.team_id == "batch-team"
    stored = json.loads(
        mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"][
            "file_object"
        ]
    )
    assert stored["output_file_id"] == unified_id


@pytest.mark.asyncio
async def test_update_batch_skips_lookup_when_db_batch_object_supplied():
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    caller_row = SimpleNamespace(
        created_by="caller-owner", team_id="caller-team", status="in_progress"
    )
    decoy_row = SimpleNamespace(
        created_by="decoy-owner", team_id="decoy-team", status="in_progress"
    )
    response = _build_batch_response(
        status="cancelling",
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )
    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    mock_prisma = _build_prisma_mock(db_batch_object=decoy_row)

    await update_batch_in_database(
        batch_id="batch_managed_ids_test",
        unified_batch_id="litellm_proxy;model_id:my-model;llm_batch_id:batch_managed_ids_test",
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=caller_row,
        operation="retrieve",
    )

    mock_prisma.db.litellm_managedobjecttable.find_first.assert_not_called()
    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs[
        "user_api_key_dict"
    ]
    assert forwarded_auth.user_id == "caller-owner"
    assert forwarded_auth.team_id == "caller-team"


@pytest.mark.asyncio
async def test_update_batch_derives_model_id_from_unified_batch_id():
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(output_file_id="file-raw-output", hidden_params={})
    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    mock_prisma = _build_prisma_mock()

    await update_batch_in_database(
        batch_id="batch_managed_ids_test",
        unified_batch_id="litellm_proxy;model_id:model-from-batch-id;llm_batch_id:batch_managed_ids_test",
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert (
        mock_managed_files.get_unified_output_file_id.call_args.kwargs["model_id"]
        == "model-from-batch-id"
    )
    assert response.output_file_id == unified_id


@pytest.mark.asyncio
async def test_ensure_batch_response_normalizes_error_file_id():
    """Both output_file_id and error_file_id must be normalized to managed IDs."""
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(
        output_file_id="file-raw-output",
        error_file_id="file-raw-error",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    mock_prisma = _build_prisma_mock()

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert response.output_file_id == unified_id
    assert response.error_file_id == unified_id
    assert mock_managed_files.get_unified_output_file_id.call_count == 2


@pytest.mark.asyncio
async def test_ensure_batch_response_swallows_conversion_errors():
    """When the managed-files conversion raises, the failure is logged, not propagated."""
    raw_output_file_id = "file-raw-output"
    response = _build_batch_response(
        output_file_id=raw_output_file_id,
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = MagicMock()
    mock_managed_files.get_unified_output_file_id = MagicMock(
        side_effect=RuntimeError("boom")
    )
    mock_managed_files.store_unified_file_id = AsyncMock()

    mock_logger = MagicMock()
    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=mock_logger,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert response.output_file_id == raw_output_file_id
    mock_logger.warning.assert_called()


@pytest.mark.asyncio
async def test_ensure_batch_response_builds_auth_from_db_batch_object():
    """If user_api_key_dict is omitted, fall back to created_by/team_id on db_batch_object."""
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    db_batch_object = SimpleNamespace(
        created_by="user-from-db", team_id="team-from-db", status="completed"
    )

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
    )

    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs[
        "user_api_key_dict"
    ]
    assert forwarded_auth.user_id == "user-from-db"
    assert forwarded_auth.team_id == "team-from-db"


@pytest.mark.asyncio
async def test_ensure_batch_response_resolves_model_name_from_unified_file_id():
    """When hidden_params lacks model_name, derive it from unified_file_id."""
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={
            "model_id": "my-model",
            "unified_file_id": "litellm_proxy:application/octet-stream;unified_id,abc;target_model_names,gpt-4o-mini,gemini-2.0-flash",
        },
    )

    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert (
        mock_managed_files.get_unified_output_file_id.call_args.kwargs["model_name"]
        == "gpt-4o-mini,gemini-2.0-flash"
    )


@pytest.mark.asyncio
async def test_ensure_batch_response_returns_early_without_managed_files_obj():
    """Without managed_files_obj, the helper is a no-op (no conversion attempted)."""
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=None,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert response.output_file_id == "file-raw-output"


@pytest.mark.asyncio
async def test_ensure_batch_response_returns_early_without_model_id():
    """Without model_id in hidden_params, the helper cannot create managed IDs."""
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={"model_name": "openai/gpt-4o"},
    )
    mock_managed_files = _build_managed_files_mock()

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert response.output_file_id == "file-raw-output"
    mock_managed_files.get_unified_output_file_id.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_batch_response_returns_early_without_auth():
    """Without user_api_key_dict or db_batch_object, no conversion is attempted."""
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )
    mock_managed_files = _build_managed_files_mock()

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
    )

    assert response.output_file_id == "file-raw-output"
    mock_managed_files.get_unified_output_file_id.assert_not_called()


def _in_memory_managed_files():
    """Build a real _PROXY_LiteLLMManagedFiles whose prisma upsert hits an in-memory row."""
    from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    store: dict = {}

    async def _upsert(where, data):
        key = where["unified_object_id"]
        if key in store:
            store[key].update(data["update"])
        else:
            store[key] = dict(data["create"])

    table = MagicMock()
    table.upsert = AsyncMock(side_effect=_upsert)
    prisma = MagicMock()
    prisma.db.litellm_managedobjecttable = table

    cache = MagicMock()
    cache.async_set_cache = AsyncMock()

    return (
        _PROXY_LiteLLMManagedFiles(internal_usage_cache=cache, prisma_client=prisma),
        store,
    )


@pytest.mark.asyncio
async def test_store_unified_object_id_persists_key_and_tags_on_create():
    """Regression (spend loss): the batch create persists the creating key hash and tags so
    CheckBatchCost can write an attributed spend row instead of a blank one the DB drops."""
    instance, store = _in_memory_managed_files()
    creator = UserAPIKeyAuth(user_id="alice", team_id="team-alpha", api_key="hash-alice")

    await instance.store_unified_object_id(
        unified_object_id="unified-b",
        file_object=_build_batch_response(batch_id="b", status="validating"),
        litellm_parent_otel_span=None,
        model_object_id="b",
        file_purpose="batch",
        user_api_key_dict=creator,
        request_tags=["env:prod"],
        persist_attribution=True,
    )

    row = store["unified-b"]
    assert row["api_key"] == "hash-alice"
    assert row["created_by"] == "alice"
    assert row["team_id"] == "team-alpha"
    assert row["request_tags"].data == ["env:prod"]


@pytest.mark.asyncio
async def test_store_unified_object_id_omits_key_and_tags_without_persist_attribution():
    """Regression (spend redirect): a caller that is not the batch create (a poll, or the
    generic post-call hook on a retrieve) carries a real hashed key, but must never have it
    recorded as the batch's paying key. created_by/team_id keep their existing behavior."""
    instance, store = _in_memory_managed_files()
    poller = UserAPIKeyAuth(user_id="bob", team_id="team-bravo", api_key="hash-bob")

    await instance.store_unified_object_id(
        unified_object_id="unified-b",
        file_object=_build_batch_response(batch_id="b", status="in_progress"),
        litellm_parent_otel_span=None,
        model_object_id="b",
        file_purpose="batch",
        user_api_key_dict=poller,
        request_tags=["env:dev"],
    )

    row = store["unified-b"]
    assert "api_key" not in row
    assert "request_tags" not in row
    assert row["created_by"] == "bob"


@pytest.mark.asyncio
async def test_store_unified_object_id_attribution_columns_are_write_once():
    """Identity is written only in the upsert create branch, so a later store for the same
    batch (a status update, a poll) can neither reassign the paying key nor clear it."""
    instance, store = _in_memory_managed_files()
    creator = UserAPIKeyAuth(user_id="alice", team_id="team-alpha", api_key="hash-alice")
    poller = UserAPIKeyAuth(user_id="bob", team_id="team-bravo", api_key="hash-bob")

    await instance.store_unified_object_id(
        unified_object_id="unified-b",
        file_object=_build_batch_response(batch_id="b", status="validating"),
        litellm_parent_otel_span=None,
        model_object_id="b",
        file_purpose="batch",
        user_api_key_dict=creator,
        request_tags=["env:prod"],
        persist_attribution=True,
    )
    await instance.store_unified_object_id(
        unified_object_id="unified-b",
        file_object=_build_batch_response(batch_id="b", status="completed"),
        litellm_parent_otel_span=None,
        model_object_id="b",
        file_purpose="batch",
        user_api_key_dict=poller,
        request_tags=["poller-tag"],
        persist_attribution=True,
    )

    row = store["unified-b"]
    assert row["api_key"] == "hash-alice"
    assert row["created_by"] == "alice"
    assert row["status"] == "completed"

    upsert_data = instance.prisma_client.db.litellm_managedobjecttable.upsert.call_args.kwargs["data"]
    assert "api_key" not in upsert_data["update"]
    assert "request_tags" not in upsert_data["update"]


@pytest.mark.asyncio
async def test_store_unified_object_id_omits_unset_columns():
    """A batch created with no tags (the common case) still registers: the optional columns
    are omitted rather than passed as None, which prisma rejects for the Json column."""
    instance, store = _in_memory_managed_files()
    creator = UserAPIKeyAuth(user_id="alice", team_id="team-alpha", api_key=None)

    await instance.store_unified_object_id(
        unified_object_id="unified-b",
        file_object=_build_batch_response(batch_id="b", status="validating"),
        litellm_parent_otel_span=None,
        model_object_id="b",
        file_purpose="batch",
        user_api_key_dict=creator,
        request_tags=None,
        persist_attribution=True,
    )

    create_data = instance.prisma_client.db.litellm_managedobjecttable.upsert.call_args.kwargs["data"]["create"]
    assert "api_key" not in create_data
    assert "request_tags" not in create_data
    assert "unified-b" in store
