"""Regression: update_batch_in_database must not persist raw provider output_file_id."""

import json
from types import SimpleNamespace
from typing import Optional
import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from unittest.mock import patch

from litellm.proxy.openai_files_endpoints.common_utils import (
    ensure_batch_response_managed_file_ids,
    get_batch_from_database,
    read_stored_batch_attribution,
    strip_internal_batch_attribution,
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


def _build_prisma_mock():
    mock = MagicMock()
    mock.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
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


_BATCH_ATTRIBUTION = {
    "user_api_key": "hashed-key-abc",
    "user_api_key_user_id": "user-real",
    "user_api_key_team_id": "team-real",
    "user_api_key_alias": "prod-key",
    "user_api_key_team_alias": "prod-team",
    "user_api_key_user_email": "real@corp.com",
    "request_tags": ["feature-x"],
}


def _db_batch_object_with_snapshot(
    *, batch_id: str, status: str, file_object=None
) -> SimpleNamespace:
    if file_object is None:
        file_object = json.dumps(
            {
                "id": batch_id,
                "status": status,
                "metadata": {"litellm_batch_attribution": _BATCH_ATTRIBUTION},
            }
        )
    return SimpleNamespace(
        created_by="user-real",
        team_id="team-real",
        status=status,
        file_object=file_object,
    )


@pytest.mark.asyncio
async def test_update_batch_in_database_preserves_attribution_snapshot():
    """Regression: overwriting file_object on a status change must not erase the create-time
    litellm_batch_attribution snapshot; the provider response never carries it, so it has to be
    carried forward from the stored row or the completed batch loses key-level spend attribution
    """
    batch_id = "batch_attr_preserve"
    unified_batch_id = "litellm_proxy;model_id:my-model;llm_batch_id:batch_attr_preserve"

    response = _build_batch_response(
        batch_id=batch_id,
        status="completed",
        output_file_id="file-rawoutput789",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )
    assert response.metadata is None

    db_batch_object = _db_batch_object_with_snapshot(
        batch_id=batch_id, status="in_progress"
    )
    mock_prisma = _build_prisma_mock()

    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=response,
        managed_files_obj=_build_managed_files_mock(),
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-real"),
    )

    stored = json.loads(
        mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"][
            "file_object"
        ]
    )
    assert stored["metadata"]["litellm_batch_attribution"] == _BATCH_ATTRIBUTION
    # The snapshot must land only in the persisted row, never on the object returned to the caller
    assert response.metadata is None


@pytest.mark.asyncio
async def test_update_batch_in_database_no_snapshot_leaves_response_metadata_untouched():
    """When the stored row has no snapshot, nothing is injected into the response metadata."""
    batch_id = "batch_attr_absent"
    unified_batch_id = "litellm_proxy;model_id:my-model;llm_batch_id:batch_attr_absent"

    response = _build_batch_response(
        batch_id=batch_id,
        status="completed",
        output_file_id="file-rawoutput789",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    db_batch_object = _db_batch_object_with_snapshot(
        batch_id=batch_id,
        status="in_progress",
        file_object=json.dumps({"id": batch_id, "status": "in_progress"}),
    )
    mock_prisma = _build_prisma_mock()

    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=response,
        managed_files_obj=_build_managed_files_mock(),
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-real"),
    )

    stored = json.loads(
        mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"][
            "file_object"
        ]
    )
    assert "litellm_batch_attribution" not in (stored.get("metadata") or {})


def test_read_stored_batch_attribution_from_json_string():
    row = _db_batch_object_with_snapshot(batch_id="b1", status="in_progress")
    assert read_stored_batch_attribution(row) == _BATCH_ATTRIBUTION


def test_read_stored_batch_attribution_from_parsed_dict():
    row = SimpleNamespace(
        file_object={"metadata": {"litellm_batch_attribution": _BATCH_ATTRIBUTION}}
    )
    assert read_stored_batch_attribution(row) == _BATCH_ATTRIBUTION


@pytest.mark.parametrize(
    "row",
    [
        None,
        SimpleNamespace(file_object=None),
        SimpleNamespace(file_object="not-json"),
        SimpleNamespace(file_object=json.dumps({"id": "b1"})),
        SimpleNamespace(file_object=json.dumps({"metadata": {}})),
        SimpleNamespace(file_object=json.dumps({"metadata": {"litellm_batch_attribution": "wrong-type"}})),
    ],
)
def test_read_stored_batch_attribution_returns_none_when_absent(row):
    assert read_stored_batch_attribution(row) is None


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


def test_strip_internal_batch_attribution_removes_only_internal_key():
    file_object_data = {
        "id": "batch_1",
        "status": "completed",
        "metadata": {"user_tag": "keep-me", "litellm_batch_attribution": _BATCH_ATTRIBUTION},
    }
    cleaned = strip_internal_batch_attribution(file_object_data)
    assert cleaned["metadata"] == {"user_tag": "keep-me"}
    # Original is not mutated (the raw row must keep the snapshot for the poller)
    assert "litellm_batch_attribution" in file_object_data["metadata"]


def test_strip_internal_batch_attribution_nulls_metadata_when_only_internal_key():
    cleaned = strip_internal_batch_attribution(
        {"id": "batch_1", "metadata": {"litellm_batch_attribution": _BATCH_ATTRIBUTION}}
    )
    assert cleaned["metadata"] is None


@pytest.mark.parametrize(
    "file_object_data",
    [
        {"id": "batch_1"},
        {"id": "batch_1", "metadata": None},
        {"id": "batch_1", "metadata": {"user_tag": "keep-me"}},
    ],
)
def test_strip_internal_batch_attribution_noop_without_internal_key(file_object_data):
    assert strip_internal_batch_attribution(file_object_data) == file_object_data


@pytest.mark.asyncio
async def test_get_batch_from_database_does_not_leak_attribution_to_caller():
    """Regression: the batch returned from the stored row must not expose litellm_batch_attribution,
    while the raw db row handed back for internal use keeps it
    """
    batch_id = "litellm_proxy;model_id:m;llm_batch_id:b"
    stored_file_object = {
        "id": "b",
        "object": "batch",
        "status": "completed",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-input",
        "completion_window": "24h",
        "created_at": 1234567890,
        "metadata": {"litellm_batch_attribution": _BATCH_ATTRIBUTION},
    }
    row = SimpleNamespace(file_object=json.dumps(stored_file_object))

    with patch(
        "litellm.proxy.openai_files_endpoints.common_utils.ManagedObjectRepository"
    ) as mock_repo:
        mock_repo.return_value.table.find_first = AsyncMock(return_value=row)
        db_batch_object, response = await get_batch_from_database(
            batch_id=batch_id,
            unified_batch_id=batch_id,
            managed_files_obj=MagicMock(),
            prisma_client=_build_prisma_mock(),
            verbose_proxy_logger=MagicMock(),
        )

    assert "litellm_batch_attribution" not in (response.metadata or {})
    # The raw row still carries the snapshot so update_batch_in_database can preserve it
    assert read_stored_batch_attribution(db_batch_object) == _BATCH_ATTRIBUTION
