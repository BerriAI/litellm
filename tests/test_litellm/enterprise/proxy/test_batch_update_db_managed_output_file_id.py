"""Regression: update_batch_in_database must not persist raw provider output_file_id."""

import base64
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
from litellm.types.utils import LiteLLMBatch, SpecialEnums


def _make_unified_file_id(
    raw_output_file_id: str = "file-managed-output",
    model_id: str = "my-model",
    model_name: str = "openai/gpt-4o",
) -> str:
    """Build a base64 unified file id the way get_unified_output_file_id does, so
    _is_base64_encoded_unified_file_id recognizes it (a fake id that fails that
    check would be redacted as fail-closed)."""
    plain = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
        "application/json",
        "00000000-0000-0000-0000-000000000abc",
        model_name,
        raw_output_file_id,
        model_id,
    )
    return base64.urlsafe_b64encode(plain.encode()).decode().rstrip("=")


UNIFIED_FILE_ID = _make_unified_file_id()


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


def _build_managed_files_mock(unified_id: str = UNIFIED_FILE_ID):
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
    unified_output_file_id = UNIFIED_FILE_ID
    batch_id = "batch_managed_ids_test"
    unified_batch_id = "litellm_proxy;model_id:my-model;llm_batch_id:batch_managed_ids_test"

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

    stored = json.loads(mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"]["file_object"])
    assert stored["output_file_id"] == unified_output_file_id
    assert stored["output_file_id"] != raw_output_file_id


@pytest.mark.asyncio
async def test_ensure_batch_response_normalizes_error_file_id():
    """Both output_file_id and error_file_id must be normalized to managed IDs."""
    unified_id = UNIFIED_FILE_ID
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
async def test_ensure_batch_response_redacts_id_when_conversion_errors():
    """Fail closed: when conversion raises, the raw provider id must be redacted,
    not returned, so it can't be replayed against /v1/files to bypass ownership."""
    raw_output_file_id = "file-raw-output"
    response = _build_batch_response(
        output_file_id=raw_output_file_id,
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = MagicMock()
    mock_managed_files.get_unified_output_file_id = MagicMock(side_effect=RuntimeError("boom"))
    mock_managed_files.store_unified_file_id = AsyncMock()

    mock_logger = MagicMock()
    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=mock_logger,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-abc"),
    )

    assert response.output_file_id is None
    mock_logger.warning.assert_called()


@pytest.mark.asyncio
async def test_ensure_batch_response_builds_auth_from_db_batch_object():
    """If user_api_key_dict is omitted, fall back to created_by/team_id on db_batch_object."""
    unified_id = UNIFIED_FILE_ID
    response = _build_batch_response(
        output_file_id="file-raw-output",
        hidden_params={"model_id": "my-model", "model_name": "openai/gpt-4o"},
    )

    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)
    db_batch_object = SimpleNamespace(created_by="user-from-db", team_id="team-from-db", status="completed")

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
    )

    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs["user_api_key_dict"]
    assert forwarded_auth.user_id == "user-from-db"
    assert forwarded_auth.team_id == "team-from-db"


@pytest.mark.asyncio
async def test_ensure_batch_response_resolves_model_name_from_unified_file_id():
    """When hidden_params lacks model_name, derive it from unified_file_id."""
    unified_id = UNIFIED_FILE_ID
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
        mock_managed_files.get_unified_output_file_id.call_args.kwargs["model_name"] == "gpt-4o-mini,gemini-2.0-flash"
    )


@pytest.mark.asyncio
async def test_ensure_batch_response_derives_model_id_from_unified_batch_id():
    """Terminal DB rows have empty _hidden_params (rehydrated from stored JSON).

    Regression for GH #33989: the helper must still register the raw output/error
    ids as managed files by deriving model_id from the unified batch id, otherwise
    the terminal-retrieve short-circuit leaks raw provider file ids.
    """
    unified_id = UNIFIED_FILE_ID
    response = _build_batch_response(
        output_file_id="file-raw-output",
        error_file_id="file-raw-error",
        hidden_params={},
    )
    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=SimpleNamespace(created_by="user-from-db", team_id=None, status="failed"),
        unified_batch_id="litellm_proxy;model_id:deployment-42;llm_batch_id:batch_abc",
    )

    assert response.output_file_id == unified_id
    assert response.error_file_id == unified_id
    assert mock_managed_files.get_unified_output_file_id.call_count == 2
    assert mock_managed_files.get_unified_output_file_id.call_args.kwargs["model_id"] == "deployment-42"


@pytest.mark.asyncio
async def test_ensure_batch_response_redacts_raw_id_without_managed_files_obj():
    """Fail closed: without the managed-files hook, conversion is impossible, so the
    raw provider id must be redacted rather than returned."""
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

    assert response.output_file_id is None


@pytest.mark.asyncio
async def test_ensure_batch_response_redacts_raw_id_without_model_id():
    """Fail closed: without a resolvable model_id no managed id can be built, so the
    raw provider id must be redacted rather than returned."""
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

    assert response.output_file_id is None
    mock_managed_files.get_unified_output_file_id.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_batch_response_redacts_raw_id_without_auth():
    """Fail closed: without any owner identity no row can be registered, so the raw
    provider id must be redacted rather than returned."""
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

    assert response.output_file_id is None
    mock_managed_files.get_unified_output_file_id.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_batch_response_owns_registered_files_by_batch_creator_not_requester():
    """Regression for GH #33989 review: when a non-creator retrieves a terminal batch
    before its files are registered, the new managed-file rows must be owned by the
    batch creator (from db_batch_object), not the requester, otherwise the requester
    could grant themselves access and the real creator couldn't manage the file."""
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
        user_api_key_dict=UserAPIKeyAuth(user_id="attacker-user", team_id="attacker-team"),
        db_batch_object=SimpleNamespace(created_by="victim-user", team_id="victim-team", status="failed"),
    )

    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs["user_api_key_dict"]
    assert forwarded_auth.user_id == "victim-user"
    assert forwarded_auth.team_id == "victim-team"
    assert response.output_file_id == UNIFIED_FILE_ID


@pytest.mark.asyncio
async def test_ensure_batch_response_mirrors_none_owner_instead_of_synthetic_default_user():
    """Regression for GH #33989 review: a batch created by a key without a user
    (created_by is None) must register its files with created_by=None so the
    creator's team ACL still matches; a synthetic "default-user-id" owner matches
    no real caller and locks the creator out of their own terminal batch files."""
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
        user_api_key_dict=None,
        db_batch_object=SimpleNamespace(created_by=None, team_id="victim-team", status="failed"),
    )

    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs["user_api_key_dict"]
    assert forwarded_auth.user_id is None
    assert forwarded_auth.team_id == "victim-team"
    assert response.output_file_id == UNIFIED_FILE_ID
