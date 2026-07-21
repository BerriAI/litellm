"""Regression for #33989: terminal batch retrieve must register missing managed-file rows."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    ensure_batch_response_managed_file_ids,
    get_batch_from_database,
)
from litellm.types.utils import LiteLLMBatch


UNIFIED_BATCH_ID = "litellm_proxy;model_id:my-model;llm_batch_id:batch-raw-123"
ENCODED_UNIFIED_BATCH_ID = (
    base64.urlsafe_b64encode(UNIFIED_BATCH_ID.encode()).decode().rstrip("=")
)


def _build_batch_response(
    *,
    output_file_id: str = "file-rawoutput789",
    error_file_id: str | None = None,
) -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch-raw-123",
        object="batch",
        status="completed",
        endpoint="/v1/chat/completions",
        input_file_id="file-input123",
        output_file_id=output_file_id,
        error_file_id=error_file_id,
        completion_window="24h",
        created_at=1234567890,
    )


def _build_managed_files_mock(unified_id: str = "file-bWFuYWdlZF9vdXRwdXRfaWQ="):
    mock = MagicMock()
    mock.get_unified_output_file_id = MagicMock(return_value=unified_id)
    mock.store_unified_file_id = AsyncMock()
    return mock


def _build_prisma_mock():
    mock = MagicMock()
    mock.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    return mock


@pytest.mark.asyncio
async def test_ensure_batch_response_derives_model_id_from_unified_batch_id():
    raw_output_file_id = "file-rawoutput789"
    unified_output_file_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(output_file_id=raw_output_file_id)

    mock_managed_files = _build_managed_files_mock(unified_id=unified_output_file_id)
    db_batch_object = SimpleNamespace(
        created_by="batch-owner", team_id="team-owner", status="completed"
    )

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=db_batch_object,
        unified_batch_id=UNIFIED_BATCH_ID,
    )

    assert response.output_file_id == unified_output_file_id
    mock_managed_files.store_unified_file_id.assert_called_once()
    store_kwargs = mock_managed_files.store_unified_file_id.call_args.kwargs
    assert store_kwargs["model_mappings"] == {"my-model": raw_output_file_id}
    assert store_kwargs["user_api_key_dict"].user_id == "batch-owner"
    assert store_kwargs["user_api_key_dict"].team_id == "team-owner"


@pytest.mark.asyncio
async def test_ensure_batch_response_registers_output_and_error_file_ids():
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(
        output_file_id="file-raw-output",
        error_file_id="file-raw-error",
    )
    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=SimpleNamespace(created_by="batch-owner", team_id=None),
        unified_batch_id=UNIFIED_BATCH_ID,
    )

    assert response.output_file_id == unified_id
    assert response.error_file_id == unified_id
    assert mock_managed_files.store_unified_file_id.call_count == 2
    mappings = [
        call.kwargs["model_mappings"]
        for call in mock_managed_files.store_unified_file_id.call_args_list
    ]
    assert {"my-model": "file-raw-output"} in mappings
    assert {"my-model": "file-raw-error"} in mappings


@pytest.mark.asyncio
async def test_get_batch_from_database_registers_missing_output_file_id():
    raw_output_file_id = "file-output-raw"
    unified_output_file_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    batch_data = {
        "id": "batch-raw-123",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-input-raw",
        "object": "batch",
        "status": "completed",
        "output_file_id": raw_output_file_id,
    }

    batch_db_record = SimpleNamespace(
        file_object=json.dumps(batch_data),
        created_by="batch-owner",
        team_id="team-owner",
        status="complete",
    )
    prisma = MagicMock()
    prisma.db.litellm_managedobjecttable.find_first = AsyncMock(
        return_value=batch_db_record
    )
    prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)

    mock_managed_files = _build_managed_files_mock(unified_id=unified_output_file_id)

    _, response = await get_batch_from_database(
        batch_id=ENCODED_UNIFIED_BATCH_ID,
        unified_batch_id=UNIFIED_BATCH_ID,
        managed_files_obj=mock_managed_files,
        prisma_client=prisma,
        verbose_proxy_logger=MagicMock(),
    )

    assert response is not None
    assert response.output_file_id == unified_output_file_id
    mock_managed_files.store_unified_file_id.assert_called_once()
    store_kwargs = mock_managed_files.store_unified_file_id.call_args.kwargs
    assert store_kwargs["model_mappings"] == {"my-model": raw_output_file_id}
    assert store_kwargs["user_api_key_dict"].user_id == "batch-owner"


@pytest.mark.asyncio
async def test_ensure_batch_response_uses_batch_owner_when_db_batch_object_present():
    response = _build_batch_response(output_file_id="file-raw-output")
    mock_managed_files = _build_managed_files_mock()

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="other-user", team_id="other-team"),
        db_batch_object=SimpleNamespace(
            created_by="batch-owner", team_id="team-owner", status="completed"
        ),
        unified_batch_id=UNIFIED_BATCH_ID,
    )

    # batch owner from db_batch_object wins over the caller auth context
    forwarded_auth = mock_managed_files.store_unified_file_id.call_args.kwargs[
        "user_api_key_dict"
    ]
    assert forwarded_auth.user_id == "batch-owner"
    assert forwarded_auth.team_id == "team-owner"


@pytest.mark.asyncio
async def test_ensure_batch_response_derives_model_id_from_encoded_response_id():
    unified_id = "file-bWFuYWdlZF9vdXRwdXRfaWQ="
    response = _build_batch_response(output_file_id="file-raw-output")
    response.id = ENCODED_UNIFIED_BATCH_ID

    mock_managed_files = _build_managed_files_mock(unified_id=unified_id)

    await ensure_batch_response_managed_file_ids(
        response=response,
        managed_files_obj=mock_managed_files,
        prisma_client=_build_prisma_mock(),
        verbose_proxy_logger=MagicMock(),
        db_batch_object=SimpleNamespace(created_by="batch-owner", team_id=None),
    )

    assert response.output_file_id == unified_id
    assert (
        mock_managed_files.get_unified_output_file_id.call_args.kwargs["model_id"]
        == "my-model"
    )
