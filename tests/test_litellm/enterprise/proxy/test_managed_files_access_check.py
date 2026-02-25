"""
Tests for managed files access control in batch polling context.

Regression test for: batch polling job running as default_user_id gets 403
when trying to access managed files created by a real user.

The fix (Option C) makes check_batch_cost call litellm.afile_content directly
with deployment credentials, bypassing the managed files access-control hooks.
"""

import base64
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth


def _make_user_api_key_dict(user_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        parent_otel_span=None,
    )


def _make_unified_file_id() -> str:
    """Create a base64-encoded unified file ID that passes _is_base64_encoded_unified_file_id."""
    raw = "litellm_proxy:application/octet-stream;unified_id,test-123;target_model_names,azure-gpt-4"
    return base64.b64encode(raw.encode()).decode()


def _make_managed_files_instance(file_created_by: str, unified_file_id: str):
    """Create a _PROXY_LiteLLMManagedFiles with a mocked DB that returns a file owned by file_created_by."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_db_record = MagicMock()
    mock_db_record.created_by = file_created_by

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=mock_db_record
    )

    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=mock_prisma,
    )
    return instance


# --- Access control unit tests (document existing behavior) ---


@pytest.mark.asyncio
async def test_should_allow_file_owner_access():
    """File owner can access their own file — baseline sanity check."""
    unified_file_id = _make_unified_file_id()
    managed_files = _make_managed_files_instance(
        file_created_by="user-A",
        unified_file_id=unified_file_id,
    )
    user = _make_user_api_key_dict("user-A")
    data = {"file_id": unified_file_id}

    result = await managed_files.check_managed_file_id_access(data, user)
    assert result is True


@pytest.mark.asyncio
async def test_should_block_different_user_access():
    """A different regular user cannot access another user's file — correct behavior."""
    unified_file_id = _make_unified_file_id()
    managed_files = _make_managed_files_instance(
        file_created_by="user-A",
        unified_file_id=unified_file_id,
    )
    user = _make_user_api_key_dict("user-B")
    data = {"file_id": unified_file_id}

    with pytest.raises(HTTPException) as exc_info:
        await managed_files.check_managed_file_id_access(data, user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_should_block_default_user_id_access():
    """
    default_user_id is correctly blocked by the access check.
    This documents the existing behavior that the Option C fix works around.
    """
    unified_file_id = _make_unified_file_id()
    managed_files = _make_managed_files_instance(
        file_created_by="user-A",
        unified_file_id=unified_file_id,
    )
    system_user = _make_user_api_key_dict("default_user_id")
    data = {"file_id": unified_file_id}

    with pytest.raises(HTTPException) as exc_info:
        await managed_files.check_managed_file_id_access(data, system_user)
    assert exc_info.value.status_code == 403


# --- Option C fix test: check_batch_cost bypasses managed files hook ---


@pytest.mark.asyncio
async def test_check_batch_cost_should_call_afile_content_directly_with_credentials():
    """
    check_batch_cost should call litellm.afile_content directly with deployment
    credentials, bypassing managed_files_obj.afile_content and its access-control
    hooks. This avoids the 403 that occurs when the background job runs as
    default_user_id.
    """
    from litellm_enterprise.proxy.common_utils.check_batch_cost import CheckBatchCost

    # Build a unified object ID in the expected format:
    # litellm_proxy;model_id:{};llm_batch_id:{};llm_output_file_id:{}
    unified_raw = "litellm_proxy;model_id:model-deploy-xyz;llm_batch_id:batch-123;llm_output_file_id:file-raw-output"
    unified_object_id = base64.b64encode(unified_raw.encode()).decode()

    # Mock a pending job from the DB
    mock_job = MagicMock()
    mock_job.unified_object_id = unified_object_id
    mock_job.created_by = "user-A"
    mock_job.id = "job-1"

    # Mock prisma
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedobjecttable.find_many = AsyncMock(
        return_value=[mock_job]
    )
    mock_prisma.db.litellm_managedobjecttable.update_many = AsyncMock()

    # Mock proxy_logging_obj — should NOT be called for file content
    mock_proxy_logging = MagicMock()
    mock_managed_files_hook = MagicMock()
    mock_managed_files_hook.afile_content = AsyncMock()
    mock_proxy_logging.get_proxy_hook = MagicMock(return_value=mock_managed_files_hook)

    # Mock the batch response (completed, with output file)
    from litellm.types.utils import LiteLLMBatch
    batch_response = LiteLLMBatch(
        id="batch-123",
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input",
        object="batch",
        status="completed",
        output_file_id="file-raw-output",
    )

    # Mock router
    mock_router = MagicMock()
    mock_router.aretrieve_batch = AsyncMock(return_value=batch_response)
    mock_router.get_deployment_credentials_with_provider = MagicMock(
        return_value={
            "api_key": "test-key",
            "api_base": "https://test.azure.com/",
            "custom_llm_provider": "azure",
        }
    )

    mock_deployment = MagicMock()
    mock_deployment.litellm_params.custom_llm_provider = "azure"
    mock_deployment.litellm_params.model = "azure/gpt-4"
    mock_router.get_deployment = MagicMock(return_value=mock_deployment)

    checker = CheckBatchCost(
        proxy_logging_obj=mock_proxy_logging,
        prisma_client=mock_prisma,
        llm_router=mock_router,
    )

    mock_file_content = MagicMock()
    mock_file_content.content = b'{"id":"req-1","response":{"status_code":200,"body":{"id":"cmpl-1","object":"chat.completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}}}\n'

    with patch(
        "litellm.files.main.afile_content",
        new_callable=AsyncMock,
        return_value=mock_file_content,
    ) as mock_direct_afile_content:
        await checker.check_batch_cost()

        # afile_content should be called directly (not through managed_files_obj)
        mock_direct_afile_content.assert_called_once()
        call_kwargs = mock_direct_afile_content.call_args.kwargs

        assert call_kwargs.get("api_key") == "test-key", (
            f"afile_content should receive api_key from deployment credentials. "
            f"Got: {call_kwargs}"
        )

        # managed_files_obj.afile_content should NOT have been called
        mock_managed_files_hook.afile_content.assert_not_called()
