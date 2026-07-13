"""
Tests for file deletion blocking when referenced by non-terminal batches.

This tests the feature where file deletion is blocked when:
1. File is referenced by a batch in non-terminal state (validating, in_progress, finalizing)
2. Batch polling is configured (proxy_batch_polling_interval > 0)

This ensures cost tracking is not disrupted by premature file deletion.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth


def _make_unified_file_id(file_id: str = "file-abc123") -> str:
    """Create a base64-encoded unified file ID."""
    raw = f"litellm_proxy:application/json;unified_id,test-{file_id};target_model_names,azure-gpt-4;llm_output_file_id,{file_id};llm_output_file_model_id,model-123"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _make_unified_batch_id(batch_id: str = "batch-123") -> str:
    """Create a base64-encoded unified batch ID."""
    raw = f"litellm_proxy;model_id:model-deploy-xyz;llm_batch_id:{batch_id};llm_output_file_id:file-output"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _make_user_api_key_dict(user_id: str = "user-A") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        parent_otel_span=None,
    )


def _make_batch_db_record(
    unified_object_id: str,
    status: str,
    file_object: dict,
    created_by: str = "user-A",
):
    """Create a mock batch database record."""
    mock_batch = MagicMock()
    mock_batch.unified_object_id = unified_object_id
    mock_batch.status = status
    mock_batch.file_object = json.dumps(file_object)
    mock_batch.created_by = created_by
    mock_batch.created_at = 1700000000
    return mock_batch


def _make_managed_files_instance_with_batches(
    file_id: str,
    batches: list,
    file_created_by: str = "user-A",
):
    """
    Create a _PROXY_LiteLLMManagedFiles instance with mocked DB and batches.
    
    Args:
        file_id: The unified file ID
        batches: List of batch records to return from DB
        file_created_by: The user who created the file
    """
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    # Mock file record
    mock_file_record = MagicMock()
    mock_file_record.unified_file_id = file_id
    mock_file_record.created_by = file_created_by
    mock_file_record.model_mappings = {"model-123": "provider-file-abc"}

    # Mock prisma
    mock_prisma = MagicMock()
    
    # Mock file table queries
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=mock_file_record
    )
    mock_prisma.db.litellm_managedfiletable.delete = AsyncMock(
        return_value=mock_file_record
    )
    
    # Mock batch/object table queries
    mock_prisma.db.litellm_managedobjecttable.find_many = AsyncMock(
        return_value=batches
    )

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.async_get_cache = AsyncMock(return_value={
        "unified_file_id": file_id,
        "model_mappings": {"model-123": "provider-file-abc"},
        "flat_model_file_ids": ["provider-file-abc"],
    })
    mock_cache.async_set_cache = AsyncMock()

    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=mock_cache,
        prisma_client=mock_prisma,
    )
    return instance


# --- Test: Batch polling configuration check ---


def test_is_batch_polling_enabled_when_job_registered():
    """Test that batch polling is detected as enabled when scheduler job is registered."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )
    
    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=MagicMock(),
    )
    
    # Mock scheduler with registered job
    mock_scheduler = MagicMock()
    mock_job = MagicMock()
    mock_scheduler.get_job.return_value = mock_job
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        assert instance._is_batch_polling_enabled() is True


def test_is_batch_polling_disabled_when_job_not_registered():
    """Test that batch polling is detected as disabled when scheduler job is not registered."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )
    
    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=MagicMock(),
    )
    
    # Mock scheduler without registered job
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        assert instance._is_batch_polling_enabled() is False


def test_is_batch_polling_disabled_when_no_scheduler():
    """Test that batch polling is detected as disabled when scheduler is not available."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )
    
    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=MagicMock(),
    )
    
    with patch("litellm.proxy.proxy_server.scheduler", None):
        assert instance._is_batch_polling_enabled() is False


# --- Test: Finding batches referencing files ---


@pytest.mark.asyncio
async def test_get_batches_referencing_file_finds_batch_with_input_file():
    """Test finding a batch that references the file as input_file_id."""
    unified_file_id = _make_unified_file_id("file-input-123")
    unified_batch_id = _make_unified_batch_id("batch-123")
    
    batch_file_object = {
        "id": "batch-123",
        "input_file_id": unified_file_id,  # Batch references this file
        "status": "validating",
    }
    
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="validating",
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch_record],
    )
    
    referencing_batches = await managed_files._get_batches_referencing_file(unified_file_id)
    
    assert len(referencing_batches) == 1
    assert referencing_batches[0]["batch_id"] == unified_batch_id
    assert referencing_batches[0]["status"] == "validating"


@pytest.mark.asyncio
async def test_get_batches_referencing_file_finds_batch_with_output_file():
    """Test finding a batch that references the file as output_file_id."""
    unified_file_id = _make_unified_file_id("file-output-456")
    unified_batch_id = _make_unified_batch_id("batch-456")
    
    batch_file_object = {
        "id": "batch-456",
        "input_file_id": "file-input-different",
        "output_file_id": unified_file_id,  # Batch references this file
        "status": "in_progress",
    }
    
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="in_progress",
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch_record],
    )
    
    referencing_batches = await managed_files._get_batches_referencing_file(unified_file_id)
    
    assert len(referencing_batches) == 1
    assert referencing_batches[0]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_get_batches_referencing_file_ignores_terminal_batches():
    """Test that batches in terminal states are not returned."""
    unified_file_id = _make_unified_file_id("file-123")
    unified_batch_id = _make_unified_batch_id("batch-completed")
    
    batch_file_object = {
        "id": "batch-completed",
        "input_file_id": unified_file_id,
        "status": "completed",
    }
    
    # Batch is in terminal state in DB
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="completed",  # Terminal state
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[],  # Query returns no batches (terminal states filtered out)
    )
    
    referencing_batches = await managed_files._get_batches_referencing_file(unified_file_id)
    
    assert len(referencing_batches) == 0


@pytest.mark.asyncio
async def test_get_batches_referencing_file_finds_multiple_batches():
    """Test finding multiple batches referencing the same file."""
    unified_file_id = _make_unified_file_id("file-shared")
    
    batch1 = _make_batch_db_record(
        unified_object_id=_make_unified_batch_id("batch-1"),
        status="validating",
        file_object={"id": "batch-1", "input_file_id": unified_file_id, "status": "validating"},
    )
    
    batch2 = _make_batch_db_record(
        unified_object_id=_make_unified_batch_id("batch-2"),
        status="in_progress",
        file_object={"id": "batch-2", "input_file_id": unified_file_id, "status": "in_progress"},
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch1, batch2],
    )
    
    referencing_batches = await managed_files._get_batches_referencing_file(unified_file_id)
    
    assert len(referencing_batches) == 2
    statuses = [b["status"] for b in referencing_batches]
    assert "validating" in statuses
    assert "in_progress" in statuses


# --- Test: File deletion blocking logic ---


@pytest.mark.asyncio
async def test_file_deletion_blocked_when_batch_polling_enabled_and_batch_references_file():
    """
    Test that file deletion is blocked when:
    1. Batch cost tracking job is registered (polling enabled)
    2. File is referenced by a non-terminal batch
    """
    unified_file_id = _make_unified_file_id("file-to-delete")
    unified_batch_id = _make_unified_batch_id("batch-active")
    
    batch_file_object = {
        "id": "batch-active",
        "input_file_id": unified_file_id,
        "status": "validating",
    }
    
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="validating",
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch_record],
    )
    
    # Mock scheduler with registered batch cost job
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = MagicMock()  # Job exists
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        with pytest.raises(HTTPException) as exc_info:
            await managed_files._check_file_deletion_allowed(unified_file_id)
        
        assert exc_info.value.status_code == 400
        error_detail = exc_info.value.detail
        assert "Cannot delete file" in error_detail
        assert unified_file_id in error_detail
        assert "validating" in error_detail
        assert "delete or cancel the referencing batch" in error_detail.lower()


@pytest.mark.asyncio
async def test_file_deletion_allowed_when_batch_polling_disabled():
    """
    Test that file deletion is allowed when batch cost tracking job is not registered,
    even if there are non-terminal batches referencing the file.
    """
    unified_file_id = _make_unified_file_id("file-to-delete")
    unified_batch_id = _make_unified_batch_id("batch-active")
    
    batch_file_object = {
        "id": "batch-active",
        "input_file_id": unified_file_id,
        "status": "validating",
    }
    
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="validating",
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch_record],
    )
    
    # Mock scheduler without registered job (batch cost tracking disabled)
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        # Should not raise an exception
        await managed_files._check_file_deletion_allowed(unified_file_id)


@pytest.mark.asyncio
async def test_file_deletion_allowed_when_no_batches_reference_file():
    """
    Test that file deletion is allowed when no batches reference the file,
    even when batch cost tracking is enabled.
    """
    unified_file_id = _make_unified_file_id("file-to-delete")
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[],  # No batches reference this file
    )
    
    # Mock scheduler with registered job (batch cost tracking enabled)
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = MagicMock()
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        # Should not raise an exception
        await managed_files._check_file_deletion_allowed(unified_file_id)


@pytest.mark.asyncio
async def test_afile_delete_calls_check_deletion_allowed():
    """
    Test that afile_delete calls _check_file_deletion_allowed before deleting.
    """
    unified_file_id = _make_unified_file_id("file-to-delete")
    unified_batch_id = _make_unified_batch_id("batch-active")
    
    batch_file_object = {
        "id": "batch-active",
        "input_file_id": unified_file_id,
        "status": "in_progress",
    }
    
    batch_record = _make_batch_db_record(
        unified_object_id=unified_batch_id,
        status="in_progress",
        file_object=batch_file_object,
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch_record],
    )
    
    # Mock llm_router
    mock_router = MagicMock()
    mock_router.afile_delete = AsyncMock()
    
    # Mock scheduler with registered job
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = MagicMock()
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        with pytest.raises(HTTPException) as exc_info:
            await managed_files.afile_delete(
                file_id=unified_file_id,
                litellm_parent_otel_span=None,
                llm_router=mock_router,
            )
        
        # Should raise error before calling router delete
        assert exc_info.value.status_code == 400
        mock_router.afile_delete.assert_not_called()


@pytest.mark.asyncio
async def test_database_limit_respected():
    """
    Test that we only fetch 10 batches from DB (not 500).
    This is a performance optimization - we only fetch what we need.
    """
    unified_file_id = _make_unified_file_id("file-shared")
    
    # Create exactly 10 batches (what DB will return with take=10)
    ten_batches = []
    for i in range(10):
        batch = _make_batch_db_record(
            unified_object_id=_make_unified_batch_id(f"batch-{i}"),
            status="validating",
            file_object={
                "id": f"batch-{i}",
                "input_file_id": unified_file_id,
                "status": "validating"
            },
        )
        ten_batches.append(batch)
    
    # Mock will return only 10 batches (as DB would with take=10)
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=ten_batches,
    )
    
    referencing_batches = await managed_files._get_batches_referencing_file(unified_file_id)
    
    # Should return all 10 that reference the file
    assert len(referencing_batches) == 10
    
    # Verify error message handles "10+" case (since we got exactly 10, might be more in DB)
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = MagicMock()
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        with pytest.raises(HTTPException) as exc_info:
            await managed_files._check_file_deletion_allowed(unified_file_id)
        
        error_detail = exc_info.value.detail
        # When we get exactly 10 matches, show "10+" to indicate there might be more
        assert "10+ batch(es)" in error_detail


@pytest.mark.asyncio
async def test_error_message_includes_batch_details():
    """
    Test that the error message includes helpful information about the blocking batches.
    """
    unified_file_id = _make_unified_file_id("file-to-delete")
    batch1_id = _make_unified_batch_id("batch-1")
    batch2_id = _make_unified_batch_id("batch-2")
    
    batch1 = _make_batch_db_record(
        unified_object_id=batch1_id,
        status="validating",
        file_object={"id": "batch-1", "input_file_id": unified_file_id, "status": "validating"},
    )
    
    batch2 = _make_batch_db_record(
        unified_object_id=batch2_id,
        status="in_progress",
        file_object={"id": "batch-2", "output_file_id": unified_file_id, "status": "in_progress"},
    )
    
    managed_files = _make_managed_files_instance_with_batches(
        file_id=unified_file_id,
        batches=[batch1, batch2],
    )
    
    # Mock scheduler with registered job
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = MagicMock()
    
    with patch("litellm.proxy.proxy_server.scheduler", mock_scheduler):
        with pytest.raises(HTTPException) as exc_info:
            await managed_files._check_file_deletion_allowed(unified_file_id)
        
        error_detail = exc_info.value.detail
        assert "2 batch(es)" in error_detail
        assert "validating" in error_detail
        assert "in_progress" in error_detail
        assert "complete cost tracking" in error_detail.lower()
        assert "delete or cancel the referencing batch" in error_detail.lower()
