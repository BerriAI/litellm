"""
Regression test: deleted managed files should return 404, not 403.

When a managed file's DB record has been deleted, can_user_call_unified_file_id()
raises HTTPException(404) directly — rather than returning True (which would
weaken access control) or False (which would cause a misleading 403).
"""

import base64

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth


def _make_user_api_key_dict(user_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=user_id,
        parent_otel_span=None,
    )


def _make_unified_file_id() -> str:
    raw = "litellm_proxy:application/octet-stream;unified_id,test-deleted-file;target_model_names,azure-gpt-4"
    return base64.b64encode(raw.encode()).decode()


def _make_managed_files_with_no_db_record():
    """Create a _PROXY_LiteLLMManagedFiles where the DB returns None (file was deleted)."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)

    return _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=mock_prisma,
    )


@pytest.mark.asyncio
async def test_should_raise_404_for_deleted_file():
    """
    When a managed file record has been deleted from the DB,
    check_managed_file_id_access should raise 404 (not 403).
    """
    unified_file_id = _make_unified_file_id()
    managed_files = _make_managed_files_with_no_db_record()
    user = _make_user_api_key_dict("any-user")
    data = {"file_id": unified_file_id}

    with pytest.raises(HTTPException) as exc_info:
        await managed_files.check_managed_file_id_access(data, user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_should_allow_owner_access_when_record_exists():
    """Baseline: file owner can access their own file."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    unified_file_id = _make_unified_file_id()

    mock_db_record = MagicMock()
    mock_db_record.created_by = "user-A"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=mock_db_record
    )

    managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=mock_prisma,
    )

    user = _make_user_api_key_dict("user-A")
    data = {"file_id": unified_file_id}

    result = await managed_files.check_managed_file_id_access(data, user)
    assert result is True


@pytest.mark.asyncio
async def test_should_block_different_user_when_record_exists():
    """Baseline: different user cannot access another user's file."""
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    unified_file_id = _make_unified_file_id()

    mock_db_record = MagicMock()
    mock_db_record.created_by = "user-A"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=mock_db_record
    )

    managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=mock_prisma,
    )

    user = _make_user_api_key_dict("user-B")
    data = {"file_id": unified_file_id}

    with pytest.raises(HTTPException) as exc_info:
        await managed_files.check_managed_file_id_access(data, user)
    assert exc_info.value.status_code == 403
