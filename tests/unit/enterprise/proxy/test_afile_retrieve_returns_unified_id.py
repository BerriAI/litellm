"""
Test that managed_files.afile_retrieve returns the unified file ID, not the
raw provider file ID, when file_object is already stored in the database.

Bug: managed_files.py Case 2 returns stored_file_object.file_object directly
without replacing .id with the unified ID. Case 3 (fetch from provider) does
it correctly at line 1028.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import LiteLLM_ManagedFileTable
from litellm.types.llms.openai import OpenAIFileObject


def _make_managed_files_instance():
    from litellm_enterprise.proxy.hooks.managed_files import (
        _PROXY_LiteLLMManagedFiles,
    )

    instance = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=MagicMock(),
        prisma_client=MagicMock(),
    )
    return instance


@pytest.mark.asyncio
async def test_should_return_unified_id_when_file_object_exists_in_db():
    """
    When get_unified_file_id returns a stored file_object (Case 2),
    afile_retrieve must set .id to the unified file ID before returning.
    """
    unified_id = "bGl0ZWxsbV9wcm94eTp1bmlmaWVkX291dHB1dF9maWxl"
    raw_provider_id = "batch_20260214-output-file-1"

    stored = LiteLLM_ManagedFileTable(
        unified_file_id=unified_id,
        file_object=OpenAIFileObject(
            id=raw_provider_id,
            bytes=489,
            created_at=1700000000,
            filename="batch_output.jsonl",
            object="file",
            purpose="batch_output",
            status="processed",
        ),
        model_mappings={"model-abc": raw_provider_id},
        flat_model_file_ids=[raw_provider_id],
        created_by="test-user",
        updated_by="test-user",
    )

    managed_files = _make_managed_files_instance()
    managed_files.get_unified_file_id = AsyncMock(return_value=stored)

    result = await managed_files.afile_retrieve(
        file_id=unified_id,
        litellm_parent_otel_span=None,
        llm_router=None,
    )

    assert result.id == unified_id, (
        f"afile_retrieve should return the unified ID '{unified_id}', "
        f"but got raw provider ID '{result.id}'"
    )
