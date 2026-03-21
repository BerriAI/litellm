"""
Test that get_batch_from_database resolves raw input_file_id to the
unified/managed file ID when reading a batch from the database.

Bug: The batch retrieve path stores the raw provider input_file_id in the
DB (via async_post_call_success_hook on the retrieve endpoint). When the
batch is later read from DB, get_batch_from_database returns the raw ID
without resolving it to the unified ID.
"""

import json
import pytest
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy.openai_files_endpoints.common_utils import get_batch_from_database


def _mock_prisma(batch_json: str, managed_file_record=None):
    """Create a mock prisma client with canned responses."""
    prisma = MagicMock()

    batch_db_record = MagicMock()
    batch_db_record.file_object = batch_json

    prisma.db.litellm_managedobjecttable.find_first = AsyncMock(
        return_value=batch_db_record
    )

    prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=managed_file_record
    )

    return prisma


@pytest.mark.asyncio
async def test_should_resolve_raw_input_file_id_to_unified_id():
    """
    When input_file_id in the stored batch is a raw provider ID,
    get_batch_from_database must look up the unified ID from the
    managed files table.
    """
    unified_batch_id = "bGl0ZWxsbV9wcm94eTpiYXRjaF9pZA"
    unified_input_file_id = "bGl0ZWxsbV9wcm94eTp1bmlmaWVkX2lucHV0"
    raw_input_file_id = "file-abc123-raw"

    batch_data = {
        "id": "batch-raw-123",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": raw_input_file_id,
        "object": "batch",
        "status": "completed",
        "output_file_id": "file-output-raw",
    }

    managed_file_record = MagicMock()
    managed_file_record.unified_file_id = unified_input_file_id

    prisma = _mock_prisma(
        batch_json=json.dumps(batch_data),
        managed_file_record=managed_file_record,
    )

    _, response = await get_batch_from_database(
        batch_id=unified_batch_id,
        unified_batch_id="decoded_unified_batch_id",
        managed_files_obj=MagicMock(),
        prisma_client=prisma,
        verbose_proxy_logger=MagicMock(),
    )

    assert response is not None
    assert response.input_file_id == unified_input_file_id, (
        f"input_file_id should be resolved to '{unified_input_file_id}', "
        f"got raw: '{response.input_file_id}'"
    )

    prisma.db.litellm_managedfiletable.find_first.assert_called_once_with(
        where={"flat_model_file_ids": {"has": raw_input_file_id}}
    )


@pytest.mark.asyncio
async def test_should_preserve_already_managed_input_file_id():
    """
    When input_file_id is already a managed/unified ID, it should
    not be modified.
    """
    import base64

    unified_batch_id = "bGl0ZWxsbV9wcm94eTpiYXRjaF9pZA"
    decoded_unified = "litellm_proxy:application/octet-stream;unified_id,test-123"
    base64_input_file_id = base64.urlsafe_b64encode(decoded_unified.encode()).decode().rstrip("=")

    batch_data = {
        "id": "batch-raw-123",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": base64_input_file_id,
        "object": "batch",
        "status": "completed",
    }

    prisma = _mock_prisma(batch_json=json.dumps(batch_data))

    _, response = await get_batch_from_database(
        batch_id=unified_batch_id,
        unified_batch_id="decoded_unified_batch_id",
        managed_files_obj=MagicMock(),
        prisma_client=prisma,
        verbose_proxy_logger=MagicMock(),
    )

    assert response is not None
    assert response.input_file_id == base64_input_file_id, (
        f"input_file_id was already managed, should be preserved as '{base64_input_file_id}', "
        f"got: '{response.input_file_id}'"
    )

    prisma.db.litellm_managedfiletable.find_first.assert_not_called()
