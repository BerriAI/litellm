"""
Test that update_batch_in_database is called AFTER post_call_success_hook,
ensuring the database stores encoded output_file_id (not raw provider IDs).

Bug: update_batch_in_database was called before post_call_success_hook, so the
database stored the raw output_file_id. On subsequent polls, the terminal-state
path read the raw ID from the DB and returned it without re-encoding, because
_hidden_params had no model_id set.

Fix: Move update_batch_in_database to after post_call_success_hook so the DB
always stores already-encoded IDs.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
)


def _make_unified_batch_id(model_id: str, raw_batch_id: str) -> str:
    s = f"litellm_proxy;model_id:{model_id};llm_batch_id:{raw_batch_id}"
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def _make_unified_output_file_id(model_id: str, raw_file_id: str) -> str:
    s = (
        f"litellm_proxy:application/json;unified_id,test-uuid;"
        f"target_model_names,test-model;llm_output_file_id,{raw_file_id};"
        f"llm_output_file_model_id,{model_id};"
    )
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


MODEL_ID = "model-xyz"
RAW_BATCH_ID = "batch-raw-abc"
RAW_OUTPUT_FILE_ID = "file-batch-output-e2d18352e87b"
UNIFIED_BATCH_ID = _make_unified_batch_id(MODEL_ID, RAW_BATCH_ID)
UNIFIED_OUTPUT_FILE_ID = _make_unified_output_file_id(MODEL_ID, RAW_OUTPUT_FILE_ID)


@pytest.mark.asyncio
async def test_update_batch_in_database_stores_encoded_output_file_id():
    """
    update_batch_in_database should receive the post-hook encoded response so
    the DB never stores a raw output_file_id. If the DB stores a raw ID,
    the next terminal-state read will return it un-encoded to the client.
    """
    from litellm.proxy.openai_files_endpoints.common_utils import update_batch_in_database
    from litellm.types.utils import LiteLLMBatch

    unified_batch_id = _is_base64_encoded_unified_file_id(UNIFIED_BATCH_ID)
    assert unified_batch_id, "Test setup: UNIFIED_BATCH_ID must decode as a unified ID"

    encoded_response = LiteLLMBatch(
        id=UNIFIED_BATCH_ID,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id="file-input-raw",
        object="batch",
        status="completed",
        output_file_id=UNIFIED_OUTPUT_FILE_ID,
    )
    encoded_response._hidden_params = {}

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedobjecttable.update = AsyncMock()

    mock_db_object = MagicMock()
    mock_db_object.status = "in_progress"

    await update_batch_in_database(
        batch_id=UNIFIED_BATCH_ID,
        unified_batch_id=unified_batch_id,
        response=encoded_response,
        managed_files_obj=MagicMock(),
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
        db_batch_object=mock_db_object,
        operation="retrieve",
    )

    mock_prisma.db.litellm_managedobjecttable.update.assert_called_once()
    stored_json = mock_prisma.db.litellm_managedobjecttable.update.call_args.kwargs["data"]["file_object"]
    stored = json.loads(stored_json)

    assert stored["output_file_id"] == UNIFIED_OUTPUT_FILE_ID, (
        f"DB should store encoded output_file_id '{UNIFIED_OUTPUT_FILE_ID}', "
        f"got raw: '{stored['output_file_id']}'"
    )
    assert _is_base64_encoded_unified_file_id(stored["output_file_id"]), (
        f"Stored output_file_id must be a managed ID, got: '{stored['output_file_id']}'"
    )


@pytest.mark.asyncio
async def test_terminal_state_db_read_returns_encoded_output_file_id():
    """
    When the DB stores an encoded output_file_id (post-fix behavior),
    get_batch_from_database returns it correctly so the terminal-state path
    can return it to the client without needing a re-encoding step.
    """
    from litellm.proxy.openai_files_endpoints.common_utils import get_batch_from_database

    unified_batch_id = _is_base64_encoded_unified_file_id(UNIFIED_BATCH_ID)

    batch_data = {
        "id": UNIFIED_BATCH_ID,
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-input-raw",
        "object": "batch",
        "status": "completed",
        "output_file_id": UNIFIED_OUTPUT_FILE_ID,
    }

    mock_db_object = MagicMock()
    mock_db_object.file_object = json.dumps(batch_data)

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=mock_db_object)
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)

    _, response = await get_batch_from_database(
        batch_id=UNIFIED_BATCH_ID,
        unified_batch_id=unified_batch_id,
        managed_files_obj=MagicMock(),
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
    )

    assert response is not None
    assert response.output_file_id == UNIFIED_OUTPUT_FILE_ID, (
        f"Terminal-state DB read should return encoded output_file_id "
        f"'{UNIFIED_OUTPUT_FILE_ID}', got: '{response.output_file_id}'"
    )
    assert _is_base64_encoded_unified_file_id(response.output_file_id), (
        f"output_file_id from DB must be a managed ID, got: '{response.output_file_id}'"
    )
