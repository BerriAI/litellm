"""
Test that batch retrieve endpoint resolves raw input_file_id to the
unified managed file ID before returning.

Bug: After batch completion, batches.retrieve returns the raw provider
input_file_id instead of the LiteLLM unified ID.
"""

import base64
import json

import pytest
from fastapi import Response
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
)


DECODED_UNIFIED_INPUT_FILE_ID = "litellm_proxy:application/octet-stream;unified_id,test-uuid;target_model_names,azure-gpt-4"
B64_UNIFIED_INPUT_FILE_ID = base64.urlsafe_b64encode(DECODED_UNIFIED_INPUT_FILE_ID.encode()).decode().rstrip("=")
RAW_INPUT_FILE_ID = "file-raw-provider-abc123"

DECODED_UNIFIED_BATCH_ID = "litellm_proxy;model_id:model-xyz;llm_batch_id:batch-123"
B64_UNIFIED_BATCH_ID = base64.urlsafe_b64encode(DECODED_UNIFIED_BATCH_ID.encode()).decode().rstrip("=")


@pytest.mark.asyncio
async def test_should_resolve_raw_input_file_id_to_unified():
    """
    When a completed batch has a raw input_file_id and the managed file table
    contains a record for that raw ID, the retrieve endpoint should resolve
    it to the unified file ID.
    """
    unified_batch_id = _is_base64_encoded_unified_file_id(B64_UNIFIED_BATCH_ID)
    assert unified_batch_id, "Test setup: batch_id should decode as unified"

    from litellm.types.utils import LiteLLMBatch

    batch_data = {
        "id": B64_UNIFIED_BATCH_ID,
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": RAW_INPUT_FILE_ID,
        "object": "batch",
        "status": "completed",
        "output_file_id": "file-output-xyz",
    }

    mock_db_object = MagicMock()
    mock_db_object.file_object = json.dumps(batch_data)

    mock_managed_file = MagicMock()
    mock_managed_file.unified_file_id = B64_UNIFIED_INPUT_FILE_ID

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedobjecttable.find_first = AsyncMock(return_value=mock_db_object)
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(return_value=mock_managed_file)

    from litellm.proxy.openai_files_endpoints.common_utils import get_batch_from_database

    _, response = await get_batch_from_database(
        batch_id=B64_UNIFIED_BATCH_ID,
        unified_batch_id=unified_batch_id,
        managed_files_obj=MagicMock(),
        prisma_client=mock_prisma,
        verbose_proxy_logger=MagicMock(),
    )

    assert response is not None, "Batch should be found in DB"
    assert response.input_file_id == B64_UNIFIED_INPUT_FILE_ID, (
        f"input_file_id should be unified '{B64_UNIFIED_INPUT_FILE_ID}', "
        f"got raw '{response.input_file_id}'"
    )


@pytest.mark.asyncio
async def test_should_restore_model_id_for_terminal_db_batch_before_post_hook():
    """
    When retrieve_batch returns a terminal batch directly from DB, it should
    restore hidden_params.model_id from the unified batch ID before running
    post_call_success_hook.
    """
    from litellm.proxy.batches_endpoints.endpoints import retrieve_batch
    from litellm.types.utils import LiteLLMBatch

    terminal_batch = LiteLLMBatch(
        id=B64_UNIFIED_BATCH_ID,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id=B64_UNIFIED_INPUT_FILE_ID,
        object="batch",
        status="completed",
        output_file_id="file-raw-provider-output-123",
    )
    terminal_batch._hidden_params = {}

    async def _post_call_success_hook(data, user_api_key_dict, response):
        assert response._hidden_params.get("model_id") == "model-xyz"
        response.output_file_id = "b64-unified-output-file-id"
        return response

    mock_proxy_logging_obj = MagicMock(
        post_call_success_hook=AsyncMock(side_effect=_post_call_success_hook),
        update_request_status=AsyncMock(),
    )
    mock_user_api_key_dict = MagicMock()
    mock_user_api_key_dict.parent_otel_span = None
    mock_user_api_key_dict.allowed_model_region = ""

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_request.query_params = {}
    mock_request.url = MagicMock()
    mock_request.url.port = 4000
    mock_request.method = "GET"
    mock_request.url.path = f"/v1/batches/{B64_UNIFIED_BATCH_ID}"

    with (
        patch(
            "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic",
            new=AsyncMock(return_value=({"batch_id": B64_UNIFIED_BATCH_ID}, MagicMock())),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
            return_value={},
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.get_batch_from_database",
            new=AsyncMock(return_value=(MagicMock(), terminal_batch)),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.resolve_input_file_id_to_unified",
            new=AsyncMock(),
        ),
        patch(
            "litellm.proxy.batches_endpoints.endpoints.resolve_output_file_ids_to_unified",
            new=AsyncMock(),
        ),
        patch("litellm.proxy.proxy_server.general_settings", {}),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_config", MagicMock()),
        patch("litellm.proxy.proxy_server.version", "1.0.0"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj),
    ):
        response = await retrieve_batch(
            request=mock_request,
            fastapi_response=Response(),
            user_api_key_dict=mock_user_api_key_dict,
            provider=None,
            batch_id=B64_UNIFIED_BATCH_ID,
        )

    assert response.output_file_id == "b64-unified-output-file-id"
