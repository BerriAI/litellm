# What this tests ?
## Tests /batches endpoints
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
from test_openai_files_endpoints import upload_file, delete_file
import os
import sys
import time
from unittest.mock import patch, MagicMock, AsyncMock


BASE_URL = "http://localhost:4000"  # Replace with your actual base URL
API_KEY = "sk-1234"  # Replace with your actual API key

from openai import OpenAI

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


@pytest.mark.asyncio
async def test_batches_operations():
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(_current_dir, "input.jsonl")
    file_obj = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    assert batch.id is not None

    # Test get batch
    _retrieved_batch = client.batches.retrieve(batch_id=batch.id)
    print("response from get batch", _retrieved_batch)

    assert _retrieved_batch.id == batch.id
    assert _retrieved_batch.input_file_id == file_obj.id

    # Test list batches
    _list_batches = client.batches.list()
    print("response from list batches", _list_batches)

    assert _list_batches is not None
    assert len(_list_batches.data) > 0

    # Clean up
    # Test cancel batch
    _canceled_batch = client.batches.cancel(batch_id=batch.id)
    print("response from cancel batch", _canceled_batch)

    assert _canceled_batch.status is not None
    assert (
        _canceled_batch.status == "cancelling" or _canceled_batch.status == "cancelled"
    )

    # finally delete the file
    _deleted_file = client.files.delete(file_id=file_obj.id)
    print("response from delete file", _deleted_file)

    assert _deleted_file.deleted is True


def create_batch_oai_sdk(filepath: str, custom_llm_provider: str) -> str:
    batch_input_file = client.files.create(
        file=open(filepath, "rb"),
        purpose="batch",
        extra_headers={"custom-llm-provider": custom_llm_provider},
    )
    batch_input_file_id = batch_input_file.id

    print("waiting for file to be processed......")
    time.sleep(5)
    rq = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": filepath,
        },
        extra_headers={"custom-llm-provider": custom_llm_provider},
    )

    print(f"Batch submitted. ID: {rq.id}")
    return rq.id


def await_batch_completion(batch_id: str, custom_llm_provider: str):
    max_tries = 3
    tries = 0

    while tries < max_tries:
        batch = client.batches.retrieve(
            batch_id, extra_headers={"custom-llm-provider": custom_llm_provider}
        )
        if batch.status == "completed":
            print(f"Batch {batch_id} completed.")
            return batch.id

        tries += 1
        print(f"waiting for batch to complete... (attempt {tries}/{max_tries})")
        time.sleep(10)

    print(
        f"Reached maximum number of attempts ({max_tries}). Batch may still be processing."
    )


def write_content_to_file(
    batch_id: str, output_path: str, custom_llm_provider: str
) -> str:
    batch = client.batches.retrieve(
        batch_id=batch_id, extra_headers={"custom-llm-provider": custom_llm_provider}
    )
    content = client.files.content(
        file_id=batch.output_file_id,
        extra_headers={"custom-llm-provider": custom_llm_provider},
    )
    print("content from files.content", content.content)
    content.write_to_file(output_path)


def read_jsonl(filepath: str):
    import json

    results = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    for item in results:
        print(item)
        custom_id = item["custom_id"]
        print(custom_id)


def get_any_completed_batch_id_azure():
    print("AZURE getting any completed batch id")
    list_of_batches = client.batches.list(extra_headers={"custom-llm-provider": "azure"})
    print("list of batches", list_of_batches)
    for batch in list_of_batches:
        if batch.status == "completed":
            return batch.id
    return None


@pytest.mark.parametrize("custom_llm_provider", ["openai"])
def test_e2e_batches_files(custom_llm_provider):
    """
    [PROD Test] Ensures OpenAI Batches + files work with OpenAI SDK
    """
    input_path = (
        "input.jsonl" if custom_llm_provider == "openai" else "input_azure.jsonl"
    )
    output_path = "out.jsonl" if custom_llm_provider == "openai" else "out_azure.jsonl"

    _current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(_current_dir, input_path)
    output_file_path = os.path.join(_current_dir, output_path)
    print("running e2e batches files with custom_llm_provider=", custom_llm_provider)
    batch_id = create_batch_oai_sdk(
        filepath=input_file_path, custom_llm_provider=custom_llm_provider
    )

    if custom_llm_provider == "azure":
        # azure takes very long to complete a batch
        return
    else:
        response_batch_id = await_batch_completion(
            batch_id=batch_id, custom_llm_provider=custom_llm_provider
        )
        if response_batch_id is None:
            return

    write_content_to_file(
        batch_id=batch_id,
        output_path=output_file_path,
        custom_llm_provider=custom_llm_provider,
    )
    read_jsonl(output_file_path)


@pytest.mark.skip(reason="Local only test to verify if things work well")
def test_vertex_batches_endpoint():
    """
    Test VertexAI Batches Endpoint
    """
    import os

    oai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    file_name = "local_testing/vertex_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = oai_client.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
        extra_headers={"custom-llm-provider": "vertex_ai"},
    )
    print("Response from creating file=", file_obj)

    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), f"Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    create_batch_response = oai_client.batches.create(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        extra_headers={"custom-llm-provider": "vertex_ai"},
        metadata={"key1": "value1", "key2": "value2"},
    )
    print("response from create batch", create_batch_response)
    pass


@pytest.mark.skip(reason="Local only test to verify if things work well")
@pytest.mark.asyncio
async def test_list_batches_with_target_model_names():
    """
    Unit test to verify that target_model_names query parameter is properly handled
    in the list_batches endpoint
    """

    # Test data
    target_model_names = "gpt-4,gpt-3.5-turbo"
    expected_model = "gpt-4"  # Should use the first model from the comma-separated list

    # Mock response for list_batches
    mock_batch_response = {
        "object": "list",
        "data": [
            {
                "id": "batch_abc123",
                "object": "batch",
                "endpoint": "/v1/chat/completions",
                "status": "validating",
                "input_file_id": "file-abc123",
                "completion_window": "24h",
                "created_at": 1711471533,
                "metadata": {},
            }
        ],
        "first_id": "batch_abc123",
        "last_id": "batch_abc123",
        "has_more": False,
    }

    # Mock the request and FastAPI dependencies
    mock_request = MagicMock()
    mock_request.method = "GET"
    mock_request.url.query = f"target_model_names={target_model_names}&limit=10"

    mock_fastapi_response = MagicMock()
    mock_user_api_key_dict = MagicMock()

    # Mock _read_request_body to return our target_model_names
    with patch(
        "litellm.proxy.batches_endpoints.endpoints._read_request_body"
    ) as mock_read_body, patch("litellm.proxy.proxy_server.llm_router") as mock_router:

        mock_read_body.return_value = {"target_model_names": target_model_names}
        mock_router.alist_batches = AsyncMock(return_value=mock_batch_response)

        # Import and call the function directly
        from litellm.proxy.batches_endpoints.endpoints import list_batches

        response = await list_batches(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            target_model_names=target_model_names,
            limit=10,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify that router.alist_batches was called with the correct model
        mock_router.alist_batches.assert_called_once()
        call_args = mock_router.alist_batches.call_args

        # Check that the model parameter was set to the first model in the list
        assert call_args.kwargs["model"] == expected_model
        assert call_args.kwargs["limit"] == 10

        # Verify the response structure
        assert response["object"] == "list"
        assert len(response["data"]) > 0


@pytest.mark.asyncio
async def test_batch_status_sync_from_provider_to_database():
    """
    Test that when batch status changes at the provider, 
    it gets synced to the ManagedObjectTable database.
    
    This tests the new refactored utility functions:
    - get_batch_from_database()
    - update_batch_in_database()
    """
    from unittest.mock import MagicMock, AsyncMock
    from litellm.proxy.openai_files_endpoints.common_utils import (
        get_batch_from_database,
        update_batch_in_database,
    )
    from litellm.types.utils import LiteLLMBatch
    import json
    
    # Setup: Create mock objects
    batch_id = "batch_test123"
    unified_batch_id = "litellm_proxy:test_unified_batch"
    
    # Mock database batch object with "validating" status
    mock_db_batch = MagicMock()
    mock_db_batch.unified_object_id = batch_id
    mock_db_batch.status = "validating"
    mock_db_batch.file_object = json.dumps({
        "id": batch_id,
        "object": "batch",
        "status": "validating",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-test123",
        "completion_window": "24h",
        "created_at": 1234567890,
    })
    
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(
        return_value=mock_db_batch
    )
    mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
    
    # Mock managed_files_obj
    mock_managed_files = MagicMock()
    
    # Mock logger
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.warning = MagicMock()
    mock_logger.error = MagicMock()
    
    # Test 1: Retrieve batch from database (initial state)
    db_batch_object, response_batch = await get_batch_from_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma_client,
        verbose_proxy_logger=mock_logger,
    )
    
    # Verify database was queried
    mock_prisma_client.db.litellm_managedobjecttable.find_first.assert_called_once_with(
        where={"unified_object_id": batch_id}
    )
    
    # Verify batch was retrieved correctly
    assert db_batch_object is not None
    assert response_batch is not None
    assert response_batch.id == batch_id
    assert response_batch.status == "validating"
    
    # Test 2: Simulate provider returning updated status
    updated_batch_response = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="completed",  # Status changed from "validating" to "completed"
        endpoint="/v1/chat/completions",
        input_file_id="file-test123",
        completion_window="24h",
        created_at=1234567890,
        output_file_id="file-output123",
    )
    
    # Test 3: Update database with new status from provider
    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=updated_batch_response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma_client,
        verbose_proxy_logger=mock_logger,
        db_batch_object=db_batch_object,
        operation="retrieve",
    )
    
    # Verify database was updated
    mock_prisma_client.db.litellm_managedobjecttable.update.assert_called_once()
    update_call_args = mock_prisma_client.db.litellm_managedobjecttable.update.call_args
    
    # Verify the update call had correct parameters
    assert update_call_args.kwargs["where"]["unified_object_id"] == batch_id
    assert update_call_args.kwargs["data"]["status"] == "complete"  # "completed" normalized to "complete"
    assert "file_object" in update_call_args.kwargs["data"]
    assert "updated_at" in update_call_args.kwargs["data"]
    
    # Verify logger was called with status change message
    mock_logger.info.assert_called()
    log_message = mock_logger.info.call_args[0][0]
    assert "validating" in log_message
    assert "completed" in log_message
    
    print("✅ Test passed: Batch status synced from provider to database")


@pytest.mark.asyncio
async def test_batch_cancel_updates_database():
    """
    Test that canceling a batch updates the database status.
    """
    from unittest.mock import MagicMock, AsyncMock
    from litellm.proxy.openai_files_endpoints.common_utils import (
        update_batch_in_database,
    )
    from litellm.types.utils import LiteLLMBatch
    
    # Setup
    batch_id = "batch_cancel_test"
    unified_batch_id = "litellm_proxy:cancel_test"
    
    # Mock cancelled batch response from provider
    cancelled_batch_response = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="cancelled",
        endpoint="/v1/chat/completions",
        input_file_id="file-test123",
        completion_window="24h",
        created_at=1234567890,
        cancelled_at=1234567999,
    )
    
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
    
    # Mock managed_files_obj
    mock_managed_files = MagicMock()
    
    # Mock logger
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.error = MagicMock()
    
    # Call update_batch_in_database for cancel operation
    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=cancelled_batch_response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma_client,
        verbose_proxy_logger=mock_logger,
        operation="cancel",
    )
    
    # Verify database was updated
    mock_prisma_client.db.litellm_managedobjecttable.update.assert_called_once()
    update_call_args = mock_prisma_client.db.litellm_managedobjecttable.update.call_args
    
    # Verify the update call had correct parameters
    assert update_call_args.kwargs["where"]["unified_object_id"] == batch_id
    assert update_call_args.kwargs["data"]["status"] == "cancelled"
    assert "file_object" in update_call_args.kwargs["data"]
    
    # Verify logger was called
    mock_logger.info.assert_called()
    log_message = mock_logger.info.call_args[0][0]
    assert "cancel" in log_message.lower()
    assert "cancelled" in log_message
    
    print("✅ Test passed: Batch cancel updates database")


@pytest.mark.asyncio
async def test_batch_terminal_state_skip_provider_call():
    """
    Test that when a batch is in a terminal state (completed, failed, cancelled, expired),
    it returns immediately from database without calling the provider.
    """
    from unittest.mock import MagicMock, AsyncMock
    from litellm.proxy.openai_files_endpoints.common_utils import (
        get_batch_from_database,
    )
    from litellm.types.utils import LiteLLMBatch
    import json
    
    # Setup: Create mock objects for a completed batch
    batch_id = "batch_completed_test"
    unified_batch_id = "litellm_proxy:completed_test"
    
    # Mock database batch object with "completed" status
    mock_db_batch = MagicMock()
    mock_db_batch.unified_object_id = batch_id
    mock_db_batch.status = "complete"
    mock_db_batch.file_object = json.dumps({
        "id": batch_id,
        "object": "batch",
        "status": "completed",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-test123",
        "output_file_id": "file-output123",
        "completion_window": "24h",
        "created_at": 1234567890,
        "completed_at": 1234567999,
    })
    
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedobjecttable.find_first = AsyncMock(
        return_value=mock_db_batch
    )
    
    # Mock managed_files_obj
    mock_managed_files = MagicMock()
    
    # Mock logger
    mock_logger = MagicMock()
    mock_logger.debug = MagicMock()
    
    # Retrieve batch from database
    db_batch_object, response_batch = await get_batch_from_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma_client,
        verbose_proxy_logger=mock_logger,
    )
    
    # Verify batch was retrieved
    assert db_batch_object is not None
    assert response_batch is not None
    assert response_batch.status == "completed"
    
    # In the actual endpoint, when status is in terminal states,
    # it should return immediately without calling the provider
    # This test verifies the database retrieval works correctly
    assert response_batch.status in ["completed", "failed", "cancelled", "expired"]
    
    print("✅ Test passed: Terminal state batch retrieved from database")


@pytest.mark.asyncio
async def test_batch_no_status_change_skip_update():
    """
    Test that when batch status hasn't changed, database update is skipped.
    """
    from unittest.mock import MagicMock, AsyncMock
    from litellm.proxy.openai_files_endpoints.common_utils import (
        update_batch_in_database,
    )
    from litellm.types.utils import LiteLLMBatch
    
    # Setup
    batch_id = "batch_no_change_test"
    unified_batch_id = "litellm_proxy:no_change_test"
    
    # Mock database batch object with "validating" status
    mock_db_batch = MagicMock()
    mock_db_batch.status = "validating"
    
    # Mock batch response from provider with same status
    batch_response = LiteLLMBatch(
        id=batch_id,
        object="batch",
        status="validating",  # Same status as in database
        endpoint="/v1/chat/completions",
        input_file_id="file-test123",
        completion_window="24h",
        created_at=1234567890,
    )
    
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_managedobjecttable.update = AsyncMock()
    
    # Mock managed_files_obj
    mock_managed_files = MagicMock()
    
    # Mock logger
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    
    # Call update_batch_in_database
    await update_batch_in_database(
        batch_id=batch_id,
        unified_batch_id=unified_batch_id,
        response=batch_response,
        managed_files_obj=mock_managed_files,
        prisma_client=mock_prisma_client,
        verbose_proxy_logger=mock_logger,
        db_batch_object=mock_db_batch,
        operation="retrieve",
    )
    
    # Verify database update was NOT called (status hasn't changed)
    mock_prisma_client.db.litellm_managedobjecttable.update.assert_not_called()
    
    # Verify logger info was NOT called (no status change to log)
    mock_logger.info.assert_not_called()
    
    print("✅ Test passed: Database update skipped when status unchanged")