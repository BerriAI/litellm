# What this tests ?
## Tests /batches endpoints
import pytest
import asyncio
import aiohttp, openai
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Union
import os
import sys
import time
from unittest.mock import patch, MagicMock, AsyncMock

from litellm.proxy.batches_endpoints.endpoints import create_batch


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


@pytest.mark.parametrize("custom_llm_provider", ["azure", "openai"])
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


@pytest.mark.asyncio
async def test_create_batch_respects_custom_llm_provider_header():
    request_body = {
        "input_file_id": "file-batch-123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }

    response_object = MagicMock()
    response_object._hidden_params = {}
    response_object.id = "batch-id"
    response_object.input_file_id = request_body["input_file_id"]

    mock_user_api_key_dict = MagicMock(
        tpm_limit=0,
        rpm_limit=0,
        max_budget=0,
        spend=0,
        allowed_model_region="",
    )
    mock_user_api_key_dict.metadata = {}

    request = MagicMock()
    request.headers = {"custom-llm-provider": "vertex_ai"}
    request.query_params = {}
    request.method = "POST"

    fastapi_response = MagicMock()
    fastapi_response.headers = {}

    logging_obj = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
        return_value=response_object
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()
    mock_proxy_logging_obj.update_request_status = AsyncMock()

    with patch(
        "litellm.proxy.batches_endpoints.endpoints._read_request_body",
        new_callable=AsyncMock,
    ) as mock_read_body, patch(
        "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic",
        new_callable=AsyncMock,
    ) as mock_common_processing, patch(
        "litellm.proxy.batches_endpoints.endpoints.litellm.acreate_batch",
        new_callable=AsyncMock,
    ) as mock_acreate_batch, patch(
        "litellm.proxy.proxy_server.general_settings",
        new={},
    ), patch(
        "litellm.proxy.proxy_server.proxy_config",
        new={},
    ), patch(
        "litellm.proxy.proxy_server.version",
        new="test-version",
    ), patch(
        "litellm.proxy.proxy_server.llm_router",
        new=None,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        new=mock_proxy_logging_obj,
    ), patch(
        "litellm.proxy.batches_endpoints.endpoints.asyncio.create_task",
        new=lambda coro: asyncio.ensure_future(coro),
    ):
        mock_read_body.return_value = dict(request_body)
        mock_common_processing.return_value = (dict(request_body), logging_obj)
        mock_acreate_batch.return_value = response_object

        response = await create_batch(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
        )

    mock_acreate_batch.assert_awaited_once()
    called_kwargs = mock_acreate_batch.call_args.kwargs
    assert called_kwargs["custom_llm_provider"] == "vertex_ai"
    assert response is response_object


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