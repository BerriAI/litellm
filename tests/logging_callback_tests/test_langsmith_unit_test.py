import io
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))

import asyncio
import gzip
import json
import logging
import time
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from datetime import datetime, timezone
from litellm.integrations.langsmith import (
    LangsmithLogger,
    LangsmithQueueObject,
    CredentialsKey,
    BatchGroup,
)

import litellm


# Test get_credentials_from_env
@pytest.mark.asyncio
async def test_get_credentials_from_env():
    # Test with direct parameters
    logger = LangsmithLogger(
        langsmith_api_key="test-key",
        langsmith_project="test-project",
        langsmith_base_url="http://test-url",
    )

    credentials = logger.get_credentials_from_env(
        langsmith_api_key="custom-key",
        langsmith_project="custom-project",
        langsmith_base_url="http://custom-url",
    )

    assert credentials["LANGSMITH_API_KEY"] == "custom-key"
    assert credentials["LANGSMITH_PROJECT"] == "custom-project"
    assert credentials["LANGSMITH_BASE_URL"] == "http://custom-url"

    # assert that the default api base is used if not provided
    credentials = logger.get_credentials_from_env()
    assert credentials["LANGSMITH_BASE_URL"] == "https://api.smith.langchain.com"

    # Test with tenant_id
    credentials = logger.get_credentials_from_env(
        langsmith_tenant_id="test-tenant-id"
    )
    assert credentials["LANGSMITH_TENANT_ID"] == "test-tenant-id"

    # Test tenant_id from environment variable
    import os
    os.environ["LANGSMITH_TENANT_ID"] = "env-tenant-id"
    credentials = logger.get_credentials_from_env()
    assert credentials["LANGSMITH_TENANT_ID"] == "env-tenant-id"
    del os.environ["LANGSMITH_TENANT_ID"]


@pytest.mark.asyncio
async def test_group_batches_by_credentials():

    logger = LangsmithLogger(langsmith_api_key="test-key")

    # Create test queue objects
    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2]

    grouped = logger._group_batches_by_credentials()

    # Check grouping
    assert len(grouped) == 1  # Should have one group since credentials are same
    key = list(grouped.keys())[0]
    assert isinstance(key, CredentialsKey)
    assert len(grouped[key].queue_objects) == 2


@pytest.mark.asyncio
async def test_group_batches_by_credentials_multiple_credentials():

    # Test with multiple different credentials
    logger = LangsmithLogger(langsmith_api_key="test-key")

    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key2",  # Different API key
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    queue_obj3 = LangsmithQueueObject(
        data={"test": "data3"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj2",  # Different project
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": None,
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2, queue_obj3]

    grouped = logger._group_batches_by_credentials()

    # Check grouping
    assert len(grouped) == 3  # Should have three groups since credentials differ
    for key, batch_group in grouped.items():
        assert isinstance(key, CredentialsKey)
        assert len(batch_group.queue_objects) == 1  # Each group should have one object


@pytest.mark.asyncio
async def test_group_batches_by_credentials_with_tenant_id():

    # Test that different tenant_ids create separate groups
    logger = LangsmithLogger(langsmith_api_key="test-key")

    queue_obj1 = LangsmithQueueObject(
        data={"test": "data1"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant1",
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant2",  # Different tenant_id
        },
    )

    queue_obj3 = LangsmithQueueObject(
        data={"test": "data3"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
            "LANGSMITH_TENANT_ID": "tenant1",  # Same as queue_obj1
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2, queue_obj3]

    grouped = logger._group_batches_by_credentials()

    # Should have two groups: one for tenant1 (queue_obj1 and queue_obj3), one for tenant2 (queue_obj2)
    assert len(grouped) == 2
    for key, batch_group in grouped.items():
        assert isinstance(key, CredentialsKey)
        assert key.tenant_id in ["tenant1", "tenant2"]
        if key.tenant_id == "tenant1":
            assert len(batch_group.queue_objects) == 2
        else:
            assert len(batch_group.queue_objects) == 1


# Test make_dot_order
@pytest.mark.asyncio
async def test_make_dot_order():
    logger = LangsmithLogger(langsmith_api_key="test-key")
    run_id = "729cff0e-f30c-4336-8b79-45d6b61c64b4"
    dot_order = logger.make_dot_order(run_id)

    print("dot_order=", dot_order)

    # Check format: YYYYMMDDTHHMMSSfffZ + run_id
    # Check the timestamp portion (first 23 characters)
    timestamp_part = dot_order[:-36]  # 36 is length of run_id
    assert len(timestamp_part) == 22
    assert timestamp_part[8] == "T"  # Check T separator
    assert timestamp_part[-1] == "Z"  # Check Z suffix

    # Verify timestamp format
    try:
        # Parse the timestamp portion (removing the Z)
        datetime.strptime(timestamp_part[:-1], "%Y%m%dT%H%M%S%f")
    except ValueError:
        pytest.fail("Timestamp portion is not in correct format")

    # Verify run_id portion
    assert dot_order[-36:] == run_id


# Test is_serializable
@pytest.mark.asyncio
async def test_is_serializable():
    from litellm.integrations.langsmith import is_serializable
    from pydantic import BaseModel

    # Test basic types
    assert is_serializable("string") is True
    assert is_serializable(123) is True
    assert is_serializable({"key": "value"}) is True

    # Test non-serializable types
    async def async_func():
        pass

    assert is_serializable(async_func) is False

    class TestModel(BaseModel):
        field: str

    assert is_serializable(TestModel(field="test")) is False


@pytest.mark.asyncio
async def test_async_send_batch():
    logger = LangsmithLogger(langsmith_api_key="test-key")

    # Mock the httpx client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    logger.async_httpx_client = AsyncMock()
    logger.async_httpx_client.post.return_value = mock_response

    # Add test data to queue
    logger.log_queue = [
        LangsmithQueueObject(
            data={"test": "data"}, credentials=logger.default_credentials
        )
    ]

    await logger.async_send_batch()

    # Verify the API call
    logger.async_httpx_client.post.assert_called_once()
    call_args = logger.async_httpx_client.post.call_args
    assert "runs/batch" in call_args[1]["url"]
    assert "x-api-key" in call_args[1]["headers"]
    # tenant_id should not be in headers if not provided
    assert "x-tenant-id" not in call_args[1]["headers"]


@pytest.mark.asyncio
async def test_async_send_batch_with_tenant_id():
    logger = LangsmithLogger(
        langsmith_api_key="test-key",
        langsmith_tenant_id="test-tenant-id"
    )

    # Mock the httpx client
    mock_response = AsyncMock()
    mock_response.status_code = 200
    logger.async_httpx_client = AsyncMock()
    logger.async_httpx_client.post.return_value = mock_response

    # Add test data to queue
    logger.log_queue = [
        LangsmithQueueObject(
            data={"test": "data"}, credentials=logger.default_credentials
        )
    ]

    await logger.async_send_batch()

    # Verify the API call includes tenant_id header
    logger.async_httpx_client.post.assert_called_once()
    call_args = logger.async_httpx_client.post.call_args
    assert "runs/batch" in call_args[1]["url"]
    assert "x-api-key" in call_args[1]["headers"]
    assert "x-tenant-id" in call_args[1]["headers"]
    assert call_args[1]["headers"]["x-tenant-id"] == "test-tenant-id"


@pytest.mark.asyncio
async def test_langsmith_key_based_logging():
    """
    In key based logging langsmith_api_key and langsmith_project are passed directly to litellm.acompletion
    """
    try:
        # Mock the httpx post request
        # We need to mock get_async_httpx_client to return a mock AsyncHTTPHandler
        # because LangsmithLogger creates its own instance
        mock_async_httpx_handler = AsyncMock()
        mock_response = MagicMock()  # Use MagicMock for response to allow sync methods
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()  # raise_for_status is sync in httpx
        mock_response.text = ""
        mock_async_httpx_handler.post = AsyncMock(return_value=mock_response)
        
        mock_get_client = patch(
            "litellm.integrations.langsmith.get_async_httpx_client",
            return_value=mock_async_httpx_handler
        )
        mock_get_client.start()
        
        litellm.set_verbose = True
        litellm.DEFAULT_FLUSH_INTERVAL_SECONDS = 1

        litellm.callbacks = [LangsmithLogger()]
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test message"}],
            max_tokens=10,
            temperature=0.2,
            mock_response="This is a mock response",
            langsmith_api_key="fake_key_project2",
            langsmith_project="fake_project2",
        )
        print("Waiting for logs to be flushed to Langsmith.....")
        await asyncio.sleep(3)

        print("done sleeping 3 seconds...")

        # Verify the post request was made with correct parameters
        mock_async_httpx_handler.post.assert_called_once()
        call_args = mock_async_httpx_handler.post.call_args

        print("call_args", call_args)

        # Check URL contains /runs/batch
        assert "/runs/batch" in call_args[1]["url"]

        # Check headers contain the correct API key
        assert call_args[1]["headers"]["x-api-key"] == "fake_key_project2"
        # tenant_id should not be in headers if not provided
        assert "x-tenant-id" not in call_args[1]["headers"]

        # Verify the request body contains the expected data
        request_body = call_args[1]["json"]
        assert "post" in request_body
        assert len(request_body["post"]) == 1  # Should contain one run

        # EXPECTED BODY
        expected_body = {
            "post": [
                {
                    "name": "LLMRun",
                    "run_type": "llm",
                    "inputs": {
                        "id": "chatcmpl-82699ee4-7932-4fc0-9585-76abc8caeafa",
                        "call_type": "acompletion",
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Test message"}],
                        "model_parameters": {
                            "temperature": 0.2,
                            "max_tokens": 10,
                        },
                    },
                    "outputs": {
                        "id": "chatcmpl-82699ee4-7932-4fc0-9585-76abc8caeafa",
                        "model": "gpt-3.5-turbo",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "content": "This is a mock response",
                                    "role": "assistant",
                                    "tool_calls": None,
                                    "function_call": None,
                                },
                            }
                        ],
                        "usage": {
                            "completion_tokens": 20,
                            "prompt_tokens": 10,
                            "total_tokens": 30,
                        },
                    },
                    "session_name": "fake_project2",
                }
            ]
        }

        # Print both bodies for debugging
        actual_body = call_args[1]["json"]
        print("\nExpected body:")
        print(json.dumps(expected_body, indent=2))
        print("\nActual body:")
        print(json.dumps(actual_body, indent=2))

        assert len(actual_body["post"]) == 1

        # Assert only the critical parts we care about
        assert actual_body["post"][0]["name"] == expected_body["post"][0]["name"]
        assert (
            actual_body["post"][0]["run_type"] == expected_body["post"][0]["run_type"]
        )
        assert (
            actual_body["post"][0]["inputs"]["messages"]
            == expected_body["post"][0]["inputs"]["messages"]
        )
        assert (
            actual_body["post"][0]["inputs"]["model_parameters"]
            == expected_body["post"][0]["inputs"]["model_parameters"]
        )
        assert (
            actual_body["post"][0]["outputs"]["choices"]
            == expected_body["post"][0]["outputs"]["choices"]
        )
        assert (
            actual_body["post"][0]["outputs"]["usage"]["completion_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["completion_tokens"]
        )
        assert (
            actual_body["post"][0]["outputs"]["usage"]["prompt_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["prompt_tokens"]
        )
        assert (
            actual_body["post"][0]["outputs"]["usage"]["total_tokens"]
            == expected_body["post"][0]["outputs"]["usage"]["total_tokens"]
        )
        assert (
            actual_body["post"][0]["session_name"]
            == expected_body["post"][0]["session_name"]
        )
        
        mock_get_client.stop()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_langsmith_queue_logging():
    try:
        # Initialize LangsmithLogger
        test_langsmith_logger = LangsmithLogger()

        litellm.callbacks = [test_langsmith_logger]
        test_langsmith_logger.batch_size = 6
        litellm.set_verbose = True

        # Make multiple calls to ensure we don't hit the batch size
        for _ in range(5):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        await asyncio.sleep(3)

        # Check that logs are in the queue
        assert len(test_langsmith_logger.log_queue) == 5

        # Now make calls to exceed the batch size
        for _ in range(3):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test message"}],
                max_tokens=10,
                temperature=0.2,
                mock_response="This is a mock response",
            )

        # Wait a short time for any asynchronous operations to complete
        await asyncio.sleep(1)

        print(
            "Length of langsmith log queue: {}".format(
                len(test_langsmith_logger.log_queue)
            )
        )
        # Check that the queue was flushed after exceeding batch size
        assert len(test_langsmith_logger.log_queue) < 5

        # Clean up
        for cb in litellm.callbacks:
            if isinstance(cb, LangsmithLogger):
                await cb.async_httpx_client.client.aclose()

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
