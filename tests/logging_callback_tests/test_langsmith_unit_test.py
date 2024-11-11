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
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
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
        },
    )

    queue_obj2 = LangsmithQueueObject(
        data={"test": "data2"},
        credentials={
            "LANGSMITH_API_KEY": "key2",  # Different API key
            "LANGSMITH_PROJECT": "proj1",
            "LANGSMITH_BASE_URL": "url1",
        },
    )

    queue_obj3 = LangsmithQueueObject(
        data={"test": "data3"},
        credentials={
            "LANGSMITH_API_KEY": "key1",
            "LANGSMITH_PROJECT": "proj2",  # Different project
            "LANGSMITH_BASE_URL": "url1",
        },
    )

    logger.log_queue = [queue_obj1, queue_obj2, queue_obj3]

    grouped = logger._group_batches_by_credentials()

    # Check grouping
    assert len(grouped) == 3  # Should have three groups since credentials differ
    for key, batch_group in grouped.items():
        assert isinstance(key, CredentialsKey)
        assert len(batch_group.queue_objects) == 1  # Each group should have one object


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
