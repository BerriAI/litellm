"""
Test cases for spend_logs_metadata propagation in batch and files endpoints.

Tests that litellm_metadata with spend_logs_metadata is properly:
1. Accepted in files.create() via extra_body (multipart form data)
2. Accepted in batches.create() via extra_body (JSON body)
3. Parsed from JSON string format
4. Merged into request metadata
5. Appears in logging output
"""

import pytest
import asyncio
import aiohttp
import json
import os
from openai import OpenAI, AsyncOpenAI
from unittest.mock import patch, MagicMock

BASE_URL = "http://localhost:4000"
API_KEY = "sk-1234"


@pytest.fixture
def spend_logs_metadata():
    """Sample spend logs metadata for testing."""
    return {
        "owner": "team-data-ai-ml",
        "product": "litellm",
        "feature": "test_batching",
        "environment": "development",
    }


@pytest.mark.asyncio
async def test_files_create_with_litellm_metadata(spend_logs_metadata):
    """
    Test that files.create() properly handles litellm_metadata in extra_body.
    
    This tests the fix for multipart form data handling where litellm_metadata
    is sent as a form field and needs to be parsed from JSON string.
    """
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Create a simple JSONL file content for batch
    file_content = b'{"custom_id": "test-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]}}'
    
    # Mock the logging to capture metadata
    with patch('litellm.proxy.proxy_server.proxy_logging_obj') as mock_logging:
        mock_logging.post_call_success_hook = AsyncMock(return_value=None)
        
        # Upload file with litellm_metadata
        uploaded_file = await client.files.create(
            purpose="batch",
            file=file_content,
            extra_body={
                "litellm_metadata": {
                    "spend_logs_metadata": spend_logs_metadata,
                }
            },
        )
        
        assert uploaded_file.id is not None
        print(f"✓ File created with ID: {uploaded_file.id}")
        
        # Clean up
        await client.files.delete(file_id=uploaded_file.id)
        
        # Verify the logging hook was called (metadata should be in the call)
        assert mock_logging.post_call_success_hook.called


@pytest.mark.asyncio
async def test_batches_create_with_litellm_metadata(spend_logs_metadata):
    """
    Test that batches.create() properly handles litellm_metadata in extra_body.
    
    This tests JSON body handling where litellm_metadata is part of the request data.
    """
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # First create a file for the batch
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(_current_dir, "input.jsonl")
    
    # Create file without metadata first
    file_obj = await client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch",
    )
    
    # Create batch with litellm_metadata
    batch = await client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        extra_body={
            "litellm_metadata": {
                "spend_logs_metadata": spend_logs_metadata,
            }
        }
    )
    
    assert batch.id is not None
    print(f"✓ Batch created with ID: {batch.id}")
    
    # Clean up
    await client.batches.cancel(batch_id=batch.id)
    await client.files.delete(file_id=file_obj.id)


@pytest.mark.asyncio
async def test_files_create_with_raw_http_request(spend_logs_metadata):
    """
    Test files.create() with litellm_metadata using raw HTTP to verify form data handling.
    
    This directly tests that the form field 'litellm_metadata' is properly
    extracted and parsed from the multipart form data.
    """
    async with aiohttp.ClientSession() as session:
        url = f"{BASE_URL}/v1/files"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        data = aiohttp.FormData()
        data.add_field("purpose", "batch")
        data.add_field(
            "file",
            b'{"custom_id": "test-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Test"}]}}',
            filename="test.jsonl"
        )
        # Add litellm_metadata as a JSON string (as OpenAI SDK does)
        data.add_field(
            "litellm_metadata",
            json.dumps({"spend_logs_metadata": spend_logs_metadata})
        )
        
        async with session.post(url, headers=headers, data=data) as response:
            assert response.status == 200
            result = await response.json()
            assert "id" in result
            file_id = result["id"]
            print(f"✓ File created via raw HTTP with ID: {file_id}")
            
            # Clean up
            delete_url = f"{BASE_URL}/v1/files/{file_id}"
            async with session.delete(delete_url, headers=headers) as delete_response:
                assert delete_response.status == 200


@pytest.mark.asyncio
async def test_batches_retrieve_with_header_metadata(spend_logs_metadata):
    """
    Test that batches.retrieve() properly handles spend_logs_metadata via headers.
    
    Since retrieve is a GET request, metadata must be passed via headers
    using x-litellm-spend-logs-metadata.
    """
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # First create a batch
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(_current_dir, "input.jsonl")
    
    file_obj = await client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch",
    )
    
    batch = await client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    
    # Retrieve with metadata in headers
    retrieved_batch = await client.batches.retrieve(
        batch_id=batch.id,
        extra_headers={
            "x-litellm-spend-logs-metadata": json.dumps(spend_logs_metadata)
        }
    )
    
    assert retrieved_batch.id == batch.id
    print(f"✓ Batch retrieved with metadata headers: {batch.id}")
    
    # Clean up
    await client.batches.cancel(batch_id=batch.id)
    await client.files.delete(file_id=file_obj.id)


@pytest.mark.asyncio
async def test_metadata_parsing_from_string():
    """
    Test that litellm_metadata is properly parsed when received as a JSON string.
    
    This tests the core parsing logic in add_litellm_data_to_request.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import Mock
    
    # Mock request and user_api_key_dict
    mock_request = Mock()
    mock_request.headers = {}
    mock_request.url.path = "/v1/files"
    mock_request.method = "POST"
    
    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
    )
    
    mock_proxy_config = Mock()
    
    # Test data with litellm_metadata as a string (as it comes from form data)
    test_metadata = {
        "spend_logs_metadata": {
            "owner": "test-team",
            "product": "test-product",
        }
    }
    
    data = {
        "litellm_metadata": json.dumps(test_metadata)  # String format
    }
    
    # Process the data
    result = await add_litellm_data_to_request(
        data=data,
        request=mock_request,
        user_api_key_dict=mock_user_api_key_dict,
        proxy_config=mock_proxy_config,
        general_settings={},
        version="test",
    )
    
    # Verify litellm_metadata was parsed from string to dict
    assert isinstance(result["litellm_metadata"], dict)
    assert result["litellm_metadata"]["spend_logs_metadata"]["owner"] == "test-team"
    
    # Verify it was merged into the metadata variable
    assert "spend_logs_metadata" in result["litellm_metadata"]
    assert result["litellm_metadata"]["spend_logs_metadata"]["product"] == "test-product"
    
    print("✓ Metadata parsing from string works correctly")


@pytest.mark.asyncio
async def test_metadata_merging_preserves_user_values():
    """
    Test that user-provided metadata takes precedence over defaults.
    
    When both user and team provide spend_logs_metadata, user values should win.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import Mock
    
    # Mock request
    mock_request = Mock()
    mock_request.headers = {}
    mock_request.url.path = "/v1/batches"
    mock_request.method = "POST"
    
    # Mock user with team metadata
    mock_user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_metadata={
            "spend_logs_metadata": {
                "owner": "team-default",
                "product": "team-product",
            }
        }
    )
    
    mock_proxy_config = Mock()
    
    # User provides their own spend_logs_metadata
    data = {
        "litellm_metadata": {
            "spend_logs_metadata": {
                "owner": "user-override",  # User value should win
                "feature": "user-feature",  # New key from user
            }
        }
    }
    
    # Process the data
    result = await add_litellm_data_to_request(
        data=data,
        request=mock_request,
        user_api_key_dict=mock_user_api_key_dict,
        proxy_config=mock_proxy_config,
        general_settings={},
        version="test",
    )
    
    # Verify user values take precedence
    spend_logs = result["litellm_metadata"]["spend_logs_metadata"]
    assert spend_logs["owner"] == "user-override"  # User value preserved
    assert spend_logs["feature"] == "user-feature"  # User value added
    # Team values should NOT override user values
    
    print("✓ User metadata values are preserved over defaults")


if __name__ == "__main__":
    """Run tests directly for debugging."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    asyncio.run(test_files_create_with_litellm_metadata({
        "owner": "team-data-ai-ml",
        "product": "litellm",
        "feature": "test_batching",
        "environment": "development",
    }))

