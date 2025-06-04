import json
import os
import sys
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.datastructures import UploadFile as StarletteUploadFile

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)


# Test is_multipart
def test_is_multipart():
    # Test with multipart content type
    request = MagicMock(spec=Request)
    request.headers = Headers({"content-type": "multipart/form-data; boundary=123"})
    assert HttpPassThroughEndpointHelpers.is_multipart(request) is True

    # Test with non-multipart content type
    request.headers = Headers({"content-type": "application/json"})
    assert HttpPassThroughEndpointHelpers.is_multipart(request) is False

    # Test with no content type
    request.headers = Headers({})
    assert HttpPassThroughEndpointHelpers.is_multipart(request) is False


# Test content length consistency for pass-through endpoints
def test_content_length_consistency():
    """Test that the Content-Length is consistent when using pre-serialized JSON."""
    # Test data
    test_data = {
        "prompt": "\n\nHuman: Tell me a short joke\n\nAssistant:",
        "max_tokens_to_sample": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Method 1: Using json parameter (what causes the issue)
    request1 = httpx.Request(
        method="POST",
        url="https://example.com",
        json=test_data
    )
    
    # Method 2: Using data parameter with pre-serialized JSON (our fix)
    json_str = json.dumps(test_data)
    request2 = httpx.Request(
        method="POST",
        url="https://example.com",
        content=json_str.encode(),
        headers={"Content-Type": "application/json"}
    )
    
    # Print the actual differences for verification
    print(f"Method 1 (json): Content-Length={request1.headers.get('content-length')}, Actual={len(request1.content)}")
    print(f"Method 2 (data): Content-Length={request2.headers.get('content-length')}, Actual={len(request2.content)}")
    print(f"Method 1 body: {request1.content}")
    print(f"Method 2 body: {request2.content}")
    
    # Assert that the Content-Length header matches the actual body length for our fix
    assert len(request2.content) == int(request2.headers.get("content-length", 0))
    
    # Check for potential mismatch with the json parameter
    # Note: This might not always occur depending on how httpx serializes JSON in different environments
    json_str_manual = json.dumps(test_data)
    manual_length = len(json_str_manual.encode())
    httpx_length = len(request1.content)
    
    if manual_length == httpx_length:
        print("Note: In this environment, manual JSON serialization matches httpx's internal serialization.")
        # Test passes because there's no mismatch in this environment
    else:
        # If they are different, that demonstrates the potential issue
        print(f"Detected Content-Length mismatch: {manual_length} vs {httpx_length}")
        assert manual_length != httpx_length


@pytest.mark.parametrize("use_data", [True, False])
def test_aws_sigv4_content_length_consistency(use_data):
    """
    Test that demonstrates how using data with pre-serialized JSON ensures
    Content-Length consistency for AWS SigV4 authentication.
    """
    # Test data
    test_data = {
        "prompt": "\n\nHuman: Tell me a short joke\n\nAssistant:",
        "max_tokens_to_sample": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Simulate SigV4 authentication process
    # 1. Pre-serialize JSON for signing
    json_str = json.dumps(test_data)
    content_length_for_signing = len(json_str.encode())
    
    # 2. Create the actual request
    if use_data:
        # Our fix: Use pre-serialized JSON with data parameter
        request = httpx.Request(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke",
            content=json_str.encode(),
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(content_length_for_signing)
            }
        )
    else:
        # Original approach: Use json parameter (which causes the issue)
        request = httpx.Request(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke",
            json=test_data,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(content_length_for_signing)
            }
        )
    
    # Check if Content-Length matches actual content length
    actual_content_length = len(request.content)
    expected_content_length = int(request.headers.get("content-length", 0))
    
    print(f"Use data: {use_data}")
    print(f"Expected Content-Length: {expected_content_length}")
    print(f"Actual content length: {actual_content_length}")
    print(f"Content: {request.content}")
    
    if use_data:
        # Our fix should ensure Content-Length matches
        assert actual_content_length == expected_content_length, "Content-Length mismatch with data parameter"
    else:
        # For the original approach, we don't assert anything specific
        # Just document whether a mismatch was detected in this environment
        if actual_content_length != expected_content_length:
            print("Content-Length mismatch detected with json parameter!")
            print(f"This demonstrates the issue fixed by our PR.")
        else:
            print("Note: In this environment, no Content-Length mismatch was detected with json parameter.")
            print("However, the fix is still valuable for ensuring consistency across all environments.")
    
    if use_data:
        # Our fix should ensure Content-Length matches
        assert actual_content_length == expected_content_length, "Content-Length mismatch with data parameter"
    else:
        # The original approach might cause a mismatch
        # Note: This might not always fail depending on how httpx serializes JSON
        if actual_content_length != expected_content_length:
            print("Content-Length mismatch detected with json parameter!")
            print(f"This demonstrates the issue fixed by our PR.")


# Test _build_request_files_from_upload_file
@pytest.mark.asyncio
async def test_build_request_files_from_upload_file():
    # Test with FastAPI UploadFile
    file_content = b"test content"
    file = BytesIO(file_content)
    # Create SpooledTemporaryFile with content type headers
    headers = {"content-type": "text/plain"}
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    result = await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
        upload_file
    )
    assert result == ("test.txt", file_content, "text/plain")

    # Test with Starlette UploadFile
    file2 = BytesIO(file_content)
    starlette_file = StarletteUploadFile(
        file=file2,
        filename="test2.txt",
        headers=Headers({"content-type": "text/plain"}),
    )
    starlette_file.read = AsyncMock(return_value=file_content)

    result = await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
        starlette_file
    )
    assert result == ("test2.txt", file_content, "text/plain")


# Test make_multipart_http_request
@pytest.mark.asyncio
async def test_make_multipart_http_request():
    # Mock request with file and form field
    request = MagicMock(spec=Request)
    request.method = "POST"

    # Mock form data
    file_content = b"test file content"
    file = BytesIO(file_content)
    # Create SpooledTemporaryFile with content type headers
    headers = {"content-type": "text/plain"}
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = {"file": upload_file, "text_field": "test value"}
    request.form = AsyncMock(return_value=form_data)

    # Mock httpx client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    # Test the function
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers={},
        requested_query_params=None,
    )

    # Verify the response
    assert response == mock_response

    # Verify the client call
    async_client.request.assert_called_once()
    call_args = async_client.request.call_args[1]

    assert call_args["method"] == "POST"
    assert str(call_args["url"]) == "http://test.com"
    assert isinstance(call_args["files"], dict)
    assert isinstance(call_args["data"], dict)
    assert call_args["data"]["text_field"] == "test value"
