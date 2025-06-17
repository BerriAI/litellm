import json
import os
import sys
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, QueryParams
from starlette.datastructures import UploadFile as StarletteUploadFile

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    pass_through_request,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
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


@pytest.mark.asyncio
async def test_pass_through_request_failure_handler():
    """
    Test that the failure handler is called when pass_through_request fails

    Critical Test: When a users pass through endpoint request fails, we must log the failure code, exception in litellm spend logs.
    """
    print("running test_pass_through_request_failure_handler")
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.llms.custom_httpx.http_handler.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                # Setup mock for post_call_failure_hook and pre_call_hook
                mock_proxy_logging.post_call_failure_hook = AsyncMock()
                mock_proxy_logging.pre_call_hook = AsyncMock()

                # Setup mock for httpx client
                mock_client = MagicMock()
                mock_client.client = MagicMock()
                mock_client.client.request = AsyncMock(
                    side_effect=httpx.HTTPError("Request failed")
                )
                mock_get_client.return_value = mock_client

                # Mock headers for custom headers
                mock_processing.get_custom_headers.return_value = {}

                # Create mock request
                mock_request = MagicMock(spec=Request)
                mock_request.method = "POST"
                mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
                mock_request.headers = Headers({})

                # Create a simple empty QueryParams
                mock_request.query_params = QueryParams({})

                # Create mock user API key dict
                mock_user_api_key_dict = MagicMock()

                # Call the function with a target that will trigger an HTTPError
                with pytest.raises(Exception):
                    await pass_through_request(
                        request=mock_request,
                        target="http://test.com",
                        custom_headers={},
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                # Assert post_call_failure_hook was called
                mock_proxy_logging.post_call_failure_hook.assert_called_once()

                # Verify the arguments to post_call_failure_hook
                call_args = mock_proxy_logging.post_call_failure_hook.call_args[1]
                assert call_args["user_api_key_dict"] == mock_user_api_key_dict
                assert isinstance(
                    call_args["original_exception"], TypeError
                )  # Now expecting TypeError
                assert "traceback_str" in call_args


def test_is_langfuse_route():
    """
    Test that the is_langfuse_route method correctly identifies Langfuse routes
    """
    handler = PassThroughEndpointLogging()

    # Test positive cases
    assert (
        handler.is_langfuse_route("http://localhost:4000/langfuse/api/public/traces")
        == True
    )
    assert (
        handler.is_langfuse_route(
            "https://proxy.example.com/langfuse/api/public/sessions"
        )
        == True
    )
    assert handler.is_langfuse_route("/langfuse/api/public/ingestion") == True
    assert handler.is_langfuse_route("http://localhost:4000/langfuse/") == True

    # Test negative cases
    assert (
        handler.is_langfuse_route("https://api.openai.com/v1/chat/completions") == False
    )
    assert (
        handler.is_langfuse_route("http://localhost:4000/anthropic/v1/messages")
        == False
    )
    assert handler.is_langfuse_route("https://example.com/other") == False
    assert handler.is_langfuse_route("") == False


@pytest.mark.asyncio
async def test_langfuse_passthrough_no_logging():
    """
    Test that langfuse pass-through requests skip logging by returning early
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Mock httpx response for langfuse request
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"status": "success"}'

    # Create langfuse URL
    langfuse_url = "http://localhost:4000/langfuse/api/public/traces"

    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=langfuse_url,
        request_body={"test": "data"},
        request_method="POST",
    )

    # Call the success handler with langfuse route
    result = await handler.pass_through_async_success_handler(
        httpx_response=mock_response,
        response_body={"status": "success"},
        logging_obj=mock_logging_obj,
        url_route=langfuse_url,
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"test": "data"},
        passthrough_logging_payload=passthrough_logging_payload,
    )

    # Should return None (early return) and not proceed with logging
    assert result is None

    # Verify that the passthrough_logging_payload was still set (this happens before the langfuse check)
    assert (
        mock_logging_obj.model_call_details["passthrough_logging_payload"]
        == passthrough_logging_payload
    )
