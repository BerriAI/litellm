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
    headers = Headers({"content-type": "text/plain"})
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
    headers = Headers({"content-type": "text/plain"})
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
async def test_make_multipart_http_request_removes_content_type_header():
    """
    Test that make_multipart_http_request removes the content-type header
    to prevent boundary mismatch errors.

    When forwarding multipart requests, the original content-type header contains
    a boundary that doesn't match the new boundary httpx generates. This test
    verifies that the content-type header is removed so httpx can set it correctly.
    """
    # Mock request with form data
    request = MagicMock(spec=Request)
    request.method = "POST"

    # Mock form data with both file and regular field
    file_content = b"test file content"
    file = BytesIO(file_content)
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = {"file": upload_file, "key": "value"}
    request.form = AsyncMock(return_value=form_data)

    # Mock httpx client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    # Headers with content-type containing old boundary (this is what causes the issue)
    original_headers = {
        "content-type": "multipart/form-data; boundary=--------------------------416423083260054165225918",
        "user-agent": "PostmanRuntime/7.49.0",
        "Authorization": "bearer sk-1234",
    }

    # Test the function
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers=original_headers,
        requested_query_params={"param": "value"},
    )

    # Verify the response
    assert response == mock_response

    # Verify the client call
    async_client.request.assert_called_once()
    call_args = async_client.request.call_args[1]

    # CRITICAL ASSERTION: content-type header should be removed
    assert "content-type" not in call_args["headers"]

    # Other headers should be preserved
    assert call_args["headers"]["user-agent"] == "PostmanRuntime/7.49.0"
    assert call_args["headers"]["Authorization"] == "bearer sk-1234"

    # Verify other parameters are correct
    assert call_args["method"] == "POST"
    assert str(call_args["url"]) == "http://test.com"
    assert isinstance(call_args["files"], dict)
    assert isinstance(call_args["data"], dict)
    assert call_args["data"]["key"] == "value"
    assert call_args["params"] == {"param": "value"}

    # Verify the original headers dict was not modified (copy was used)
    assert "content-type" in original_headers


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


def test_construct_target_url_with_subpath():
    """
    Test that construct_target_url_with_subpath correctly constructs target URLs
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        HttpPassThroughEndpointHelpers,
    )

    # Test with include_subpath=False
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="api/v1", include_subpath=False
    )
    assert result == "http://example.com"

    # Test with include_subpath=True and no subpath
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="", include_subpath=True
    )
    assert result == "http://example.com"

    # Test with include_subpath=True and subpath
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with base_target already ending with /
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com/", subpath="api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with subpath starting with /
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com", subpath="/api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"

    # Test with both conditions
    result = HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
        base_target="http://example.com/", subpath="/api/v1", include_subpath=True
    )
    assert result == "http://example.com/api/v1"


def test_add_exact_path_route():
    """
    Test that add_exact_path_route correctly adds exact path routes
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test data
    path = "/test/path"
    target = "http://example.com"
    custom_headers = {"x-custom": "header"}
    forward_headers = True
    merge_query_params = False
    dependencies = []

    # Call the function
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=mock_app,
        path=path,
        target=target,
        custom_headers=custom_headers,
        forward_headers=forward_headers,
        merge_query_params=merge_query_params,
        dependencies=dependencies,
        cost_per_request=None,
        endpoint_id="test-endpoint-id",
    )

    # Verify add_api_route was called with correct parameters
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    assert call_args["path"] == path
    assert call_args["methods"] == ["GET", "POST", "PUT", "DELETE", "PATCH"]
    assert call_args["dependencies"] == dependencies
    assert callable(call_args["endpoint"])


def test_add_subpath_route():
    """
    Test that add_subpath_route correctly adds wildcard routes for sub-paths
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test data
    path = "/test/path"
    target = "http://example.com"
    custom_headers = {"x-custom": "header"}
    forward_headers = True
    merge_query_params = False
    dependencies = []

    # Call the function
    InitPassThroughEndpointHelpers.add_subpath_route(
        app=mock_app,
        path=path,
        target=target,
        custom_headers=custom_headers,
        forward_headers=forward_headers,
        merge_query_params=merge_query_params,
        dependencies=dependencies,
        cost_per_request=None,
        endpoint_id="test-endpoint-id",
    )

    # Verify add_api_route was called with correct parameters
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Should have wildcard path
    expected_wildcard_path = f"{path}/{{subpath:path}}"
    assert call_args["path"] == expected_wildcard_path
    assert call_args["methods"] == ["GET", "POST", "PUT", "DELETE", "PATCH"]
    assert call_args["dependencies"] == dependencies
    assert callable(call_args["endpoint"])


@pytest.mark.asyncio
async def test_initialize_pass_through_endpoints_with_include_subpath():
    """
    Test that initialize_pass_through_endpoints adds wildcard routes when include_subpath is True
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        initialize_pass_through_endpoints,
    )

    # Mock the helper functions directly
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_exact_path_route"
    ) as mock_add_exact_route:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_subpath_route"
        ) as mock_add_subpath_route:
            with patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ):
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header"
                ) as mock_set_env:
                    mock_set_env.return_value = {}

                    # Test endpoint with include_subpath=True
                    endpoints = [
                        {
                            "path": "/test/endpoint",
                            "target": "http://example.com",
                            "include_subpath": True,
                        }
                    ]

                    await initialize_pass_through_endpoints(endpoints)

                    # Should be called once for exact path and once for subpath
                    mock_add_exact_route.assert_called_once()
                    mock_add_subpath_route.assert_called_once()

                    # Verify exact path route call
                    exact_call_args = mock_add_exact_route.call_args[1]
                    assert exact_call_args["path"] == "/test/endpoint"
                    assert exact_call_args["target"] == "http://example.com"

                    # Verify subpath route call
                    subpath_call_args = mock_add_subpath_route.call_args[1]
                    assert subpath_call_args["path"] == "/test/endpoint"
                    assert subpath_call_args["target"] == "http://example.com"


@pytest.mark.asyncio
async def test_initialize_pass_through_endpoints_without_include_subpath():
    """
    Test that initialize_pass_through_endpoints only adds exact route when include_subpath is False
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        initialize_pass_through_endpoints,
    )

    # Mock the helper functions directly
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_exact_path_route"
    ) as mock_add_exact_route:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.add_subpath_route"
        ) as mock_add_subpath_route:
            with patch(
                "litellm.proxy.proxy_server.premium_user",
                True,
            ):
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header"
                ) as mock_set_env:
                    mock_set_env.return_value = {}

                    # Test endpoint with include_subpath=False (default)
                    endpoints = [
                        {
                            "path": "/test/endpoint",
                            "target": "http://example.com",
                            "include_subpath": False,
                        }
                    ]

                    await initialize_pass_through_endpoints(endpoints)

                    # Should be called only once for exact path
                    mock_add_exact_route.assert_called_once()
                    mock_add_subpath_route.assert_not_called()

                    # Verify exact path route call
                    exact_call_args = mock_add_exact_route.call_args[1]
                    assert exact_call_args["path"] == "/test/endpoint"
                    assert exact_call_args["target"] == "http://example.com"


def test_set_cost_per_request():
    """
    Test that _set_cost_per_request correctly sets the cost in logging object and kwargs
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

    # Test with cost_per_request set
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=0.50,
    )

    kwargs = {"some_existing_key": "value"}

    # Call the method
    result_kwargs = handler._set_cost_per_request(
        logging_obj=mock_logging_obj,
        passthrough_logging_payload=passthrough_logging_payload,
        kwargs=kwargs,
    )

    # Verify that response_cost is set in kwargs and logging object
    assert result_kwargs["response_cost"] == 0.50
    assert mock_logging_obj.model_call_details["response_cost"] == 0.50
    assert result_kwargs["some_existing_key"] == "value"  # Existing kwargs preserved


def test_set_cost_per_request_none():
    """
    Test that _set_cost_per_request does nothing when cost_per_request is None
    """
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()

    # Mock the logging object
    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    # Test with cost_per_request not set (None)
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=None,
    )

    kwargs = {"some_existing_key": "value"}

    # Call the method
    result_kwargs = handler._set_cost_per_request(
        logging_obj=mock_logging_obj,
        passthrough_logging_payload=passthrough_logging_payload,
        kwargs=kwargs,
    )

    # Verify that response_cost is not set
    assert "response_cost" not in result_kwargs
    assert "response_cost" not in mock_logging_obj.model_call_details
    assert result_kwargs["some_existing_key"] == "value"  # Existing kwargs preserved


@pytest.mark.asyncio
async def test_pass_through_success_handler_with_cost_per_request():
    """
    Test that the success handler correctly processes cost_per_request
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

    # Mock the _handle_logging method to capture the call
    handler._handle_logging = AsyncMock()

    # Mock httpx response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"status": "success", "data": "test"}'

    # Create passthrough logging payload with cost_per_request
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url="http://example.com/api",
        request_body={"test": "data"},
        request_method="POST",
        cost_per_request=1.25,
    )

    start_time = datetime.now()
    end_time = datetime.now()

    # Call the success handler
    result = await handler.pass_through_async_success_handler(
        httpx_response=mock_response,
        response_body={"status": "success", "data": "test"},
        logging_obj=mock_logging_obj,
        url_route="http://example.com/api",
        result="",
        start_time=start_time,
        end_time=end_time,
        cache_hit=False,
        request_body={"test": "data"},
        passthrough_logging_payload=passthrough_logging_payload,
    )

    # Verify that the logging object has the cost set
    assert mock_logging_obj.model_call_details["response_cost"] == 1.25

    # Verify that _handle_logging was called with the correct kwargs
    handler._handle_logging.assert_called_once()
    call_kwargs = handler._handle_logging.call_args[1]
    assert call_kwargs["response_cost"] == 1.25


@pytest.mark.asyncio
async def test_create_pass_through_route_with_cost_per_request():
    """
    Test that create_pass_through_route correctly passes cost_per_request to the endpoint function
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    # Create the endpoint function with cost_per_request
    endpoint_func = create_pass_through_route(
        endpoint="/test/path",
        target="http://example.com",
        custom_headers={},
        _forward_headers=True,
        _merge_query_params=False,
        dependencies=[],
        cost_per_request=3.75,
    )

    # Mock the pass_through_request function to capture its call
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
    ) as mock_pass_through:
        mock_pass_through.return_value = MagicMock()

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})

        # Call the endpoint function
        # Create a proper UserAPIKeyAuth mock
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        await endpoint_func(
            request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
            fastapi_response=MagicMock(),
        )

        # Verify that pass_through_request was called with cost_per_request
        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]
        assert call_kwargs["cost_per_request"] == 3.75


def test_initialize_pass_through_endpoints_with_cost_per_request():
    """
    Test that initialize_pass_through_endpoints correctly passes cost_per_request to route creation
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    # Mock FastAPI app
    mock_app = MagicMock()

    # Test exact path route with cost_per_request
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=mock_app,
        path="/test/path",
        target="http://example.com",
        custom_headers={},
        forward_headers=True,
        merge_query_params=False,
        dependencies=[],
        cost_per_request=5.00,
        endpoint_id="test-endpoint-id-1",
    )

    # Verify add_api_route was called
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Verify the endpoint function was created with cost_per_request
    # We can't directly test the internal cost_per_request value, but we can verify
    # that the endpoint function was created properly
    assert call_args["path"] == "/test/path"
    assert callable(call_args["endpoint"])

    # Reset mock for subpath test
    mock_app.reset_mock()

    # Test subpath route with cost_per_request
    InitPassThroughEndpointHelpers.add_subpath_route(
        app=mock_app,
        path="/test/path",
        target="http://example.com",
        custom_headers={},
        forward_headers=True,
        merge_query_params=False,
        dependencies=[],
        cost_per_request=7.50,
        endpoint_id="test-endpoint-id-2",
    )

    # Verify add_api_route was called for subpath
    mock_app.add_api_route.assert_called_once()
    call_args = mock_app.add_api_route.call_args[1]

    # Verify the wildcard path and endpoint function
    assert call_args["path"] == "/test/path/{subpath:path}"
    assert callable(call_args["endpoint"])


@pytest.mark.asyncio
async def test_pass_through_request_contains_proxy_server_request_in_kwargs():
    """
    Test that pass_through_request (parent method) correctly includes proxy_server_request
    in kwargs passed to the success handler.

    Critical Test: Ensures that when pass_through_request is called, the kwargs passed to
    downstream methods contain the proxy server request details (url, method, body).
    """
    print("running test_pass_through_request_contains_proxy_server_request_in_kwargs")

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler"
        ) as mock_http_handler:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    with patch(
                        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_response_body"
                    ) as mock_get_response_body:
                        # Setup mock for pre_call_hook and post_call_failure_hook
                        mock_proxy_logging.pre_call_hook = AsyncMock(
                            return_value={"test": "data"}
                        )
                        mock_proxy_logging.post_call_failure_hook = AsyncMock()

                        # Setup mock for http response
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.headers = {}
                        mock_response.aread = AsyncMock(
                            return_value=b'{"success": true}'
                        )
                        mock_response.text = '{"success": true}'
                        mock_response.raise_for_status = MagicMock()

                        # Mock the HTTP request handler directly
                        mock_http_handler.return_value = mock_response

                        # Mock response body parser
                        mock_get_response_body.return_value = {"success": True}

                        # Mock headers for custom headers
                        mock_processing.get_custom_headers.return_value = {}

                        # Mock success handler to capture kwargs
                        mock_success_handler.return_value = None

                        # Create mock request
                        mock_request = MagicMock(spec=Request)
                        mock_request.method = "POST"
                        mock_request.url = "http://test-proxy.com/api/endpoint"
                        mock_request.body = AsyncMock(
                            return_value=b'{"message": "test request"}'
                        )
                        mock_request.headers = Headers({})
                        mock_request.query_params = QueryParams({})

                        # Create mock user API key dict
                        mock_user_api_key_dict = MagicMock()
                        mock_user_api_key_dict.api_key = "test-api-key"
                        mock_user_api_key_dict.key_alias = "test-alias"
                        mock_user_api_key_dict.user_email = "test@example.com"
                        mock_user_api_key_dict.user_id = "test-user-id"
                        mock_user_api_key_dict.team_id = "test-team-id"
                        mock_user_api_key_dict.org_id = "test-org-id"
                        mock_user_api_key_dict.team_alias = "test-team-alias"
                        mock_user_api_key_dict.end_user_id = "test-end-user-id"
                        mock_user_api_key_dict.request_route = "/api/endpoint"

                        # Call pass_through_request (the parent method)
                        result = await pass_through_request(
                            request=mock_request,
                            target="http://target-api.com/endpoint",
                            custom_headers={"X-Custom": "header"},
                            user_api_key_dict=mock_user_api_key_dict,
                        )

                        # Verify the success handler was called
                        mock_success_handler.assert_called_once()

                        # Extract the kwargs passed to the success handler
                        call_kwargs = mock_success_handler.call_args[1]

                        # Verify that litellm_params exists in kwargs
                        assert "litellm_params" in call_kwargs
                        litellm_params = call_kwargs["litellm_params"]

                        # Verify that proxy_server_request exists in litellm_params
                        assert "proxy_server_request" in litellm_params
                        proxy_server_request = litellm_params["proxy_server_request"]

                        # Verify the proxy_server_request contains expected fields
                        assert "url" in proxy_server_request
                        assert "method" in proxy_server_request
                        assert "body" in proxy_server_request

                        # Verify the values match the original request
                        assert proxy_server_request["url"] == str(mock_request.url)
                        assert proxy_server_request["method"] == mock_request.method
                        # The body should be the value returned by pre_call_hook, not the original request body
                        assert proxy_server_request["body"] == {"test": "data"}

                        # Verify other required kwargs are present
                        assert "call_type" in call_kwargs
                        assert call_kwargs["call_type"] == "pass_through_endpoint"
                        assert "litellm_call_id" in call_kwargs
                        assert "passthrough_logging_payload" in call_kwargs

                        # Verify metadata contains user information
                        assert "metadata" in litellm_params
                        metadata = litellm_params["metadata"]
                        assert metadata["user_api_key_hash"] == "test-api-key"
                        assert metadata["user_api_key_alias"] == "test-alias"
                        assert metadata["user_api_key_user_email"] == "test@example.com"
                        assert metadata["user_api_key_user_id"] == "test-user-id"


@pytest.mark.asyncio
async def test_create_pass_through_endpoint():
    """
    Test creating a new pass-through endpoint

    This test verifies that the create_pass_through_endpoints function:
    1. Accepts a PassThroughGenericEndpoint object
    2. Auto-generates an ID if not provided
    3. Adds the endpoint to the database
    4. Returns the created endpoint with the generated ID
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        ConfigFieldUpdate,
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Mock existing config (empty list)
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=[]
            )

            # Create test endpoint data
            test_endpoint = PassThroughGenericEndpoint(
                path="/test/endpoint",
                target="http://example.com/api",
                headers={"Authorization": "Bearer test-token"},
                include_subpath=True,
                cost_per_request=0.50,
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the create function
            result = await create_pass_through_endpoints(
                data=test_endpoint, user_api_key_dict=mock_user_api_key_dict
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            created_endpoint = result.endpoints[0]
            assert created_endpoint.path == "/test/endpoint"
            assert created_endpoint.target == "http://example.com/api"
            assert created_endpoint.headers == {"Authorization": "Bearer test-token"}
            assert created_endpoint.include_subpath is True
            assert created_endpoint.cost_per_request == 0.50
            assert created_endpoint.id is not None  # Should be auto-generated

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            assert len(update_call_args["data"].field_value) == 1
            assert update_call_args["data"].field_value[0]["path"] == "/test/endpoint"
            assert update_call_args["data"].field_value[0]["id"] is not None


@pytest.mark.asyncio
async def test_update_pass_through_endpoint():
    """
    Test updating an existing pass-through endpoint

    This test verifies that the update_pass_through_endpoints function:
    1. Finds the existing endpoint by ID
    2. Updates only the provided fields (partial updates)
    3. Preserves the existing ID
    4. Updates the database with the modified endpoint
    5. Returns the updated endpoint
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        ConfigFieldUpdate,
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Create existing endpoint data
            existing_endpoint_id = "test-endpoint-123"
            existing_endpoints = [
                {
                    "id": existing_endpoint_id,
                    "path": "/test/endpoint",
                    "target": "http://example.com/api",
                    "headers": {"Authorization": "Bearer old-token"},
                    "include_subpath": False,
                    "cost_per_request": 0.25,
                }
            ]

            # Mock existing config
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=existing_endpoints
            )

            # Create update data (partial update)
            update_data = PassThroughGenericEndpoint(
                path="/test/endpoint",  # Keep same path
                target="http://newapi.com/v2",  # Update target
                headers={
                    "Authorization": "Bearer new-token",
                    "X-Custom": "header",
                },  # Update headers
                cost_per_request=0.75,  # Update cost
                # include_subpath not provided - should preserve existing value
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the update function
            result = await update_pass_through_endpoints(
                endpoint_id=existing_endpoint_id,
                data=update_data,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            updated_endpoint = result.endpoints[0]
            assert updated_endpoint.id == existing_endpoint_id  # ID preserved
            assert updated_endpoint.path == "/test/endpoint"
            assert updated_endpoint.target == "http://newapi.com/v2"  # Updated
            assert updated_endpoint.headers == {
                "Authorization": "Bearer new-token",
                "X-Custom": "header",
            }  # Updated
            assert updated_endpoint.include_subpath is False  # Preserved existing value
            assert updated_endpoint.cost_per_request == 0.75  # Updated

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            assert len(update_call_args["data"].field_value) == 1
            updated_data = update_call_args["data"].field_value[0]
            assert updated_data["id"] == existing_endpoint_id
            assert updated_data["target"] == "http://newapi.com/v2"
            assert updated_data["cost_per_request"] == 0.75


@pytest.mark.asyncio
async def test_update_pass_through_endpoint_not_found():
    """
    Test updating a non-existent pass-through endpoint raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock existing config with different endpoint
        existing_endpoints = [
            {
                "id": "different-endpoint-456",
                "path": "/different/endpoint",
                "target": "http://different.com/api",
                "headers": {},
                "include_subpath": False,
                "cost_per_request": 0.0,
            }
        ]

        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        # Create update data
        update_data = PassThroughGenericEndpoint(
            path="/test/endpoint", target="http://newapi.com/v2"
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the update function with non-existent ID
        with pytest.raises(HTTPException) as exc_info:
            await update_pass_through_endpoints(
                endpoint_id="non-existent-endpoint-123",
                data=update_data,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Verify the exception
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint():
    """
    Test deleting an existing pass-through endpoint

    This test verifies that the delete_pass_through_endpoints function:
    1. Finds the existing endpoint by ID
    2. Removes it from the database
    3. Returns the deleted endpoint
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        ConfigFieldUpdate,
        PassThroughEndpointResponse,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        with patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config:
            # Create existing endpoint data
            endpoint_to_delete_id = "test-endpoint-123"
            other_endpoint_id = "other-endpoint-456"
            existing_endpoints = [
                {
                    "id": endpoint_to_delete_id,
                    "path": "/test/endpoint",
                    "target": "http://example.com/api",
                    "headers": {"Authorization": "Bearer test-token"},
                    "include_subpath": True,
                    "cost_per_request": 0.50,
                },
                {
                    "id": other_endpoint_id,
                    "path": "/other/endpoint",
                    "target": "http://other.com/api",
                    "headers": {},
                    "include_subpath": False,
                    "cost_per_request": 0.25,
                },
            ]

            # Mock existing config
            mock_get_config.return_value = ConfigFieldInfo(
                field_name="pass_through_endpoints", field_value=existing_endpoints
            )

            # Mock user API key dict
            mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

            # Call the delete function
            result = await delete_pass_through_endpoints(
                endpoint_id=endpoint_to_delete_id,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify the result
            assert isinstance(result, PassThroughEndpointResponse)
            assert len(result.endpoints) == 1

            deleted_endpoint = result.endpoints[0]
            assert deleted_endpoint.id == endpoint_to_delete_id
            assert deleted_endpoint.path == "/test/endpoint"
            assert deleted_endpoint.target == "http://example.com/api"
            assert deleted_endpoint.headers == {"Authorization": "Bearer test-token"}
            assert deleted_endpoint.include_subpath is True
            assert deleted_endpoint.cost_per_request == 0.50

            # Verify database calls
            mock_get_config.assert_called_once_with(
                field_name="pass_through_endpoints",
                user_api_key_dict=mock_user_api_key_dict,
            )

            mock_update_config.assert_called_once()
            update_call_args = mock_update_config.call_args[1]
            assert update_call_args["data"].field_name == "pass_through_endpoints"
            # Should only have the other endpoint remaining
            assert len(update_call_args["data"].field_value) == 1
            remaining_endpoint = update_call_args["data"].field_value[0]
            assert remaining_endpoint["id"] == other_endpoint_id
            assert remaining_endpoint["path"] == "/other/endpoint"


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint_not_found():
    """
    Test deleting a non-existent pass-through endpoint raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import ConfigFieldInfo, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock existing config with different endpoint
        existing_endpoints = [
            {
                "id": "different-endpoint-456",
                "path": "/different/endpoint",
                "target": "http://different.com/api",
                "headers": {},
                "include_subpath": False,
                "cost_per_request": 0.0,
            }
        ]

        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the delete function with non-existent ID
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="non-existent-endpoint-123",
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Verify the exception
        assert exc_info.value.status_code == 400
        assert "not found" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_delete_pass_through_endpoint_empty_list():
    """
    Test deleting from an empty endpoint list raises HTTPException
    """
    from fastapi import HTTPException

    from litellm.proxy._types import ConfigFieldInfo, UserAPIKeyAuth
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        delete_pass_through_endpoints,
    )

    # Mock the database functions
    with patch(
        "litellm.proxy.proxy_server.get_config_general_settings"
    ) as mock_get_config:
        # Mock empty config
        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

        # Mock user API key dict
        mock_user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

        # Call the delete function
        with pytest.raises(HTTPException) as exc_info:
            await delete_pass_through_endpoints(
                endpoint_id="any-endpoint-123", user_api_key_dict=mock_user_api_key_dict
            )

        # Verify the exception
        assert exc_info.value.status_code == 400
        assert "no pass-through endpoints setup" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_pass_through_request_query_params_forwarding():
    """
    Test that query parameters from the original request are properly forwarded to the target URL.

    This test verifies the fix for the bug where query parameters like api-version were being lost
    when forwarding requests to Azure OpenAI and other pass-through endpoints.
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler"
        ) as mock_http_handler:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    with patch(
                        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_response_body"
                    ) as mock_get_response_body:
                        # Setup mock for pre_call_hook
                        test_body = {"name": "Azure Assistant", "model": "gpt-4o"}
                        mock_proxy_logging.pre_call_hook = AsyncMock(
                            return_value=test_body
                        )

                        # Setup mock for http response
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.headers = {"content-type": "application/json"}
                        mock_response.aread = AsyncMock(
                            return_value=b'{"id": "asst_123", "object": "assistant"}'
                        )
                        mock_response.text = '{"id": "asst_123", "object": "assistant"}'
                        mock_response.raise_for_status = MagicMock()

                        # Mock the HTTP request handler to capture the call
                        mock_http_handler.return_value = mock_response

                        # Mock response body parser
                        mock_get_response_body.return_value = {
                            "id": "asst_123",
                            "object": "assistant",
                        }

                        # Mock headers for custom headers
                        mock_processing.get_custom_headers.return_value = {}

                        # Mock success handler
                        mock_success_handler.return_value = None

                        # Create mock request with query parameters (Azure API version)
                        mock_request = MagicMock(spec=Request)
                        mock_request.method = "POST"
                        mock_request.url = (
                            "http://localhost:4000/azure-assistant/openai/assistants"
                        )
                        mock_request.body = AsyncMock(
                            return_value=json.dumps(test_body).encode()
                        )
                        mock_request.headers = Headers(
                            {"Content-Type": "application/json"}
                        )

                        # Create QueryParams with api-version parameter
                        mock_request.query_params = QueryParams(
                            [("api-version", "2025-01-01-preview")]
                        )

                        # Create mock user API key dict
                        mock_user_api_key_dict = MagicMock()
                        mock_user_api_key_dict.api_key = "sk-1234"

                        # Call pass_through_request
                        result = await pass_through_request(
                            request=mock_request,
                            target="https://krris-m2f9a9i7-eastus2.openai.azure.com/openai/assistants",
                            custom_headers={"Authorization": "Bearer azure_token"},
                            user_api_key_dict=mock_user_api_key_dict,
                        )

                        # Verify the HTTP handler was called
                        mock_http_handler.assert_called_once()

                        # Extract the call arguments to verify query parameters were passed
                        call_kwargs = mock_http_handler.call_args[1]

                        # The key assertion: query parameters should be preserved and passed to the HTTP handler
                        assert "requested_query_params" in call_kwargs
                        assert call_kwargs["requested_query_params"] == {
                            "api-version": "2025-01-01-preview"
                        }

                        # Verify the target URL is correct
                        assert (
                            str(call_kwargs["url"])
                            == "https://krris-m2f9a9i7-eastus2.openai.azure.com/openai/assistants"
                        )

                        # Verify the request body is preserved
                        assert call_kwargs["_parsed_body"] == test_body


@pytest.mark.asyncio
async def test_pass_through_with_httpbin_redirect():
    """
    Integration test using httpbin.org redirect endpoint to test real redirect handling.
    This tests the actual redirect handling capability end-to-end using the full pass_through_request function.
    """
    from unittest.mock import MagicMock

    from fastapi import Request
    from starlette.datastructures import Headers, QueryParams

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        pass_through_request,
    )

    # Create mock request
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.headers = Headers({})
    mock_request.query_params = QueryParams("")

    # Mock the body method to return empty bytes for GET request
    async def mock_body():
        return b""

    mock_request.body = mock_body

    # Mock user API key dict
    mock_user_api_key_dict = MagicMock()

    try:
        # Test with httpbin.org redirect endpoint
        # This will redirect to httpbin.org/get
        response = await pass_through_request(
            request=mock_request,
            target="https://httpbin.org/redirect/1",
            custom_headers={},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Should get the final response (200) from /get endpoint, not the redirect (302)
        assert response.status_code == 200

        # The response should be from the /get endpoint
        response_content = bytes(response.body).decode("utf-8")

        # httpbin.org/get returns JSON with info about the request
        assert '"url": "https://httpbin.org/get"' in response_content
        print("GOT A Response from HTTPBIN=", response_content)
    except Exception as e:
        # If httpbin.org is not accessible, skip the test
        import pytest

        pytest.skip(f"Could not reach httpbin.org for integration test: {e}")


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_with_filter():
    """
    Test that _filter_endpoints_by_team_allowed_routes correctly filters endpoints
    when team has allowed_passthrough_routes in metadata
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/allowed1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/allowed2", target="http://example.com/api2"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-3", path="/api/notallowed", target="http://example.com/api3"
        ),
    ]

    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {
        "allowed_passthrough_routes": ["/api/allowed1", "/api/allowed2"]
    }
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should only return allowed endpoints
    assert len(result) == 2
    assert result[0].path == "/api/allowed1"
    assert result[1].path == "/api/allowed2"

    # Verify database call
    mock_prisma_client.db.litellm_teamtable.find_unique.assert_called_once_with(
        where={"team_id": "test-team-123"}
    )


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_team_not_found():
    """
    Test that _filter_endpoints_by_team_allowed_routes raises HTTPException
    when team is not found
    """
    from fastapi import HTTPException

    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test", target="http://example.com/api"
        ),
    ]

    # Mock prisma client to return None (team not found)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    # Call the function and expect HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await _filter_endpoints_by_team_allowed_routes(
            team_id="non-existent-team",
            pass_through_endpoints=endpoints,
            prisma_client=mock_prisma_client,
        )

    # Verify the exception
    assert exc_info.value.status_code == 404
    assert "Team not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_no_metadata():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns all endpoints
    when team has no metadata
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has None metadata
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = None
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return all endpoints when no metadata
    assert len(result) == 2
    assert result[0].path == "/api/test1"
    assert result[1].path == "/api/test2"


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_no_allowed_routes_key():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns all endpoints
    when team metadata doesn't have allowed_passthrough_routes key
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has metadata but no allowed_passthrough_routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"some_other_key": "some_value"}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return all endpoints when allowed_passthrough_routes key doesn't exist
    assert len(result) == 2
    assert result[0].path == "/api/test1"
    assert result[1].path == "/api/test2"


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_empty_allowed_list():
    """
    Test that _filter_endpoints_by_team_allowed_routes returns empty list
    when team has empty allowed_passthrough_routes list
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/test1", target="http://example.com/api1"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2", path="/api/test2", target="http://example.com/api2"
        ),
    ]

    # Mock prisma client with team that has empty allowed_passthrough_routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"allowed_passthrough_routes": []}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return empty list when allowed_passthrough_routes is empty
    assert len(result) == 0


@pytest.mark.asyncio
async def test_filter_endpoints_by_team_allowed_routes_partial_match():
    """
    Test that _filter_endpoints_by_team_allowed_routes correctly filters
    when only some endpoints match allowed routes
    """
    from litellm.proxy._types import PassThroughGenericEndpoint
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _filter_endpoints_by_team_allowed_routes,
    )

    # Create test endpoints
    endpoints = [
        PassThroughGenericEndpoint(
            id="endpoint-1", path="/api/openai", target="http://example.com/openai"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-2",
            path="/api/anthropic",
            target="http://example.com/anthropic",
        ),
        PassThroughGenericEndpoint(
            id="endpoint-3", path="/api/azure", target="http://example.com/azure"
        ),
        PassThroughGenericEndpoint(
            id="endpoint-4", path="/api/cohere", target="http://example.com/cohere"
        ),
    ]

    # Mock prisma client with team that allows only 2 routes
    mock_prisma_client = MagicMock()
    mock_team = MagicMock()
    mock_team.metadata = {"allowed_passthrough_routes": ["/api/openai", "/api/azure"]}
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team
    )

    # Call the function
    result = await _filter_endpoints_by_team_allowed_routes(
        team_id="test-team-123",
        pass_through_endpoints=endpoints,
        prisma_client=mock_prisma_client,
    )

    # Should return only the 2 allowed endpoints
    assert len(result) == 2
    assert result[0].path == "/api/openai"
    assert result[1].path == "/api/azure"


@pytest.mark.asyncio
async def test_bedrock_router_passthrough_metadata_initialization():
    """
    Test that bedrock router passthrough properly initializes metadata for hooks.

    This test verifies the fix for issue #15826 where metadata.headers and
    litellm_params.proxy_server_request were missing for /bedrock passthrough
    requests with router models.

    The fix ensures router bedrock models use the same common processing path
    as non-router models, which properly initializes all metadata structures.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        handle_bedrock_passthrough_router_model,
    )

    # Mock ProxyBaseLLMRequestProcessing to verify it's used
    with patch(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing"
    ) as mock_processing_class:
        # Setup mock instance
        mock_processor = MagicMock()
        mock_processing_class.return_value = mock_processor

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aread = AsyncMock(
            return_value=b'{"content": [{"text": "Hello"}]}'
        )
        mock_processor.base_passthrough_process_llm_request = AsyncMock(
            return_value=mock_response
        )

        # Create mock request with headers
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "http://localhost:4000/bedrock/model/my-model/invoke"
        mock_request.headers = Headers(
            {
                "content-type": "application/json",
                "authorization": "Bearer sk-test-key",
                "x-custom-header": "test-value",
            }
        )
        mock_request.query_params = QueryParams({})

        # Create mock user API key dict with all required fields
        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "sk-test-key"
        mock_user_api_key_dict.key_alias = "test-alias"
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.team_id = "team-123"

        # Mock other required dependencies
        mock_router = MagicMock()
        mock_proxy_logging = MagicMock()
        mock_general_settings = {}
        mock_proxy_config = MagicMock()
        mock_select_data_generator = MagicMock()

        request_body = {
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "anthropic_version": "bedrock-2023-05-31",
        }

        # Call the function
        result = await handle_bedrock_passthrough_router_model(
            model="my-bedrock-model",
            endpoint="/model/my-bedrock-model/invoke",
            request=mock_request,
            request_body=request_body,
            llm_router=mock_router,
            user_api_key_dict=mock_user_api_key_dict,
            proxy_logging_obj=mock_proxy_logging,
            general_settings=mock_general_settings,
            proxy_config=mock_proxy_config,
            select_data_generator=mock_select_data_generator,
            user_model=None,
            user_temperature=None,
            user_request_timeout=None,
            user_max_tokens=None,
            user_api_base=None,
            version="1.0",
        )

        # Verify that ProxyBaseLLMRequestProcessing was instantiated
        # This is the KEY assertion - router models now use the common processing path
        mock_processing_class.assert_called_once()

        # Verify that base_passthrough_process_llm_request was called
        # This proves we're using the common processing path that initializes metadata
        mock_processor.base_passthrough_process_llm_request.assert_called_once()

        # Verify the call included all required parameters for proper metadata initialization
        call_kwargs = mock_processor.base_passthrough_process_llm_request.call_args[1]

        # These are the critical parameters that ensure metadata is properly initialized:
        assert (
            call_kwargs["request"] == mock_request
        ), "Request must be passed for header extraction"
        assert (
            call_kwargs["user_api_key_dict"] == mock_user_api_key_dict
        ), "User API key dict needed for metadata"
        assert (
            call_kwargs["proxy_logging_obj"] == mock_proxy_logging
        ), "Logging obj needed for hooks"
        assert (
            call_kwargs["llm_router"] == mock_router
        ), "Router needed for model routing"
        assert call_kwargs["model"] == "my-bedrock-model", "Model name must be passed"

        # Verify response was returned
        assert result == mock_response


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_adds_headers_to_metadata():
    """
    Test that add_litellm_data_to_request adds headers to metadata for guardrails.

    This test verifies the fix for issue #17477 where guardrails couldn't access
    request headers (like User-Agent) on Bedrock pass-through endpoints.

    The fix ensures headers are available in data["metadata"]["headers"] so
    guardrails can validate User-Agent, API keys, and other header-based checks.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy._types import UserAPIKeyAuth

    # Create mock request with headers including User-Agent
    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.url = MagicMock()
    mock_request.url.path = "/bedrock/model/my-model/converse"
    mock_request.headers = Headers(
        {
            "content-type": "application/json",
            "user-agent": "claude-cli/2.0.69 (external, cli)",
            "authorization": "Bearer sk-test-key",
            "x-custom-header": "test-value",
        }
    )
    mock_request.query_params = QueryParams({})

    # Create mock user API key dict
    mock_user_api_key_dict = UserAPIKeyAuth()

    # Create mock proxy config
    mock_proxy_config = MagicMock()
    mock_proxy_config.pass_through_endpoints = []

    # Initial data dict (simulating Bedrock pass-through)
    data = {
        "model": "my-bedrock-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=mock_request,
        user_api_key_dict=mock_user_api_key_dict,
        proxy_config=mock_proxy_config,
        general_settings={},
        version="1.0",
    )

    # Verify headers are added to metadata for guardrails
    assert "metadata" in result, "metadata should be present in result"
    assert "headers" in result["metadata"], "headers should be present in metadata"
    assert isinstance(
        result["metadata"]["headers"], dict
    ), "headers should be a dictionary"

    # Verify specific headers are accessible (important for guardrails)
    headers = result["metadata"]["headers"]
    assert (
        "user-agent" in headers or "User-Agent" in headers
    ), "User-Agent header should be accessible in metadata"

    # Also verify proxy_server_request has headers (original location)
    assert "proxy_server_request" in result
    assert "headers" in result["proxy_server_request"]
