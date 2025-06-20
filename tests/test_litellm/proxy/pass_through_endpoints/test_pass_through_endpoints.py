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
                        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={"test": "data"})
                        mock_proxy_logging.post_call_failure_hook = AsyncMock()
                        
                        # Setup mock for http response
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.headers = {}
                        mock_response.aread = AsyncMock(return_value=b'{"success": true}')
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
                        mock_request.body = AsyncMock(return_value=b'{"message": "test request"}')
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
