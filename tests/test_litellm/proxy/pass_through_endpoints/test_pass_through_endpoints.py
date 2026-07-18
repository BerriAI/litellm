import asyncio
import json
import logging
import os
import sys
from contextlib import ExitStack
from io import BytesIO
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request, UploadFile
from starlette.datastructures import FormData, Headers, QueryParams
from starlette.datastructures import UploadFile as StarletteUploadFile

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS,
    HttpPassThroughEndpointHelpers,
    InitPassThroughEndpointHelpers,
    LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY,
    _registered_pass_through_routes,
    create_pass_through_route,
    initialize_pass_through_endpoints,
    pass_through_request,
    resolve_pass_through_request_timeout,
    resolve_llm_passthrough_timeout,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY,
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

    form_data = FormData([("file", upload_file), ("text_field", "test value")])
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
    assert call_args["files"] == [("file", ("test.txt", file_content, "text/plain"))]
    assert call_args["data"] == {"text_field": ["test value"]}


@pytest.mark.asyncio
async def test_make_multipart_http_request_forwards_repeated_fields():
    """
    Regression: a client sending several parts under the same field name
    (e.g. ``-F file=@a.pdf -F file=@b.pdf``) must have every part forwarded.
    Starlette's ``FormData.items()`` collapses duplicate keys to the last value,
    so the handler must read ``multi_items()`` and emit one httpx files tuple per
    file plus a list value per repeated non-file field.
    """
    request = MagicMock(spec=Request)
    request.method = "POST"

    def _upload(filename: str, content: bytes) -> UploadFile:
        f = UploadFile(
            file=BytesIO(content),
            filename=filename,
            headers=Headers({"content-type": "application/pdf"}),
        )
        f.read = AsyncMock(return_value=content)
        return f

    form_data = FormData(
        [
            ("file", _upload("a.pdf", b"PDF-ONE")),
            ("file", _upload("b.pdf", b"PDF-TWO")),
            ("other_parameter", "xxx"),
            ("other_parameter", "yyy"),
        ]
    )
    request.form = AsyncMock(return_value=form_data)

    mock_response = MagicMock()
    mock_response.status_code = 200
    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers={},
        requested_query_params=None,
    )

    call_args = async_client.request.call_args[1]

    assert call_args["files"] == [
        ("file", ("a.pdf", b"PDF-ONE", "application/pdf")),
        ("file", ("b.pdf", b"PDF-TWO", "application/pdf")),
    ]
    assert call_args["data"] == {"other_parameter": ["xxx", "yyy"]}


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

    form_data = FormData([("file", upload_file), ("key", "value")])
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
    assert call_args["files"] == [("file", ("test.txt", file_content, "text/plain"))]
    assert call_args["data"] == {"key": ["value"]}
    assert call_args["params"] == {"param": "value"}

    # Verify the original headers dict was not modified (copy was used)
    assert "content-type" in original_headers


@pytest.mark.asyncio
async def test_non_streaming_http_request_handler_multipart_with_non_empty_parsed_body():
    """
    Regression: pass_through_request injects litellm_logging_obj into _parsed_body before
    forwarding. Multipart uploads must still use files=, not json=_parsed_body.
    """
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = Headers(
        {"content-type": "multipart/form-data; boundary=------------------------test"}
    )

    file_content = b"test file content"
    file = BytesIO(file_content)
    upload_headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=upload_headers)
    upload_file.read = AsyncMock(return_value=file_content)
    request.form = AsyncMock(return_value=FormData([("file", upload_file)]))

    mock_response = MagicMock()
    mock_response.status_code = 200
    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=mock_response)

    await HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com"),
        headers={},
        requested_query_params=None,
        _parsed_body={"litellm_logging_obj": MagicMock()},
        forward_multipart=True,
    )

    async_client.request.assert_called_once()
    call_args = async_client.request.call_args[1]
    assert "files" in call_args
    assert "json" not in call_args
    assert call_args["files"] == [("file", ("test.txt", file_content, "text/plain"))]


@pytest.mark.asyncio
async def test_pass_through_request_failure_handler():
    """
    Test that the failure handler is called when pass_through_request fails

    Critical Test: When a users pass through endpoint request fails, we must log the failure code, exception in litellm spend logs.
    """
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
        is True
    )
    assert (
        handler.is_langfuse_route(
            "https://proxy.example.com/langfuse/api/public/sessions"
        )
        is True
    )
    assert handler.is_langfuse_route("/langfuse/api/public/ingestion") is True
    assert handler.is_langfuse_route("http://localhost:4000/langfuse/") is True

    # Test negative cases
    assert (
        handler.is_langfuse_route("https://api.openai.com/v1/chat/completions") is False
    )
    assert (
        handler.is_langfuse_route("http://localhost:4000/anthropic/v1/messages")
        is False
    )
    assert handler.is_langfuse_route("https://example.com/other") is False
    assert handler.is_langfuse_route("") is False


def test_is_vertex_route_ignores_plain_predict_path_segment():
    """
    Regression for LIT-4527: a custom (non-Vertex) passthrough URL whose path
    contains a plain `predict`/`search` segment must not be classified as a
    Vertex route, otherwise its success logging is routed into the Vertex
    handler, the Vertex-shaped transform fails, and no log row is recorded.

    Real Vertex custom methods are invoked with the GCP `resource:method` colon
    syntax, so only the colon form should count as Vertex.
    """
    handler = PassThroughEndpointLogging()

    assert (
        handler.is_vertex_route(
            "https://upstream.example.com/ml/api/v1/time-series-forecast/predict"
        )
        is False
    )
    assert handler.is_vertex_route("https://upstream.example.com/api/v1/search") is False
    assert (
        handler.is_vertex_route("https://upstream.example.com/predict/generateContent")
        is False
    )

    assert (
        handler.is_vertex_route(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/m:predict"
        )
        is True
    )
    assert (
        handler.is_vertex_route(
            "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/anthropic/models/claude:rawPredict"
        )
        is True
    )
    assert (
        handler.is_vertex_route(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/m:generateContent"
        )
        is True
    )
    assert (
        handler.is_vertex_route(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/m:predictLongRunning"
        )
        is True
    )
    assert (
        handler.is_vertex_route("https://discoveryengine.googleapis.com/v1/x:search")
        is True
    )

    assert (
        handler.is_vertex_route(
            "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/batchPredictionJobs"
        )
        is True
    )


@pytest.mark.asyncio
async def test_custom_passthrough_predict_path_logs_via_generic_handler():
    """
    Regression for LIT-4527: an upstream-successful custom passthrough request to
    a `/predict` path (non-Vertex body) must still produce a log row. Before the
    fix it was routed into VertexPassthroughLoggingHandler, which failed on the
    non-Vertex body and dropped the log entirely.
    """
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    handler = PassThroughEndpointLogging()
    handler._handle_logging = AsyncMock()

    mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging_obj.model_call_details = {}

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"forecast": [1, 2, 3]}'

    url_route = "https://upstream.example.com/ml/api/v1/time-series-forecast/predict"
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=url_route,
        request_body={"series": [1, 2]},
        request_method="POST",
    )

    with patch(
        "litellm.proxy.pass_through_endpoints.success_handler.VertexPassthroughLoggingHandler.vertex_passthrough_handler"
    ) as mock_vertex_handler:
        await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body={"forecast": [1, 2, 3]},
            logging_obj=mock_logging_obj,
            url_route=url_route,
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"series": [1, 2]},
            passthrough_logging_payload=passthrough_logging_payload,
        )

    mock_vertex_handler.assert_not_called()
    handler._handle_logging.assert_awaited_once()
    logged_object = handler._handle_logging.call_args.kwargs[
        "standard_logging_response_object"
    ]
    assert logged_object == {"response": '{"forecast": [1, 2, 3]}'}


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
async def test_pass_through_handler_rejects_unregistered_method():
    """
    Stale FastAPI routes can remain after an endpoint is updated from all methods
    to a restricted method list. The handler must enforce the current registry.
    """
    from fastapi import HTTPException

    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    endpoint_func = create_pass_through_route(
        endpoint="/test/path",
        target="http://example.com",
    )
    request = MagicMock(spec=Request)
    request.method = "GET"

    with (
        patch.dict(os.environ, {"SERVER_ROOT_PATH": ""}),
        patch(
            "litellm.proxy.auth.auth_utils.get_request_route",
            return_value="/test/path",
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type",
            new_callable=AsyncMock,
            return_value=({}, {}, None, False),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            {
                "test-endpoint-id:exact:/test/path:POST": {
                    "endpoint_id": "test-endpoint-id",
                    "path": "/test/path",
                    "type": "exact",
                    "methods": ["POST"],
                    "passthrough_params": {
                        "target": "http://example.com",
                        "custom_headers": {},
                        "forward_headers": False,
                        "merge_query_params": False,
                    },
                }
            },
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await endpoint_func(
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
            )

    assert exc_info.value.status_code == 405


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
    await handler.pass_through_async_success_handler(
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
    unique_path = "/test/path/unique/cost_per_request"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com",
        custom_headers={},
        _forward_headers=True,
        _merge_query_params=False,
        dependencies=[],
        cost_per_request=3.75,
    )

    # Mock the pass_through_request function to capture its call
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
        ) as mock_pass_through,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route"
        ) as mock_is_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
        ) as mock_get_registered,
    ):
        mock_pass_through.return_value = MagicMock()
        mock_is_registered.return_value = True
        mock_get_registered.return_value = None

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
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


def test_resolve_pass_through_request_timeout_precedence():
    assert resolve_pass_through_request_timeout(endpoint_timeout=900) == 900.0

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"pass_through_request_timeout": 1200},
    ):
        assert resolve_pass_through_request_timeout() == 1200.0
        assert resolve_pass_through_request_timeout(endpoint_timeout=800) == 800.0

    with patch("litellm.proxy.proxy_server.general_settings", {}):
        assert (
            resolve_pass_through_request_timeout()
            == DEFAULT_PASS_THROUGH_REQUEST_TIMEOUT_SECONDS
        )


def test_resolve_llm_passthrough_timeout_precedence():
    assert resolve_llm_passthrough_timeout(kwargs={"timeout": 45}) == 45.0
    assert (
        resolve_llm_passthrough_timeout(
            kwargs={"request_timeout": 30},
            litellm_params={"timeout": 60},
        )
        == 30.0
    )
    assert (
        resolve_llm_passthrough_timeout(
            litellm_params={"timeout": 90},
        )
        == 90.0
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"pass_through_request_timeout": 6},
    ):
        assert resolve_llm_passthrough_timeout() == 6.0


@pytest.mark.asyncio
async def test_pass_through_request_uses_resolved_timeout():
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            mock_proxy_logging.pre_call_hook = AsyncMock(
                side_effect=lambda **kwargs: kwargs["data"]
            )

            mock_client = MagicMock()
            mock_client.client = MagicMock()
            mock_client.client.request = AsyncMock(
                side_effect=httpx.HTTPError("Request failed")
            )
            mock_get_client.return_value = mock_client

            mock_request = MagicMock(spec=Request)
            mock_request.method = "POST"
            mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
            mock_request.headers = Headers({})
            mock_request.query_params = QueryParams({})

            mock_user_api_key_dict = MagicMock()

            with pytest.raises(Exception):
                await pass_through_request(
                    request=mock_request,
                    target="http://test.com",
                    custom_headers={},
                    user_api_key_dict=mock_user_api_key_dict,
                    timeout=1500,
                )

            mock_get_client.assert_called_once()
            assert mock_get_client.call_args[1]["params"]["timeout"] == 1500


@pytest.mark.asyncio
async def test_create_pass_through_route_forwards_timeout():
    unique_path = "/test/path/unique/timeout"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com",
        custom_headers={},
        _forward_headers=True,
        _merge_query_params=False,
        dependencies=[],
        timeout=1800,
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
        ) as mock_pass_through,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route"
        ) as mock_is_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
        ) as mock_get_registered,
    ):
        mock_pass_through.return_value = MagicMock()
        mock_is_registered.return_value = True
        mock_get_registered.return_value = None

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        await endpoint_func(
            request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
            fastapi_response=MagicMock(),
        )

        call_kwargs = mock_pass_through.call_args[1]
        assert call_kwargs["timeout"] == 1800


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
                        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                            return_value={"x-callback-test": "value"}
                        )

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
                        await pass_through_request(
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
async def test_pass_through_request_streaming_marks_logging_obj_as_stream():
    """
    Regression: a streaming pass-through request must flag its logging object as
    streaming (logging_obj.stream and model_call_details["stream"]) before the
    response is dispatched, so cost/success callbacks treat it as a stream and the
    streaming dedup guard fires instead of double-logging.
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughStreamingHandler.chunk_processor"
            ) as mock_chunk_processor:
                mock_proxy_logging.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-3", "stream": True}
                )
                mock_proxy_logging.post_call_failure_hook = AsyncMock()
                mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                    return_value={"x-callback-test": "value"}
                )

                upstream_response = MagicMock()
                upstream_response.status_code = 200
                upstream_response.headers = {}
                upstream_response.raise_for_status = MagicMock()

                async_client = MagicMock()
                async_client.build_request = MagicMock(return_value=MagicMock())
                async_client.send = AsyncMock(return_value=upstream_response)
                mock_get_client.return_value = MagicMock(client=async_client)

                async def _empty_chunks(*args, **kwargs):
                    return
                    yield  # pragma: no cover

                mock_chunk_processor.return_value = _empty_chunks()

                mock_request = MagicMock(spec=Request)
                mock_request.method = "POST"
                mock_request.url = "http://test-proxy.com/v1/messages"
                mock_request.body = AsyncMock(
                    return_value=b'{"model": "claude-3", "stream": true}'
                )
                mock_request.headers = Headers({})
                mock_request.query_params = QueryParams({})

                await pass_through_request(
                    request=mock_request,
                    target="http://target-api.com/v1/messages",
                    custom_headers={},
                    user_api_key_dict=MagicMock(),
                    stream=True,
                )

                async_client.send.assert_awaited_once()
                assert async_client.send.call_args.kwargs["stream"] is True

                mock_chunk_processor.assert_called_once()
                logging_obj = mock_chunk_processor.call_args.kwargs[
                    "litellm_logging_obj"
                ]
                assert logging_obj.stream is True
                assert logging_obj.model_call_details["stream"] is True


@pytest.mark.asyncio
async def test_pass_through_request_sse_response_marks_logging_obj_as_stream():
    """
    Regression: a request that is not flagged as streaming up front but whose
    upstream response comes back as an SSE stream (content-type text/event-stream)
    must still flag its logging object as streaming before dispatch. Otherwise the
    cost/success callbacks treat the assembled stream as a non-stream and the dedup
    guard never fires, double-logging the request.
    """
    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughStreamingHandler.chunk_processor"
            ) as mock_chunk_processor:
                mock_proxy_logging.pre_call_hook = AsyncMock(
                    return_value={"model": "claude-3"}
                )
                mock_proxy_logging.post_call_failure_hook = AsyncMock()
                mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                    return_value={"x-callback-test": "value"}
                )

                upstream_response = MagicMock()
                upstream_response.status_code = 200
                upstream_response.headers = {"content-type": "text/event-stream"}
                upstream_response.raise_for_status = MagicMock()

                async_client = MagicMock()
                async_client.build_request = MagicMock(return_value=MagicMock())
                async_client.send = AsyncMock(return_value=upstream_response)
                mock_get_client.return_value = MagicMock(client=async_client)

                async def _empty_chunks(*args, **kwargs):
                    return
                    yield  # pragma: no cover

                mock_chunk_processor.return_value = _empty_chunks()

                mock_request = MagicMock(spec=Request)
                mock_request.method = "POST"
                mock_request.url = "http://test-proxy.com/v1/messages"
                mock_request.body = AsyncMock(return_value=b'{"model": "claude-3"}')
                mock_request.headers = Headers({})
                mock_request.query_params = QueryParams({})

                await pass_through_request(
                    request=mock_request,
                    target="http://target-api.com/v1/messages",
                    custom_headers={},
                    user_api_key_dict=MagicMock(),
                    stream=False,
                )

                async_client.send.assert_awaited_once()

                mock_chunk_processor.assert_called_once()
                logging_obj = mock_chunk_processor.call_args.kwargs[
                    "litellm_logging_obj"
                ]
                assert logging_obj.stream is True
                assert logging_obj.model_call_details["stream"] is True


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
                data=test_endpoint,
                request=MagicMock(spec=Request),
                user_api_key_dict=mock_user_api_key_dict,
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
                request=MagicMock(spec=Request),
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
async def test_create_pass_through_endpoint_auth_true_enforces_allowlist():
    """
    Regression: a pass-through endpoint created through the management API with
    auth=true (the model default) must be treated as allowlist-enforced. The
    create path registers FastAPI routes with dependencies=None, so deriving
    enforcement from dependency metadata let a key with broad llm_api_routes
    access call the route without an allowed_passthrough_routes match.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_endpoints,
    )

    registry: dict = {}

    with (
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings"
        ) as mock_get_config,
        patch("litellm.proxy.proxy_server.update_config_general_settings"),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            registry,
        ),
    ):
        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=[]
        )

        # auth is not passed -> defaults to True on PassThroughGenericEndpoint
        endpoint = PassThroughGenericEndpoint(
            path="/secure-passthrough",
            target="http://example.com/api",
            methods=["POST"],
        )
        await create_pass_through_endpoints(
            data=endpoint,
            request=MagicMock(spec=Request),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        )

        assert any(value.get("auth") is True for value in registry.values())
        assert (
            RouteChecks.is_auth_enforced_pass_through_route(
                route="/secure-passthrough", method="POST"
            )
            is True
        )

        post_request = MagicMock(spec=Request)
        post_request.method = "POST"

        without_allowlist = UserAPIKeyAuth(
            user_id="u", allowed_routes=["llm_api_routes"]
        )
        with pytest.raises(HTTPException) as exc_info:
            RouteChecks.is_virtual_key_allowed_to_call_route(
                route="/secure-passthrough",
                valid_token=without_allowlist,
                request=post_request,
            )
        assert exc_info.value.status_code == 403
        assert "allowed_passthrough_routes" in exc_info.value.detail

        with_allowlist = UserAPIKeyAuth(
            user_id="u",
            allowed_routes=["llm_api_routes"],
            metadata={"allowed_passthrough_routes": ["/secure-passthrough"]},
        )
        assert (
            RouteChecks.is_virtual_key_allowed_to_call_route(
                route="/secure-passthrough",
                valid_token=with_allowlist,
                request=post_request,
            )
            is True
        )


@pytest.mark.asyncio
async def test_update_pass_through_endpoint_auth_true_enforces_allowlist():
    """
    Regression: editing a pass-through endpoint through the management API must
    keep an auth=true route allowlist-enforced. remove_endpoint_routes drops the
    old registry entry, so the re-registration has to record the auth flag.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    registry: dict = {}
    existing_endpoint_id = "edit-me-123"
    existing_endpoints = [
        {
            "id": existing_endpoint_id,
            "path": "/edited-passthrough",
            "target": "http://example.com/api",
            "auth": True,
            "methods": ["POST"],
        }
    ]

    with (
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings"
        ) as mock_get_config,
        patch("litellm.proxy.proxy_server.update_config_general_settings"),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            registry,
        ),
    ):
        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        update_data = PassThroughGenericEndpoint(
            path="/edited-passthrough",
            target="http://newapi.com/v2",
            methods=["POST"],
        )
        await update_pass_through_endpoints(
            endpoint_id=existing_endpoint_id,
            data=update_data,
            request=MagicMock(spec=Request),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        )

        assert (
            RouteChecks.is_auth_enforced_pass_through_route(
                route="/edited-passthrough", method="POST"
            )
            is True
        )

        post_request = MagicMock(spec=Request)
        post_request.method = "POST"

        without_allowlist = UserAPIKeyAuth(
            user_id="u", allowed_routes=["llm_api_routes"]
        )
        with pytest.raises(HTTPException) as exc_info:
            RouteChecks.is_virtual_key_allowed_to_call_route(
                route="/edited-passthrough",
                valid_token=without_allowlist,
                request=post_request,
            )
        assert exc_info.value.status_code == 403
        assert "allowed_passthrough_routes" in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_pass_through_endpoint_preserves_auth_false():
    """
    Regression: editing an unrelated field on an auth=false pass-through must not
    silently flip it to auth=true. auth defaults to True on the request model, so a
    naive exclude_none merge would overwrite the stored auth=false and start
    rejecting every team/key that lacks allowed_passthrough_routes.
    """
    from litellm.proxy._types import (
        ConfigFieldInfo,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        update_pass_through_endpoints,
    )

    registry: dict = {}
    existing_endpoint_id = "public-forwarder-123"
    existing_endpoints = [
        {
            "id": existing_endpoint_id,
            "path": "/public-passthrough",
            "target": "http://example.com/api",
            "auth": False,
            "methods": ["POST"],
        }
    ]

    with (
        patch(
            "litellm.proxy.proxy_server.get_config_general_settings"
        ) as mock_get_config,
        patch(
            "litellm.proxy.proxy_server.update_config_general_settings"
        ) as mock_update_config,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._registered_pass_through_routes",
            registry,
        ),
    ):
        mock_get_config.return_value = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=existing_endpoints
        )

        update_data = PassThroughGenericEndpoint(
            path="/public-passthrough",
            target="http://newapi.com/v2",
            methods=["POST"],
        )
        result = await update_pass_through_endpoints(
            endpoint_id=existing_endpoint_id,
            data=update_data,
            request=MagicMock(spec=Request),
            user_api_key_dict=MagicMock(spec=UserAPIKeyAuth),
        )

        assert result.endpoints[0].auth is False

        persisted = mock_update_config.call_args[1]["data"].field_value[0]
        assert persisted["auth"] is False

        assert (
            RouteChecks.is_auth_enforced_pass_through_route(
                route="/public-passthrough", method="POST"
            )
            is False
        )


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
                request=MagicMock(spec=Request),
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
async def test_get_pass_through_endpoints_includes_config_and_db():
    """
    Test that get_pass_through_endpoints returns both config-defined and DB endpoints,
    with correct is_from_config flag. Config-only endpoints have is_from_config=True,
    DB endpoints have is_from_config=False. When same path exists in both, DB overrides.
    """
    from litellm.proxy._types import (
        PassThroughEndpointResponse,
        PassThroughGenericEndpoint,
        UserAPIKeyAuth,
    )
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        get_pass_through_endpoints,
    )

    # Config-defined endpoints (from config file)
    config_endpoints = [
        {
            "path": "/v1/rerank",
            "target": "https://api.cohere.com/v1/rerank",
            "headers": {"content-type": "application/json"},
        },
        {
            "path": "/v1/config-only",
            "target": "https://config.example.com/api",
            "headers": {},
        },
    ]

    # DB endpoints (one overlaps with config path, one is DB-only)
    db_endpoints = [
        {
            "id": "db-endpoint-1",
            "path": "/v1/rerank",  # Same as config - DB should override
            "target": "https://db-override.com/v1/rerank",
            "headers": {},
            "include_subpath": False,
        },
        {
            "id": "db-endpoint-2",
            "path": "/db/only",
            "target": "https://db-only.example.com/api",
            "headers": {},
            "include_subpath": False,
        },
    ]

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        MagicMock(),
    ):
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._get_pass_through_endpoints_from_db",
            new_callable=AsyncMock,
        ) as mock_get_db:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints._get_pass_through_endpoints_from_config"
            ) as mock_get_config:
                db_objects = [
                    PassThroughGenericEndpoint(**ep, is_from_config=False)
                    for ep in db_endpoints
                ]
                config_objects = [
                    PassThroughGenericEndpoint(**ep, is_from_config=True)
                    for ep in config_endpoints
                ]
                mock_get_db.return_value = db_objects
                mock_get_config.return_value = config_objects

                mock_user = MagicMock(spec=UserAPIKeyAuth)

                result = await get_pass_through_endpoints(
                    endpoint_id=None,
                    user_api_key_dict=mock_user,
                    team_id=None,
                )

    assert isinstance(result, PassThroughEndpointResponse)
    # config_only: /v1/config-only (not in db_paths)
    # db: /v1/rerank (overrides config), /db/only
    # So we should have: /v1/config-only (from config) + /v1/rerank + /db/only (from db)
    assert len(result.endpoints) == 3

    # Check is_from_config values
    by_path = {ep.path: ep for ep in result.endpoints}
    assert by_path["/v1/config-only"].is_from_config is True
    assert by_path["/v1/rerank"].is_from_config is False  # DB overrides
    assert by_path["/db/only"].is_from_config is False

    # Verify DB override: /v1/rerank should have DB target
    assert by_path["/v1/rerank"].target == "https://db-override.com/v1/rerank"


def test_get_pass_through_endpoints_from_config_skips_malformed():
    """
    Test that _get_pass_through_endpoints_from_config skips malformed endpoints
    and returns only valid ones, without raising.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        _get_pass_through_endpoints_from_config,
    )

    # Mix of valid and malformed config endpoints
    config_passthrough_endpoints = [
        {"path": "/valid/1", "target": "https://valid1.example.com"},
        {},  # Missing required path and target
        {"path": "/missing-target"},  # Missing required target
        {"target": "https://example.com"},  # Missing required path
        {"path": "/valid/2", "target": "https://valid2.example.com", "headers": {}},
    ]

    with patch(
        "litellm.proxy.proxy_server.config_passthrough_endpoints",
        config_passthrough_endpoints,
    ):
        result = _get_pass_through_endpoints_from_config()

    # Only the 2 valid endpoints should be returned
    assert len(result) == 2
    paths = {ep.path for ep in result}
    assert "/valid/1" in paths
    assert "/valid/2" in paths
    for ep in result:
        assert ep.is_from_config is True


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
                        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                            return_value={"x-callback-test": "value"}
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
                        await pass_through_request(
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
                        assert call_kwargs.get("forward_multipart") is False

                        # Verify the target URL is correct
                        assert (
                            str(call_kwargs["url"])
                            == "https://krris-m2f9a9i7-eastus2.openai.azure.com/openai/assistants"
                        )

                        # Verify the request body is preserved
                        assert call_kwargs["_parsed_body"] == test_body


class _FakeManagedFilesHook:
    def __init__(self, file_row: SimpleNamespace):
        self._file_row = file_row

    async def get_unified_file_id(self, file_id: str, litellm_parent_otel_span=None) -> SimpleNamespace:
        return self._file_row


async def _run_pass_through_and_capture_wire_url(
    target: str,
    incoming_query: str,
    merge_query_params: bool = False,
    default_query_params: Optional[dict] = None,
    custom_llm_provider: Optional[str] = None,
    managed_files_hook: Optional[_FakeManagedFilesHook] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
) -> httpx.URL:
    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    recorded_requests = []

    def transport_handler(upstream_request: httpx.Request) -> httpx.Response:
        recorded_requests.append(upstream_request)
        return httpx.Response(200, json={"ok": True})

    real_handler = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(None)},
    )
    cache_dict = litellm.in_memory_llm_clients_cache.cache_dict
    cache_key = next((key for key, cached in cache_dict.items() if cached is real_handler), None)
    assert cache_key is not None, (
        "PassThroughEndpoint client not found in in_memory_llm_clients_cache; "
        "get_async_httpx_client may not be caching this provider."
    )
    cache_dict[cache_key] = SimpleNamespace(
        client=httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    )

    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.headers = Headers({})
    mock_request.query_params = QueryParams(incoming_query)
    mock_request.body = AsyncMock(return_value=b"")

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(
        side_effect=lambda user_api_key_dict, data, call_type: data
    )
    mock_proxy_logging.post_call_failure_hook = AsyncMock()
    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})
    mock_proxy_logging.get_proxy_hook = MagicMock(return_value=managed_files_hook)

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging)
            )
            if managed_files_hook is not None:
                stack.enter_context(
                    patch(
                        "litellm.proxy.proxy_server.general_settings",
                        {"passthrough_managed_object_ids": True},
                    )
                )
                stack.enter_context(patch("litellm.proxy.proxy_server.prisma_client", None))
            response = await pass_through_request(
                request=mock_request,
                target=target,
                custom_headers={},
                user_api_key_dict=user_api_key_dict if user_api_key_dict is not None else MagicMock(),
                merge_query_params=merge_query_params,
                default_query_params=default_query_params,
                custom_llm_provider=custom_llm_provider,
            )
    finally:
        cache_dict[cache_key] = real_handler

    assert response.status_code == 200
    assert len(recorded_requests) == 1
    return recorded_requests[0].url


@pytest.mark.asyncio
async def test_pass_through_request_merge_query_params_preserves_target_query_on_wire():
    """
    Regression test: with merge_query_params=True, the target URL's own query
    params must survive on the final outgoing request. Passing the incoming
    params via httpx's params= replaces the URL's entire query string, which
    used to silently drop the merged target params.
    """
    wire_url = await _run_pass_through_and_capture_wire_url(
        target="https://www.bing.com/search?setLang=en-US&mkt=en-US",
        incoming_query="q=litellm",
        merge_query_params=True,
    )
    assert dict(wire_url.params) == {
        "setLang": "en-US",
        "mkt": "en-US",
        "q": "litellm",
    }


@pytest.mark.asyncio
async def test_pass_through_request_default_query_params_reach_the_wire():
    """
    default_query_params are sent with every request and can be overridden
    per-key by client-provided query params; params the client does not
    override must not be dropped from the outgoing request.
    """
    wire_url = await _run_pass_through_and_capture_wire_url(
        target="https://example.com/api",
        incoming_query="limit=5&api-version=client-version",
        default_query_params={"api-version": "2024-01-01", "setLang": "en-US"},
    )
    assert dict(wire_url.params) == {
        "api-version": "client-version",
        "setLang": "en-US",
        "limit": "5",
    }


@pytest.mark.asyncio
async def test_pass_through_request_without_merge_replaces_target_query():
    wire_url = await _run_pass_through_and_capture_wire_url(
        target="https://www.bing.com/search?setLang=en-US",
        incoming_query="q=litellm",
    )
    assert dict(wire_url.params) == {"q": "litellm"}


@pytest.mark.asyncio
async def test_pass_through_request_merge_query_params_rewrites_managed_ids_on_the_wire():
    """
    Regression test: on merge-enabled endpoints the managed-ID rewrite must see
    the incoming query params before they are folded into the URL. Folding
    first bakes the un-rewritten managed ID into the URL and hands the rewriter
    None, leaking the managed ID upstream.
    """
    from litellm.proxy.pass_through_endpoints.managed_id_codec import new_managed_id

    managed_id = new_managed_id("openai", "file-raw-123")
    hook = _FakeManagedFilesHook(SimpleNamespace(created_by="user-1", team_id=None))
    wire_url = await _run_pass_through_and_capture_wire_url(
        target="https://api.openai.com/v1/files/content?api-version=preview",
        incoming_query=f"file_id={managed_id}",
        merge_query_params=True,
        custom_llm_provider="openai",
        managed_files_hook=hook,
        user_api_key_dict=UserAPIKeyAuth(user_id="user-1"),
    )
    assert dict(wire_url.params) == {
        "api-version": "preview",
        "file_id": "file-raw-123",
    }


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
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

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

    # Verify headers are added to litellm_metadata for guardrails.
    # Bedrock passthrough uses litellm_metadata to prevent key-level
    # tags from leaking into the provider payload (GH#30629).
    assert "litellm_metadata" in result, "litellm_metadata should be present in result"
    assert (
        "headers" in result["litellm_metadata"]
    ), "headers should be present in litellm_metadata"
    assert isinstance(
        result["litellm_metadata"]["headers"], dict
    ), "headers should be a dictionary"

    # Verify specific headers are accessible (important for guardrails)
    headers = result["litellm_metadata"]["headers"]
    assert (
        "user-agent" in headers or "User-Agent" in headers
    ), "User-Agent header should be accessible in metadata"

    # Also verify proxy_server_request has headers (original location)
    assert "proxy_server_request" in result
    assert "headers" in result["proxy_server_request"]


@pytest.mark.asyncio
async def test_create_pass_through_route_custom_body_url_target():
    """
    Test that programmatic callers (e.g. Bedrock proxy) can attach a JSON body via
    request.state[LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY]; it is forwarded to
    pass_through_request and takes precedence over the request-parsed body.

    We cannot use a `custom_body: dict` route parameter: FastAPI would treat it as
    the HTTP body and reject multipart/form-data before the handler runs.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    unique_path = "/test/path/unique/custom_body_url"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="https://bedrock-agent-runtime.us-east-1.amazonaws.com",
        custom_headers=Headers(
            {
                "Authorization": "AWS4-HMAC-SHA256 signed",
                "Content-Type": "application/json",
            }
        ),
        _forward_headers=True,
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
        ) as mock_pass_through,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route"
        ) as mock_is_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
        ) as mock_get_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type"
        ) as mock_parse_request,
    ):
        mock_pass_through.return_value = MagicMock()
        mock_is_registered.return_value = True
        mock_get_registered.return_value = None
        # Simulate the request parser returning a different body
        mock_parse_request.return_value = (
            {},  # query_params_data
            {"parsed_from_request": True},  # custom_body_data (from request)
            None,  # file_data
            False,  # stream
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})
        mock_request.state = SimpleNamespace()

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        # The caller-supplied body (e.g. from bedrock_proxy_route)
        bedrock_body = {
            "retrievalQuery": {"text": "What is in the knowledge base?"},
        }

        setattr(
            mock_request.state, LITELLM_PASS_THROUGH_CUSTOM_BODY_STATE_KEY, bedrock_body
        )

        await endpoint_func(
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]

        # The critical assertion: custom_body takes precedence over
        # the body parsed from the raw request
        assert call_kwargs["custom_body"] == bedrock_body
        # HeadersDict-like custom_headers (e.g. botocore SigV4) must be coerced
        # to a plain dict so signed headers actually reach the upstream.
        assert call_kwargs["custom_headers"] == {
            "authorization": "AWS4-HMAC-SHA256 signed",
            "content-type": "application/json",
        }


@pytest.mark.asyncio
async def test_pass_through_request_non_streaming_uses_content_for_state_raw_body():
    """
    Bedrock SigV4 path: exact signed bytes live on request.state; upstream must receive
    content=... even if pre_call_hook mutates the parsed dict (would change json=).
    """
    # Bytes that were signed (simulated); parsed body + hook will diverge on purpose.
    raw_signed = b'{"retrievalQuery":{"text":"signed"},"sig":"intact"}'
    parsed_from_wire = {"retrievalQuery": {"text": "signed"}, "sig": "intact"}

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.query_params = QueryParams({})
    mock_request.headers = Headers({"Content-Type": "application/json"})
    mock_request.state = SimpleNamespace()
    setattr(mock_request.state, LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY, raw_signed)
    mock_request.body = AsyncMock(
        return_value=json.dumps(parsed_from_wire).encode("utf-8")
    )

    mock_user = MagicMock()
    mock_user.api_key = "sk-test"

    upstream = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=b'{"ok": true}',
        request=httpx.Request(
            "POST",
            "https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/KB/retrieve",
        ),
    )

    mock_async_client = AsyncMock()
    mock_async_client.build_request = MagicMock(return_value=MagicMock())
    mock_async_client.send = AsyncMock(return_value=upstream)
    mock_client_obj = MagicMock()
    mock_client_obj.client = mock_async_client

    async def _hook_mutates_body(**kwargs):
        data = kwargs["data"]
        if isinstance(data, dict):
            data["hook_mutated"] = True
        return data

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=mock_client_obj,
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=_hook_mutates_body),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ),
    ):
        await pass_through_request(
            request=mock_request,
            target="https://bedrock-agent-runtime.us-east-1.amazonaws.com/knowledgebases/KB/retrieve",
            custom_headers={"content-type": "application/json"},
            user_api_key_dict=mock_user,
            stream=False,
        )

    mock_async_client.build_request.assert_called_once()
    build_kw = mock_async_client.build_request.call_args[1]
    assert build_kw.get("content") == raw_signed
    assert "json" not in build_kw
    mock_async_client.send.assert_awaited_once()
    assert mock_async_client.send.call_args.kwargs.get("stream") is True


@pytest.mark.asyncio
async def test_pass_through_request_streaming_uses_content_for_state_raw_body():
    """Streaming pass-through with state raw body must use build_request(..., content=...)."""
    raw_signed = b'{"model":"m","stream":true}'
    parsed_from_wire = {"model": "m", "stream": True}

    mock_request = MagicMock(spec=Request)
    mock_request.method = "POST"
    mock_request.query_params = QueryParams({})
    mock_request.headers = Headers({"Content-Type": "application/json"})
    mock_request.state = SimpleNamespace()
    setattr(mock_request.state, LITELLM_PASS_THROUGH_RAW_BODY_STATE_KEY, raw_signed)
    mock_request.body = AsyncMock(
        return_value=json.dumps(parsed_from_wire).encode("utf-8")
    )

    mock_user = MagicMock()
    mock_user.api_key = "sk-test"

    mock_built = MagicMock()
    mock_async_client = AsyncMock()
    mock_async_client.build_request = MagicMock(return_value=mock_built)
    stream_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"data: {}\n\n",
        request=httpx.Request("POST", "https://example.com/v1/messages"),
    )
    mock_async_client.send = AsyncMock(return_value=stream_resp)
    mock_client_obj = MagicMock()
    mock_client_obj.client = mock_async_client

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
            return_value=mock_client_obj,
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
            new=AsyncMock(side_effect=lambda **kw: kw["data"]),
        ),
        patch(
            "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
            new=AsyncMock(),
        ),
    ):
        response = await pass_through_request(
            request=mock_request,
            target="https://example.com/v1/messages",
            custom_headers={"Authorization": "Bearer x"},
            user_api_key_dict=mock_user,
            stream=None,
        )

    from fastapi.responses import StreamingResponse

    assert isinstance(response, StreamingResponse)
    mock_async_client.build_request.assert_called_once()
    br_kw = mock_async_client.build_request.call_args[1]
    assert br_kw.get("content") == raw_signed
    assert "json" not in br_kw


@pytest.mark.asyncio
async def test_create_pass_through_route_no_custom_body_falls_back():
    """
    Test that the URL-based endpoint_func falls back to the request-parsed body
    when custom_body is not provided.

    This ensures the default pass-through behavior is preserved — only the
    Bedrock proxy route (and similar callers) supply a pre-built body.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        create_pass_through_route,
    )

    unique_path = "/test/path/unique/no_custom_body"
    endpoint_func = create_pass_through_route(
        endpoint=unique_path,
        target="http://example.com/api",
        custom_headers={},
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_request"
        ) as mock_pass_through,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route"
        ) as mock_is_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route"
        ) as mock_get_registered,
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints._parse_request_data_by_content_type"
        ) as mock_parse_request,
    ):
        mock_pass_through.return_value = MagicMock()
        mock_is_registered.return_value = True
        mock_get_registered.return_value = None
        request_parsed_body = {"key": "from_request"}
        mock_parse_request.return_value = (
            {},  # query_params_data
            request_parsed_body,  # custom_body_data
            None,  # file_data
            False,  # stream
        )

        mock_request = MagicMock(spec=Request)
        mock_request.url = MagicMock()
        mock_request.url.path = unique_path
        mock_request.path_params = {}
        mock_request.query_params = QueryParams({})
        mock_request.state = SimpleNamespace()

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"

        # Call without state body — should use the request-parsed body
        await endpoint_func(
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=mock_user_api_key_dict,
        )

        mock_pass_through.assert_called_once()
        call_kwargs = mock_pass_through.call_args[1]

        # Should fall back to the body parsed from the request
        assert call_kwargs["custom_body"] == request_parsed_body


def test_is_registered_pass_through_route_with_custom_root():
    """
    Registry stores bare paths; incoming routes may be bare (get_request_route)
    or prefixed (request.url.path). Both should resolve via normalization.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        _registered_pass_through_routes,
    )

    # Clear the registry first
    _registered_pass_through_routes.clear()

    # Register a pass-through route with endpoint format: {endpoint_id}:exact:{path}
    endpoint_id = "test-endpoint-123"
    path = "/api/endpoint"
    route_key = f"{endpoint_id}:exact:{path}"
    _registered_pass_through_routes[route_key] = {
        "target": "http://example.com",
        "headers": {},
    }

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/proxy"):
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/proxy/api/endpoint"
            )
            is True
        )
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/api/endpoint"
            )
            is True
        )

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"):
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/api/endpoint"
            )
            is True
        )
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/proxy/api/endpoint"
            )
            is False
        )

    # Clean up
    _registered_pass_through_routes.clear()


def test_get_registered_pass_through_route_with_custom_root():
    """
    get_registered_pass_through_route matches bare registry paths against
    bare or SERVER_ROOT_PATH-prefixed incoming routes.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        _registered_pass_through_routes,
    )

    # Clear the registry first
    _registered_pass_through_routes.clear()

    # Register a pass-through route
    endpoint_id = "test-endpoint-456"
    path = "/chat/completions"
    target_config = {
        "target": "http://api.example.com/v1/chat/completions",
        "headers": {"Authorization": "Bearer token123"},
        "forward_headers": True,
    }
    route_key = f"{endpoint_id}:exact:{path}"
    _registered_pass_through_routes[route_key] = target_config

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"):
        # Prefixed incoming route
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/litellm/chat/completions"
        )
        assert result is not None
        assert result["target"] == "http://api.example.com/v1/chat/completions"
        assert result["headers"]["Authorization"] == "Bearer token123"

        # Bare incoming route (get_request_route convention)
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/chat/completions"
        )
        assert result is not None
        assert result["target"] == "http://api.example.com/v1/chat/completions"

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/"):
        result = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
            "/chat/completions"
        )
        assert result is not None
        assert result["target"] == "http://api.example.com/v1/chat/completions"

    # Clean up
    _registered_pass_through_routes.clear()


@pytest.mark.parametrize(
    "server_root_path,route_type,incoming_route,should_match",
    [
        ("", "subpath", "/ml/api/v1/time-series-forecast/predict", True),
        ("", "exact", "/ml", True),
        ("", "exact", "/ml/extra", False),
        ("/llmproxy", "subpath", "/ml/api/v1/time-series-forecast/predict", True),
        (
            "/llmproxy",
            "subpath",
            "/llmproxy/ml/api/v1/time-series-forecast/predict",
            True,
        ),
        ("/llmproxy", "exact", "/ml", True),
        ("/llmproxy", "exact", "/llmproxy/ml", True),
        ("/llmproxy", "subpath", "/other/api", False),
    ],
)
def test_db_registered_pass_through_route_bare_path_convention(
    server_root_path, route_type, incoming_route, should_match
):
    """
    Regression: #28547 / SERVER_ROOT_PATH — registry stores bare /ml paths;
    get_request_route() supplies bare paths; prefixed url.path must still match.
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
        _registered_pass_through_routes,
    )

    _registered_pass_through_routes.clear()
    endpoint_id = "customer-ml"
    path = "/ml"
    route_key = f"{endpoint_id}:{route_type}:{path}:GET,POST"
    _registered_pass_through_routes[route_key] = {
        "endpoint_id": endpoint_id,
        "path": path,
        "type": route_type,
        "target": "https://example.com",
        "methods": ["GET", "POST"],
    }

    with patch(
        "litellm.proxy.utils.get_server_root_path",
        return_value=server_root_path,
    ):
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                incoming_route
            )
            is should_match
        )

    _registered_pass_through_routes.clear()


def test_mapped_pass_through_routes_with_server_root_path():
    """
    Mapped passthrough routes (vertex_ai, bedrock, etc) should match
    even when SERVER_ROOT_PATH is set and the incoming route is prefixed.

    Regression test for https://github.com/BerriAI/litellm/issues/22272
    """
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
        InitPassThroughEndpointHelpers,
    )

    with patch("litellm.proxy.utils.get_server_root_path", return_value="/litellm"):
        # prefixed route should match mapped routes like /vertex_ai
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/litellm/vertex_ai/v1/projects/foo"
            )
            is True
        )
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/litellm/bedrock/model/invoke"
            )
            is True
        )

        # bare route without prefix should not match when root is set
        assert (
            InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                "/vertex_ai/v1/projects/foo"
            )
            is False
        )


@pytest.mark.asyncio
async def test_multipart_passthrough_preserves_boundary():
    """
    Test that multipart/form-data requests through passthrough preserve the boundary
    and can be correctly parsed by the upstream server.

    Regression test for multipart boundary stripping issue.
    """
    from io import BytesIO

    # Mock the httpx request to verify files are passed correctly
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({"content-type": "application/json"})
    mock_response.aread = AsyncMock(
        return_value=b'{"filename": "test.txt", "size": 17}'
    )
    mock_response.text = '{"filename": "test.txt", "size": 17}'

    async def mock_httpx_request(method, url, **kwargs):
        # Verify that files parameter is passed (not json)
        assert "files" in kwargs, "Files should be passed for multipart requests"
        file_parts = [
            value for name, value in kwargs["files"] if name == "file"
        ]
        assert len(file_parts) == 1, "File field should be in files"

        # Verify content-type is NOT in headers (httpx will set it with correct boundary)
        headers = kwargs.get("headers", {})
        assert (
            "content-type" not in headers
        ), "content-type should be removed for multipart"

        filename, content, content_type = file_parts[0]
        assert filename == "test.txt"
        assert content == b"test file content"
        assert content_type == "text/plain"

        return mock_response

    async_client = MagicMock()
    async_client.request = AsyncMock(side_effect=mock_httpx_request)

    # Create mock request
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.headers = Headers({"content-type": "multipart/form-data; boundary=test123"})

    # Mock form data
    file_content = b"test file content"
    file = BytesIO(file_content)
    headers = Headers({"content-type": "text/plain"})
    upload_file = UploadFile(file=file, filename="test.txt", headers=headers)
    upload_file.read = AsyncMock(return_value=file_content)

    form_data = FormData([("file", upload_file)])
    request.form = AsyncMock(return_value=form_data)

    # Test the multipart handler directly
    response = await HttpPassThroughEndpointHelpers.make_multipart_http_request(
        request=request,
        async_client=async_client,
        url=httpx.URL("http://test.com/upload"),
        headers={},
        requested_query_params=None,
    )

    # Verify the response
    assert response.status_code == 200
    async_client.request.assert_called_once()


def test_get_response_headers_strips_server_and_date():
    """Regression: forwarding the upstream's Server/Date headers causes
    uvicorn to add its own and strict HTTP parsers (aiohttp) reject the
    response with 'Duplicate Server header found'. The helper must strip
    headers that the ASGI server writes itself."""
    upstream_headers = httpx.Headers(
        {
            "server": "cloudflare",
            "date": "Fri, 24 Apr 2026 23:26:19 GMT",
            "content-type": "application/json",
            "content-length": "123",
            "transfer-encoding": "chunked",
            "content-encoding": "gzip",
            "connection": "keep-alive",
            "keep-alive": "timeout=5",
            "x-request-id": "req_abc",
            "anthropic-ratelimit-requests-remaining": "100",
        }
    )

    result = HttpPassThroughEndpointHelpers.get_response_headers(upstream_headers)

    lowered_keys = {k.lower() for k in result}
    for stripped in (
        "server",
        "date",
        "content-length",
        "transfer-encoding",
        "content-encoding",
        "connection",
        "keep-alive",
    ):
        assert (
            stripped not in lowered_keys
        ), f"{stripped!r} must not be forwarded by passthrough"

    # Application/business headers must still pass through.
    lowered = {k.lower(): v for k, v in result.items()}
    assert lowered["content-type"] == "application/json"
    assert lowered["x-request-id"] == "req_abc"
    assert lowered["anthropic-ratelimit-requests-remaining"] == "100"


class TestStaleRouteCleanupOnReload:
    """Regression tests for the PERF-13 / issue #19921 reload leak.

    ``initialize_pass_through_endpoints`` is re-run every 30s by the
    ``add_deployment_job`` scheduler. Endpoints sourced from the DB/config
    without a persisted ``id`` get a fresh UUID each cycle, so their route key
    ("{id}:{type}:{path}:{methods}") changes every reload. The old cleanup
    called ``remove_endpoint_routes(route_key)`` which matches on ``endpoint_id``
    and therefore never matched a route key, so ``_registered_pass_through_routes``
    grew without bound. That unbounded dict turned the O(n) per-cycle cleanup and
    the per-request ``is_registered_pass_through_route`` scan into a CPU sink.
    """

    def setup_method(self):
        _registered_pass_through_routes.clear()

    def teardown_method(self):
        _registered_pass_through_routes.clear()

    @staticmethod
    def _patches():
        stack = ExitStack()
        stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.SafeRouteAdder.add_api_route_if_not_exists"
            )
        )
        stack.enter_context(patch("litellm.proxy.proxy_server.premium_user", True))
        mock_set_env = stack.enter_context(
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.set_env_variables_in_header"
            )
        )
        mock_set_env.return_value = {}
        return stack

    @staticmethod
    def _paths_in_registry():
        return sorted(v["path"] for v in _registered_pass_through_routes.values())

    @pytest.mark.asyncio
    async def test_registry_stays_bounded_across_reloads_for_idless_endpoint(self):
        """A DB/config endpoint with no id must not grow the registry per reload.

        Mutation check: with the old ``remove_endpoint_routes`` call this asserts
        2 but the registry holds ``2 * num_cycles`` entries, so it fails.
        """
        num_cycles = 25
        with self._patches():
            for _ in range(num_cycles):
                # Fresh dict each cycle mirrors the DB loader rebuilding objects;
                # a reused dict would cache the minted id and hide the bug.
                await initialize_pass_through_endpoints(
                    [
                        {
                            "path": "/vertex-passthrough",
                            "target": "http://example.com",
                            "include_subpath": True,
                        }
                    ]
                )

        assert len(_registered_pass_through_routes) == 2
        assert self._paths_in_registry() == [
            "/vertex-passthrough",
            "/vertex-passthrough",
        ]

    @pytest.mark.asyncio
    async def test_departed_endpoint_is_removed_on_next_reload(self):
        """A route present in one cycle but absent the next is dropped.

        Mutation check: the old cleanup leaves the departed ``/a`` key behind,
        so the registry would hold both paths instead of only ``/b``.
        """
        with self._patches():
            await initialize_pass_through_endpoints(
                [{"path": "/a", "target": "http://example.com"}]
            )
            assert self._paths_in_registry() == ["/a"]

            await initialize_pass_through_endpoints(
                [{"path": "/b", "target": "http://example.com"}]
            )

        assert self._paths_in_registry() == ["/b"]

    @pytest.mark.asyncio
    async def test_live_route_survives_reload_and_stays_resolvable(self):
        """The currently-registered route must remain after the stale-key sweep.

        Guards against a cleanup that over-removes (e.g. stripping the shared
        path of the freshly re-registered endpoint).
        """
        with self._patches():
            for _ in range(3):
                await initialize_pass_through_endpoints(
                    [
                        {
                            "path": "/live-passthrough",
                            "target": "http://example.com",
                            "include_subpath": True,
                        }
                    ]
                )

        assert InitPassThroughEndpointHelpers.is_registered_pass_through_route(
            "/live-passthrough"
        )
        assert InitPassThroughEndpointHelpers.is_registered_pass_through_route(
            "/live-passthrough/some/subpath"
        )


# Regression (LIT-3538): a pre-call guardrail block on a passthrough endpoint
# must be logged at WARNING without a traceback, not as an ERROR with a full
# stack trace. The generic ``except Exception`` in ``pass_through_request`` used
# to call ``verbose_proxy_logger.exception(...)`` for every exception, so an
# intentional guardrail block (which the rest of the codebase already classifies
# via ``CustomGuardrail._is_guardrail_intervention``) produced scary error noise
# even though the client correctly receives the 4xx.
from fastapi import HTTPException as _FastAPIHTTPException

from litellm.exceptions import (
    BlockedPiiEntityError,
    GuardrailRaisedException,
)

_PT_MODULE = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"


def _lit3538_user_api_key_dict():
    d = MagicMock()
    d.api_key = "sk-test"
    d.user_id = "user-1"
    d.team_id = "team-1"
    d.org_id = None
    d.metadata = {}
    d.team_metadata = {}
    d.parent_otel_span = None
    d.request_route = "/mock/echo"
    return d


def _lit3538_request():
    r = MagicMock()
    r.method = "POST"
    r.query_params = {}
    r.url = "http://testserver/mock/echo"
    r.state = SimpleNamespace()
    headers = MagicMock()
    headers.copy.return_value = {}
    r.headers = headers
    return r


async def _drive_pass_through_block(raised_exception):
    """Drive the real ``pass_through_request`` so its pre_call_hook raises
    ``raised_exception``, returning (status_code, logger_mock)."""
    proxy_logging = MagicMock()
    proxy_logging.pre_call_hook = AsyncMock(side_effect=raised_exception)
    proxy_logging.post_call_failure_hook = AsyncMock()
    proxy_logging.get_proxy_hook = MagicMock(return_value=None)

    logger = MagicMock()

    patches = [
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
        patch(f"{_PT_MODULE}.verbose_proxy_logger", logger),
        patch(
            f"{_PT_MODULE}._read_request_body",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(f"{_PT_MODULE}._safe_get_request_headers", return_value={}),
        patch(
            "litellm.proxy.pass_through_endpoints.passthrough_guardrails."
            "PassthroughGuardrailHandler.collect_guardrails",
            return_value=[],
        ),
    ]

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        status_code = None
        try:
            await pass_through_request(
                request=_lit3538_request(),
                target="https://upstream.example/echo",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_lit3538_user_api_key_dict(),
                stream=False,
            )
        except Exception as e:  # ProxyException carrying the original status
            status_code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return status_code, logger


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guardrail_exception, expected_code",
    [
        (
            GuardrailRaisedException(guardrail_name="g", message="blocked"),
            400,
        ),
        (
            BlockedPiiEntityError(entity_type="EMAIL", guardrail_name="presidio"),
            400,
        ),
        (
            _FastAPIHTTPException(
                status_code=400, detail={"error": "Violated moderation policy"}
            ),
            400,
        ),
    ],
)
async def test_pre_call_guardrail_block_logs_warning_not_exception(
    guardrail_exception, expected_code
):
    status_code, logger = await _drive_pass_through_block(guardrail_exception)

    assert int(status_code) == expected_code
    assert (
        logger.exception.call_count == 0
    ), "guardrail block must not be logged as an ERROR with a traceback"
    assert (
        logger.warning.call_count == 1
    ), "guardrail block must be logged once at WARNING"


@pytest.mark.asyncio
async def test_non_guardrail_exception_still_logs_with_traceback():
    status_code, logger = await _drive_pass_through_block(
        RuntimeError("upstream connection reset")
    )

    assert int(status_code) == 500
    assert (
        logger.exception.call_count == 1
    ), "a genuine failure must still be logged via verbose_proxy_logger.exception"
    assert (
        logger.warning.call_count == 0
    ), "a genuine failure must not be downgraded to WARNING"


# Regression: generic config-based passthrough (`pass_through_request`) used to
# call `response.raise_for_status()` on upstream errors and re-raise as an
# `HTTPException`, which the outer `except` block then reshaped into a
# `ProxyException` (`{"error": {"message": "<stringified upstream body>", ...}}`).
# Upstream error responses must reach the client byte-for-byte, with the
# original status code, exactly like success responses already do.
_UPSTREAM_ERROR_BODY = {
    "error": "Permission denied",
    "error_code": "ACCESS_DENIED",
    "request_id": "req_mock_403",
    "trace_id": "trace_mock_403",
}


@pytest.mark.asyncio
async def test_pass_through_request_non_streaming_upstream_error_returned_unchanged():
    upstream_content = json.dumps(_UPSTREAM_ERROR_BODY).encode("utf-8")
    upstream_response = httpx.Response(
        status_code=403,
        headers={"content-type": "application/json"},
        content=upstream_content,
        request=httpx.Request("POST", "http://target-api.com/api/denied"),
    )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
                    mock_proxy_logging.post_call_failure_hook = AsyncMock()
                    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                        return_value=None
                    )
                    mock_processing.get_custom_headers.return_value = {}
                    mock_success_handler.return_value = None

                    async_client = MagicMock()
                    async_client.build_request = MagicMock(return_value=MagicMock())
                    async_client.send = AsyncMock(return_value=upstream_response)
                    mock_get_client.return_value = MagicMock(client=async_client)

                    mock_request = MagicMock(spec=Request)
                    mock_request.method = "POST"
                    mock_request.url = "http://test-proxy.com/mock-upstream/api/denied"
                    mock_request.body = AsyncMock(return_value=b'{"action": "read"}')
                    mock_request.headers = Headers({"content-type": "application/json"})
                    mock_request.query_params = QueryParams({})

                    response = await pass_through_request(
                        request=mock_request,
                        target="http://target-api.com/api/denied",
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                    )
                    await asyncio.sleep(0)

    assert response.status_code == 403
    body = json.loads(response.body)
    # Exact dict equality proves the upstream body was forwarded verbatim,
    # not stringified into a ProxyException's `error.message` field.
    assert body == _UPSTREAM_ERROR_BODY
    assert set(body.keys()) != {"error"} or not isinstance(body["error"], dict)

    # Regression: the success handler has no status-code awareness, so it must
    # never be called for a 4xx/5xx upstream response - otherwise the same
    # request gets recorded as both a failure and a success in SpendLogs.
    mock_success_handler.assert_not_called()

    # Regression: post_call_failure_hook (spend-tracking, alerting callbacks)
    # must still fire for upstream errors even though the client-facing
    # response is unchanged and no ProxyException is raised.
    from fastapi import HTTPException

    mock_proxy_logging.post_call_failure_hook.assert_called_once()
    failure_call_kwargs = mock_proxy_logging.post_call_failure_hook.call_args.kwargs
    # Must be reported as HTTPException, not the raw httpx error: ProxyLogging's
    # alerting only excludes HTTPException/ProxyException from its "High"
    # severity llm_exceptions alert, so a raw HTTPStatusError here would page
    # ops for every routine upstream 4xx returned through passthrough.
    assert isinstance(failure_call_kwargs["original_exception"], HTTPException)
    assert failure_call_kwargs["original_exception"].status_code == 403

    # Regression: the failure-hook log payload's response_body must reflect
    # the upstream error JSON, not None, so downstream spend-tracking/logging
    # integrations can see what the upstream actually returned.
    assert failure_call_kwargs["request_data"]["response_body"] == _UPSTREAM_ERROR_BODY


@pytest.mark.asyncio
async def test_pass_through_request_upstream_error_failure_hook_exception_is_swallowed():
    """
    A broken failure-hook callback (e.g. a misconfigured alerting integration)
    must never take down the passthrough response - the upstream error body
    must still reach the client unchanged, and the callback's exception must
    only be logged, not raised.
    """
    upstream_content = json.dumps(_UPSTREAM_ERROR_BODY).encode("utf-8")
    upstream_response = httpx.Response(
        status_code=403,
        headers={"content-type": "application/json"},
        content=upstream_content,
        request=httpx.Request("POST", "http://target-api.com/api/denied"),
    )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
                    mock_proxy_logging.post_call_failure_hook = AsyncMock(
                        side_effect=RuntimeError("alerting integration misconfigured")
                    )
                    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                        return_value=None
                    )
                    mock_processing.get_custom_headers.return_value = {}
                    mock_success_handler.return_value = None

                    async_client = MagicMock()
                    async_client.build_request = MagicMock(return_value=MagicMock())
                    async_client.send = AsyncMock(return_value=upstream_response)
                    mock_get_client.return_value = MagicMock(client=async_client)

                    mock_request = MagicMock(spec=Request)
                    mock_request.method = "POST"
                    mock_request.url = "http://test-proxy.com/mock-upstream/api/denied"
                    mock_request.body = AsyncMock(return_value=b'{"action": "read"}')
                    mock_request.headers = Headers({"content-type": "application/json"})
                    mock_request.query_params = QueryParams({})

                    response = await pass_through_request(
                        request=mock_request,
                        target="http://target-api.com/api/denied",
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                    )
                    await asyncio.sleep(0)

    mock_proxy_logging.post_call_failure_hook.assert_called_once()
    assert response.status_code == 403
    assert json.loads(response.body) == _UPSTREAM_ERROR_BODY


@pytest.mark.asyncio
async def test_pass_through_request_streaming_upstream_error_returned_unchanged():
    from fastapi.responses import StreamingResponse

    upstream_content = json.dumps(_UPSTREAM_ERROR_BODY).encode("utf-8")
    upstream_response = httpx.Response(
        status_code=403,
        headers={"content-type": "application/json"},
        content=upstream_content,
        request=httpx.Request("GET", "http://target-api.com/api/stream-denied"),
    )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
            ) as mock_success_handler:
                mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
                mock_proxy_logging.post_call_failure_hook = AsyncMock()
                mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                    return_value=None
                )
                mock_success_handler.return_value = None

                async_client = MagicMock()
                async_client.build_request = MagicMock(return_value=MagicMock())
                async_client.send = AsyncMock(return_value=upstream_response)
                mock_get_client.return_value = MagicMock(client=async_client)

                mock_request = MagicMock(spec=Request)
                mock_request.method = "GET"
                mock_request.url = "http://test-proxy.com/mock-upstream/api/stream-denied"
                mock_request.body = AsyncMock(return_value=b"")
                mock_request.headers = Headers({})
                mock_request.query_params = QueryParams({})

                response = await pass_through_request(
                    request=mock_request,
                    target="http://target-api.com/api/stream-denied",
                    custom_headers={},
                    user_api_key_dict=MagicMock(),
                    stream=True,
                )

    assert isinstance(response, StreamingResponse)
    assert response.status_code == 403

    streamed_chunks = [chunk async for chunk in response.body_iterator]
    await asyncio.sleep(0)
    streamed_bytes = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
        for chunk in streamed_chunks
    )
    assert streamed_bytes == upstream_content
    assert json.loads(streamed_bytes) == _UPSTREAM_ERROR_BODY

    # Regression: chunk_processor's end-of-stream success logging has no
    # status-code awareness, so it must never fire for a 4xx/5xx upstream
    # response - otherwise the same request gets recorded as both a failure
    # (via the hook below) and a success in SpendLogs.
    mock_success_handler.assert_not_called()

    # Regression: post_call_failure_hook must still fire for streaming
    # upstream errors, mirroring the non-streaming behavior, and must also
    # report an HTTPException (not the raw httpx error) to avoid triggering
    # a "High" severity llm_exceptions alert for a routine upstream 4xx.
    from fastapi import HTTPException

    mock_proxy_logging.post_call_failure_hook.assert_called_once()
    failure_call_kwargs = mock_proxy_logging.post_call_failure_hook.call_args.kwargs
    assert isinstance(failure_call_kwargs["original_exception"], HTTPException)
    assert failure_call_kwargs["original_exception"].status_code == 403


@pytest.mark.asyncio
async def test_pass_through_request_non_streaming_success_unchanged():
    """Success (2xx) passthrough behavior must remain unchanged by the error fix."""
    upstream_success_body = {"status": "ok", "message": "mock upstream success"}
    upstream_content = json.dumps(upstream_success_body).encode("utf-8")
    upstream_response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=upstream_content,
        request=httpx.Request("GET", "http://target-api.com/api/success"),
    )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        with patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client"
        ) as mock_get_client:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.ProxyBaseLLMRequestProcessing"
            ) as mock_processing:
                with patch(
                    "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
                ) as mock_success_handler:
                    mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
                    mock_proxy_logging.post_call_failure_hook = AsyncMock()
                    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(
                        return_value=None
                    )
                    mock_processing.get_custom_headers.return_value = {}
                    mock_success_handler.return_value = None

                    async_client = MagicMock()
                    async_client.build_request = MagicMock(return_value=MagicMock())
                    async_client.send = AsyncMock(return_value=upstream_response)
                    mock_get_client.return_value = MagicMock(client=async_client)

                    mock_request = MagicMock(spec=Request)
                    mock_request.method = "GET"
                    mock_request.url = "http://test-proxy.com/mock-upstream/api/success"
                    mock_request.body = AsyncMock(return_value=b"")
                    mock_request.headers = Headers({})
                    mock_request.query_params = QueryParams({})

                    response = await pass_through_request(
                        request=mock_request,
                        target="http://target-api.com/api/success",
                        custom_headers={},
                        user_api_key_dict=MagicMock(),
                    )
                    await asyncio.sleep(0)

    assert response.status_code == 200
    assert json.loads(response.body) == upstream_success_body
    # Regression guard: the failure hook must only fire for upstream errors,
    # never for a successful upstream response.
    mock_proxy_logging.post_call_failure_hook.assert_not_called()
    # ...and the success handler must still fire exactly once for a 2xx,
    # proving the status_code gate doesn't also swallow real successes.
    mock_success_handler.assert_called_once()


@pytest.mark.asyncio
async def test_pass_through_request_internal_failure_still_raises_proxy_exception():
    """
    Internal proxy failures (e.g. a hook raising before any upstream request is
    made) must still surface as ProxyException, distinct from upstream
    passthrough errors which are now returned unchanged.
    """
    from litellm.proxy._types import ProxyException

    with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging:
        mock_proxy_logging.pre_call_hook = AsyncMock(
            side_effect=RuntimeError("auth backend unavailable")
        )
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = "http://test-proxy.com/mock-upstream/api/success"
        mock_request.body = AsyncMock(return_value=b"")
        mock_request.headers = Headers({})
        mock_request.query_params = QueryParams({})

        with pytest.raises(ProxyException) as exc_info:
            await pass_through_request(
                request=mock_request,
                target="http://target-api.com/api/success",
                custom_headers={},
                user_api_key_dict=MagicMock(),
            )

    assert int(exc_info.value.code) == 500
    assert "auth backend unavailable" in exc_info.value.message


class _RecordingUpstreamByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks
        self.chunks_served = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            self.chunks_served += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class _FakeUpstreamTransport(httpx.AsyncBaseTransport):
    def __init__(self, status_code, headers, stream):
        self._status_code = status_code
        self._headers = headers
        self._stream = stream

    async def handle_async_request(self, request):
        return httpx.Response(
            status_code=self._status_code,
            headers=self._headers,
            stream=self._stream,
            request=request,
        )


def _inject_fake_passthrough_client(transport, timeout):
    """Dependency-inject a fake upstream via the client cache that
    get_async_httpx_client resolves passthrough clients from (no monkeypatching
    of the HTTP layer). The cache entry is located by calling the production
    get_async_httpx_client and identity-scanning the cache for the handler it
    returned, so the internal cache-key format is never duplicated here. Must
    run inside the test's event loop because cache keys are loop-scoped.
    Returns (client, cleanup)."""
    import litellm
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    real_handler = get_async_httpx_client(
        httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": resolve_pass_through_request_timeout(timeout)},
    )
    cache = litellm.in_memory_llm_clients_cache
    cache_key = next(
        (key for key, cached in cache.cache_dict.items() if cached is real_handler),
        None,
    )
    assert cache_key is not None, (
        "PassThroughEndpoint client not found in in_memory_llm_clients_cache; "
        "get_async_httpx_client may not be caching this provider."
    )
    fake_client = httpx.AsyncClient(transport=transport)
    cache.cache_dict[cache_key] = SimpleNamespace(client=fake_client)

    def _cleanup():
        cache.cache_dict.pop(cache_key, None)

    return fake_client, _cleanup


def _enter_relay_logging_mocks(stack, parsed_body):
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    mock_proxy_logging = stack.enter_context(
        patch("litellm.proxy.proxy_server.proxy_logging_obj")
    )
    mock_proxy_logging.pre_call_hook = AsyncMock(return_value=parsed_body)
    mock_proxy_logging.post_call_failure_hook = AsyncMock()
    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value=None)
    mock_success_handler = stack.enter_context(
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler"
        )
    )
    mock_success_handler.return_value = None
    stack.enter_context(
        patch.object(
            GLOBAL_LOGGING_WORKER, "ensure_initialized_and_enqueue", new=MagicMock()
        )
    )
    return mock_proxy_logging, mock_success_handler


def _relay_client_request(method="GET"):
    mock_request = MagicMock(spec=Request)
    mock_request.method = method
    mock_request.url = "http://localhost:4000/passthrough-relay/results"
    mock_request.body = AsyncMock(return_value=b"")
    mock_request.headers = Headers({})
    mock_request.query_params = QueryParams({})
    return mock_request


@pytest.mark.asyncio
async def test_pass_through_request_relays_non_json_body_without_buffering():
    """
    Regression (LIT-4009): non-SSE passthrough responses used to be fully
    buffered in proxy memory (content = await response.aread()) before a single
    byte reached the client, ballooning proxy RSS to a multiple of the body size
    for large non-JSON downloads (e.g. Anthropic batch results .jsonl files) and
    producing near-total TTFB dead air that let intermediaries kill the silent
    connection mid-download.

    A non-JSON 2xx body must be relayed as a StreamingResponse whose chunks are
    pulled from the upstream one at a time, with zero chunks consumed before the
    handler returns, upstream status/headers plus x-litellm-* headers preserved,
    and the success-handler logging fired with response_body=None once the
    stream completes. Pre-fix, the handler returned a plain Response after
    reading the entire body, so these assertions fail on the old code.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth

    upstream_chunks = (
        b'{"custom_id": "a", "result": {}}\n',
        b'{"custom_id": "b", "result": {}}\n',
        b'{"custom_id": "c", "result": {}}\n',
    )
    upstream_stream = _RecordingUpstreamByteStream(upstream_chunks)
    fake_client, cleanup = _inject_fake_passthrough_client(
        _FakeUpstreamTransport(
            status_code=200,
            headers={
                "content-type": "application/x-jsonl",
                "x-upstream-marker": "batch-results",
                "content-length": str(sum(len(c) for c in upstream_chunks)),
            },
            stream=upstream_stream,
        ),
        timeout=311.0,
    )
    try:
        with ExitStack() as stack:
            _, mock_success_handler = _enter_relay_logging_mocks(stack, {})

            response = await pass_through_request(
                request=_relay_client_request(),
                target="http://upstream.test/v1/messages/batches/b1/results",
                custom_headers={},
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-relay-test"),
                timeout=311.0,
            )

            assert isinstance(response, StreamingResponse)
            assert upstream_stream.chunks_served == 0
            mock_success_handler.assert_not_called()

            iterator = response.body_iterator
            first_chunk = await iterator.__anext__()
            assert first_chunk == upstream_chunks[0]
            assert upstream_stream.chunks_served == 1

            remaining = [chunk async for chunk in iterator]
            assert b"".join([first_chunk, *remaining]) == b"".join(upstream_chunks)
            assert upstream_stream.closed is True

            assert response.status_code == 200
            assert response.headers["x-upstream-marker"] == "batch-results"
            assert "x-litellm-call-id" in response.headers
            assert "content-length" not in response.headers

            mock_success_handler.assert_called_once()
            success_kwargs = mock_success_handler.call_args.kwargs
            assert success_kwargs["response_body"] is None
            assert (
                success_kwargs["url_route"]
                == "http://upstream.test/v1/messages/batches/b1/results"
            )
    finally:
        cleanup()
        await fake_client.aclose()


@pytest.mark.asyncio
async def test_pass_through_request_json_response_stays_buffered_for_logging():
    """
    JSON responses (content-type application/json) must keep the buffered
    behavior: spend logging and guardrails inspect the parsed body, so the
    handler reads the full upstream body and passes the parsed dict to the
    success handler.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth

    upstream_chunks = (b'{"id": "file-123"', b', "status": "processed"}')
    upstream_stream = _RecordingUpstreamByteStream(upstream_chunks)
    fake_client, cleanup = _inject_fake_passthrough_client(
        _FakeUpstreamTransport(
            status_code=200,
            headers={"content-type": "application/json"},
            stream=upstream_stream,
        ),
        timeout=312.0,
    )
    try:
        with ExitStack() as stack:
            _, mock_success_handler = _enter_relay_logging_mocks(stack, {})

            response = await pass_through_request(
                request=_relay_client_request(),
                target="http://upstream.test/v1/files/file-123",
                custom_headers={},
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-relay-test"),
                timeout=312.0,
            )

            assert not isinstance(response, StreamingResponse)
            assert response.status_code == 200
            assert response.body == b"".join(upstream_chunks)
            assert upstream_stream.chunks_served == len(upstream_chunks)

            mock_success_handler.assert_called_once()
            success_kwargs = mock_success_handler.call_args.kwargs
            assert success_kwargs["response_body"] == {
                "id": "file-123",
                "status": "processed",
            }
    finally:
        cleanup()
        await fake_client.aclose()


@pytest.mark.asyncio
async def test_pass_through_request_upstream_error_body_stays_buffered():
    """
    Upstream errors are never relayed as a stream, whatever their content-type:
    the body must stay available for the failure hook and reach the client
    buffered with the upstream status code, exactly as before the fix.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth

    upstream_stream = _RecordingUpstreamByteStream((b"upstream ", b"exploded"))
    fake_client, cleanup = _inject_fake_passthrough_client(
        _FakeUpstreamTransport(
            status_code=502,
            headers={"content-type": "application/x-jsonl"},
            stream=upstream_stream,
        ),
        timeout=313.0,
    )
    try:
        with ExitStack() as stack:
            mock_proxy_logging, mock_success_handler = _enter_relay_logging_mocks(
                stack, {}
            )

            response = await pass_through_request(
                request=_relay_client_request(),
                target="http://upstream.test/v1/messages/batches/b1/results",
                custom_headers={},
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-relay-test"),
                timeout=313.0,
            )

            assert not isinstance(response, StreamingResponse)
            assert response.status_code == 502
            assert response.body == b"upstream exploded"
            mock_proxy_logging.post_call_failure_hook.assert_called_once()
            mock_success_handler.assert_not_called()
    finally:
        cleanup()
        await fake_client.aclose()


_PARTIAL_RELAY_WARNING_MARKER = "ended before upstream body was fully relayed"


@pytest.mark.asyncio
async def test_pass_through_relay_client_disconnect_logs_partial_relay_warning(caplog):
    """
    Regression: when the client disconnects mid-relay (GeneratorExit), the
    proxy log must record that the upstream body was only partially delivered,
    including the route and the byte count that reached the client, while the
    success handler still fires so the partial delivery produces a spend-log
    row. Pre-fix, the finally block fired the success handler silently and a
    partial delivery was indistinguishable from a complete one.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth

    upstream_chunks = (b'{"custom_id": "a"}\n', b'{"custom_id": "b"}\n')
    upstream_stream = _RecordingUpstreamByteStream(upstream_chunks)
    fake_client, cleanup = _inject_fake_passthrough_client(
        _FakeUpstreamTransport(
            status_code=200,
            headers={"content-type": "application/x-jsonl"},
            stream=upstream_stream,
        ),
        timeout=314.0,
    )
    try:
        with ExitStack() as stack:
            _, mock_success_handler = _enter_relay_logging_mocks(stack, {})

            response = await pass_through_request(
                request=_relay_client_request(),
                target="http://upstream.test/v1/messages/batches/b1/results",
                custom_headers={},
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-relay-test"),
                timeout=314.0,
            )

            assert isinstance(response, StreamingResponse)
            iterator = response.body_iterator
            first_chunk = await iterator.__anext__()
            assert first_chunk == upstream_chunks[0]

            with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
                await iterator.aclose()

            partial_relay_warnings = [
                record.getMessage()
                for record in caplog.records
                if record.levelno == logging.WARNING
                and _PARTIAL_RELAY_WARNING_MARKER in record.getMessage()
            ]
            assert len(partial_relay_warnings) == 1
            assert (
                "http://upstream.test/v1/messages/batches/b1/results"
                in partial_relay_warnings[0]
            )
            assert (
                f"{len(first_chunk)} bytes were sent to the client"
                in partial_relay_warnings[0]
            )

            assert upstream_stream.closed is True
            mock_success_handler.assert_called_once()
            assert mock_success_handler.call_args.kwargs["response_body"] is None
    finally:
        cleanup()
        await fake_client.aclose()


@pytest.mark.asyncio
async def test_pass_through_relay_full_consumption_logs_no_partial_relay_warning(caplog):
    """
    A fully consumed relay must not be reported as a partial delivery: the
    success handler fires and no partial-relay warning is logged.
    """
    from fastapi.responses import StreamingResponse

    from litellm.proxy._types import UserAPIKeyAuth

    upstream_chunks = (b'{"custom_id": "a"}\n', b'{"custom_id": "b"}\n')
    upstream_stream = _RecordingUpstreamByteStream(upstream_chunks)
    fake_client, cleanup = _inject_fake_passthrough_client(
        _FakeUpstreamTransport(
            status_code=200,
            headers={"content-type": "application/x-jsonl"},
            stream=upstream_stream,
        ),
        timeout=315.0,
    )
    try:
        with ExitStack() as stack:
            _, mock_success_handler = _enter_relay_logging_mocks(stack, {})

            response = await pass_through_request(
                request=_relay_client_request(),
                target="http://upstream.test/v1/messages/batches/b1/results",
                custom_headers={},
                user_api_key_dict=UserAPIKeyAuth(api_key="sk-relay-test"),
                timeout=315.0,
            )

            assert isinstance(response, StreamingResponse)
            with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
                relayed = [chunk async for chunk in response.body_iterator]

            assert b"".join(relayed) == b"".join(upstream_chunks)
            assert not any(
                _PARTIAL_RELAY_WARNING_MARKER in record.getMessage()
                for record in caplog.records
            )
            mock_success_handler.assert_called_once()
    finally:
        cleanup()
        await fake_client.aclose()
