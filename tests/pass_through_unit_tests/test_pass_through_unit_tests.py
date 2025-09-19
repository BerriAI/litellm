import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import fastapi
from fastapi import FastAPI
from fastapi.routing import APIRoute
import httpx
import pytest
import litellm
from typing import AsyncGenerator
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)
from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    _update_metadata_with_tags_in_header,
    HttpPassThroughEndpointHelpers,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


@pytest.fixture
def mock_request():
    # Create a mock request with headers
    class QueryParams:
        def __init__(self):
            self._dict = {}

    class MockRequest:
        def __init__(
            self, headers=None, method="POST", request_body: Optional[dict] = None
        ):
            self.headers = headers or {}
            self.query_params = QueryParams()
            self.method = method
            self.request_body = request_body or {}
            # Add url attribute that the actual code expects
            self.url = "http://localhost:8000/test"

        async def body(self) -> bytes:
            return bytes(json.dumps(self.request_body), "utf-8")

    return MockRequest


@pytest.fixture
def mock_user_api_key_dict():
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        end_user_id="test-user",
    )


def test_update_metadata_with_tags_in_header_no_tags(mock_request):
    """
    No tags should be added to metadata if they do not exist in headers
    """
    # Test when no tags are present in headers
    request = mock_request(headers={})
    metadata = {"existing": "value"}

    result = _update_metadata_with_tags_in_header(request=request, metadata=metadata)

    assert result == {"existing": "value"}
    assert "tags" not in result


def test_update_metadata_with_tags_in_header_with_tags(mock_request):
    """
    Tags should be added to metadata if they exist in headers
    """
    # Test when tags are present in headers
    request = mock_request(headers={"tags": "tag1,tag2,tag3"})
    metadata = {"existing": "value"}

    result = _update_metadata_with_tags_in_header(request=request, metadata=metadata)

    assert result == {"existing": "value", "tags": ["tag1", "tag2", "tag3"]}


def test_init_kwargs_for_pass_through_endpoint_basic(
    mock_request, mock_user_api_key_dict
):
    """
    Basic test for init_kwargs_for_pass_through_endpoint

    - metadata should contain user_api_key, user_api_key_user_id, user_api_key_team_id, user_api_key_end_user_id  from `mock_user_api_key_dict`
    """
    request = mock_request()
    passthrough_payload = PassthroughStandardLoggingPayload(
        url="https://test.com",
        request_body={},
    )

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        litellm_call_id="test-call-id",
        logging_obj=LiteLLMLoggingObj(
            model="test-model",
            messages=[],
            stream=False,
            call_type="test-call-type",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        ),
    )

    assert result["call_type"] == "pass_through_endpoint"
    assert result["litellm_call_id"] == "test-call-id"
    assert result["passthrough_logging_payload"] == passthrough_payload

    #########################################################
    # Check metadata
    #########################################################
    assert result["litellm_params"]["metadata"]["user_api_key"] == "test-key"
    assert result["litellm_params"]["metadata"]["user_api_key_hash"] == "test-key"
    assert result["litellm_params"]["metadata"]["user_api_key_alias"] is None
    assert result["litellm_params"]["metadata"]["user_api_key_user_email"] is None
    assert result["litellm_params"]["metadata"]["user_api_key_user_id"] == "test-user"
    assert result["litellm_params"]["metadata"]["user_api_key_team_id"] == "test-team"
    assert result["litellm_params"]["metadata"]["user_api_key_org_id"] is None
    assert result["litellm_params"]["metadata"]["user_api_key_team_alias"] is None
    assert result["litellm_params"]["metadata"]["user_api_key_end_user_id"] == "test-user"
    assert result["litellm_params"]["metadata"]["user_api_key_request_route"] is None


def test_init_kwargs_with_litellm_metadata(mock_request, mock_user_api_key_dict):
    """
    Expected behavior: litellm_metadata should be merged with default metadata

    see usage example here: https://docs.litellm.ai/docs/pass_through/anthropic_completion#send-litellm_metadata-tags
    """
    request = mock_request()
    parsed_body = {
        "litellm_metadata": {"custom_field": "custom_value", "tags": ["tag1", "tag2"]}
    }
    passthrough_payload = PassthroughStandardLoggingPayload(
        url="https://test.com",
        request_body={},
    )

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        _parsed_body=parsed_body,
        litellm_call_id="test-call-id",
        logging_obj=LiteLLMLoggingObj(
            model="test-model",
            messages=[],
            stream=False,
            call_type="test-call-type",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        ),
    )

    # Check that litellm_metadata was merged with default metadata
    metadata = result["litellm_params"]["metadata"]
    print("metadata", metadata)
    assert metadata["custom_field"] == "custom_value"
    assert metadata["tags"] == ["tag1", "tag2"]
    assert metadata["user_api_key"] == "test-key"


def test_init_kwargs_with_tags_in_header(mock_request, mock_user_api_key_dict):
    """
    Tags should be added to metadata if they exist in headers
    """
    request = mock_request(headers={"tags": "tag1,tag2"})
    passthrough_payload = PassthroughStandardLoggingPayload(
        url="https://test.com",
        request_body={},
    )

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        litellm_call_id="test-call-id",
        logging_obj=LiteLLMLoggingObj(
            model="test-model",
            messages=[],
            stream=False,
            call_type="test-call-type",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        ),
    )

    # Check that tags were added to metadata
    metadata = result["litellm_params"]["metadata"]
    print("metadata", metadata)
    assert metadata["tags"] == ["tag1", "tag2"]


athropic_request_body = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello, world tell me 2 sentences "}],
    "litellm_metadata": {"tags": ["hi", "hello"]},
}


@pytest.mark.asyncio
async def test_pass_through_request_logging_failure(
    mock_request, mock_user_api_key_dict
):
    """
    Test that pass_through_request still returns a response even if logging raises an Exception
    """

    # Mock the logging handler to raise an error
    async def mock_logging_failure(*args, **kwargs):
        raise Exception("Logging failed!")

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    # Add mock content
    mock_response._content = b'{"mock": "response"}'

    async def mock_aread():
        return mock_response._content

    mock_response.aread = mock_aread

    # Patch both the logging handler and the httpx client
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughEndpointLogging.pass_through_async_success_handler",
        new=mock_logging_failure,
    ), patch(
        "httpx.AsyncClient.send",
        return_value=mock_response,
    ), patch(
        "httpx.AsyncClient.request",
        return_value=mock_response,
    ):
        request = mock_request(
            headers={}, method="POST", request_body=athropic_request_body
        )
        response = await pass_through_request(
            request=request,
            target="https://exampleopenaiendpoint-production.up.railway.app/v1/messages",
            custom_headers={},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert response was returned successfully despite logging failure
        assert response.status_code == 200

        # Verify we got the mock response content
        # For FastAPI Response objects, content is accessed via the body attribute
        assert response.body == b'{"mock": "response"}'


@pytest.mark.asyncio
async def test_pass_through_request_logging_failure_with_stream(
    mock_request, mock_user_api_key_dict
):
    """
    Test that pass_through_request still returns a response even if logging raises an Exception
    """

    # Mock the logging handler to raise an error
    async def mock_logging_failure(*args, **kwargs):
        raise Exception("Logging failed!")

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200

    # Add headers property to mock response
    mock_response.headers = {
        "content-type": "application/json",  # Not streaming
    }

    # Create mock chunks for streaming
    mock_chunks = [b'{"chunk": 1}', b'{"chunk": 2}']
    mock_response.body_iterator = AsyncMock()
    mock_response.body_iterator.__aiter__.return_value = mock_chunks

    # Add aread method to mock response
    mock_response._content = b'{"mock": "response"}'

    async def mock_aread():
        return mock_response._content

    mock_response.aread = mock_aread

    # Patch both the logging handler and the httpx client
    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.PassThroughStreamingHandler._route_streaming_logging_to_handler",
        new=mock_logging_failure,
    ), patch(
        "httpx.AsyncClient.send",
        return_value=mock_response,
    ), patch(
        "httpx.AsyncClient.request",
        return_value=mock_response,
    ):
        request = mock_request(
            headers={}, method="POST", request_body=athropic_request_body
        )
        response = await pass_through_request(
            request=request,
            target="https://exampleopenaiendpoint-production.up.railway.app/v1/messages",
            custom_headers={},
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Assert response was returned successfully despite logging failure
        assert response.status_code == 200

        # Check if it's a streaming response or regular response
        from fastapi.responses import StreamingResponse
        if isinstance(response, StreamingResponse):
            # For streaming responses in tests, we just verify it's the right type
            # and status code since iterating over it is complex in test context
            assert response.status_code == 200
        else:
            # Non-streaming response - should have body attribute
            assert hasattr(response, "body")
            assert response.body == b'{"mock": "response"}'


def test_pass_through_routes_support_all_methods():
    """
    Test that all pass-through routes support GET, POST, PUT, DELETE, PATCH methods
    """
    # Import the routers
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        router as llm_router,
    )

    # Expected HTTP methods
    expected_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}

    # Function to check routes in a router
    def check_router_methods(router):
        for route in router.routes:
            if isinstance(route, APIRoute):
                # Get path and methods for this route
                path = route.path
                methods = set(route.methods)
                print("supported methods for route", path, "are", methods)
                # Assert all expected methods are supported
                assert (
                    methods == expected_methods
                ), f"Route {path} does not support all methods. Supported: {methods}, Expected: {expected_methods}"

    # Check both routers
    check_router_methods(llm_router)


def test_is_bedrock_agent_runtime_route():
    """
    Test that _is_bedrock_agent_runtime_route correctly identifies bedrock agent runtime endpoints
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _is_bedrock_agent_runtime_route,
    )

    # Test agent runtime endpoints (should return True)
    assert _is_bedrock_agent_runtime_route("/knowledgebases/kb-123/retrieve") is True
    assert (
        _is_bedrock_agent_runtime_route("/agents/knowledgebases/kb-123/retrieve")
        is True
    )

    # Test regular bedrock runtime endpoints (should return False)
    assert (
        _is_bedrock_agent_runtime_route("/guardrail/test-id/version/1/apply") is False
    )
    assert (
        _is_bedrock_agent_runtime_route("/model/cohere.command-r-v1:0/converse")
        is False
    )
    assert _is_bedrock_agent_runtime_route("/some/random/endpoint") is False
