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
import httpx
import pytest
import litellm
from typing import AsyncGenerator
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.types import EndpointType
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
    _init_kwargs_for_pass_through_endpoint,
    _update_metadata_with_tags_in_header,
)
from litellm.proxy.pass_through_endpoints.types import PassthroughStandardLoggingPayload


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

        async def body(self) -> bytes:
            return bytes(json.dumps(self.request_body), "utf-8")

    return MockRequest


@pytest.fixture
def mock_user_api_key_dict():
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
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

    result = _init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        litellm_call_id="test-call-id",
    )

    assert result["call_type"] == "pass_through_endpoint"
    assert result["litellm_call_id"] == "test-call-id"
    assert result["passthrough_logging_payload"] == passthrough_payload

    # Check metadata
    expected_metadata = {
        "user_api_key": "test-key",
        "user_api_key_user_id": "test-user",
        "user_api_key_team_id": "test-team",
        "user_api_key_end_user_id": "test-user",
    }
    assert result["litellm_params"]["metadata"] == expected_metadata


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

    result = _init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        _parsed_body=parsed_body,
        litellm_call_id="test-call-id",
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

    result = _init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        litellm_call_id="test-call-id",
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

    # Patch only the logging handler
    with patch(
        "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughEndpointLogging.pass_through_async_success_handler",
        new=mock_logging_failure,
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
        print("response", response)
        print(vars(response))


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

    athropic_request_body["stream"] = True
    # Patch only the logging handler
    with patch(
        "litellm.proxy.pass_through_endpoints.streaming_handler.PassThroughStreamingHandler._route_streaming_logging_to_handler",
        new=mock_logging_failure,
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

        print(vars(response))
        print(dir(response))
        body_iterator = response.body_iterator
        async for chunk in body_iterator:
            assert chunk

        print("response", response)
        print(vars(response))
