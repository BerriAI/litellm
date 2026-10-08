import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional


import fastapi
from fastapi import FastAPI
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
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
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

        def __iter__(self):
            return iter(self._dict.items())

        def items(self):
            return self._dict.items()

        def keys(self):
            return self._dict.keys()

        def values(self):
            return self._dict.values()

    class MockRequest:
        def __init__(
            self, headers=None, method="POST", request_body: Optional[dict] = None
        ):
            self.headers = headers or {}
            self.query_params = QueryParams()
            self.method = method
            self.request_body = request_body or {}
            # Add url attribute that the actual code expects
            self.url = httpx.URL("http://localhost:8000/test")
            self.scope = {"type": "http", "method": method, "path": "/test"}
            # Add state attribute that FastAPI requests have
            self.state = type("State", (), {})()

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
    assert (
        result["litellm_params"]["metadata"]["user_api_key_end_user_id"] == "test-user"
    )
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
    "model": "claude-sonnet-4-5-20250929",
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
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.pass_through_endpoints.PassThroughEndpointLogging.pass_through_async_success_handler",
            new=mock_logging_failure,
        ),
        patch(
            "httpx.AsyncClient.send",
            return_value=mock_response,
        ),
        patch(
            "httpx.AsyncClient.request",
            return_value=mock_response,
        ),
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
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler.PassThroughStreamingHandler.route_streaming_logging_to_handler",
            new=mock_logging_failure,
        ),
        patch(
            "httpx.AsyncClient.send",
            return_value=mock_response,
        ),
        patch(
            "httpx.AsyncClient.request",
            return_value=mock_response,
        ),
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


def test_init_kwargs_filters_pricing_params(mock_request, mock_user_api_key_dict):
    """
    Test that pricing parameters are properly filtered out from the request body
    and don't get sent to the provider API.

    This ensures that custom pricing parameters like:
    - cache_read_input_token_cost
    - input_cost_per_token_batches
    - output_cost_per_token_batches
    - cache_creation_input_token_cost
    etc. are removed from the request body before sending to provider.

    Regression test for: LIT-1221
    """
    request = mock_request()

    # Create a parsed body with pricing parameters that should be filtered out
    parsed_body = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "test"}],
        # Standard pricing params (should be filtered)
        "input_cost_per_token": 0.00002,
        "output_cost_per_token": 0.00002,
        "input_cost_per_second": 0.00001,
        "output_cost_per_second": 0.00001,
        # Cache-related pricing params (should be filtered)
        "cache_read_input_token_cost": 0.00005,
        "cache_creation_input_token_cost": 0.00003,
        "cache_creation_input_token_cost_above_1hr": 0.00004,
        # Batch pricing params (should be filtered)
        "input_cost_per_token_batches": 0.00005,
        "output_cost_per_token_batches": 0.00006,
        # Other pricing params (should be filtered)
        "input_cost_per_audio_token": 0.00001,
        "output_cost_per_audio_token": 0.00001,
        "input_cost_per_character": 0.000001,
        "output_cost_per_character": 0.000001,
        "input_cost_per_image": 0.001,
        "output_cost_per_image": 0.001,
        # Tiered pricing
        "tiered_pricing": [{"input_cost_per_token": 0.00001}],
        # This should NOT be filtered (it's a valid OpenAI parameter)
        "temperature": 0.7,
        "max_tokens": 100,
    }

    passthrough_payload = PassthroughStandardLoggingPayload(
        url="https://api.openai.com/v1/chat/completions",
        request_body=parsed_body.copy(),
    )

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=mock_user_api_key_dict,
        passthrough_logging_payload=passthrough_payload,
        _parsed_body=parsed_body,
        litellm_call_id="test-call-id",
        logging_obj=LiteLLMLoggingObj(
            model="gpt-5.5",
            messages=[{"role": "user", "content": "test"}],
            stream=False,
            call_type="completion",
            start_time=datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        ),
    )

    # Verify pricing parameters were filtered out from parsed_body
    assert "input_cost_per_token" not in parsed_body
    assert "output_cost_per_token" not in parsed_body
    assert "input_cost_per_second" not in parsed_body
    assert "output_cost_per_second" not in parsed_body
    assert "cache_read_input_token_cost" not in parsed_body
    assert "cache_creation_input_token_cost" not in parsed_body
    assert "cache_creation_input_token_cost_above_1hr" not in parsed_body
    assert "input_cost_per_token_batches" not in parsed_body
    assert "output_cost_per_token_batches" not in parsed_body
    assert "input_cost_per_audio_token" not in parsed_body
    assert "output_cost_per_audio_token" not in parsed_body
    assert "input_cost_per_character" not in parsed_body
    assert "output_cost_per_character" not in parsed_body
    assert "input_cost_per_image" not in parsed_body
    assert "output_cost_per_image" not in parsed_body
    assert "tiered_pricing" not in parsed_body

    # Verify valid OpenAI parameters remain in parsed_body
    assert parsed_body["model"] == "gpt-5.5"
    assert parsed_body["messages"] == [{"role": "user", "content": "test"}]
    assert parsed_body["temperature"] == 0.7
    assert parsed_body["max_tokens"] == 100

    # Verify pricing parameters are stored in litellm_params for internal use
    litellm_params = result["litellm_params"]
    assert litellm_params["input_cost_per_token"] == 0.00002
    assert litellm_params["output_cost_per_token"] == 0.00002
    # Note: Other pricing params are also stored but we test the key ones that caused the regression


def test_init_kwargs_client_metadata_cannot_spoof_authenticated_identity(
    mock_request, mock_user_api_key_dict
):
    request = mock_request()
    passthrough_payload = PassthroughStandardLoggingPayload(
        url="https://test.com",
        request_body={},
    )
    authenticated_key = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        end_user_id="test-user",
        key_alias="real-key",
        team_alias="Real Team",
        user_email="real@example.com",
        org_id="real-org",
    )

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=request,
        user_api_key_dict=authenticated_key,
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
        _parsed_body={
            "litellm_metadata": {
                "user_api_key_org_id": "victim-org",
                "user_api_key_end_user_id": "victim-end-user",
                "user_api_key_user_id": "victim-user",
                "user_api_key_team_id": "victim-team",
                "user_api_key_team_alias": "Victim Team",
                "user_api_key_alias": "victim-key",
                "user_api_key_user_email": "victim@example.com",
            }
        },
    )

    metadata = result["litellm_params"]["metadata"]
    assert metadata["user_api_key_user_id"] == "test-user"
    assert metadata["user_api_key_team_id"] == "test-team"
    assert metadata["user_api_key_team_alias"] == "Real Team"
    assert metadata["user_api_key_alias"] == "real-key"
    assert metadata["user_api_key_user_email"] == "real@example.com"
    assert metadata["user_api_key_org_id"] == "real-org"
    assert metadata["user_api_key_end_user_id"] == "test-user"


def test_init_kwargs_no_authenticated_identity_field_is_client_settable(
    mock_request, mock_user_api_key_dict
):
    authenticated_key = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
        end_user_id="test-end-user",
        key_alias="real-key",
        team_alias="Real Team",
        user_email="real@example.com",
        org_id="real-org",
        organization_alias="Real Org",
        project_id="real-project",
        project_alias="Real Project",
        spend=1.5,
        max_budget=10.0,
        user_spend=2.5,
        user_max_budget=20.0,
        team_spend=3.5,
        team_max_budget=30.0,
        metadata={"real": "auth-metadata"},
    )
    expected = dict(
        LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
            user_api_key_dict=authenticated_key
        )
    )
    assert len(expected) >= 20

    spoofed = {key: f"SPOOFED-{key}" for key in expected}

    result = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        request=mock_request(),
        user_api_key_dict=authenticated_key,
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url="https://test.com", request_body={}
        ),
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
        _parsed_body={"litellm_metadata": dict(spoofed), "metadata": dict(spoofed)},
    )

    metadata = result["litellm_params"]["metadata"]
    survived = {
        key: metadata.get(key)
        for key in expected
        if metadata.get(key) != expected[key]
    }
    assert survived == {}, f"client-supplied values survived for: {sorted(survived)}"
