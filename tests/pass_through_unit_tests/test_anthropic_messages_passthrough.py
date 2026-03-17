import json
import os
import sys
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
import pytest
from dotenv import load_dotenv
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages,
)

from typing import Optional
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router import Router
import importlib
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from base_anthropic_unified_messages_test import BaseAnthropicMessagesTest

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):  # Add event_loop as a dependency
    curr_dir = os.getcwd()
    sys.path.insert(0, os.path.abspath("../.."))

    import litellm
    from litellm import Router

    importlib.reload(litellm)

    # Set the event loop from the fixture
    asyncio.set_event_loop(event_loop)

    print(litellm)
    yield

    # Clean up any pending tasks
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()

    # Run the event loop until all tasks are cancelled
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


def _validate_anthropic_response(response: Dict[str, Any]):
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"


class TestAnthropicDirectAPI(BaseAnthropicMessagesTest):
    """Tests for direct Anthropic API calls"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "claude-3-haiku-20240307",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "claude-3-haiku-20240307"


class TestAnthropicBedrockAPI(BaseAnthropicMessagesTest):
    """Tests for Anthropic via Bedrock"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"


class TestAnthropicOpenAIAPI(BaseAnthropicMessagesTest):
    """Tests for OpenAI via Anthropic messages interface"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "openai/gpt-4o-mini",
            "client": None,
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_anthropic_messages_litellm_router_streaming_with_logging(self):
        """
        Test the anthropic_messages with streaming request
        """
        pass


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_with_bad_request():
    """
    Test the anthropic_messages with streaming request
    """
    try:
        response = await litellm.anthropic.messages.acreate(
            messages=[{"role": "user", "content": "hi"}],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",
            max_tokens=100,
            stream=True,
        )
        print(response)
        if isinstance(response, AsyncIterator):
            async for chunk in response:
                print("chunk=", chunk)
    except Exception as e:
        print("got exception", e)
        print("vars", vars(e))
        if hasattr(e, "status_code"):
            assert getattr(e, "status_code") == 400
        else:
            assert isinstance(e, Exception)


@pytest.mark.asyncio
async def test_anthropic_messages_router_streaming_with_bad_request():
    """
    Test the anthropic_messages with streaming request
    """
    try:
        router = Router(
            model_list=[
                {
                    "model_name": "claude-special-alias",
                    "litellm_params": {
                        "model": "claude-3-haiku-20240307",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                }
            ]
        )

        response = await router.aanthropic_messages(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-special-alias",
            max_tokens=100,
            stream=True,
        )
        print(response)
        if isinstance(response, AsyncIterator):
            async for chunk in response:
                print("chunk=", chunk)
    except Exception as e:
        print("got exception", e)
        print("vars", vars(e))
        if hasattr(e, "status_code"):
            assert getattr(e, "status_code") == 400
        else:
            assert isinstance(e, Exception)


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_non_streaming():
    """
    Test the anthropic_messages with non-streaming request
    """
    litellm._turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "claude-special-alias",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            }
        ]
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model="claude-special-alias",
        max_tokens=100,
    )

    # Verify response
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")
    return response


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_routing_strategy():
    """
    Test the anthropic_messages with routing strategy + non-streaming request
    """
    litellm._turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "claude-special-alias",
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            }
        ],
        routing_strategy="latency-based-routing",
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model="claude-special-alias",
        max_tokens=100,
        metadata={
            "user_id": "hello",
        },
    )

    # Verify response
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")
    return response

@pytest.mark.asyncio
async def test_anthropic_messages_fallbacks():
    """
    E2E test the anthropic_messages fallbacks from Anthropic API to Bedrock
    """
    litellm._turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-opus-4-20250514",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-20250514",
                    "api_key": "bad-key",
                },
            },
            {
                "model_name": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                },
            }
        ],
        fallbacks=[
            {
                "anthropic/claude-opus-4-20250514": 
                ["bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"]
            }
        ]
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model="anthropic/claude-opus-4-20250514",
        max_tokens=100,
        metadata={
            "user_id": "hello",
        },
    )

    # Verify response
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")
    return response


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_latency_metadata_tracking():
    """
    Test the anthropic_messages with routing strategy and verify that _latency_per_deployment
    field is passed in litellm_metadata when calling litellm.anthropic_messages
    """
    with unittest.mock.patch("litellm.anthropic_messages") as mock_anthropic_messages:
        # Mock the return value
        mock_response = {
            "id": "msg_123456",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Here's a joke for you!"}],
            "model": "claude-3-haiku-20240307",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_anthropic_messages.return_value = mock_response
        # Set the __name__ attribute that the router expects
        mock_anthropic_messages.__name__ = "anthropic_messages"

        MODEL_GROUP = "claude-special-alias"
        router = Router(
            model_list=[
                {
                    "model_name": MODEL_GROUP,
                    "litellm_params": {
                        "model": "claude-3-haiku-20240307",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                }
            ],
            routing_strategy="latency-based-routing",
        )

        # Set up test parameters
        messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

        # Call the handler
        response = await router.aanthropic_messages(
            messages=messages,
            model=MODEL_GROUP,
            max_tokens=100,
            metadata={
                "user_id": "hello",
            },
        )

        # Verify response
        assert response == mock_response

        # Verify that litellm.anthropic_messages was called
        mock_anthropic_messages.assert_called_once()

        # Get the call arguments
        call_args = mock_anthropic_messages.call_args
        call_kwargs = call_args.kwargs

        print("Call kwargs:", json.dumps(call_kwargs, indent=2, default=str))

        # Verify that litellm_metadata was passed and contains _latency_per_deployment
        assert (
            "litellm_metadata" in call_kwargs
        ), "litellm_metadata should be passed to anthropic_messages"

        litellm_metadata = call_kwargs["litellm_metadata"]
        assert litellm_metadata is not None, "litellm_metadata should not be None"
        assert isinstance(
            litellm_metadata, dict
        ), "litellm_metadata should be a dictionary"

        # Verify _latency_per_deployment is present
        assert (
            "_latency_per_deployment" in litellm_metadata
        ), "litellm_metadata should contain _latency_per_deployment field"

        # Verify the structure of _latency_per_deployment
        latency_per_deployment = litellm_metadata["_latency_per_deployment"]
        assert isinstance(
            latency_per_deployment, dict
        ), "_latency_per_deployment should be a dictionary"

        print(f"✅ Latency per deployment data: {latency_per_deployment}")

        # Verify other expected fields in litellm_metadata
        assert "model_group" in litellm_metadata
        assert litellm_metadata["model_group"] == MODEL_GROUP
        assert "deployment" in litellm_metadata
        assert "model_info" in litellm_metadata

        # Verify other call parameters
        assert call_kwargs["model"] == "claude-3-haiku-20240307"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["metadata"] == {"user_id": "hello"}

        print(
            "✅ Successfully verified that _latency_per_deployment is passed in litellm_metadata to anthropic_messages"
        )

        return response


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("inside async_log_success_event")
        self.logged_standard_logging_payload = kwargs.get("standard_logging_object")

        pass


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_non_streaming_with_logging():
    """
    Test the anthropic_messages with non-streaming request

    - Ensure Cost + Usage is tracked
    """
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    litellm._turn_on_debug()
    MODEL_GROUP = "claude-special-alias"
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            }
        ]
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model=MODEL_GROUP,
        max_tokens=100,
    )

    # Verify response
    _validate_anthropic_response(response)

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")

    await asyncio.sleep(1)

    assert (
        test_custom_logger.logged_standard_logging_payload is not None
    ), "Logging payload should not be None"
    print(
        "tracked standard logging payload",
        json.dumps(
            test_custom_logger.logged_standard_logging_payload, indent=4, default=str
        ),
    )
    assert test_custom_logger.logged_standard_logging_payload["messages"] == messages
    assert test_custom_logger.logged_standard_logging_payload["response"] is not None
    assert (
        test_custom_logger.logged_standard_logging_payload["model"]
        == "claude-3-haiku-20240307"
    )

    # check logged usage + spend
    assert test_custom_logger.logged_standard_logging_payload["response_cost"] > 0
    assert (
        test_custom_logger.logged_standard_logging_payload["prompt_tokens"]
        == response["usage"]["input_tokens"]
    )
    assert (
        test_custom_logger.logged_standard_logging_payload["completion_tokens"]
        == response["usage"]["output_tokens"]
    )

    # assert model_group
    assert (
        test_custom_logger.logged_standard_logging_payload["model_group"] == MODEL_GROUP
    )


@pytest.mark.asyncio
async def test_anthropic_messages_with_extra_headers():
    """
    Test the anthropic_messages with extra headers
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "fake-api-key")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]
    extra_headers = {
        "anthropic-version": "custom-version-for-test",
    }

    # Create a mock response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_123456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Why did the chicken cross the road? To get to the other side!",
            }
        ],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Call the handler with extra_headers and our mocked client
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
        client=mock_client,
        provider_specific_header={
            "custom_llm_provider": "anthropic",
            "extra_headers": extra_headers,
        },
    )

    # Verify the post method was called with the right parameters
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs

    # Verify headers were passed correctly
    headers = call_kwargs.get("headers", {})
    print("HEADERS IN REQUEST", headers)
    for key, value in extra_headers.items():
        assert key in headers
        assert headers[key] == value

    # Verify the response was processed correctly
    assert response == mock_response.json.return_value

    return response


# @pytest.mark.asyncio
# async def test_bedrock_messages_api_header_forwarding():
#     """
#     Test that headers from kwargs (set by proxy's add_headers_to_llm_call_by_model_group)
#     are correctly passed to validate_anthropic_messages_environment for Bedrock Invoke API.
    
#     This verifies that forward_client_headers_to_llm_api works for Bedrock Invoke API (Messages API).
    
#     Issue: When calling Anthropic models via the Messages API, LiteLLM makes a call to 
#     Bedrock's Invoke API, and custom headers were not being forwarded, even though
#     they worked correctly for Chat Completions API with Bedrock's Converse API.
#     """
#     from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
#     from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
#     from litellm.types.router import GenericLiteLLMParams
    
#     handler = BaseLLMHTTPHandler()
    
#     # Headers that would be set by the proxy when forward_client_headers_to_llm_api is configured
#     custom_headers = {
#         "X-Custom-Header": "CustomValue",
#         "X-Request-ID": "req-123",
#     }
    
#     # Mock the provider config
#     mock_provider_config = MagicMock()
    
#     # We'll check what headers are passed to this method
#     mock_provider_config.validate_anthropic_messages_environment.return_value = (
#         {"Authorization": "Bearer test"},
#         "https://bedrock-runtime.us-east-1.amazonaws.com/invoke"
#     )
#     mock_provider_config.transform_anthropic_messages_request.return_value = {"model": "test"}
#     mock_provider_config.get_complete_url.return_value = "https://test.com"
#     mock_provider_config.sign_request.return_value = ({}, None)
#     mock_provider_config.transform_anthropic_messages_response.return_value = {"id": "test"}
    
#     # Mock HTTP client to prevent actual network calls
#     with unittest.mock.patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client") as mock_get_client:
#         mock_http_client = AsyncMock()
#         mock_response = MagicMock()
#         mock_response.status_code = 200
#         mock_response.json.return_value = {"id": "test", "content": []}
#         mock_response.text = "{}"
#         mock_http_client.post.return_value = mock_response
#         mock_get_client.return_value = mock_http_client
        
#         # Mock logging object
#         mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
#         mock_logging_obj.model_call_details = {}
        
#         # Call the handler with headers in kwargs
#         try:
#             await handler.async_anthropic_messages_handler(
#                 model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
#                 messages=[{"role": "user", "content": "Hello"}],
#                 anthropic_messages_provider_config=mock_provider_config,
#                 anthropic_messages_optional_request_params={"max_tokens": 100},
#                 custom_llm_provider="bedrock",
#                 litellm_params=GenericLiteLLMParams(
#                     api_key="test-key",
#                     aws_region_name="us-east-1"
#                 ),
#                 logging_obj=mock_logging_obj,
#                 api_key="test-key",
#                 stream=False,
#                 kwargs={"headers": custom_headers}  # Headers set by proxy
#             )
#         except Exception:
#             pass  # Ignore errors, we're only checking if headers were passed
        
#         # Verify that validate_anthropic_messages_environment was called
#         assert mock_provider_config.validate_anthropic_messages_environment.called
        
#         # Get the headers that were passed
#         call_args = mock_provider_config.validate_anthropic_messages_environment.call_args
#         passed_headers = call_args[1]["headers"]
        
#         # The custom headers from kwargs should be in the passed headers
#         assert "X-Custom-Header" in passed_headers or "x-custom-header" in passed_headers
#         assert "X-Request-ID" in passed_headers or "x-request-id" in passed_headers


@pytest.mark.asyncio
async def test_anthropic_messages_with_thinking():
    """
    Test the anthropic_messages with thinking
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "fake-api-key")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Create a mock response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_123456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Why did the chicken cross the road? To get to the other side!",
            }
        ],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Call the handler with extra_headers and our mocked client
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
        client=mock_client,
        thinking={"budget_tokens": 100},
    )

    # Verify the post method was called with the right parameters
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs
    print("CALL KWARGS", call_kwargs)

    # Verify headers were passed correctly
    request_body = json.loads(call_kwargs.get("data", {}))
    print("REQUEST BODY", request_body)
    assert request_body["max_tokens"] == 100
    assert request_body["model"] == "claude-3-haiku-20240307"
    assert request_body["messages"] == messages
    assert request_body["thinking"] == {"budget_tokens": 100}

    # Verify the response was processed correctly
    assert response == mock_response.json.return_value

    return response


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_credentials_passthrough():
    """
    Test that AWS credentials are correctly passed through to BaseAWSLLM.get_credentials
    when using anthropic.messages.acreate with a bedrock model
    """
    # Mock the get_credentials method
    with unittest.mock.patch.object(
        BaseAWSLLM, "get_credentials"
    ) as mock_get_credentials:
        # Create a proper mock for credentials with the necessary attributes
        mock_credentials = unittest.mock.MagicMock()
        mock_credentials.access_key = "mock_access_key"
        mock_credentials.secret_key = "mock_secret_key"
        mock_credentials.token = "mock_session_token"
        mock_get_credentials.return_value = mock_credentials

        # We also need to mock the actual AWS request signing to avoid real API calls
        with unittest.mock.patch("botocore.auth.SigV4Auth.add_auth"):
            # Set up mock for AsyncHTTPHandler.post to avoid actual API calls
            with unittest.mock.patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
            ) as mock_post:
                # Configure mock response
                mock_response = unittest.mock.MagicMock()
                mock_response.raise_for_status = unittest.mock.MagicMock()
                mock_response.json.return_value = {
                    "id": "msg_bedrock_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a mock response"}],
                    "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }
                mock_post.return_value = mock_response

                # Test AWS credentials parameters - separate from function call parameters
                aws_params = {
                    "aws_access_key_id": "test_access_key",
                    "aws_secret_access_key": "test_secret_key",
                    "aws_session_token": "test_session_token",
                    "aws_region_name": "us-west-2",
                    "aws_role_name": "test_role_name",
                    "aws_session_name": "test_session_name",
                    "aws_profile_name": "test_profile",
                    "aws_web_identity_token": "test_web_identity_token",
                    "aws_sts_endpoint": "https://sts.test-region.amazonaws.com",
                }

                # Call the function with AWS credentials
                await litellm.anthropic.messages.acreate(
                    messages=[{"role": "user", "content": "Hello, test credentials"}],
                    model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    max_tokens=100,
                    **aws_params,
                )

                # Verify get_credentials was called with the correct parameters
                mock_get_credentials.assert_called_once()
                call_args = mock_get_credentials.call_args[1]

                # Assert that our test credentials were passed correctly
                for param_name, param_value in aws_params.items():
                    assert (
                        call_args[param_name] == param_value
                    ), f"Parameter {param_name} was not passed correctly"


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_dynamic_region():
    """
    Test that when aws_region_name is provided, it is used in request url
    """
    # Mock the HTTP response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_bedrock_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a mock response"}],
        "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Patch necessary AWS components
    with unittest.mock.patch(
        "botocore.auth.SigV4Auth.add_auth"
    ), unittest.mock.patch.object(
        BaseAWSLLM, "get_credentials"
    ) as mock_get_credentials:

        # Setup mock credentials
        mock_credentials = unittest.mock.MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_session_token"
        mock_get_credentials.return_value = mock_credentials

        # Test with specific region
        test_region = "us-east-1"

        # Call anthropic.messages.acreate with aws_region_name
        response = await litellm.anthropic.messages.acreate(
            messages=[{"role": "user", "content": "Hello, test region"}],
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=100,
            aws_region_name=test_region,
            client=mock_client,
        )

        # Verify response
        assert response == mock_response.json.return_value

        # Verify the post method was called with the correct URL containing the region
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check that the URL contains the correct region
        url = call_args.kwargs.get("url", "")
        assert (
            f"bedrock-runtime.{test_region}.amazonaws.com" in url
        ), f"URL does not contain the correct region. URL: {url}"

        # Verify get_credentials was called with the correct region
        mock_get_credentials.assert_called_once()
        credentials_args = mock_get_credentials.call_args.kwargs
        assert credentials_args.get("aws_region_name") == test_region


def test_sync_openai_messages():
    """
    Test the anthropic_messages with sync request
    """
    litellm._turn_on_debug()
    response = litellm.anthropic.messages.create(
        messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
        model="openai/gpt-4o-mini",
        max_tokens=100,
    )
    print("ANT response", response)

    assert response is not None
    assert isinstance(response, dict)
    assert response["content"][0]["text"] is not None
