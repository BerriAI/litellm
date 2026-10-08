import json
import os
from datetime import datetime
from typing import Dict, Any
import asyncio
import unittest.mock
from unittest.mock import MagicMock

import litellm
import pytest
from dotenv import load_dotenv
from litellm.llms.anthropic.pass_through.messages.handler import (
    anthropic_messages,
)

from typing import Optional
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
import importlib
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

    test_non_streaming_base = None

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "claude-haiku-4-5-20251001",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "claude-haiku-4-5-20251001"


class TestAnthropicBedrockAPI(BaseAnthropicMessagesTest):
    """Tests for Anthropic via Bedrock"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class TestAnthropicOpenAIAPI(BaseAnthropicMessagesTest):
    """Tests for OpenAI via Anthropic messages interface"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "openai/gpt-4.1-mini",
            "client": None,
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "gpt-4.1-mini"

    @pytest.mark.asyncio
    async def test_anthropic_messages_litellm_router_streaming_with_logging(self):
        """
        Test the anthropic_messages with streaming request
        """
        pass


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_non_streaming():
    """
    Test the anthropic_messages with non-streaming request
    """
    litellm.turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "claude-special-alias",
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
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
    litellm.turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "claude-special-alias",
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
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
    litellm.turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "anthropic/claude-opus-4-7",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-7",
                    "api_key": "bad-key",
                },
            },
            {
                "model_name": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
            },
        ],
        fallbacks=[
            {
                "anthropic/claude-opus-4-7": [
                    "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
                ]
            }
        ],
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model="anthropic/claude-opus-4-7",
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
    litellm.turn_on_debug()
    MODEL_GROUP = "claude-special-alias"
    router = Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
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
        == "claude-haiku-4-5-20251001"
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
#                 model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
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


def test_sync_openai_messages():
    """
    Test the anthropic_messages with sync request
    """
    litellm.turn_on_debug()
    response = litellm.anthropic.messages.create(
        messages=[{"role": "user", "content": "Hello, can you tell me a short joke?"}],
        model="openai/gpt-4.1-mini",
        max_tokens=100,
    )
    print("ANT response", response)

    assert response is not None
    assert isinstance(response, dict)
    assert response["content"][0]["text"] is not None
