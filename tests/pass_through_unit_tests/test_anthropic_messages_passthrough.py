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


@pytest.mark.asyncio
async def test_anthropic_messages_non_streaming():
    """
    Test the anthropic_messages with non-streaming request
    """
    litellm._turn_on_debug()
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
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
async def test_anthropic_messages_streaming():
    """
    Test the anthropic_messages with streaming request
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    async_httpx_client = AsyncHTTPHandler()
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
        stream=True,
        client=async_httpx_client,
    )

    if isinstance(response, AsyncIterator):
        async for chunk in response:
            print("chunk=", chunk)


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_with_bad_request():
    """
    Test the anthropic_messages with streaming request
    """
    try:
        response = await litellm.anthropic.messages.acreate(
            messages=["hi"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",
            max_tokens=100,
            stream=True,
        )
        print(response)
        async for chunk in response:
            print("chunk=", chunk)
    except Exception as e:
        print("got exception", e)
        print("vars", vars(e))
        assert e.status_code == 400


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
            messages=["hi"],
            model="claude-special-alias",
            max_tokens=100,
            stream=True,
        )
        print(response)
        async for chunk in response:
            print("chunk=", chunk)
    except Exception as e:
        print("got exception", e)
        print("vars", vars(e))
        assert e.status_code == 400


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
    _validate_anthropic_response(response)

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")

    await asyncio.sleep(1)
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


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_streaming_with_logging():
    """
    Test the anthropic_messages with streaming request

    - Ensure Cost + Usage is tracked
    """
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    # litellm._turn_on_debug()
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
        stream=True,
    )

    response_prompt_tokens = 0
    response_completion_tokens = 0
    all_anthropic_usage_chunks = []

    async for chunk in response:
        # Decode chunk if it's bytes
        print("chunk=", chunk)

        # Handle SSE format chunks
        if isinstance(chunk, bytes):
            chunk_str = chunk.decode("utf-8")
            # Extract the JSON data part from SSE format
            for line in chunk_str.split("\n"):
                if line.startswith("data: "):
                    try:
                        json_data = json.loads(line[6:])  # Skip the 'data: ' prefix
                        print(
                            "\n\nJSON data:",
                            json.dumps(json_data, indent=4, default=str),
                        )

                        # Extract usage information
                        if (
                            json_data.get("type") == "message_start"
                            and "message" in json_data
                        ):
                            if "usage" in json_data["message"]:
                                usage = json_data["message"]["usage"]
                                all_anthropic_usage_chunks.append(usage)
                                print(
                                    "USAGE BLOCK",
                                    json.dumps(usage, indent=4, default=str),
                                )
                        elif "usage" in json_data:
                            usage = json_data["usage"]
                            all_anthropic_usage_chunks.append(usage)
                            print(
                                "USAGE BLOCK", json.dumps(usage, indent=4, default=str)
                            )
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON from: {line[6:]}")
        elif hasattr(chunk, "message"):
            if chunk.message.usage:
                print(
                    "USAGE BLOCK",
                    json.dumps(chunk.message.usage, indent=4, default=str),
                )
                all_anthropic_usage_chunks.append(chunk.message.usage)
        elif hasattr(chunk, "usage"):
            print("USAGE BLOCK", json.dumps(chunk.usage, indent=4, default=str))
            all_anthropic_usage_chunks.append(chunk.usage)

    print(
        "all_anthropic_usage_chunks",
        json.dumps(all_anthropic_usage_chunks, indent=4, default=str),
    )

    # Extract token counts from usage data
    if all_anthropic_usage_chunks:
        response_prompt_tokens = max(
            [usage.get("input_tokens", 0) for usage in all_anthropic_usage_chunks]
        )
        response_completion_tokens = max(
            [usage.get("output_tokens", 0) for usage in all_anthropic_usage_chunks]
        )

    print("input_tokens_anthropic_api", response_prompt_tokens)
    print("output_tokens_anthropic_api", response_completion_tokens)

    await asyncio.sleep(4)

    print(
        "logged_standard_logging_payload",
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
        == response_prompt_tokens
    )
    assert (
        test_custom_logger.logged_standard_logging_payload["completion_tokens"]
        == response_completion_tokens
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
        "anthropic-beta": "very-custom-beta-value",
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
