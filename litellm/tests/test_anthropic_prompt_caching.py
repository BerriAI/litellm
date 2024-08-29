import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.prompt_templates.factory import anthropic_messages_pt

# litellm.num_retries =3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    print(f"user_model_dict: {user_model_dict}")


@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []


@pytest.mark.asyncio
async def test_litellm_anthropic_prompt_caching_tools():
    # Arrange: Set up the MagicMock for the httpx.AsyncClient
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet-20240620",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }

    mock_response.json = return_val

    litellm.set_verbose = True
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.acompletion function
        response = await litellm.acompletion(
            api_key="mock_api_key",
            model="anthropic/claude-3-5-sonnet-20240620",
            messages=[
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                }
            ],
            extra_headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
        )

        # Print what was called on the mock
        print("call args=", mock_post.call_args)

        expected_url = "https://api.anthropic.com/v1/messages"
        expected_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "x-api-key": "mock_api_key",
        }

        expected_json = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather like in Boston today?",
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "cache_control": {"type": "ephemeral"},
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            "max_tokens": 4096,
            "model": "claude-3-5-sonnet-20240620",
        }

        mock_post.assert_called_once_with(
            expected_url, json=expected_json, headers=expected_headers, timeout=600.0
        )


@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_basic():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement"
                        * 400,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
            },
            # The final turn is marked with cache-control, for continuing in followups.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
        temperature=0.2,
        max_tokens=10,
        extra_headers={
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
    )

    print("response=", response)

    assert "cache_read_input_tokens" in response.usage
    assert "cache_creation_input_tokens" in response.usage

    # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
    assert (response.usage.cache_read_input_tokens > 0) or (
        response.usage.cache_creation_input_tokens > 0
    )


@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_no_headers():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement"
                        * 400,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
            },
            # The final turn is marked with cache-control, for continuing in followups.
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the key terms and conditions in this agreement?",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
        temperature=0.2,
        max_tokens=10,
    )

    print("response=", response)

    assert "cache_read_input_tokens" in response.usage
    assert "cache_creation_input_tokens" in response.usage

    # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
    assert (response.usage.cache_read_input_tokens > 0) or (
        response.usage.cache_creation_input_tokens > 0
    )


@pytest.mark.asyncio
async def test_litellm_anthropic_prompt_caching_system():
    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#prompt-caching-examples
    # LArge Context Caching Example
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet-20240620",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }

    mock_response.json = return_val

    litellm.set_verbose = True
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Act: Call the litellm.acompletion function
        response = await litellm.acompletion(
            api_key="mock_api_key",
            model="anthropic/claude-3-5-sonnet-20240620",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant tasked with analyzing legal documents.",
                        },
                        {
                            "type": "text",
                            "text": "Here is the full text of a complex legal agreement",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": "what are the key terms and conditions in this agreement?",
                },
            ],
            extra_headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
            },
        )

        # Print what was called on the mock
        print("call args=", mock_post.call_args)

        expected_url = "https://api.anthropic.com/v1/messages"
        expected_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "x-api-key": "mock_api_key",
        }

        expected_json = {
            "system": [
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing legal documents.",
                },
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "what are the key terms and conditions in this agreement?",
                        }
                    ],
                }
            ],
            "max_tokens": 4096,
            "model": "claude-3-5-sonnet-20240620",
        }

        mock_post.assert_called_once_with(
            expected_url, json=expected_json, headers=expected_headers, timeout=600.0
        )
