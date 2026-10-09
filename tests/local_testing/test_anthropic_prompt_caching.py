import json
import os
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_amazing_vertex_completion import load_vertex_ai_credentials
from test_streaming import streaming_format_tests

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

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


@pytest.fixture
def anthropic_messages():
    return [
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 500,
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
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_anthropic_vertex_ai_prompt_caching(anthropic_messages, sync_mode):
    litellm.turn_on_debug()

    load_vertex_ai_credentials()

    client = HTTPHandler() if sync_mode else AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            if sync_mode:
                response = completion(
                    model="vertex_ai/claude-3-5-sonnet-v2@20241022 ",
                    messages=anthropic_messages,
                    client=client,
                )
            else:
                response = await litellm.acompletion(
                    model="vertex_ai/claude-3-5-sonnet-v2@20241022 ",
                    messages=anthropic_messages,
                    client=client,
                )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs["headers"])
        assert "anthropic-beta" not in mock_post.call_args.kwargs["headers"]


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_basic():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement"
                        * 100,
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
        },
    )

    print("response=", response)

    assert "cache_read_input_tokens" in response.usage
    assert "cache_creation_input_tokens" in response.usage

    # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
    assert (response.usage.cache_read_input_tokens > 0) or (
        response.usage.cache_creation_input_tokens > 0
    )


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_basic_with_cache_creation():
    from uuid import uuid4

    random_id = uuid4()

    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement {}".format(
                            random_id
                        )
                        * 100,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
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
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
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
        },
    )

    print("response=", response)

    assert "cache_read_input_tokens" in response.usage
    assert "cache_creation_input_tokens" in response.usage

    # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
    assert (response.usage.cache_read_input_tokens > 0) or (
        response.usage.cache_creation_input_tokens > 0
    )


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_no_headers():
    litellm.set_verbose = True
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement"
                        * 100,
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


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_streaming():
    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[
            # System Message
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement"
                        * 100,
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
        stream=True,
        stream_options={"include_usage": True},
    )

    idx = 0
    is_cache_read_input_tokens_in_usage = False
    is_cache_creation_input_tokens_in_usage = False
    async for chunk in response:
        streaming_format_tests(idx=idx, chunk=chunk)
        # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
        if hasattr(chunk, "usage"):
            print("Received final usage - {}".format(chunk.usage))
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "cache_read_input_tokens"):
            is_cache_read_input_tokens_in_usage = True
        if hasattr(chunk, "usage") and hasattr(
            chunk.usage, "cache_creation_input_tokens"
        ):
            is_cache_creation_input_tokens_in_usage = True

        idx += 1

    print("response=", response)

    assert (
        is_cache_read_input_tokens_in_usage and is_cache_creation_input_tokens_in_usage
    )
