import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

from test_streaming import streaming_format_tests

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from test_amazing_vertex_completion import load_vertex_ai_credentials

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
    mock_response.headers = {"key": "value"}

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


@pytest.fixture
def anthropic_messages():
    return [
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 400,
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
    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

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
async def test_anthropic_api_prompt_caching_basic_with_cache_creation():
    from uuid import uuid4

    random_id = uuid4()

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
                        "text": "Here is the full text of a complex legal agreement {}".format(
                            random_id
                        )
                        * 400,
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
async def test_anthropic_api_prompt_caching_with_content_str():
    system_message = [
        {
            "role": "system",
            "content": "Here is the full text of a complex legal agreement",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    translated_system_message = litellm.AnthropicConfig().translate_system_message(
        messages=system_message
    )

    assert translated_system_message == [
        # System Message
        {
            "type": "text",
            "text": "Here is the full text of a complex legal agreement",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_messages = [
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": "What are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": "What are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(
        messages=user_messages,
        model="claude-3-5-sonnet-20240620",
        llm_provider="anthropic",
    )

    expected_messages = [
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
            "content": [
                {
                    "type": "text",
                    "text": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
                }
            ],
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

    assert len(translated_messages) == len(expected_messages)
    for idx, i in enumerate(translated_messages):
        assert (
            i == expected_messages[idx]
        ), "Error on idx={}. Got={}, Expected={}".format(idx, i, expected_messages[idx])


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


@pytest.mark.asyncio()
@pytest.mark.flaky(retries=3, delay=1)
async def test_anthropic_api_prompt_caching_streaming():
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
    mock_response.headers = {"key": "value"}

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


def test_is_prompt_caching_enabled(anthropic_messages):
    assert litellm.utils.is_prompt_caching_valid_prompt(
        messages=anthropic_messages,
        tools=None,
        custom_llm_provider="anthropic",
        model="anthropic/claude-3-5-sonnet-20240620",
    )


@pytest.mark.parametrize(
    "messages, expected_model_id",
    [("anthropic_messages", True), ("normal_messages", False)],
)
@pytest.mark.asyncio()
@pytest.mark.skip(
    reason="BETA FEATURE - skipping since this led to a latency impact, beta feature that is not used as yet"
)
async def test_router_prompt_caching_model_stored(
    messages, expected_model_id, anthropic_messages
):
    """
    If a model is called with prompt caching supported, then the model id should be stored in the router cache.
    """
    import asyncio
    from litellm.router import Router
    from litellm.router_utils.prompt_caching_cache import PromptCachingCache

    router = Router(
        model_list=[
            {
                "model_name": "claude-model",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                },
                "model_info": {"id": "1234"},
            }
        ]
    )

    if messages == "anthropic_messages":
        _messages = anthropic_messages
    else:
        _messages = [{"role": "user", "content": "Hello"}]

    await router.acompletion(
        model="claude-model",
        messages=_messages,
        mock_response="The sky is blue.",
    )
    await asyncio.sleep(1)
    cache = PromptCachingCache(
        cache=router.cache,
    )

    cached_model_id = cache.get_model_id(messages=_messages, tools=None)

    if expected_model_id:
        assert cached_model_id["model_id"] == "1234"
    else:
        assert cached_model_id is None


@pytest.mark.asyncio()
# @pytest.mark.skip(
#     reason="BETA FEATURE - skipping since this led to a latency impact, beta feature that is not used as yet"
# )
async def test_router_with_prompt_caching(anthropic_messages):
    """
    if prompt caching supported model called with prompt caching valid prompt,
    then 2nd call should go to the same model.
    """
    from litellm.router import Router
    import asyncio
    from litellm.router_utils.prompt_caching_cache import PromptCachingCache

    router = Router(
        model_list=[
            {
                "model_name": "claude-model",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                    "mock_response": "The sky is blue.",
                },
            },
            {
                "model_name": "claude-model",
                "litellm_params": {
                    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "mock_response": "The sky is green.",
                },
            },
        ],
        optional_pre_call_checks=["prompt_caching"],
    )

    response = await router.acompletion(
        messages=anthropic_messages,
        model="claude-model",
        mock_response="The sky is blue.",
    )
    print("response=", response)

    initial_model_id = response._hidden_params["model_id"]

    await asyncio.sleep(1)
    cache = PromptCachingCache(
        cache=router.cache,
    )

    cached_model_id = cache.get_model_id(messages=anthropic_messages, tools=None)

    assert cached_model_id is not None
    prompt_caching_cache_key = PromptCachingCache.get_prompt_caching_cache_key(
        messages=anthropic_messages, tools=None
    )
    print(f"prompt_caching_cache_key: {prompt_caching_cache_key}")
    assert cached_model_id["model_id"] == initial_model_id

    new_messages = anthropic_messages + [
        {"role": "user", "content": "What is the weather in SF?"}
    ]

    for _ in range(20):
        response = await router.acompletion(
            messages=new_messages,
            model="claude-model",
            mock_response="The sky is blue.",
        )
        print("response=", response)

        assert response._hidden_params["model_id"] == initial_model_id
