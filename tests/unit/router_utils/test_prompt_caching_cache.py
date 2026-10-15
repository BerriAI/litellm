import asyncio
from collections.abc import Mapping
from typing import Final, cast

import pytest

from litellm.router_utils.prompt_caching_cache import PromptCachingCache
from litellm.types.llms.openai import AllMessageValues


async def _first_non_none(read):
    while True:
        value = await read()
        if value is not None:
            return value
        await asyncio.sleep(0)


def test_extract_cacheable_prefix_with_string_content_and_message_level_cache_control():
    """
    Test that extract_cacheable_prefix correctly handles messages where:
    - content is a string (not a list of content blocks)
    - cache_control is a sibling key at the message level

    This is a valid message format per LiteLLM's ChatCompletionUserMessage type:
    {"role": "user", "content": "...", "cache_control": {"type": "ephemeral"}}

    Regression test for issue #19228.
    """
    # Test case 1: Single message with string content and message-level cache_control
    messages_string_content = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "This is a large message that should be cached",
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        },
    ]

    result = PromptCachingCache.extract_cacheable_prefix(messages_string_content)

    # Should return both messages (system + user with cache_control)
    assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "This is a large message that should be cached"
    assert result[1].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}


def test_extract_cacheable_prefix_with_string_content_no_cache_control():
    """
    Test that extract_cacheable_prefix returns empty list when:
    - content is a string
    - no cache_control is present
    """
    messages_no_cache = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]

    result = PromptCachingCache.extract_cacheable_prefix(messages_no_cache)

    # Should return empty list (no cacheable content)
    assert len(result) == 0, f"Expected 0 messages, got {len(result)}"


def test_extract_cacheable_prefix_mixed_string_and_list_content():
    """
    Test that extract_cacheable_prefix handles messages with a mix of:
    - String content with message-level cache_control
    - List content with block-level cache_control

    The last cache_control (regardless of format) should determine the cacheable prefix.
    """
    # Message with string content + cache_control, followed by message with list content + cache_control
    messages_mixed = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "First cached message",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Second cached message in list format",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "This should not be in the prefix"},
    ]

    result = PromptCachingCache.extract_cacheable_prefix(messages_mixed)

    # Should include first 3 messages (up to and including the last cache_control)
    assert len(result) == 3, f"Expected 3 messages, got {len(result)}"
    assert result[0]["role"] == "system"
    assert result[1]["content"] == "First cached message"
    assert isinstance(result[2]["content"], list)


def test_prompt_cache_breakpoint_marks_the_same_cacheable_prefix_as_cache_control():
    def _messages(marker: Mapping[str, Mapping[str, str]]) -> list[AllMessageValues]:
        return cast(  # cast-ok: the text part type does not declare the prompt_cache_breakpoint key
            list[AllMessageValues],
            [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Here is the full text of a complex legal agreement" * 400, **marker}
                    ],
                },
                {"role": "user", "content": "what are the key terms?"},
            ],
        )

    marked_key: Final = PromptCachingCache.get_prompt_caching_cache_key(
        _messages({"prompt_cache_breakpoint": {"mode": "explicit"}}), None
    )

    assert marked_key is not None
    assert marked_key == PromptCachingCache.get_prompt_caching_cache_key(
        _messages({"cache_control": {"type": "ephemeral"}}), None
    )
    assert PromptCachingCache.get_prompt_caching_cache_key(_messages({}), None) is None


@pytest.mark.asyncio
async def test_same_cacheable_prefix_routes_to_same_deployment() -> None:
    from litellm import Router

    def _messages(user_content: str):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant tasked with analyzing legal documents.",
                    },
                    {
                        "type": "text",
                        "text": "Here is the full text of a complex legal agreement" * 400,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": user_content},
        ]

    router: Final = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-5-mini",
                    "api_key": f"test-key-{i}",
                },
                "model_info": {"id": f"deployment-{i}"},
            }
            for i in range(1, 4)
        ],
        routing_strategy="simple-shuffle",
        optional_pre_call_checks=["prompt_caching"],
    )

    messages1: Final = _messages("what are the key terms and conditions in this agreement?")
    messages2: Final = _messages("how many words are there?")
    messages3: Final = _messages("how many sentences are there?")

    key1: Final = PromptCachingCache.get_prompt_caching_cache_key(messages1, None)
    key2: Final = PromptCachingCache.get_prompt_caching_cache_key(messages2, None)
    key3: Final = PromptCachingCache.get_prompt_caching_cache_key(messages3, None)

    assert key1 is not None
    assert key1 == key2 == key3

    response1: Final = await router.acompletion(
        model="test-model", messages=messages1, mock_response="ok"
    )
    model_id_1: Final = response1._hidden_params["model_id"]
    assert model_id_1 in {"deployment-1", "deployment-2", "deployment-3"}

    from litellm.integrations.anthropic_cache_control_hook import (
        AnthropicCacheControlHook,
    )

    affinity_messages: Final = AnthropicCacheControlHook.messages_with_default_injections(
        messages=messages2,
        models=("gpt-5-mini",),
        tools=None,
        enable_prompt_caching=None,
        request_kwargs={},
    )
    cache: Final = PromptCachingCache(cache=router.cache)
    cached_2: Final = await asyncio.wait_for(
        _first_non_none(lambda: cache.async_get_model_id(affinity_messages, None)),
        timeout=10,
    )
    assert cached_2 == {"model_id": model_id_1}

    response2: Final = await router.acompletion(
        model="test-model", messages=messages2, mock_response="ok"
    )
    response3: Final = await router.acompletion(
        model="test-model", messages=messages3, mock_response="ok"
    )
    model_id_2: Final = response2._hidden_params["model_id"]
    model_id_3: Final = response3._hidden_params["model_id"]

    assert model_id_1 == model_id_2 == model_id_3
