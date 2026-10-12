import asyncio
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.router_utils.prompt_caching_cache import (
    PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
    PROMPT_CACHE_PIN_TTL_SECONDS,
    PromptCachingCache,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + seconds


def _pin_cache(clock: _ManualClock) -> PromptCachingCache:
    return PromptCachingCache(cache=DualCache(in_memory_cache=InMemoryCache(clock=clock)))


def _system_block_messages(
    cache_control: dict[str, str], document: str = "a complex legal agreement"
) -> list[AllMessageValues]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the full text of {document}" * 200,
                    "cache_control": cache_control,
                }
            ],
        },
        {"role": "user", "content": "Reply with the single word ok."},
    ]


def _message_level_messages(cache_control: dict[str, str]) -> list[AllMessageValues]:
    return [
        {
            "role": "system",
            "content": "Here is the full text of a complex legal agreement" * 200,
            "cache_control": cache_control,
        },
        {"role": "user", "content": "Reply with the single word ok."},
    ]


def _tool(cache_control: dict[str, str] | None) -> ChatCompletionToolParam:
    tool: ChatCompletionToolParam = {
        "type": "function",
        "function": {"name": "lookup_clause", "parameters": {"type": "object", "properties": {}}},
    }
    return {**tool, "cache_control": cache_control} if cache_control else tool


@pytest.mark.parametrize(
    ("messages", "tools", "expected_ttl"),
    [
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "1h"}),
            None,
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="content-block-1h",
        ),
        pytest.param(
            _message_level_messages({"type": "ephemeral", "ttl": "1h"}),
            None,
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="message-level-1h",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "5m"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="explicit-5m",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="default-ttl",
        ),
        pytest.param(
            [
                *_system_block_messages({"type": "ephemeral", "ttl": "1h"}),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Again.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
            ],
            None,
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="1h-system-then-5m-turn",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral"}),
            [_tool({"type": "ephemeral", "ttl": "1h"})],
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="1h-on-tools-only",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral"}),
            [_tool(None)],
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="unmarked-tools",
        ),
        pytest.param(
            [
                *_system_block_messages({"type": "ephemeral", "ttl": "5m"}),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Again.", "cache_control": {"type": "something_else", "ttl": "1h"}},
            ],
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="non-ephemeral-1h-ignored",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "3600s"}),
            None,
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="gemini-style-3600s",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "2h"}),
            None,
            2 * PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="longer-than-1h",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "90s"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="shorter-than-5m-keeps-the-floor",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "soon"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="unparseable-ttl-keeps-the-floor",
        ),
    ],
)
def test_pin_ttl_follows_the_longest_breakpoint_ttl(
    messages: list[AllMessageValues],
    tools: list[ChatCompletionToolParam] | None,
    expected_ttl: int,
) -> None:
    assert PromptCachingCache.pin_ttl_seconds(messages, tools) == expected_ttl


@pytest.mark.asyncio
async def test_one_hour_prefix_pin_outlives_the_five_minute_window() -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _system_block_messages({"type": "ephemeral", "ttl": "1h"}, document="a lease")
    five_minutes: Final = _system_block_messages({"type": "ephemeral"}, document="a will")

    await cache.async_add_model_id(model_id="deployment-1h", messages=one_hour, tools=None)
    await cache.async_add_model_id(model_id="deployment-5m", messages=five_minutes, tools=None)

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    assert await cache.async_get_model_id(messages=five_minutes, tools=None) is None
    pinned_after_five_minutes: Final = await cache.async_get_model_id(messages=one_hour, tools=None)
    assert pinned_after_five_minutes is not None
    assert pinned_after_five_minutes["model_id"] == "deployment-1h"

    clock.advance(PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS - PROMPT_CACHE_PIN_TTL_SECONDS)
    assert await cache.async_get_model_id(messages=one_hour, tools=None) is None


def test_sync_one_hour_message_level_pin_outlives_the_five_minute_window() -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _message_level_messages({"type": "ephemeral", "ttl": "1h"})

    cache.add_model_id(model_id="deployment-1h", messages=one_hour, tools=None)

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    pinned: Final = cache.get_model_id(messages=one_hour, tools=None)
    assert pinned is not None
    assert pinned["model_id"] == "deployment-1h"

    clock.advance(PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS - PROMPT_CACHE_PIN_TTL_SECONDS)
    assert cache.get_model_id(messages=one_hour, tools=None) is None


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
    cache: Final = PromptCachingCache(cache=router._prompt_caching_pin_cache)
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
