import asyncio
from typing import Final, cast

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


def _function_marked_tool(cache_control: dict[str, str]) -> ChatCompletionToolParam:
    tool: Final = _tool(None)
    return {**tool, "function": {**tool["function"], "cache_control": cache_control}}


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
            [_function_marked_tool({"type": "ephemeral", "ttl": "1h"})],
            PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS,
            id="1h-nested-under-function",
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
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="seconds-form-keeps-the-floor",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "2h"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="longer-than-1h-keeps-the-floor",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "99999999h"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="huge-ttl-keeps-the-floor",
        ),
        pytest.param(
            _system_block_messages({"type": "ephemeral", "ttl": "9" * 400 + "h"}),
            None,
            PROMPT_CACHE_PIN_TTL_SECONDS,
            id="overflowing-ttl-keeps-the-floor",
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

    await cache.async_add_model_id(model_id="deployment-1h", messages=one_hour, tools=None, honors_one_hour_ttl=True)
    await cache.async_add_model_id(
        model_id="deployment-5m", messages=five_minutes, tools=None, honors_one_hour_ttl=True
    )

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    assert await cache.async_get_model_id(messages=five_minutes, tools=None) is None
    pinned_after_five_minutes: Final = await cache.async_get_model_id(messages=one_hour, tools=None)
    assert pinned_after_five_minutes is not None
    assert pinned_after_five_minutes["model_id"] == "deployment-1h"

    clock.advance(PROMPT_CACHE_ONE_HOUR_PIN_TTL_SECONDS - PROMPT_CACHE_PIN_TTL_SECONDS)
    assert await cache.async_get_model_id(messages=one_hour, tools=None) is None


@pytest.mark.asyncio
async def test_one_hour_prefix_on_a_deployment_without_one_hour_caching_keeps_the_five_minute_pin() -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _system_block_messages({"type": "ephemeral", "ttl": "1h"})

    await cache.async_add_model_id(
        model_id="deployment-5m-only", messages=one_hour, tools=None, honors_one_hour_ttl=False
    )

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    assert await cache.async_get_model_id(messages=one_hour, tools=None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_write", "second_write", "pinned_past_five_minutes"),
    [
        pytest.param(
            ("deployment-1h", True),
            ("deployment-5m-only", False),
            None,
            id="five-minute-deployment-after-a-one-hour-pin",
        ),
        pytest.param(
            ("deployment-5m-only", False),
            ("deployment-1h", True),
            {"model_id": "deployment-1h"},
            id="one-hour-deployment-after-a-five-minute-pin",
        ),
    ],
)
async def test_latest_write_sets_the_pin_lifetime_in_a_mixed_group(
    first_write: tuple[str, bool],
    second_write: tuple[str, bool],
    pinned_past_five_minutes: dict[str, str] | None,
) -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _system_block_messages({"type": "ephemeral", "ttl": "1h"})

    await cache.async_add_model_id(
        model_id=first_write[0], messages=one_hour, tools=None, honors_one_hour_ttl=first_write[1]
    )
    await cache.async_add_model_id(
        model_id=second_write[0], messages=one_hour, tools=None, honors_one_hour_ttl=second_write[1]
    )

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    assert await cache.async_get_model_id(messages=one_hour, tools=None) == pinned_past_five_minutes


def test_sync_five_minute_deployment_after_a_one_hour_pin_keeps_the_five_minute_pin() -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _message_level_messages({"type": "ephemeral", "ttl": "1h"})

    cache.add_model_id(model_id="deployment-1h", messages=one_hour, tools=None, honors_one_hour_ttl=True)
    cache.add_model_id(model_id="deployment-5m-only", messages=one_hour, tools=None, honors_one_hour_ttl=False)

    clock.advance(PROMPT_CACHE_PIN_TTL_SECONDS + 1)
    assert cache.get_model_id(messages=one_hour, tools=None) is None


def test_sync_one_hour_message_level_pin_outlives_the_five_minute_window() -> None:
    clock: Final = _ManualClock()
    cache: Final = _pin_cache(clock)
    one_hour: Final = _message_level_messages({"type": "ephemeral", "ttl": "1h"})

    cache.add_model_id(model_id="deployment-1h", messages=one_hour, tools=None, honors_one_hour_ttl=True)

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
    cache: Final = PromptCachingCache(cache=router.prompt_caching_pin_cache)
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


def _marked_tool(
    name: str, cache_control: dict | None = None, *, nested: bool = False
) -> ChatCompletionToolParam:
    tool: Final = {
        "type": "function",
        "function": {"name": name, "description": "word " * 40, "parameters": {"type": "object", "properties": {}}},
    }
    if cache_control is not None and nested:
        return cast(
            ChatCompletionToolParam,
            {**tool, "function": {**tool["function"], "cache_control": cache_control}},
        )
    if cache_control is not None:
        return cast(ChatCompletionToolParam, {**tool, "cache_control": cache_control})
    return cast(ChatCompletionToolParam, tool)


def test_prepend_system_prompt_dedupes_the_string_standard_logging_prepends():
    """
    The success event logs the string system prompt already prepended as a system message, and the
    request still carries it in `system`. Prepending unconditionally would hash the prefix twice on
    one side, so the copy already sitting at messages[0] must be recognized as the same prompt.
    """
    messages = cast(
        list[AllMessageValues],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
    )

    assert PromptCachingCache.prepend_system_prompt(messages, "be brief") is messages


def test_prepend_system_prompt_prepends_a_list_system_prompt():
    """
    Standard logging only prepends a string system, so a /v1/messages list system never reaches the
    write side's messages. Both sides have to rebuild the same list the provider caches.
    """
    system: Final = [{"type": "text", "text": "be brief", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    messages = cast(list[AllMessageValues], [{"role": "user", "content": "hi"}])

    result = PromptCachingCache.prepend_system_prompt(messages, system)

    assert result == [{"role": "system", "content": system}, *messages]


@pytest.mark.parametrize("system", [None, "", [], 0, {"type": "text"}])
def test_prepend_system_prompt_leaves_unsystem_values_untouched(system):
    messages = cast(list[AllMessageValues], [{"role": "user", "content": "hi"}])

    assert PromptCachingCache.prepend_system_prompt(messages, system) is messages


def test_prepend_system_prompt_never_mutates_the_input_messages():
    system: Final = [{"type": "text", "text": "be brief"}]
    user_message: Final = {"role": "user", "content": "hi"}
    messages = cast(list[AllMessageValues], [user_message])

    result = PromptCachingCache.prepend_system_prompt(messages, system)

    assert result is not messages
    assert messages == [user_message]


def test_a_tool_breakpoint_emits_a_position_zero_key_even_without_messages():
    """
    A request that marks only its tools still hits the provider prompt cache on the tool prefix, so
    the pin has to exist at a position before the first message block.
    """
    tools: Final = [_marked_tool("read", {"type": "ephemeral"})]

    positions = PromptCachingCache.prefix_positions([], tools)
    assert len(positions) == 1
    assert positions[0].position == 0

    with_messages = PromptCachingCache.prefix_positions(
        cast(
            list[AllMessageValues],
            [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        ),
        tools,
    )
    assert [position.position for position in with_messages] == [0, 1]


def test_a_retimed_or_moved_tool_breakpoint_keeps_the_same_key():
    """
    cache_control on a tool is a boundary marker, not part of the cached content, so a ttl change
    or a breakpoint that leaves an earlier tool unmarked hashes to the same prefix key.
    """
    marked: Final = {"type": "ephemeral"}
    retimed: Final = {"type": "ephemeral", "ttl": "1h"}
    tools_5m: Final = [_marked_tool("read", marked)]
    tools_1h: Final = [_marked_tool("read", retimed)]

    key_5m: Final = PromptCachingCache.get_prompt_caching_cache_key([], tools_5m)
    key_1h: Final = PromptCachingCache.get_prompt_caching_cache_key([], tools_1h)

    assert key_5m is not None
    assert key_5m == key_1h

    both_marked: Final = [_marked_tool("read", marked), _marked_tool("write", marked)]
    last_marked: Final = [_marked_tool("read"), _marked_tool("write", retimed)]

    key_both: Final = PromptCachingCache.get_prompt_caching_cache_key([], both_marked)
    key_last: Final = PromptCachingCache.get_prompt_caching_cache_key([], last_marked)
    assert key_both == key_last


def test_a_nested_function_cache_control_marks_a_boundary_and_strips_from_the_key():
    """
    LiteLLM also honors the OpenAI-style mark nested inside `function` (the Anthropic transform
    maps it to the tool's cache_control), so it has to mark a boundary and hash away exactly like
    the top-level form, or a client that marks that way loses affinity to a breakpoint that moved.
    """
    marked: Final = {"type": "ephemeral"}
    retimed: Final = {"type": "ephemeral", "ttl": "1h"}

    positions = PromptCachingCache.prefix_positions([], [_marked_tool("read", marked, nested=True)])
    assert len(positions) == 1
    assert positions[0].position == 0

    key_nested: Final = PromptCachingCache.get_prompt_caching_cache_key(
        [], [_marked_tool("read", marked, nested=True)]
    )
    key_nested_retimed: Final = PromptCachingCache.get_prompt_caching_cache_key(
        [], [_marked_tool("read", retimed, nested=True)]
    )
    key_toplevel: Final = PromptCachingCache.get_prompt_caching_cache_key([], [_marked_tool("read", marked)])

    assert key_nested is not None
    assert key_nested == key_nested_retimed
    assert key_nested == key_toplevel


def test_unmarked_tools_change_the_key_marked_tools_do_not_share():
    """Tools are provider-cached content, so a different tool set must never read another request's pin."""
    tools_a: Final = [_marked_tool("read", {"type": "ephemeral"})]
    tools_b: Final = [_marked_tool("write", {"type": "ephemeral"})]

    key_a: Final = PromptCachingCache.get_prompt_caching_cache_key([], tools_a)
    key_b: Final = PromptCachingCache.get_prompt_caching_cache_key([], tools_b)

    assert key_a is not None
    assert key_a != key_b


def test_a_non_dict_tool_entry_still_derives_a_key_instead_of_raising():
    """
    A client can send `tools: ["not-a-tool"]`. The provider rejects that with its own 400, so the
    routing check has to key on what it was given and let the request through, never raise a 500.
    """
    messages = cast(
        list[AllMessageValues],
        [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
    )
    tools: Final = cast(list[ChatCompletionToolParam], ["not-a-tool", _marked_tool("read", {"type": "ephemeral"})])

    positions = PromptCachingCache.prefix_positions(messages, tools)

    assert [position.position for position in positions] == [0, 1]
    assert PromptCachingCache.get_prompt_caching_cache_key(messages, tools) == positions[-1].cache_key


def test_empty_and_missing_tools_seed_the_same_key():
    """`tools=[]` and no tools at all are the same request to the provider, so they must key alike."""
    messages = cast(
        list[AllMessageValues],
        [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
    )

    key_none: Final = PromptCachingCache.get_prompt_caching_cache_key(messages, None)
    key_empty: Final = PromptCachingCache.get_prompt_caching_cache_key(messages, [])
    assert key_none == key_empty
