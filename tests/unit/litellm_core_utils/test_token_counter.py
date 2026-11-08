#### What this tests ####
#    This tests litellm.token_counter.token_counter() function
import asyncio
import base64
import importlib
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from itertools import accumulate, chain
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from typing import Final, Literal, Protocol, cast
from unittest.mock import MagicMock, patch, sentinel

import anyio.to_thread
import pytest
import tiktoken
from PIL import Image
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

import litellm
import litellm.constants
from litellm import decode, encode, get_modified_max_tokens
from litellm import token_counter as token_counter_old
from litellm.constants import (
    HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS,
    PDF_DATA_URL_PREFIX,
    PDF_MAX_XOBJECT_DRAWS_PER_PAGE,
    TOKEN_COUNTER_MAX_CONCURRENT_COUNTS,
)
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.token_counter import (
    EXTRAPOLATION_SAMPLES,
    PDF_PAGE_MAX_EXACT_CONTENT_BYTES,
    _count_inline_pdf_tokens,
    _encoding_count,
    _get_exact_count_function,
    _get_extrapolating_count_function,
    _get_tiktoken_count_function,
    _price_pdf_pages,
    _pdf_string_operand_bytes,
    calculate_img_tokens,
    estimate_image_tokens,
    high_detail_image_token_upper_bound,
    is_openai_model,
    offload_token_count,
)
from litellm.litellm_core_utils.token_counter import token_counter as token_counter_new
from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from tests.large_text import text
from tests.unit.litellm_core_utils.event_loop_lag import (
    assert_loop_stayed_free,
    timed_with_loop_lags,
    warm_tokenizer,
)
from tests.unit.litellm_core_utils.messages_with_counts import (
    MESSAGES_TEXT,
    MESSAGES_WITH_IMAGES,
    MESSAGES_WITH_TOOLS,
)


def token_counter_both_assert_same(**args):
    new = token_counter_new(**args)
    old = token_counter_old(**args)
    assert new == old, f"New token counter {new} does not match old token counter {old}"
    return new


## Choose which token_counter the test will use.

# token_counter = token_counter_new
# token_counter = token_counter_old
token_counter = token_counter_both_assert_same


def test_token_counter_basic():
    assert (
        token_counter(
            model="claude-2",
            messages=[
                {
                    "role": "user",
                    "content": "This is a long message that definitely exceeds the token limit.",
                }
            ],
        )
        == 19
    )


def test_token_counter_large_repeated_text_is_encoded_in_bounded_chunks():
    text_length: Final = 1024 * 1024
    messages: Final = [{"role": "user", "content": [{"type": "text", "text": "A" * text_length}]}]

    with patch("litellm.litellm_core_utils.token_counter._encoding_count", wraps=_encoding_count) as encoding_count:
        tokens: Final = token_counter_new(model="us.anthropic.claude-sonnet-4-6", messages=messages)

    encoded_lengths: Final = tuple(len(call.args[1]) for call in encoding_count.call_args_list)
    assert tokens > 0
    assert sum(encoded_lengths) >= text_length
    assert max(encoded_lengths) <= litellm.constants.TIKTOKEN_ENCODE_MAX_CHUNK_SIZE_CHARS


@pytest.mark.parametrize(
    "text",
    [
        "Short text",
        "This is a normal message with punctuation, numbers, and a few words.",
    ],
)
def test_token_counter_short_text_matches_tiktoken(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    expected = len(encoding.encode(text, disallowed_special=()))

    assert token_counter_new(model="us.anthropic.claude-sonnet-4-6", text=text) == expected


def test_token_counter_default_encoding_matches_cl100k():
    encoding: Final = tiktoken.get_encoding("cl100k_base")
    expected: Final = len(encoding.encode("hello world", disallowed_special=()))

    assert token_counter_new(model=None, text="hello world") == expected


def test_token_counter_text_over_chunk_boundary_stays_close_to_tiktoken():
    text = ("The quick brown fox jumps over the lazy dog. " * 30)[:1025]
    encoding = tiktoken.get_encoding("cl100k_base")
    expected = len(encoding.encode(text, disallowed_special=()))

    actual = token_counter_new(model="us.anthropic.claude-sonnet-4-6", text=text)

    assert abs(actual - expected) <= 4


@pytest.mark.parametrize(
    "configured",
    ["0", "-1", "-1024", "not-an-int", "", "   ", "999999999", "inf", "1e9"],
)
def test_invalid_chunk_size_config_stays_usable(monkeypatch, configured):
    """A misconfigured chunk size must not raise, count zero, or restore the quadratic encode cost."""
    monkeypatch.setenv("TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS", configured)
    try:
        reloaded = importlib.reload(litellm.constants)
        chunk_size = reloaded.TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS
        assert 1 <= chunk_size <= reloaded.TIKTOKEN_ENCODE_MAX_CHUNK_SIZE_CHARS

        encoding = tiktoken.get_encoding("cl100k_base")
        count_tokens = _get_tiktoken_count_function(
            lambda text: len(encoding.encode(text, disallowed_special=())),
            chunk_size=chunk_size,
        )
        assert count_tokens("The quick brown fox jumps over the lazy dog. " * 40) > 0
    finally:
        monkeypatch.delenv("TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS")
        importlib.reload(litellm.constants)


def test_valid_chunk_size_config_is_honoured(monkeypatch):
    monkeypatch.setenv("TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS", "2048")
    try:
        assert importlib.reload(litellm.constants).TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS == 2048
    finally:
        monkeypatch.delenv("TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS")
        importlib.reload(litellm.constants)


async def test_huggingface_count_in_a_worker_thread_leaves_the_event_loop_free():
    warm_tokenizer("claude-fable-5")

    tokens, took, lags = await timed_with_loop_lags(
        lambda: asyncify(token_counter_new)(model="claude-fable-5", text=text * 100)
    )

    assert tokens > 0
    assert_loop_stayed_free(took, lags)


@pytest.mark.parametrize("max_exact_chars", [64, 1_000, 2_500])
def test_count_above_the_cap_samples_the_whole_string_and_scales(max_exact_chars: int):
    count_exactly: Final = MagicMock(side_effect=lambda chunk: chunk.count("a") + len(chunk))
    front_heavy: Final = "a" * 1_000 + "b" * 4_000
    exact: Final = 1_000 + len(front_heavy)

    estimate: Final = _get_extrapolating_count_function(count_exactly, max_exact_chars=max_exact_chars)(front_heavy)

    assert abs(estimate - exact) <= exact // 100
    assert sum(len(call.args[0]) for call in count_exactly.call_args_list) <= max_exact_chars


def test_count_at_or_below_the_cap_is_exact():
    count_exactly: Final = MagicMock(side_effect=len)

    assert _get_extrapolating_count_function(count_exactly, max_exact_chars=5_000)("a" * 5_000) == 5_000
    assert count_exactly.call_args_list == [(("a" * 5_000,),)]


class _SlowEncoder:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self.in_flight = 0
        self.peak_in_flight = 0

    def encode_batch_fast(self, texts: list[str]) -> list[list[int]]:
        with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        time.sleep(0.1)
        with self._lock:
            self.in_flight -= 1
        return [[0] * len(text) for text in texts]


@pytest.mark.asyncio
async def test_offloaded_counts_do_not_borrow_from_the_shared_thread_pool():
    encoder: Final = _SlowEncoder()
    count: Final = _get_exact_count_function(None, {"type": "huggingface_tokenizer", "tokenizer": encoder})
    shared_pool: Final = anyio.to_thread.current_default_thread_limiter()
    burst: Final = 2 * TOKEN_COUNTER_MAX_CONCURRENT_COUNTS

    async def shared_pool_borrowed_until_done(counting: asyncio.Future[list[int]]) -> tuple[int, ...]:
        if counting.done():
            return ()
        await asyncio.sleep(0.01)
        return (shared_pool.borrowed_tokens, *await shared_pool_borrowed_until_done(counting))

    counting: Final = asyncio.ensure_future(asyncio.gather(*(offload_token_count(count)("abc") for _ in range(burst))))
    borrowed: Final = await shared_pool_borrowed_until_done(counting)

    assert await counting == [3] * burst
    assert len(borrowed) > 1 and max(borrowed) == 0
    assert 1 < encoder.peak_in_flight <= TOKEN_COUNTER_MAX_CONCURRENT_COUNTS


def _count_in_a_fresh_event_loop(text: str, result: Future[int]) -> None:
    def slow_count(counted: str) -> int:
        time.sleep(0.1)
        return len(counted)

    result.set_result(asyncio.run(offload_token_count(slow_count)(text)))


def test_offloaded_counts_finish_in_every_event_loop_that_shares_the_process():
    loops: Final = 2 * TOKEN_COUNTER_MAX_CONCURRENT_COUNTS
    results: Final = tuple(Future[int]() for _ in range(loops))
    threads: Final = tuple(
        threading.Thread(target=_count_in_a_fresh_event_loop, args=("a" * size, result), daemon=True)
        for size, result in enumerate(results, start=1)
    )
    for thread in threads:
        thread.start()

    _, pending = wait(results, timeout=5)

    assert not pending
    assert tuple(result.result() for result in results) == tuple(range(1, loops + 1))


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("8", 8), ("0", 4), ("not-an-int", 4)],
)
def test_max_concurrent_counts_config_is_honoured(monkeypatch: pytest.MonkeyPatch, configured: str, expected: int):
    monkeypatch.setenv("TOKEN_COUNTER_MAX_CONCURRENT_COUNTS", configured)
    try:
        assert importlib.reload(litellm.constants).TOKEN_COUNTER_MAX_CONCURRENT_COUNTS == expected
    finally:
        monkeypatch.delenv("TOKEN_COUNTER_MAX_CONCURRENT_COUNTS")
        importlib.reload(litellm.constants)


def test_token_counter_applies_the_default_cap():
    max_exact_chars: Final = litellm.constants.TOKEN_COUNTER_MAX_EXACT_CHARS
    prose: Final = ("The quick brown fox jumps over the lazy dog. " * (max_exact_chars // 45 + 1))[:max_exact_chars]
    over_the_cap: Final = prose + "a" * 200_000
    exact: Final = _get_exact_count_function("gpt-5.6")(over_the_cap)

    estimate: Final = token_counter_new(model="gpt-5.6", text=over_the_cap)

    assert estimate != exact
    assert abs(estimate - exact) <= exact // 100


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("2048", 2048), ("0", 4_000_000), ("not-an-int", 4_000_000)],
)
def test_max_exact_chars_config_is_honoured(monkeypatch: pytest.MonkeyPatch, configured: str, expected: int):
    monkeypatch.setenv("TOKEN_COUNTER_MAX_EXACT_CHARS", configured)
    try:
        assert importlib.reload(litellm.constants).TOKEN_COUNTER_MAX_EXACT_CHARS == expected
    finally:
        monkeypatch.delenv("TOKEN_COUNTER_MAX_EXACT_CHARS")
        importlib.reload(litellm.constants)


def test_token_counter_with_prefix():
    messages = [
        {"role": "user", "content": "Who won the world cup in 2022?"},
        {"role": "assistant", "content": "Argentina", "prefix": True},
    ]
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens == 22, f"Expected 22 tokens, got {tokens}"


def test_token_counter_normal_plus_function_calling():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "content1"},
        {"role": "assistant", "content": "content2"},
        {"role": "user", "content": "conten3"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_E0lOb1h6qtmflUyok4L06TgY",
                    "function": {
                        "arguments": '{"query":"search query","domain":"google.ca","gl":"ca","hl":"en"}',
                        "name": "SearchInternet",
                    },
                    "type": "function",
                }
            ],
        },
        {
            "tool_call_id": "call_E0lOb1h6qtmflUyok4L06TgY",
            "role": "tool",
            "name": "SearchInternet",
            "content": "tool content",
        },
    ]
    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens == 80


# test_token_counter_normal_plus_function_calling()


def test_token_counter_legacy_function_call_counts_arguments():
    """
    Regression for VERIA-492 (Token-counter function_call bypass).

    The legacy OpenAI assistant `function_call` field carries arbitrary text in
    `arguments`. Before the fix, `_count_messages` had no branch for
    `function_call` and fell through to the unsupported-key `continue`, so an
    assistant turn could smuggle unlimited text past `token_counter` and the
    proxy `/utils/token_counter` endpoint (and downstream pre-call budget /
    `get_modified_max_tokens` math). After the fix it must be counted the
    same as the equivalent `tool_calls` payload.
    """
    long_arg = "A" * 4000
    fc_messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "search", "arguments": long_arg},
        },
    ]
    tc_messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": long_arg},
                }
            ],
        },
    ]
    fc_tokens = token_counter(model="gpt-3.5-turbo", messages=fc_messages)
    tc_tokens = token_counter(model="gpt-3.5-turbo", messages=tc_messages)
    assert fc_tokens == tc_tokens, (
        f"function_call arguments must count like tool_calls arguments; "
        f"got function_call={fc_tokens}, tool_calls={tc_tokens}"
    )
    assert fc_tokens > 500, f"4000-char arguments payload must contribute real tokens, got {fc_tokens}"


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_TEXT,
)
def test_token_counter_textonly(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo", messages=[message_count_pair["message"]]
    )
    assert counted_tokens == message_count_pair["count"]


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_TEXT,
)
def test_token_counter_count_response_tokens(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo",
        messages=[message_count_pair["message"]],
        count_response_tokens=True,
    )
    # 3 tokens are not added because of count_response_tokens=True
    expected = message_count_pair["count"] - 3
    assert counted_tokens == expected


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_WITH_IMAGES,
)
def test_token_counter_with_images(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-4o", messages=[message_count_pair["message"]]
    )
    assert counted_tokens == message_count_pair["count"]


@pytest.mark.parametrize(
    "message_count_pair",
    MESSAGES_WITH_TOOLS,
)
def test_token_counter_with_tools(message_count_pair):
    counted_tokens = token_counter(
        model="gpt-35-turbo",
        messages=[message_count_pair["system_message"]],
        tools=message_count_pair["tools"],
        tool_choice=message_count_pair["tool_choice"],
    )
    expected_tokens = message_count_pair["count"]
    actual_diff = counted_tokens - expected_tokens

    if "count-tolerate" in message_count_pair:
        if message_count_pair["count-tolerate"] == counted_tokens:
            pass  # expected
        else:
            tolerated_diff = message_count_pair["count-tolerate"] - expected_tokens
            assert (
                actual_diff <= tolerated_diff
            ), f"Expected {expected_tokens} tokens, got {counted_tokens}. Counted tokens is only allowed to be off by {tolerated_diff} in the over-counting direction."
            if actual_diff != tolerated_diff:
                raise NeedsToleranceUpdateError(
                    f"SOMETHING BROKEN GOT FIXED! THIS is good! Adjust 'count-tolerate' from {message_count_pair['count-tolerate']} to {counted_tokens}"
                )

    else:
        assert (
            expected_tokens == counted_tokens
        ), f"Expected {expected_tokens} tokens, got {counted_tokens}."


def test_token_counter_counts_gemini_function_declarations():
    openai_tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Find current weather conditions for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and region"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    gemini_tools: Final = litellm.utils.get_optional_params(
        model="gemini-2.5-pro",
        custom_llm_provider="gemini",
        tools=openai_tools,
    )["tools"]
    camel_case_tools: Final = [{"functionDeclarations": gemini_tools[0]["function_declarations"]}]

    openai_tokens: Final = token_counter_new(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=openai_tools,
    )
    gemini_tokens: Final = token_counter_new(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=gemini_tools,
    )
    camel_case_tokens: Final = token_counter_new(
        model="gemini-2.5-pro",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=camel_case_tools,
    )

    assert openai_tokens == gemini_tokens == camel_case_tokens


def test_token_counter_skips_non_mapping_tools():
    openai_tool: Final = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Find current weather conditions for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City and region"}},
                "required": ["location"],
            },
        },
    }
    messages: Final = [{"role": "user", "content": "What's the weather?"}]
    valid_tokens: Final = token_counter_new(
        model="gemini-2.5-pro",
        messages=messages,
        tools=[openai_tool],
    )
    mixed_tokens: Final = token_counter_new(
        model="gemini-2.5-pro",
        messages=messages,
        tools=["bad", None, openai_tool],
    )

    assert mixed_tokens == valid_tokens


@pytest.mark.parametrize(
    ("shorthand", "explicit"),
    [({"type": "array"}, {"type": "array", "items": {}}), (True, {})],
    ids=["array_without_items", "boolean_subschema"],
)
def test_token_counter_counts_json_schema_shorthands_like_their_explicit_form(shorthand: object, explicit: object) -> None:
    assert _count_with_property_schema(shorthand) == _count_with_property_schema(explicit)


@pytest.mark.parametrize(
    ("malformed", "without_keyword"),
    [
        (
            {"type": "object", "properties": {"id": {"type": "string"}}, "required": True},
            {"type": "object", "properties": {"id": {"type": "string"}}},
        ),
        ({"type": "object", "properties": [{"id": {"type": "string"}}]}, {"type": "object"}),
        ({"type": "string", "enum": None}, {"type": "string"}),
        ({"type": "integer", "enum": 3}, {"type": "integer"}),
    ],
    ids=["required_true", "properties_list", "string_enum_null", "integer_enum_number"],
)
def test_token_counter_counts_a_malformed_schema_keyword_as_absent(malformed: object, without_keyword: object) -> None:
    assert _count_with_property_schema(malformed) == _count_with_property_schema(without_keyword)


def _count_with_property_schema(property_schema: object) -> int:
    tool: Final = {
        "type": "function",
        "function": {
            "name": "read_files",
            "parameters": {"type": "object", "properties": {"paths": property_schema}},
        },
    }
    return token_counter_new(model="gpt-4o", messages=[{"role": "user", "content": "Read the config files"}], tools=[tool])


_WEATHER_FUNCTION: Final = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}
_TIME_FUNCTION: Final = {
    "name": "get_time",
    "description": "Get the local time for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
_WEB_SEARCH_FUNCTION: Final = {
    "name": "web_search",
    "description": "Search the web",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


def _openai_tool(function: Mapping[str, object]) -> dict[str, object]:
    return {"type": "function", "function": function}


def _anthropic_tool(function: Mapping[str, object]) -> dict[str, object]:
    return {"name": function["name"], "description": function["description"], "input_schema": function["parameters"]}


_USER_MESSAGES: Final = ({"role": "user", "content": "What is the weather in Paris today?"},)
_SYSTEM_AND_USER_MESSAGES: Final = ({"role": "system", "content": "You are a helpful assistant."}, *_USER_MESSAGES)


@pytest.mark.parametrize("tool_shape", [_openai_tool, _anthropic_tool])
@pytest.mark.parametrize(
    ("model", "messages", "functions", "billed_prompt_tokens"),
    [
        # usage.prompt_tokens billed for this request by Azure AI Foundry DeepSeek-V4-Flash-0731,
        # Databricks databricks-qwen35-122b-a10b and DashScope qwen3.5-plus on 2026-10-17
        ("DeepSeek-V4-Flash-0731", _USER_MESSAGES, (_WEATHER_FUNCTION,), 295),
        ("DeepSeek-V4-Flash-0731", _USER_MESSAGES, (_WEATHER_FUNCTION, _TIME_FUNCTION), 348),
        ("DeepSeek-V4-Flash-0731", _SYSTEM_AND_USER_MESSAGES, (_WEATHER_FUNCTION,), 300),
        ("databricks-qwen35-122b-a10b", _USER_MESSAGES, (_WEATHER_FUNCTION,), 289),
        ("databricks-qwen35-122b-a10b", _USER_MESSAGES, (_WEATHER_FUNCTION, _TIME_FUNCTION), 352),
        ("databricks-qwen35-122b-a10b", _SYSTEM_AND_USER_MESSAGES, (_WEATHER_FUNCTION,), 296),
        ("qwen3.5-plus", _USER_MESSAGES, (_WEATHER_FUNCTION,), 284),
        ("qwen3.5-plus", _USER_MESSAGES, (_WEATHER_FUNCTION, _TIME_FUNCTION), 342),
    ],
)
def test_token_counter_tools_match_billed_prompt_for_json_tool_chat_templates(
    tool_shape: Callable[[Mapping[str, object]], dict[str, object]],
    model: str,
    messages: tuple[dict[str, str], ...],
    functions: tuple[Mapping[str, object], ...],
    billed_prompt_tokens: int,
):
    counted: Final = token_counter_new(
        model=model,
        messages=list(messages),
        tools=[tool_shape(function) for function in functions],
    )

    assert abs(counted - billed_prompt_tokens) <= billed_prompt_tokens * 0.05


@pytest.mark.parametrize(
    ("model", "counted_like"),
    [
        ("DeepSeek-V4-Pro", "DeepSeek-V4-Flash-0731"),
        ("accounts/fireworks/models/deepseek-v4-flash", "DeepSeek-V4-Flash-0731"),
        ("deepseek/deepseek-chat", "DeepSeek-V4-Flash-0731"),
        ("deepseek/deepseek-chat:free", "DeepSeek-V4-Flash-0731"),
        ("deepseek-reasoner", "DeepSeek-V4-Flash-0731"),
        ("deepseek/deepseek-flash", "DeepSeek-V4-Flash-0731"),
        ("DeepSeek-V4.1-Flash", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("databricks-deepseek-v4-1-flash", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("deepseek-ai/deepseek-v4.1-flash-maas", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("deepseek-v4p1-flash-1017", "accounts/fireworks/models/deepseek-v4p1-flash"),
        ("grok-4.1-fast-non-reasoning", "grok-4-1-fast-reasoning"),
        ("xai.grok-4.1-fast", "grok-4-1-fast-reasoning"),
        ("grok-4.3-latest", "grok-4.3"),
        ("xai.grok-4-3", "grok-4.3"),
        ("Qwen/Qwen3.5-122B-A10B", "databricks-qwen35-122b-a10b"),
        ("qwen3-235b-a22b", "mistral-large-latest"),
    ],
)
def test_token_counter_counts_tools_by_the_models_chat_template_family(model: str, counted_like: str):
    tools: Final = [_openai_tool(_WEATHER_FUNCTION), _openai_tool(_TIME_FUNCTION)]

    assert token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=tools) == token_counter_new(
        model=counted_like, messages=list(_USER_MESSAGES), tools=tools
    )


@pytest.mark.parametrize("model", ["DeepSeek-V4-Flash-0731", "databricks-qwen35-122b-a10b"])
@pytest.mark.parametrize(
    "builtin_tool",
    [
        {"type": "web_search_preview"},
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"name": "web_search"},
    ],
)
def test_token_counter_skips_tools_not_sent_as_functions_for_chat_template_families(
    model: str, builtin_tool: dict[str, object]
):
    weather_tool: Final = _openai_tool(_WEATHER_FUNCTION)

    assert token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=[builtin_tool]) == token_counter_new(
        model=model, messages=list(_USER_MESSAGES)
    )
    assert token_counter_new(
        model=model, messages=list(_USER_MESSAGES), tools=[builtin_tool, weather_tool]
    ) == token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=[weather_tool])


def _responses_tool(function: Mapping[str, object]) -> dict[str, object]:
    return {"type": "function", **function}


def _gemini_tool(function: Mapping[str, object]) -> dict[str, object]:
    return {"function_declarations": [function]}


@pytest.mark.parametrize("model", ["DeepSeek-V4-Flash-0731", "databricks-qwen35-122b-a10b"])
@pytest.mark.parametrize("function", [_WEATHER_FUNCTION, _WEB_SEARCH_FUNCTION])
@pytest.mark.parametrize("tool_shape", [_responses_tool, _gemini_tool])
def test_token_counter_counts_function_tools_like_chat_completions_tools_for_chat_template_families(
    model: str, function: Mapping[str, object], tool_shape: Callable[[Mapping[str, object]], dict[str, object]]
):
    assert token_counter_new(
        model=model, messages=list(_USER_MESSAGES), tools=[tool_shape(function)]
    ) == token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=[_openai_tool(function)])


# Azure AI Foundry DeepSeek-V4-Flash-0731 billed 295 prompt tokens for the weather tool with each of
# these tool_choice values on 2026-10-17, the same as without one
@pytest.mark.parametrize("tool_choice", ["none", "required", {"type": "function", "function": {"name": "get_weather"}}])
def test_token_counter_tool_choice_adds_nothing_for_deepseek_v4(tool_choice: object):
    tools: Final = [_openai_tool(_WEATHER_FUNCTION)]

    assert token_counter_new(
        model="DeepSeek-V4-Flash-0731", messages=list(_USER_MESSAGES), tools=tools, tool_choice=tool_choice
    ) == token_counter_new(model="DeepSeek-V4-Flash-0731", messages=list(_USER_MESSAGES), tools=tools)


# A request with a one-function tool billed 283 prompt tokens on xAI grok-4.3 without tool_choice and 297 with a
# named one, and 357 and 358 on Azure AI Foundry grok-4-1-fast-reasoning, on 2026-10-29
@pytest.mark.parametrize("model", ["grok-4.3", "grok-4-1-fast-reasoning"])
def test_token_counter_counts_a_named_tool_choice_for_grok(model: str):
    tools: Final = [_openai_tool(_WEATHER_FUNCTION)]
    named: Final = {"type": "function", "function": {"name": "get_weather"}}

    assert token_counter_new(
        model=model, messages=list(_USER_MESSAGES), tools=tools, tool_choice=named
    ) > token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=tools)


def _counted_with_strict(model: str, **strict: bool) -> int:
    return token_counter_new(
        model=model, messages=list(_USER_MESSAGES), tools=[_openai_tool({**_WEATHER_FUNCTION, **strict})]
    )


# A one-function tool billed the same prompt tokens with strict unset and false on 2026-10-29: 357 through Azure AI
# Foundry grok-4-1-fast-reasoning chat and 283 through xAI grok-4.3
@pytest.mark.parametrize("model", ["grok-4-1-fast-reasoning", "grok-4.3"])
def test_token_counter_renders_a_false_strict_like_an_unset_one_for_grok(model: str):
    assert (
        _counted_with_strict(model, strict=False)
        == _counted_with_strict(model)
        < _counted_with_strict(model, strict=True)
    )


# A one-function tool billed 5 prompt tokens more with strict false than unset, the same as with strict true, on
# 2026-10-30: 310 against 305 through Baseten DeepSeek-V4.1-Flash and 309 against 304 through Together
@pytest.mark.parametrize(
    "model", ["accounts/fireworks/models/deepseek-v4p1-flash", "deepseek-v4-flash", "qwen3.5-plus"]
)
def test_token_counter_renders_a_false_strict_like_a_true_one_for_other_chat_templates(model: str):
    assert (
        _counted_with_strict(model, strict=False)
        == _counted_with_strict(model, strict=True)
        > _counted_with_strict(model)
    )


@pytest.mark.parametrize(
    ("tool_choice", "expected_tool_choice"),
    [
        pytest.param({"type": "auto"}, None, id="anthropic-auto"),
        pytest.param({"type": "any"}, None, id="anthropic-any"),
        pytest.param({"type": "none"}, "none", id="anthropic-none"),
        pytest.param(
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
            id="anthropic-tool",
        ),
        pytest.param(
            {"type": "function", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
            id="responses-function",
        ),
    ],
)
def test_token_counter_counts_anthropic_tool_choices_like_openai_choices(
    tool_choice: Mapping[str, object], expected_tool_choice: str | Mapping[str, object] | None
) -> None:
    messages: Final = cast(list[AllMessageValues], list(_USER_MESSAGES))
    tools: Final = cast(list[ChatCompletionToolParam], [_openai_tool(_WEATHER_FUNCTION)])

    assert token_counter_new(
        model="gpt-4", messages=messages, tools=tools, tool_choice=tool_choice
    ) == token_counter_new(
        model="gpt-4", messages=messages, tools=tools, tool_choice=expected_tool_choice
    )


@pytest.mark.parametrize("model", ["DeepSeek-V4-Flash-0731", "databricks-qwen35-122b-a10b"])
@pytest.mark.parametrize(
    "anthropic_tool",
    [
        {"name": "get_weather", "input_schema": _WEATHER_FUNCTION["parameters"]},
        _anthropic_tool(_WEATHER_FUNCTION),
        {**_anthropic_tool(_WEATHER_FUNCTION), "strict": True},
        {**_anthropic_tool(_WEATHER_FUNCTION), "defer_loading": True, "input_examples": [{"location": "Paris"}]},
        {**_anthropic_tool(_WEATHER_FUNCTION), "custom": {"eager_input_streaming": True}},
        _anthropic_tool(_WEB_SEARCH_FUNCTION),
    ],
)
def test_token_counter_counts_anthropic_tools_as_sent_to_chat_template_families(
    model: str, anthropic_tool: dict[str, object]
):
    sent_tools, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai([anthropic_tool])  # pyright: ignore[reportArgumentType]  # raw Anthropic tool dict

    assert token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=[anthropic_tool]) == token_counter_new(
        model=model, messages=list(_USER_MESSAGES), tools=sent_tools
    )


# Qwen3.5's chat_template.jinja opens a system turn for the tool schemas when the request has none, 5 tokens that a
# request with a system message already pays, while DeepSeek V4's encoding_dsv4.py appends them to the first turn
# either way (both read on Hugging Face on 2026-10-17). A system message cost the same with and without a tool on
# Fireworks deepseek-v4p1-flash, Azure AI Foundry grok-4-1-fast-reasoning and xAI grok-4.3 on 2026-10-29
@pytest.mark.parametrize(
    ("model", "implicit_system_turn_tokens"),
    [
        ("DeepSeek-V4-Flash-0731", 0),
        ("databricks-qwen35-122b-a10b", 5),
        ("accounts/fireworks/models/deepseek-v4p1-flash", 0),
        ("grok-4-1-fast-reasoning", 0),
        ("grok-4.3", 0),
    ],
)
def test_token_counter_counts_the_tools_system_turn_only_without_a_system_message(
    model: str, implicit_system_turn_tokens: int
):
    tools: Final = [_openai_tool(_WEATHER_FUNCTION)]
    system_cost_with_tools: Final = token_counter_new(
        model=model, messages=list(_SYSTEM_AND_USER_MESSAGES), tools=tools
    ) - token_counter_new(model=model, messages=list(_USER_MESSAGES), tools=tools)
    system_cost_without_tools: Final = token_counter_new(
        model=model, messages=list(_SYSTEM_AND_USER_MESSAGES)
    ) - token_counter_new(model=model, messages=list(_USER_MESSAGES))

    assert system_cost_with_tools == system_cost_without_tools - implicit_system_turn_tokens


class NeedsToleranceUpdateError(Exception):
    """Custom exception to mark tests that have improved"""


# test_tokenizers()


def test_encoding_and_decoding(tmp_path: Path):
    sample_text = "Hellö World, this is my input string!"

    # openai encoding + decoding
    openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
    openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)

    assert openai_text == sample_text

    # claude encoding + decoding
    claude_tokens = encode(model="claude-3-5-haiku-20241022", text=sample_text)

    claude_text = decode(model="claude-3-5-haiku-20241022", tokens=claude_tokens)

    assert claude_text == sample_text

    # cohere encoding + decoding
    cohere_tokens = encode(model="command-nightly", text=sample_text)
    cohere_text = decode(model="command-nightly", tokens=cohere_tokens)

    assert cohere_text == sample_text

    # llama2 encoding + decoding
    words = sample_text.split()
    result = _run_in_memory_hub(
        HUB_ROUND_TRIP_SCRIPT,
        {
            "hf-internal-testing/llama-tokenizer": _word_level_tokenizer_json(
                pre_tokenizers.WhitespaceSplit(),
                vocab={"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}},
            )
        },
        sample_text,
        tmp_path,
    )

    assert result["decoded"] == sample_text
    assert result["requested"] == ["hf-internal-testing/llama-tokenizer"]
    assert len(result["tokens"]) == len(words)
    assert len(result["tokens"]) != len(encode(model="gpt-3.5-turbo", text=sample_text))


# test_encoding_and_decoding()


# test_gpt_vision_token_counting()


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4-vision-preview",
        "gpt-4o",
        "claude-3-opus-20240229",
        "command-nightly",
        "mistral/mistral-tiny",
    ],
)
def test_load_test_token_counter(model):
    """Token-count a large prompt for each supported model family."""
    messages = [{"role": "user", "content": text}] * 10
    assert token_counter(model=model, messages=messages) > 0


@pytest.mark.parametrize(
    "model, base_model, input_tokens, user_max_tokens, expected_value",
    [
        ("random-model", "random-model", 1024, 1024, 1024),
        ("gpt-3.5-turbo", "gpt-3.5-turbo", 4000, 5000, 4096),  # model max output = 4096
    ],
)
def test_get_modified_max_tokens(model, base_model, input_tokens, user_max_tokens, expected_value):
    """
    - Test when max_output is not known => expect user_max_tokens
    - Test when max_output == max_input,
        - input > max_output, no max_tokens => expect None
        - input + max_tokens > max_output => expect remainder
        - input + max_tokens < max_output => expect max_tokens
    - Test when max_tokens > max_output => expect max_output
    """
    args = locals()
    import litellm

    litellm.token_counter = MagicMock()

    def _mock_token_counter(*args, **kwargs):
        return input_tokens

    litellm.token_counter.side_effect = _mock_token_counter
    print(f"_mock_token_counter: {_mock_token_counter()}")
    messages = [{"role": "user", "content": "Hello world!"}]

    calculated_value = get_modified_max_tokens(
        model=model,
        base_model=base_model,
        messages=messages,
        user_max_tokens=user_max_tokens,
        buffer_perc=0,
        buffer_num=0,
    )

    if expected_value is None:
        assert calculated_value is None
    else:
        assert calculated_value == expected_value, f"Got={calculated_value}, Expected={expected_value}, Params={args}"


def test_empty_tools():
    messages = [{"role": "user", "content": "hey, how's it going?", "tool_calls": None}]

    result = token_counter(
        messages=messages,
    )

    print(result)


@pytest.mark.skip(
    reason="Skipping this test temporarily because it relies on a function being called that I am removing."
)
def test_gpt_4o_token_counter():
    with patch.object(
        litellm.utils, "openai_token_counter", new=MagicMock()
    ) as mock_client:
        token_counter(
            model="gpt-4o-2024-05-13", messages=[{"role": "user", "content": "Hey!"}]
        )

        mock_client.assert_called()


@pytest.mark.parametrize(
    "img_url",
    [
        "https://example.com/test-image.png",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAL0AAAC9CAMAAADRCYwCAAAAh1BMVEX///8AAAD8/Pz5+fkEBAT39/cJCQn09PRNTU3y8vIMDAwzMzPe3t7v7+8QEBCOjo7FxcXR0dHn5+elpaWGhoYYGBivr686OjocHBy0tLQtLS1TU1PY2Ni6urpaWlpERER3d3ecnJxoaGiUlJRiYmIlJSU4ODhBQUFycnKAgIDBwcFnZ2chISE7EjuwAAAI/UlEQVR4nO1caXfiOgz1bhJIyAJhX1JoSzv8/9/3LNlpYd4rhX6o4/N8Z2lKM2cURZau5JsQEhERERERERERERERERERERHx/wBjhDPC3OGN8+Cc5JeMuheaETSdO8vZFyCScHtmz2CsktoeMn7rLM1u3h0PMAEhyYX7v/Q9wQvoGdB0hlbzm45lEq/wd6y6G9aezvBk9AXwp1r3LHJIRsh6s2maxaJpmvqgvkC7WFS3loUnaFJtKRVUCEoV/RpCnHRvAsesVQ1hw+vd7Mpo+424tLs72NplkvQgcdrsvXkW/zJWqH/fA0FT84M/xnQJt4to3+ZLuanbM6X5lfXKHosO9COgREqpCR5i86pf2zPS7j9tTj+9nO7bQz3+xGEyGW9zqgQ1tyQ/VsxEDvce/4dcUPNb5OD9yXvR4Z2QisuP0xiGWPnemgugU5q/troHhGEjIF5sTOyW648aC0TssuaaCEsYEIkGzjWXOp3A0vVsf6kgRyqaDk+T7DIVWrb58b2tT5xpUucKwodOD/5LbrZC1ws6YSaBZJ/8xlh+XZSYXaMJ2ezNqjB3IPXuehPcx2U6b4t1dS/xNdFzguUt8ie7arnPeyCZroxLHzGgGdqVcspwafizPWEXBee+9G1OaufGdvNng/9C+gwgZ3PH3r87G6zXTZ5D5De2G2DeFoANXfbACkT+fxBQ22YFsTTJF9hjFVO6VbqxZXko4WJ8s52P4PnuxO5KRzu0/hlix1ySt8iXjgaQ+4IHPA9nVzNkdduM9LFT/Aacj4FtKrHA7iAw602Vnht6R8Vq1IOS+wNMKLYqayAYfRuufQPGeGb7sZogQQoLZrGPgZ6KoYn70Iw30O92BNEDpvwouCFn6wH2uS+EhRb3WF/HObZk3HuxfRQM3Y/Of/VH0n4MKNHZDiZvO9+m/ABALfkOcuar/7nOo7B95ACGVAFaz4jMiJwJhdaHBkySmzlGTu82gr6FSTik2kJvLnY9nOd/D90qcH268m3I/cgI1xg1maE5CuZYaWLH+UHANCIck0yt7Mx5zBm5vVHXHwChsZ35kKqUpmo5Svq5/fzfAI5g2vDtFPYo1HiEA85QrDeGm9g//LG7K0scO3sdpj2CBDgCa+0OFs0bkvVgnnM/QBDwllOMm+cN7vMSHlB7Uu4haHKaTwgGkv8tlK+hP8fzmFuK/RQTpaLPWvbd58yWIo66HHM0OsPoPhVqmtaEVL7N+wYcTLTbb0DLdgp23Eyy2VYJ2N7bkLFAAibtoLPe5sLt6Oa2bvU+zyeMa8wrixO0gRTn9tO9NCSThTLGqcqtsDvphlfmx/cPBZVvw24jg1LE2lPuEo35Mhi58U0I/Ga8n5w+NS8i34MAQLos5B1u0xL1ZvCVYVRw/Fs2q53KLaXJMWwOZZ/4MPYV19bAHmgGDKB6f01xoeJKFbl63q9J34KdaVNPJWztQyRkzA3KNs1AdAEDowMxh10emXTCx75CkurtbY/ZpdNDGdsn2UcHKHsQ8Ai3WZi48IfkvtjOhsLpuIRSKZTX9FA4o+0d6o/zOWqQzVJMynL9NsxhSJOaourq6nBVQBueMSyubsX2xHrmuABZN2Ns9jr5nwLFlLF/2R6atjW/67Yd11YQ1Z+kA9Zk9dPTM/o6dVo6HHVgC0JR8oUfmI93T9u3gvTG94bAH02Y5xeqRcjuwnKCK6Q2+ajl8KXJ3GSh22P3Zfx6S+n008ROhJn+JRIUVu6o7OXl8w1SeyhuqNDwNI7SjbK08QrqPxS95jy4G7nCXVq6G3HNu0LtK5J0e226CfC005WKK9sVvfxI0eUbcnzutfhWe3rpZHM0nZ/ny/N8tanKYlQ6VEW5Xuym8yV1zZX58vwGhZp/5tFfhybZabdbrQYOs8F+xEhmPsb0/nki6kIyVvzZzUASiOrTfF+Sj9bXC7DoJxeiV8tjQL6loSd0yCx7YyB6rPdLx31U2qCG3F/oXIuDuqd6LFO+4DNIJuxFZqSsU0ea88avovFnWKRYFYRQDfCfcGaBCLn4M4A1ntJ5E57vicwqq2enaZEF5nokCYu9TbKqCC5yCDfL+GhLxT4w4xEJs+anqgou8DOY2q8FMryjb2MehC1dRJ9s4g9NXeTwPkWON4RH+FhIe0AWR/S9ekvQ+t70XHeimGF78LzuU7d7PwrswdIG2VpgF8C53qVQsTDtBJc4CdnkQPbnZY9mbPdDFra3PCXBBQ5QBn2aQqtyhvlyYM4Hb2/mdhsxCUen04GZVvIJZw5PAamMOmjzq8Q+dzAKLXDQ3RUZItWsg4t7W2DP+JDrJDymoMH7E5zQtuEpG03GTIjGCW3LQqOYEsXgFc78x76NeRwY6SNM+IfQoh6myJKRBIcLYxZcwscJ/gI2isTBty2Po9IkYzP0/SS4hGlxRjFAG5z1Jt1LckiB57yWvo35EaolbvA+6fBa24xodL2YjsPpTnj3JgJOqhcgOeLVsYYwoK0wjY+m1D3rGc40CukkaHnkEjarlXrF1B9M6ECQ6Ow0V7R7N4G3LfOHAXtymoyXOb4QhaYHJ/gNBJUkxclpSs7DNcgWWDDmM7Ke5MJpGuioe7w5EOvfTunUKRzOh7G2ylL+6ynHrD54oQO3//cN3yVO+5qMVsPZq0CZIOx4TlcJ8+Vz7V5waL+7WekzUpRFMTnnTlSCq3X5usi8qmIleW/rit1+oQZn1WGSU/sKBYEqMNh1mBOc6PhK8yCfKHdUNQk8o/G19ZPTs5MYfai+DLs5vmee37zEyyH48WW3XA6Xw6+Az8lMhci7N/KleToo7PtTKm+RA887Kqc6E9dyqL/QPTugzMHLbLZtJKqKLFfzVWRNJ63c+95uWT/F7R0U5dDVvuS409AJXhJvD0EwWaWdW8UN11u/7+umaYjT8mJtzZwP/MD4r57fihiHlC5fylHfaqnJdro+Dr7DajvO+vi2EwyD70s8nCH71nzIO1l5Zl+v1DMCb5ebvCMkGHvobXy/hPumGLyX0218/3RyD1GRLOuf9u/OGQyDmto32yMiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIv7GP8YjWPR/czH2AAAAAElFTkSuQmCC",
    ],
)
def test_img_url_token_counter(img_url, monkeypatch):
    """
    Verify get_image_dimensions returns valid (width, height) for both an
    HTTPS URL and a base64 data URI. The HTTPS branch is exercised with a
    mocked HTTP fetch so the test is hermetic - it can't break when a
    third-party image URL goes away.
    """
    import base64
    from litellm.litellm_core_utils.token_counter import get_image_dimensions

    # Minimal valid 1x1 PNG, served by the mocked safe_get for the URL case.
    _tiny_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )

    if img_url.startswith(("http://", "https://")):

        class _FakeResponse:
            headers = {"Content-Length": str(len(_tiny_png))}

            def read(self):
                return _tiny_png

        monkeypatch.setattr(
            "litellm.litellm_core_utils.token_counter.safe_get",
            lambda client, url, **kw: _FakeResponse(),
        )

    width, height = get_image_dimensions(data=img_url)

    print(width, height)

    assert width is not None
    assert height is not None


def test_token_encode_disallowed_special():
    encode(model="gpt-3.5-turbo", text="Hello, world! <|endoftext|>")
    token_counter(model="gpt-3.5-turbo", text="Hello, world! <|endoftext|>")


def test_token_counter():
    try:
        messages = [{"role": "user", "content": "hi how are you what time is it"}]
        tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
        print("gpt-35-turbo")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="claude-2", messages=messages)
        print("claude-2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="gemini/chat-bison", messages=messages)
        print("gemini/chat-bison")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="ollama/llama2", messages=messages)
        print("ollama/llama2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="anthropic.claude-instant-v1", messages=messages)
        print("anthropic.claude-instant-v1")
        print(tokens)
        assert tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


import unittest

from litellm.utils import (
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES,
    _HuggingFaceTokenizerLoadFailure,
    _HuggingFaceTokenizerLoadOutcomes,
    _return_huggingface_tokenizer,
    _select_tokenizer_helper,
    claude_json_str,
    encoding,
)

# Clear the cache at module load to ensure clean state
_HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


class TestTokenizerSelection(unittest.TestCase):
    def setUp(self):
        """Clear the LRU cache before each test method.

        The HuggingFace tokenizers behind _select_tokenizer_helper are cached with
        @lru_cache, which can cause cache hits from previous tests when running with
        --dist=loadscope (tests from same file run on same worker).
        """
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()

    @patch("litellm.utils.tokenizer_dispatch.from_pretrained")
    def test_llama3_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Test with llama-3 model
        result = _select_tokenizer_helper("llama-3-7b")

        # Verify the attempt to load Llama-3 tokenizer
        mock_from_pretrained.assert_called_once_with("Xenova/llama-3-tokenizer")

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.tokenizer_dispatch.from_pretrained")
    def test_cohere_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Add Cohere model to the list for testing
        litellm.cohere_models = ["command-r-v1"]

        # Test with Cohere model
        result = _select_tokenizer_helper("command-r-v1")

        # Verify the attempt to load Cohere tokenizer
        mock_from_pretrained.assert_called_once_with(
            "Xenova/c4ai-command-r-v01-tokenizer"
        )

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.tokenizer_dispatch.anthropic")
    def test_claude_tokenizer_api_failure(self, mock_anthropic):
        # Setup mock to raise an error
        mock_anthropic.side_effect = Exception("Failed to load tokenizer")

        # Add Claude model to the list for testing
        litellm.anthropic_models = ["claude-2"]

        # Test with Claude model
        result = _select_tokenizer_helper("claude-2")

        # Verify the attempt to load Claude tokenizer
        mock_anthropic.assert_called_once_with()

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils.tokenizer_dispatch.from_pretrained")
    def test_llama2_tokenizer_api_failure(self, mock_from_pretrained):
        # Setup mock to raise an error
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")

        # Test with Llama-2 model
        result = _select_tokenizer_helper("llama-2-7b")

        # Verify the attempt to load Llama-2 tokenizer
        mock_from_pretrained.assert_called_once_with(
            "hf-internal-testing/llama-tokenizer"
        )

        # Verify fallback to OpenAI tokenizer
        self.assertEqual(result["type"], "openai_tokenizer")
        self.assertEqual(result["tokenizer"], encoding)

    @patch("litellm.utils._return_huggingface_tokenizer")
    def test_disable_hf_tokenizer_download(self, mock_return_huggingface_tokenizer):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)
        try:
            result = _select_tokenizer_helper("grok-32r22r")
            mock_return_huggingface_tokenizer.assert_not_called()
            assert result["type"] == "openai_tokenizer"
            assert result["tokenizer"] == encoding
        finally:
            monkeypatch.undo()

    @patch("litellm.utils.tokenizer_dispatch.from_pretrained")
    def test_failed_huggingface_tokenizer_download_is_not_retried_on_every_count(
        self, mock_from_pretrained: MagicMock
    ):
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
        try:
            mock_from_pretrained.side_effect = RuntimeError("HF unavailable")
            model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
            text: Final = "Count the tokens for this message."
            expected: Final = len(encoding.encode(text, disallowed_special=()))

            first_count: Final = litellm.token_counter(model=model, text=text)
            first_attempts: Final = mock_from_pretrained.call_count
            second_count: Final = litellm.token_counter(model=model, text=text)

            assert first_attempts > 0
            assert mock_from_pretrained.call_count == first_attempts
            assert first_count == expected
            assert second_count == expected
        finally:
            _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


_CODE_SAMPLE: Final = "    for item in items:\n        total += item.price * item.quantity\n" * 20


def _deepseek_token_ids(text: str) -> list[int]:
    deepseek_tokenizer: Final = Tokenizer.from_str(
        importlib.resources.files("litellm.litellm_core_utils.tokenizers")
        .joinpath("deepseek_tokenizer.json")
        .read_text(encoding="utf-8")
    )
    return deepseek_tokenizer.encode(text).ids


def _deepseek_token_count(text: str) -> int:
    return len(_deepseek_token_ids(text))


@pytest.mark.parametrize(
    "model",
    [
        "azure_ai/DeepSeek-V4-Flash-0731",
        "deepseek/deepseek-v4-pro",
        "fireworks_ai/deepseek-v4p1-flash",
        "bedrock/us-east-1/deepseek.v3.2",
        "together_ai/deepseek-ai/DeepSeek-V3.1",
        "databricks/databricks-deepseek-v4-flash-0731",
        "replicate/deepseek-ai/deepseek-v3",
        "deepseek-chat",
        "deepseek/deepseek-reasoner",
        "openrouter/deepseek/deepseek-chat-v3-0324",
        "openrouter/deepseek/deepseek-chat:free",
    ],
)
def test_token_counter_counts_deepseek_v3_and_v4_with_the_packaged_deepseek_tokenizer(model: str):
    expected: Final = _deepseek_token_count(_CODE_SAMPLE)

    assert expected != len(tiktoken.get_encoding("cl100k_base").encode(_CODE_SAMPLE))
    assert litellm.token_counter(model=model, text=_CODE_SAMPLE) == expected


@pytest.mark.parametrize(
    "model",
    ["deepinfra/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "fireworks_ai/accounts/fireworks/models/deepseek-v2p5"],
)
def test_token_counter_skips_the_deepseek_tokenizer_for_deepseek_models_built_on_other_vocabularies(model: str):
    assert litellm.token_counter(model=model, text=_CODE_SAMPLE) != _deepseek_token_count(_CODE_SAMPLE)


def test_encode_and_decode_use_deepseeks_token_ids_for_deepseek_v4():
    token_ids: Final = encode(model="deepseek/deepseek-v4-flash", text=_CODE_SAMPLE)

    assert token_ids == _deepseek_token_ids(_CODE_SAMPLE)
    assert decode(model="deepseek/deepseek-v4-flash", tokens=token_ids) == _CODE_SAMPLE


def test_disable_hf_tokenizer_download_keeps_deepseek_v4_on_cl100k(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)

    counted: Final = litellm.token_counter(model="deepseek/deepseek-v4-flash", text=_CODE_SAMPLE)

    assert counted != _deepseek_token_count(_CODE_SAMPLE)
    assert counted == litellm.token_counter(model="", text=_CODE_SAMPLE)


@pytest.mark.parametrize("model", ["llama-2-7b", "meta-llama/llama-3-8b-instruct"])
def test_offline_mode_counts_downloaded_tokenizer_models_with_tiktoken(monkeypatch: pytest.MonkeyPatch, model: str):
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")

    with patch("litellm.utils.tokenizer_dispatch.from_pretrained") as from_pretrained:
        result: Final = _select_tokenizer_helper(model)

    from_pretrained.assert_not_called()
    assert result["type"] == "openai_tokenizer"


def test_offline_mode_keeps_deepseek_v4_on_its_bundled_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")

    assert litellm.token_counter(model="deepseek/deepseek-v4-flash", text=_CODE_SAMPLE) == _deepseek_token_count(
        _CODE_SAMPLE
    )


def test_concurrent_first_deepseek_counts_share_one_loaded_tokenizer():
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    start: Final = threading.Barrier(8)

    def first_tokenizer(_: int) -> object:
        start.wait()
        return _select_tokenizer_helper("deepseek/deepseek-v4-flash")["tokenizer"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        tokenizers: Final = tuple(pool.map(first_tokenizer, range(8)))

    assert len({id(tokenizer) for tokenizer in tokenizers}) == 1


def test_remembered_tokenizer_failure_warns_once_instead_of_on_every_count(caplog: pytest.LogCaptureFixture):
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    try:
        with (
            patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=RuntimeError("HF unavailable")),
            caplog.at_level("WARNING", logger="LiteLLM"),
        ):
            types: Final = tuple(_select_tokenizer_helper(model)["type"] for _ in range(3))
        fallback_warnings: Final = [r for r in caplog.records if "token counts may be approximate" in r.getMessage()]
        assert types == ("openai_tokenizer",) * 3
        assert len(fallback_warnings) == 1
    finally:
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


def test_download_that_fails_after_another_count_loaded_the_tokenizer_counts_with_it():
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    slow_download_started: Final = threading.Event()
    loading_count_finished: Final = threading.Event()

    def download(_: str) -> object:
        if threading.current_thread().name.startswith("slow-download"):
            slow_download_started.set()
            loading_count_finished.wait(timeout=5)
            raise RuntimeError("HF rate limited")
        return sentinel.tokenizer

    try:
        with patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=download):
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="slow-download") as pool:
                slow_count: Final = pool.submit(_select_tokenizer_helper, model)
                assert slow_download_started.wait(timeout=5)
                loading_count: Final = _select_tokenizer_helper(model)
                loading_count_finished.set()
                slow_count_tokenizer: Final = slow_count.result(timeout=5)["tokenizer"]
            later_counts: Final = tuple(_select_tokenizer_helper(model) for _ in range(3))
        assert slow_count_tokenizer is sentinel.tokenizer
        assert loading_count["tokenizer"] is sentinel.tokenizer
        assert tuple(count["tokenizer"] for count in later_counts) == (sentinel.tokenizer,) * 3
    finally:
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


def test_loaded_tokenizer_serves_later_counts_without_downloading_it_again():
    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    try:
        with patch("litellm.utils.tokenizer_dispatch.from_pretrained", return_value=sentinel.tokenizer):
            loaded: Final = _select_tokenizer_helper(model)
        with patch(
            "litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=RuntimeError("HF rate limited")
        ) as from_pretrained:
            later_counts: Final = tuple(_select_tokenizer_helper(model) for _ in range(3))
        assert loaded["tokenizer"] is sentinel.tokenizer
        assert tuple(count["tokenizer"] for count in later_counts) == (sentinel.tokenizer,) * 3
        assert from_pretrained.call_count == 0
    finally:
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


def _run_now(retry: Callable[[], None]) -> None:
    retry()


def _start_retry_resolving(ended: Future[None]) -> Callable[[Callable[[], None]], None]:
    def start(retry: Callable[[], None]) -> None:
        def run_then_resolve() -> None:
            retry()
            ended.set_result(None)

        threading.Thread(target=run_then_resolve, daemon=True).start()

    return start


def _eventually(probe: Callable[[], bool]) -> bool:
    deadline: Final = time.monotonic() + 5
    while time.monotonic() < deadline:
        if probe():
            return True
        time.sleep(0.01)
    return False


def test_failed_download_is_retried_once_per_window_until_it_loads():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=299.0)
    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock, start_retry=_run_now)

    def download(_: str) -> object:
        if from_pretrained.call_count == 1:
            clock.return_value = 322.0
            raise RuntimeError("HF unavailable")
        if from_pretrained.call_count == 2:
            raise RuntimeError("HF rate limited")
        return sentinel.tokenizer

    with patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=download) as from_pretrained:
        first: Final = _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = 322.0 + HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS - 1
        inside_the_window: Final = _return_huggingface_tokenizer(model, outcomes)
        downloads_inside_the_window: Final = from_pretrained.call_count
        clock.return_value = 322.0 + HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        claimant: Final = _return_huggingface_tokenizer(model, outcomes)
        inside_the_next_window: Final = _return_huggingface_tokenizer(model, outcomes)
        downloads_inside_the_next_window: Final = from_pretrained.call_count
        clock.return_value = 322.0 + 2 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        _return_huggingface_tokenizer(model, outcomes)
        after_the_second_retry: Final = _return_huggingface_tokenizer(model, outcomes)
    assert isinstance(first, _HuggingFaceTokenizerLoadFailure)
    assert isinstance(inside_the_window, _HuggingFaceTokenizerLoadFailure)
    assert downloads_inside_the_window == 1
    assert isinstance(claimant, _HuggingFaceTokenizerLoadFailure)
    assert isinstance(inside_the_next_window, _HuggingFaceTokenizerLoadFailure)
    assert downloads_inside_the_next_window == 2
    assert from_pretrained.call_count == 3
    assert after_the_second_retry == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}


def test_retry_after_the_window_runs_in_the_background_while_every_count_answers_at_once():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock)
    retry_started: Final = threading.Event()
    retry_released: Final = threading.Event()

    def download(_: str) -> object:
        if from_pretrained.call_count == 1:
            raise RuntimeError("HF unavailable")
        retry_started.set()
        retry_released.wait(timeout=5)
        return sentinel.tokenizer

    with patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=download) as from_pretrained:
        _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = 3 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        counts_while_retry_runs: Final = tuple(_return_huggingface_tokenizer(model, outcomes) for _ in range(4))
        assert retry_started.wait(timeout=5)
        clock.return_value = 9 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        later_counts_while_retry_runs: Final = tuple(_return_huggingface_tokenizer(model, outcomes) for _ in range(2))
        downloads_while_retry_runs: Final = from_pretrained.call_count
        retry_released.set()
        retried: Final = _eventually(
            lambda: _return_huggingface_tokenizer(model, outcomes)
            == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}
        )
    assert all(
        isinstance(count, _HuggingFaceTokenizerLoadFailure)
        for count in counts_while_retry_runs + later_counts_while_retry_runs
    )
    assert downloads_while_retry_runs == 2
    assert retried
    assert from_pretrained.call_count == 2


def test_download_from_before_the_first_failure_ending_during_a_retry_leaves_it_the_only_retry():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock)
    early_download_started: Final = threading.Event()
    early_download_released: Final = threading.Event()
    retry_started: Final = threading.Event()
    retry_released: Final = threading.Event()

    def download(_: str) -> object:
        if from_pretrained.call_count == 1:
            early_download_started.set()
            early_download_released.wait(timeout=5)
            raise RuntimeError("HF unavailable")
        if from_pretrained.call_count == 2:
            raise RuntimeError("HF unavailable")
        retry_started.set()
        retry_released.wait(timeout=5)
        return sentinel.tokenizer

    with (
        patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=download) as from_pretrained,
        ThreadPoolExecutor(max_workers=1) as early_pool,
    ):
        early_count: Final = early_pool.submit(_return_huggingface_tokenizer, model, outcomes)
        assert early_download_started.wait(timeout=5)
        _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS + 1.0
        _return_huggingface_tokenizer(model, outcomes)
        assert retry_started.wait(timeout=5)
        early_download_released.set()
        early: Final = early_count.result(timeout=5)
        clock.return_value = 5 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        counts_while_retry_runs: Final = tuple(_return_huggingface_tokenizer(model, outcomes) for _ in range(3))
        downloads_while_retry_runs: Final = from_pretrained.call_count
        retry_released.set()
        retried: Final = _eventually(
            lambda: _return_huggingface_tokenizer(model, outcomes)
            == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}
        )
    assert isinstance(early, _HuggingFaceTokenizerLoadFailure)
    assert all(isinstance(count, _HuggingFaceTokenizerLoadFailure) for count in counts_while_retry_runs)
    assert downloads_while_retry_runs == 3
    assert retried


def test_retry_failing_after_an_earlier_download_loaded_the_tokenizer_keeps_it():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    retry_ended: Final = Future[None]()
    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock, start_retry=_start_retry_resolving(retry_ended))
    early_download_started: Final = threading.Event()
    early_download_released: Final = threading.Event()
    retry_started: Final = threading.Event()
    retry_released: Final = threading.Event()

    def download(_: str) -> object:
        if from_pretrained.call_count == 1:
            early_download_started.set()
            early_download_released.wait(timeout=5)
            return sentinel.tokenizer
        if from_pretrained.call_count == 2:
            raise RuntimeError("HF unavailable")
        retry_started.set()
        retry_released.wait(timeout=5)
        raise RuntimeError("HF rate limited")

    with (
        patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=download) as from_pretrained,
        ThreadPoolExecutor(max_workers=1) as early_pool,
    ):
        early_count: Final = early_pool.submit(_return_huggingface_tokenizer, model, outcomes)
        assert early_download_started.wait(timeout=5)
        _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        _return_huggingface_tokenizer(model, outcomes)
        assert retry_started.wait(timeout=5)
        early_download_released.set()
        early: Final = early_count.result(timeout=5)
        retry_released.set()
        retry_ended.result(timeout=5)
        later_counts: Final = tuple(_return_huggingface_tokenizer(model, outcomes) for _ in range(3))
    loaded: Final = {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}
    assert early == loaded
    assert from_pretrained.call_count == 3
    assert later_counts == (loaded,) * 3


class _NativePanic(BaseException):
    pass


def _fork_guard_error() -> BaseException:
    from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted

    return ForkedAfterNativeRuntimeStarted("native runtime started before fork")


def _native_panic() -> BaseException:
    return _NativePanic("tokenizer parser panicked")


@pytest.mark.parametrize("escaping_error", (_fork_guard_error, _native_panic), ids=("fork_guard", "native_panic"))
def test_retry_that_raises_past_failure_handling_still_retries_after_the_next_window(
    escaping_error: Callable[[], BaseException],
):
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    error: Final = escaping_error()
    retries: Final = SimpleQueue[Future[None]]()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer-retry") as pool:

        def start_retry(retry: Callable[[], None]) -> None:
            retries.put(pool.submit(retry))

        outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock, start_retry=start_retry)
        with patch(
            "litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=RuntimeError("HF unavailable")
        ) as from_pretrained:
            _return_huggingface_tokenizer(model, outcomes)
            clock.return_value = HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
            from_pretrained.side_effect = error
            _return_huggingface_tokenizer(model, outcomes)
            raised: Final = retries.get(timeout=5).exception(timeout=5)
            from_pretrained.side_effect = None
            from_pretrained.return_value = sentinel.tokenizer
            inside_the_next_window: Final = _return_huggingface_tokenizer(model, outcomes)
            no_retry_inside_the_next_window: Final = retries.empty()
            clock.return_value = 2 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
            _return_huggingface_tokenizer(model, outcomes)
            retries.get(timeout=5).result(timeout=5)
            after_the_next_window: Final = _return_huggingface_tokenizer(model, outcomes)
    assert raised is error
    assert isinstance(inside_the_next_window, _HuggingFaceTokenizerLoadFailure)
    assert no_retry_inside_the_next_window
    assert from_pretrained.call_count == 3
    assert after_the_next_window == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}


def test_process_forked_during_a_retry_starts_its_own_retry_instead_of_waiting_on_the_parents():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    pid: Final = MagicMock(return_value=100)
    start_retry: Final = MagicMock()
    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock, start_retry=start_retry, pid=pid)
    with patch(
        "litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=RuntimeError("HF unavailable")
    ) as from_pretrained:
        _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        _return_huggingface_tokenizer(model, outcomes)
        _return_huggingface_tokenizer(model, outcomes)
        retries_started_by_the_parent: Final = start_retry.call_count
        pid.return_value = 101
        from_pretrained.side_effect = None
        from_pretrained.return_value = sentinel.tokenizer
        _return_huggingface_tokenizer(model, outcomes)
        start_retry.call_args.args[0]()
        in_the_forked_process: Final = _return_huggingface_tokenizer(model, outcomes)
    assert retries_started_by_the_parent == 1
    assert start_retry.call_count == 2
    assert in_the_forked_process == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}


def test_retry_thread_that_cannot_start_answers_the_count_and_retries_after_the_next_window():
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    clock: Final = MagicMock(return_value=0.0)
    thread_starts: Final = MagicMock(side_effect=(RuntimeError("can't start new thread"), None))

    def start_thread_then_retry(retry: Callable[[], None]) -> None:
        thread_starts()
        retry()

    outcomes: Final = _HuggingFaceTokenizerLoadOutcomes(clock=clock, start_retry=start_thread_then_retry)
    with patch(
        "litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=RuntimeError("HF unavailable")
    ) as from_pretrained:
        _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        when_the_thread_cannot_start: Final = _return_huggingface_tokenizer(model, outcomes)
        from_pretrained.side_effect = None
        from_pretrained.return_value = sentinel.tokenizer
        inside_the_next_window: Final = _return_huggingface_tokenizer(model, outcomes)
        clock.return_value = 2 * HUGGINGFACE_TOKENIZER_LOAD_RETRY_SECONDS
        _return_huggingface_tokenizer(model, outcomes)
        after_the_next_window: Final = _return_huggingface_tokenizer(model, outcomes)
    assert isinstance(when_the_thread_cannot_start, _HuggingFaceTokenizerLoadFailure)
    assert isinstance(inside_the_next_window, _HuggingFaceTokenizerLoadFailure)
    assert from_pretrained.call_count == 2
    assert after_the_next_window == {"type": "huggingface_tokenizer", "tokenizer": sentinel.tokenizer}


def test_fork_guard_error_while_loading_a_tokenizer_is_raised_and_not_remembered():
    from litellm.rust_bridge.fork_guard import ForkedAfterNativeRuntimeStarted

    _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()
    model: Final = "databricks/databricks-meta-llama-3-3-70b-instruct"
    forked: Final = ForkedAfterNativeRuntimeStarted("native runtime started before fork")
    try:
        with patch("litellm.utils.tokenizer_dispatch.from_pretrained", side_effect=forked):
            with pytest.raises(ForkedAfterNativeRuntimeStarted):
                _select_tokenizer_helper(model)
        with patch("litellm.utils.tokenizer_dispatch.from_pretrained", return_value=sentinel.tokenizer):
            loaded: Final = _select_tokenizer_helper(model)
        assert loaded["type"] == "huggingface_tokenizer"
        assert loaded["tokenizer"] is sentinel.tokenizer
    finally:
        _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


def test_token_counter_with_anthropic_tool_use():
    """
    Test that _count_anthropic_content() correctly handles tool_use blocks.

    Validates that:
    - 'name' field is counted (string)
    - 'input' field is counted (dict serialized to string)
    - Metadata fields ('type', 'id') are skipped
    """
    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_01234567890",  # Should be skipped
                    "name": "get_weather",  # Should be counted
                    "input": {  # Should be counted (serialized)
                        "location": "San Francisco, CA",
                        "unit": "fahrenheit",
                    },
                },
            ],
        },
    ]

    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count: user message + "I'll check" text + "get_weather" name + input dict
    assert (
        tokens > 15
    ), f"Expected reasonable token count for message with tool_use, got {tokens}"


def test_token_counter_with_anthropic_tool_result():
    """
    Test that _count_anthropic_content() correctly handles tool_result blocks.

    Validates that:
    - 'content' field (when string) is counted
    - Metadata fields ('type', 'tool_use_id') are skipped
    - Full conversation with tool_use → tool_result flow works
    """
    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01234567890",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234567890",  # Should be skipped
                    "content": "The weather in San Francisco is 65°F and sunny.",  # Should be counted
                }
            ],
        },
    ]

    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    assert (
        tokens > 25
    ), f"Expected reasonable token count for conversation with tool_result, got {tokens}"


def test_token_counter_with_nested_tool_result():
    """
    Test that _count_anthropic_content() recursively handles nested content lists.

    Validates that:
    - tool_result with 'content' as a list (not string) is handled
    - Nested content blocks are recursively counted via _count_content_list()
    - TypedDict inference correctly identifies list fields
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01234567890",
                    "content": [  # Nested list - should recursively count
                        {
                            "type": "text",
                            "text": "The weather in San Francisco is 65°F and sunny.",
                        },
                        {"type": "text", "text": "UV index is moderate."},
                    ],
                }
            ],
        }
    ]

    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count both nested text blocks
    assert (
        tokens > 15
    ), f"Expected reasonable token count for nested tool_result, got {tokens}"


def test_token_counter_tool_use_and_result_combined():
    """
    Test dynamic field inference with multiple tool_use and tool_result blocks.

    Validates that:
    - Multiple tool_use blocks in same message are handled
    - Multiple tool_result blocks in same message are handled
    - skip_fields correctly filters metadata across all blocks
    - Full realistic conversation flow works end-to-end
    """
    messages = [
        {
            "role": "user",
            "content": "What's the weather in San Francisco and New York?",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll check the weather in both cities for you.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01A",
                    "name": "get_weather",
                    "input": {"location": "San Francisco, CA"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01B",
                    "name": "get_weather",
                    "input": {"location": "New York, NY"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A",
                    "content": "San Francisco: 65°F, sunny",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01B",
                    "content": "New York: 45°F, cloudy",
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The weather in San Francisco is 65°F and sunny, while New York is cooler at 45°F and cloudy.",
        },
    ]

    tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count all text, tool names, inputs, and results
    assert (
        tokens > 60
    ), f"Expected substantial token count for full tool conversation, got {tokens}"


def test_token_counter_with_image_url():
    """
    Test that _count_image_tokens() correctly handles image_url content blocks.

    Validates that:
    - image_url as dict with 'url' and 'detail' is handled
    - image_url as string is handled
    - 'detail' field validation works ('low', 'high', 'auto')
    - calculate_img_tokens is called with correct parameters
    """
    # Test with dict format (detail: low)
    messages_dict = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "low",  # Should use low token count (85 base tokens)
                    },
                },
            ],
        }
    ]

    tokens_dict = token_counter(
        model="gpt-3.5-turbo",
        messages=messages_dict,
        use_default_image_token_count=True,  # Avoid actual HTTP request
    )
    assert tokens_dict > 0, f"Expected positive token count, got {tokens_dict}"
    assert tokens_dict > 85, f"Expected at least base image tokens, got {tokens_dict}"

    # Test with string format (defaults to auto/low)
    messages_str = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg",  # String format
                }
            ],
        }
    ]

    tokens_str = token_counter(
        model="gpt-3.5-turbo", messages=messages_str, use_default_image_token_count=True
    )
    assert (
        tokens_str > 0
    ), f"Expected positive token count for string image_url, got {tokens_str}"

    # Test invalid detail value raises error
    messages_invalid = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "invalid",  # Should raise ValueError
                    },
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="Invalid detail value") as exc_info:
        token_counter(model="gpt-3.5-turbo", messages=messages_invalid)
    e = exc_info.value
    assert "Invalid detail value" in str(
        e
    ), f"Expected detail validation error, got: {e}"


def test_token_counter_with_thinking_content():
    """
    Test that _count_content_list() correctly handles Claude's extended thinking content blocks.

    Validates that:
    - 'thinking' content type is recognized and counted
    - 'thinking' text field is counted
    - 'signature' field is skipped (opaque signature blob)
    - Full conversation with thinking blocks works
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this complex problem: who came first, chicken or egg",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "This is actually a fascinating question that touches on philosophy, biology, and semantics. Let me break this down: The egg came first from an evolutionary biology perspective.",
                    "signature": "EqcLCkYICxgCKkCrqu6lP...",  # Should be skipped
                },
                {
                    "type": "text",
                    "text": "# The Chicken-or-Egg Question: A Multi-Layered Answer\n\n## **The Short Answer: The Egg Came First**",
                },
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
    ]

    tokens = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929", messages=messages
    )
    assert tokens > 0, f"Expected positive token count, got {tokens}"
    # Should count: user message + thinking text + response text + "Thanks"
    # The thinking text alone is ~30 tokens, plus other content should be > 50 total
    assert (
        tokens > 50
    ), f"Expected substantial token count for message with thinking, got {tokens}"

    # Test that thinking block without 'thinking' field doesn't crash (edge case)
    messages_no_thinking = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    # No 'thinking' field - should count as 0 tokens
                    "signature": "EqcLCkYICxgCKkCrqu6lP...",
                },
                {"type": "text", "text": "Response"},
            ],
        }
    ]

    tokens_no_thinking = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929", messages=messages_no_thinking
    )
    assert (
        tokens_no_thinking > 0
    ), f"Expected positive token count even with empty thinking, got {tokens_no_thinking}"
    # Should only count "Response" and message overhead
    assert (
        tokens_no_thinking < 15
    ), f"Expected minimal token count for empty thinking block, got {tokens_no_thinking}"


def test_token_counter_with_redacted_thinking_content():
    """
    A replayed redacted_thinking block (Anthropic redacted reasoning, or the /v1/messages bridge's stand-in
    for a reasoning item with no summary) counts zero tokens for its encrypted payload, like a thinking
    block with no text. It used to raise, which made is_prompt_caching_valid_prompt return False and the
    prompt_caching pre-call check stop pinning the deployment that held the cached prefix.
    """
    model = "anthropic/claude-sonnet-4-5-20250929"
    reply = {"type": "text", "text": "Draw from the box labeled Mixed, because that label must be wrong."}
    redacted_block = {"type": "redacted_thinking", "data": "EqQBCkYIBRgCKkBjZ2xhc3M" * 30}
    user_turn = {"role": "user", "content": [{"type": "text", "text": "Which box do you draw from?"}]}
    follow_up = {"role": "user", "content": [{"type": "text", "text": "Restate that in one sentence."}]}

    without_block = [user_turn, {"role": "assistant", "content": [reply]}, follow_up]
    with_block = [user_turn, {"role": "assistant", "content": [redacted_block, reply]}, follow_up]

    assert token_counter(model=model, messages=with_block) == token_counter(model=model, messages=without_block)


def test_token_counter_with_compaction_block():
    """
    A replayed on-demand compaction block (Anthropic's signed summary of the earlier turns) counts its
    summary text like a text block and nothing for its signature. It used to raise, which turned every
    /v1/messages/count_tokens call on a compacted history into a 500 once the provider count fell back here.
    """
    model = "anthropic/claude-sonnet-5-5"
    summary = "The user is building a recipe app in Python and asked for one-sentence class descriptions."
    follow_up = {"role": "user", "content": [{"type": "text", "text": "Now do the same for Ingredient."}]}
    signed_block = {"type": "compaction", "content": summary, "signature": "EqQBCkYIBRgCKkBjZ2xhc3M" * 40}

    as_compaction = [{"role": "assistant", "content": [signed_block]}, follow_up]
    unsigned = [{"role": "assistant", "content": [{"type": "compaction", "content": summary}]}, follow_up]
    as_text = [{"role": "assistant", "content": [{"type": "text", "text": summary}]}, follow_up]

    assert token_counter(model=model, messages=as_compaction) == token_counter(model=model, messages=as_text)
    assert token_counter(model=model, messages=unsigned) == token_counter(model=model, messages=as_text)


def test_token_counter_counts_chat_compaction_blocks_like_their_summary_text():
    """
    Chat clients replay a compaction block in the assistant message's provider_specific_fields, and inference
    sends it to Anthropic as a content block, so the local count includes its summary text and skips its signature.
    """
    model = "anthropic/claude-sonnet-5-5"
    summary = "The user is building a recipe app in Python and asked for one-sentence class descriptions."
    follow_up = {"role": "user", "content": "Now do the same for Ingredient."}
    signed_block = {"type": "compaction", "content": summary, "signature": "EqQBCkYIBRgCKkBjZ2xhc3M" * 40}

    replayed = [
        {"role": "assistant", "content": None, "provider_specific_fields": {"compaction_blocks": [signed_block]}},
        follow_up,
    ]
    as_text = [{"role": "assistant", "content": summary}, follow_up]

    assert token_counter(model=model, messages=replayed) == token_counter(model=model, messages=as_text)


class _BosEncoder:
    def encode_batch_fast(self, texts: list[str]) -> list[list[int]]:
        return [[0] * (len(text.split()) + 1) for text in texts]


def test_token_counter_charges_nothing_for_a_chat_compaction_block_with_no_summary():
    """
    A tokenizer that prepends a BOS token charges one token for the empty string, so a replayed
    compaction block with an empty summary must not reach it, like an empty thinking block.
    """
    model = "anthropic/claude-sonnet-5-5"
    tokenizer = {"type": "huggingface_tokenizer", "tokenizer": _BosEncoder()}
    follow_up = {"role": "user", "content": "Now do the same for Ingredient."}
    replayed = [
        {
            "role": "assistant",
            "content": None,
            "provider_specific_fields": {"compaction_blocks": [{"type": "compaction", "content": ""}]},
        },
        follow_up,
    ]
    bare = [{"role": "assistant", "content": None}, follow_up]

    assert token_counter(model=model, messages=replayed, custom_tokenizer=tokenizer) == token_counter(
        model=model, messages=bare, custom_tokenizer=tokenizer
    )


def test_count_content_list_skips_the_tokenizer_for_empty_block_text():
    """
    An empty thinking block, a redacted_thinking block (no thinking text at all) and a compaction
    block with no summary carry no tokens, even under a tokenizer that charges for the empty string
    (the llama-2 tokenizer prepends its BOS token to every encode call). Counting them through the
    tokenizer added one token per block on such models.
    """
    from litellm.litellm_core_utils.token_counter import _count_content_list

    def charges_for_empty_text(text: str) -> int:
        return len(text.split()) + 1

    empty_blocks = [
        {"type": "thinking", "thinking": "", "signature": "EqcLCkYICxgCKkCrqu6lP..."},
        {"type": "thinking", "thinking": None, "signature": "EqcLCkYICxgCKkCrqu6lP..."},
        {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIw"},
        {"type": "compaction", "content": ""},
        {"type": "text", "text": ""},
    ]

    assert _count_content_list(charges_for_empty_text, empty_blocks, False, None) == 0
    assert _count_content_list(charges_for_empty_text, [{"type": "thinking", "thinking": "two words"}], False, None) == 3


def test_token_counter_with_tool_reference_block():
    """
    Regression test: a message containing an Anthropic tool-search
    `tool_reference` content block must NOT raise.

    Before the fix, token_counter raised
    `Invalid content item type: tool_reference`. On the streaming
    anthropic_messages proxy path this nulled response_cost and caused the
    SpendLogs row to be dropped, silently undercounting cost. token_counter
    must instead count the referenced tool name and return a positive count.
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look up the right tool."},
                {"type": "tool_reference", "tool_name": "search_knowledge_base"},
            ],
        }
    ]

    # Must not raise, and must produce a positive token count.
    tokens = token_counter_new(
        model="anthropic/claude-sonnet-4-5-20250929", messages=messages
    )
    assert tokens > 0, f"Expected positive token count, got {tokens}"

    # A tool_reference with no/empty tool_name must also be handled gracefully.
    messages_empty = [
        {
            "role": "assistant",
            "content": [{"type": "tool_reference", "tool_name": ""}],
        }
    ]
    tokens_empty = token_counter_new(
        model="anthropic/claude-sonnet-4-5-20250929", messages=messages_empty
    )
    assert tokens_empty >= 0


def test_count_content_list_rejects_unknown_type():
    """
    An unrecognized content block type must raise, and the error message must
    enumerate the supported types (including `tool_reference`). This pins the
    catch-all contract so a future block type isn't silently dropped.
    """
    from litellm.litellm_core_utils.token_counter import _count_content_list

    with pytest.raises(ValueError, match='Error getting number of tokens from content list: Invalid') as exc_info:
        _count_content_list(
            count_function=len,
            content_list=[{"type": "totally_unknown_block"}],
            use_default_image_token_count=False,
            default_token_count=None,
        )

    message = str(exc_info.value)
    assert "Invalid content item type: totally_unknown_block" in message
    assert "tool_reference" in message


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
        {"type": "url", "url": "https://example.com/image.png"},
        {"type": "file", "file_id": "file-abc123"},
    ],
    ids=["base64", "url", "file"],
)
def test_token_counter_with_anthropic_image_block(source: dict[str, str]):
    """Anthropic `image` blocks must count for every source variant, not raise `Invalid content item type` (which the router's context-window pre-call check swallows into an unfiltered dispatch)."""
    from litellm.constants import DEFAULT_IMAGE_TOKEN_COUNT

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "source": source},
            ],
        }
    ]

    tokens = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=messages,
        use_default_image_token_count=True,
    )
    assert tokens > DEFAULT_IMAGE_TOKEN_COUNT, (
        f"Expected the image block to contribute tokens, got {tokens}"
    )


def test_anthropic_image_block_matches_equivalent_image_url():
    """An Anthropic `image` block prices identically to the OpenAI `image_url` carrying the same bytes."""
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                }
            ],
        }
    ]
    openai_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                }
            ],
        }
    ]

    anthropic_tokens = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929", messages=anthropic_messages
    )
    openai_tokens = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929", messages=openai_messages
    )
    assert anthropic_tokens == openai_tokens


def test_anthropic_image_block_nested_in_tool_result():
    """An `image` block nested in a `tool_result.content` list is counted through the same recursion."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ],
        }
    ]

    tokens = token_counter(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=messages,
        use_default_image_token_count=True,
    )
    assert tokens > 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"}, "data:image/jpeg;base64,/9j/4AAQ"),
        ({"type": "url", "url": "https://example.com/image.png"}, "https://example.com/image.png"),
        ({"type": "file", "file_id": "file-abc123"}, ""),
    ],
    ids=["base64", "url", "file"],
)
def test_anthropic_image_source_resolves_to_what_the_image_pricer_reads(source: dict[str, str], expected: str):
    """base64 sources become a data URI, url sources pass through, file sources resolve to an empty string."""
    from litellm.litellm_core_utils.token_counter import _anthropic_image_source_data

    assert _anthropic_image_source_data(source) == expected


def test_anthropic_image_block_with_empty_base64_data():
    """A base64 source with empty `data` prices as an image rather than raising."""
    from litellm.litellm_core_utils.token_counter import _count_content_list

    tokens = _count_content_list(
        count_function=len,
        content_list=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}
        ],
        use_default_image_token_count=False,
        default_token_count=None,
    )
    assert tokens > 0


def test_token_counter_counts_anthropic_search_result_text() -> None:
    from litellm.litellm_core_utils.token_counter import _count_content_list

    search_result: Final = {
        "type": "search_result",
        "source": "https://example.invalid/notes",
        "title": "Project notes",
        "content": [{"type": "text", "text": "The code word is 7f3a."}],
        "citations": {"enabled": True},
    }

    assert _count_content_list(len, [search_result], False, None) == len(
        "https://example.invalid/notesProject notesThe code word is 7f3a."
    )


def test_token_counter_counts_a_search_result_without_a_title() -> None:
    from litellm.litellm_core_utils.token_counter import _count_content_list

    search_result: Final = {
        "type": "search_result",
        "source": "https://example.invalid/notes",
        "content": [{"type": "text", "text": "The code word is 7f3a."}],
    }

    assert _count_content_list(len, [search_result], False, None) == len(
        "https://example.invalid/notesThe code word is 7f3a."
    )


@pytest.mark.parametrize("nested", (False, True))
def test_unknown_content_type_remains_a_value_error_with_an_optional_fallback(nested: bool) -> None:
    from litellm.litellm_core_utils.token_counter import _count_content_list

    unknown: Final = {"type": "future_block"}
    content: Final = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [unknown]}] if nested else [unknown]

    with pytest.raises(ValueError, match="Invalid content item type: future_block"):
        _count_content_list(len, content, False, None)
    assert _count_content_list(len, content, False, 7) == 7


def test_anthropic_image_block_without_source_raises():
    """An `image` block with no `source` raises, matching the OpenAI `image_url`-without-`url` behavior."""
    from litellm.litellm_core_utils.token_counter import _count_content_list

    with pytest.raises(ValueError, match="Error getting number of tokens from content list"):
        _count_content_list(
            count_function=len,
            content_list=[{"type": "image"}],
            use_default_image_token_count=False,
            default_token_count=None,
        )

    # ... and `default_token_count`, the caller's opt-out from raising, still wins.
    assert (
        _count_content_list(
            count_function=len,
            content_list=[{"type": "image"}],
            use_default_image_token_count=False,
            default_token_count=7,
        )
        == 7
    )


def _count_user_content(content: list[dict], model: str = "anthropic/claude-fable-5") -> int:
    from litellm.litellm_core_utils.token_counter import token_counter

    return token_counter(
        model=model,
        messages=[{"role": "user", "content": content}],
        use_default_image_token_count=True,
    )


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"},
        {"type": "url", "url": "https://example.com/report.pdf"},
        {"type": "file", "file_id": "file-abc123"},
    ],
    ids=["base64", "url", "file"],
)
def test_anthropic_document_block_with_opaque_source_is_priced_like_an_image(source: dict[str, str]):
    """A `document` with no readable pages (a bare PDF header, a URL, a file id) is priced like an `image`, not raised on."""
    prompt = {"type": "text", "text": "Summarize this file."}

    assert _count_user_content([prompt, {"type": "document", "source": source}]) == _count_user_content(
        [prompt, {"type": "image", "source": source}]
    )


def test_anthropic_document_block_text_sources_count_their_text():
    """`text` and `content` document sources count the text they carry, as inline text blocks would."""
    prompt = {"type": "text", "text": "Summarize this file."}
    body = {"type": "text", "text": "Revenue grew eleven percent while churn fell to two percent."}
    picture = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}

    text_source = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": body["text"]}}
    assert _count_user_content([prompt, text_source]) == _count_user_content([prompt, body])

    string_content = {"type": "document", "source": {"type": "content", "content": body["text"]}}
    assert _count_user_content([prompt, string_content]) == _count_user_content([prompt, body])

    block_content = {"type": "document", "source": {"type": "content", "content": [body, picture]}}
    assert _count_user_content([prompt, block_content]) == _count_user_content([prompt, body, picture])


def test_anthropic_document_title_and_context_add_their_tokens():
    prompt = {"type": "text", "text": "Summarize this file."}
    source = {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"}
    described = {"type": "document", "source": source, "title": "Q3 board packet", "context": "Shared by finance"}

    assert _count_user_content([prompt, described]) == _count_user_content(
        [
            prompt,
            {"type": "text", "text": "Q3 board packet"},
            {"type": "text", "text": "Shared by finance"},
            {"type": "document", "source": source},
        ]
    )


def test_openai_file_block_prices_like_the_equivalent_anthropic_document():
    """An inline `file` is a `document` in the chat-completions dialect, so it must price identically, not raise.

    Before the fix `file` was missing from the content-block match even though `ChatCompletionFileObject`
    is in the union this counter accepts, so every local count of a Responses `input_file` raised
    `Invalid content item type: file` and surfaced as a 500 on /v1/responses/input_tokens.
    """
    prompt = {"type": "text", "text": "Summarize this file."}
    inline_file = {
        "type": "file",
        "file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64,JVBERi0xLjQK"},
    }
    document = {
        "type": "document",
        "title": "report.pdf",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"},
    }

    assert _count_user_content([prompt, inline_file]) == _count_user_content([prompt, document])
    assert _count_user_content([prompt, inline_file]) > _count_user_content([prompt])

    readable: Final = _pdf_base64(("Revenue grew eleven percent while churn fell to two percent.",))
    readable_file: Final = {"type": "file", "file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64," + readable}}
    readable_document: Final = {"type": "document", "title": "report.pdf", "source": _pdf_source(readable)}
    assert _count_user_content([prompt, readable_file]) == _count_user_content([prompt, readable_document])
    assert _count_user_content([prompt, readable_file]) > _count_user_content([prompt, inline_file])


def test_openai_file_block_without_inline_bytes_counts_what_it_carries():
    """A `file` block naming an uploaded file has no bytes to price, so it adds only the filename's tokens."""
    prompt = {"type": "text", "text": "Summarize this file."}

    by_id = {"type": "file", "file": {"file_id": "file-abc123"}}
    assert _count_user_content([prompt, by_id]) == _count_user_content([prompt])

    named = {"type": "file", "file": {"file_id": "file-abc123", "filename": "report.pdf"}}
    assert _count_user_content([prompt, named]) == _count_user_content(
        [prompt, {"type": "text", "text": "report.pdf"}]
    )


def _pdf_base64(
    pages: tuple[str, ...],
    width: int = 612,
    height: int = 792,
    xobjects: tuple[bytes, ...] = (),
    drawing: bytes = b"",
    page_resources: str | None = None,
) -> str:
    """`xobjects` are numbered from 4 + 2 * len(pages), every page lists the first one as its /X0 XObject, and every
    page's content stream ends with `drawing`. `page_resources` replaces every page's /Resources dictionary."""
    first_xobject: Final = 4 + 2 * len(pages)
    page_xobjects: Final = f" /XObject << /X0 {first_xobject} 0 R >>" if xobjects else ""
    resources: Final = page_resources if page_resources is not None else f"<< /Font << /F1 3 0 R >>{page_xobjects} >>"

    def page_objects(index: int, text: str) -> tuple[bytes, bytes]:
        escaped: Final = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream: Final = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1") + drawing
        content: Final = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        page: Final = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources {resources} /Contents {4 + 2 * index} 0 R >>"
        ).encode()
        return content, page

    kids: Final = " ".join(f"{5 + 2 * index} 0 R" for index in range(len(pages)))
    bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Count {len(pages)} /Kids [ {kids} ] >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        *chain.from_iterable(page_objects(index, text) for index, text in enumerate(pages)),
        *xobjects,
    )
    header: Final = b"%PDF-1.4\n"
    objects: Final = tuple(
        f"{number} 0 obj\n".encode() + body + b"\nendobj\n" for number, body in enumerate(bodies, start=1)
    )
    offsets: Final = accumulate((len(header), *(len(obj) for obj in objects[:-1])))
    xref: Final = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets
    )
    trailer: Final = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{len(header) + sum(len(obj) for obj in objects)}\n%%EOF\n"
    ).encode()
    return base64.b64encode(header + b"".join(objects) + xref + trailer).decode()


def _pdf_stream(content: bytes, dictionary: bytes = b"") -> bytes:
    dictionary_entries: Final = b" " + dictionary if dictionary else b""
    return (
        b"<< /Length " + str(len(content)).encode() + dictionary_entries + b" >>\nstream\n" + content + b"\nendstream"
    )


def _pdf_data_url_with_xobjects(
    page_content: bytes,
    page_xobjects: bytes,
    xobject_objects: tuple[bytes, ...],
    font_object: bytes = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
) -> str:
    object_bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        font_object,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 3 0 R >> /XObject " + page_xobjects + b" >> /Contents 5 0 R >>",
        _pdf_stream(page_content),
        *xobject_objects,
    )
    object_chunks: Final = tuple(
        f"{index} 0 obj\n".encode() + body + b"\nendobj\n" for index, body in enumerate(object_bodies, start=1)
    )
    header: Final = b"%PDF-1.4\n"
    offsets: Final = accumulate((len(header), *(len(chunk) for chunk in object_chunks[:-1])))
    xref: Final = f"xref\n0 {len(object_chunks) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets
    )
    trailer: Final = (
        f"trailer\n<< /Size {len(object_chunks) + 1} /Root 1 0 R >>\n"
        f"startxref\n{len(header) + sum(len(chunk) for chunk in object_chunks)}\n%%EOF\n"
    ).encode()
    return PDF_DATA_URL_PREFIX + base64.b64encode(header + b"".join(object_chunks) + xref + trailer).decode()


def _pdf_data_url_with_doubling_forms(depth: int, leaf_content: bytes) -> str:
    form_objects: Final = tuple(
        _pdf_stream(
            leaf_content if index == depth - 1 else b"/Fm Do\n/Fm Do\n",
            b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> /XObject "
            + (b"<< >>" if index == depth - 1 else b"<< /Fm " + str(7 + index).encode() + b" 0 R >>")
            + b" >>",
        )
        for index in range(depth)
    )
    return _pdf_data_url_with_xobjects(b"/Fm Do\n", b"<< /Fm 6 0 R >>", form_objects)


def _pdf_data_url_with_identity_h_font(
    page_content: bytes,
    page_xobjects: bytes = b"<< >>",
    xobject_objects: tuple[bytes, ...] = (),
) -> str:
    descendant_font_number: Final = 6 + len(xobject_objects)
    cmap_number: Final = descendant_font_number + 1
    descriptor_number: Final = cmap_number + 1
    font_object: Final = (
        b"<< /Type /Font /Subtype /Type0 /BaseFont /Test /Encoding /Identity-H /DescendantFonts ["
        + str(descendant_font_number).encode()
        + b" 0 R] /ToUnicode "
        + str(cmap_number).encode()
        + b" 0 R >>"
    )
    descendant_font: Final = (
        b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Test "
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        b"/FontDescriptor " + str(descriptor_number).encode() + b" 0 R /DW 1000 >>"
    )
    cmap: Final = (
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Test def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n"
        b"<0000> <FFFF>\n"
        b"endcodespacerange\n"
        b"1 beginbfchar\n"
        b"<0001> <00480065006C006C006F00200077006F0072006C0064>\n"
        b"endbfchar\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end\n"
        b"end"
    )
    font_descriptor: Final = (
        b"<< /Type /FontDescriptor /FontName /Test /Flags 4 /FontBBox [0 0 1000 1000] "
        b"/ItalicAngle 0 /Ascent 1000 /Descent 0 /CapHeight 700 /StemV 80 >>"
    )
    return _pdf_data_url_with_xobjects(
        page_content,
        page_xobjects,
        (*xobject_objects, descendant_font, _pdf_stream(cmap), font_descriptor),
        font_object,
    )


def _pdf_data_url_with_shared_xobjects(
    page_contents: tuple[bytes, ...],
    xobject_objects: tuple[bytes, ...],
) -> str:
    page_count: Final = len(page_contents)
    page_object_numbers: Final = tuple(5 + index for index in range(page_count))
    content_object_numbers: Final = tuple(5 + page_count + index for index in range(page_count))
    first_xobject_object_number: Final = 5 + 2 * page_count
    page_references: Final = b" ".join(f"{number} 0 R".encode() for number in page_object_numbers)
    xobject_references: Final = b" ".join(
        f"/Fm{index} {first_xobject_object_number + index} 0 R".encode() for index in range(len(xobject_objects))
    )
    page_objects: Final = tuple(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources 4 0 R /Contents "
        + f"{content_object} 0 R >>".encode()
        for content_object in content_object_numbers
    )
    content_objects: Final = tuple(_pdf_stream(content) for content in page_contents)
    object_bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + page_references + f"] /Count {page_count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Font << /F1 3 0 R >> /XObject << " + xobject_references + b" >> >>",
        *page_objects,
        *content_objects,
        *xobject_objects,
    )
    object_chunks: Final = tuple(
        f"{index} 0 obj\n".encode() + body + b"\nendobj\n" for index, body in enumerate(object_bodies, start=1)
    )
    header: Final = b"%PDF-1.4\n"
    offsets: Final = accumulate((len(header), *(len(chunk) for chunk in object_chunks[:-1])))
    xref: Final = f"xref\n0 {len(object_chunks) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets
    )
    trailer: Final = (
        f"trailer\n<< /Size {len(object_chunks) + 1} /Root 1 0 R >>\n"
        f"startxref\n{len(header) + sum(len(chunk) for chunk in object_chunks)}\n%%EOF\n"
    ).encode()
    return PDF_DATA_URL_PREFIX + base64.b64encode(header + b"".join(object_chunks) + xref + trailer).decode()


class _TestPdfMediaBox(Protocol):
    width: float
    height: float


class _TestPdfContentStream(Protocol):
    def get_data(self) -> bytes: ...


class _TestPdfDictionary(Protocol):
    def get(self, key: str) -> object | None: ...

    def items(self) -> Iterable[tuple[str, object]]: ...

    def __getitem__(self, key: str) -> object: ...


class _TextExtractionMock(Protocol):
    call_count: int
    return_value: str

    def __call__(self) -> str: ...

    def assert_not_called(self) -> None: ...


class _TestPdfPage(Protocol):
    mediabox: _TestPdfMediaBox
    extract_text: _TextExtractionMock

    def items(self) -> Iterable[tuple[str, object]]: ...

    def get(self, key: str) -> object | None: ...

    def get_contents(self) -> _TestPdfContentStream | None: ...


class _PdfTextPage(Protocol):
    def extract_text(self) -> str | None: ...


class _PdfTextPages(Protocol):
    def __getitem__(self, index: int) -> _PdfTextPage: ...


class _PdfTextReader(Protocol):
    pages: _PdfTextPages


class _PdfReaderModule(Protocol):
    PdfReader: Callable[[io.BytesIO], _PdfTextReader]


class _PypdfGenericModule(Protocol):
    DictionaryObject: Callable[[Mapping[str, object]], _TestPdfDictionary]
    NameObject: Callable[[str], str]


def _pdf_text_page(pdf_bytes: bytes) -> _PdfTextPage:
    pypdf_module: Final = cast(_PdfReaderModule, importlib.import_module("pypdf"))
    return pypdf_module.PdfReader(io.BytesIO(pdf_bytes)).pages[0]


def _pdf_extracted_text(pdf_bytes: bytes) -> str:
    return _pdf_text_page(pdf_bytes).extract_text() or ""


def _pdf_source(pdf_base64: str) -> dict[str, str]:
    return {"type": "base64", "media_type": "application/pdf", "data": pdf_base64}


def test_base64_pdf_document_counts_every_page_text_and_rendering():
    """A base64 PDF is read page by page: each page costs its text plus the image Anthropic renders it to.

    Before the fix the whole document was priced as one 85-token image, so a count_tokens call that fell back
    to the local counter answered 116 for a 12-page PDF the provider then billed at 35941 input tokens.
    """
    prompt: Final = {"type": "text", "text": "Summarize this file."}
    first: Final = "Revenue grew eleven percent while churn fell to two percent."
    second: Final = "Headcount is flat and the office lease was renewed for three years."
    base: Final = _count_user_content([prompt])
    blank_page: Final = _count_user_content([prompt, {"type": "document", "source": _pdf_source(_pdf_base64(("",)))}]) - base

    assert blank_page > 0
    assert _count_user_content([prompt, {"type": "document", "source": _pdf_source(_pdf_base64((first, second)))}]) == (
        _count_user_content([prompt, {"type": "text", "text": first}, {"type": "text", "text": second}]) + 2 * blank_page
    )
    one_page: Final = _count_user_content([prompt, {"type": "document", "source": _pdf_source(_pdf_base64((first,)))}]) - base
    twelve_pages: Final = (
        _count_user_content([prompt, {"type": "document", "source": _pdf_source(_pdf_base64((first,) * 12))}]) - base
    )
    assert one_page > _count_user_content([prompt, {"type": "text", "text": first}]) - base
    assert twelve_pages == 12 * one_page


@pytest.mark.parametrize("fields", [{}, {"title": "Q3 board packet", "context": "Shared by finance"}], ids=["bare", "described"])
def test_pdf_page_rendering_cost_follows_anthropic_image_scaling(fields: dict[str, str]):
    """Anthropic rasterizes each PDF page within its image limits (1568 px long edge, 1.15 MP) and bills
    width * height / 750 tokens for it: https://platform.claude.com/docs/en/build-with-claude/pdf-support and
    https://platform.claude.com/docs/en/build-with-claude/vision, read 2026-10-07, when Bedrock billed about
    1550 tokens per blank Letter page on both Sonnet 4.6 and Opus 4.8.
    """
    prompt: Final = {"type": "text", "text": "Summarize this file."}

    def blank_page_cost(width: int, height: int) -> int:
        with_page: Final = {"type": "document", "source": _pdf_source(_pdf_base64(("",), width, height)), **fields}
        without_page: Final = {"type": "document", "source": _pdf_source(_pdf_base64((), width, height)), **fields}
        return _count_user_content([prompt, with_page]) - _count_user_content([prompt, without_page])

    letter: Final = blank_page_cost(612, 792)
    poster: Final = blank_page_cost(2448, 3168)
    strip: Final = blank_page_cost(1000, 100)

    assert letter == poster == 1534
    assert strip == 328


def _fake_pdf_page(content_bytes: int | bytes, include_xobjects: bool = False) -> _TestPdfPage:
    content: Final = b"x" * content_bytes if isinstance(content_bytes, int) else content_bytes
    pypdf_generic: Final = cast(_PypdfGenericModule, importlib.import_module("pypdf.generic"))
    name_object: Final = pypdf_generic.NameObject
    font: Final = pypdf_generic.DictionaryObject(
        {
            name_object("/Type"): name_object("/Font"),
            name_object("/Subtype"): name_object("/Type1"),
            name_object("/BaseFont"): name_object("/Helvetica"),
        }
    )
    resources: Final = pypdf_generic.DictionaryObject(
        {
            name_object("/Font"): pypdf_generic.DictionaryObject({name_object("/F1"): font}),
            **({name_object("/XObject"): pypdf_generic.DictionaryObject({})} if include_xobjects else {}),
        }
    )

    def get_page_resource(key: str) -> object | None:
        return resources if key == "/Resources" else None

    return cast(
        _TestPdfPage,
        SimpleNamespace(
            mediabox=SimpleNamespace(width=612.0, height=792.0),
            get_contents=lambda: SimpleNamespace(get_data=lambda: content),
            extract_text=cast(
                _TextExtractionMock,
                MagicMock(return_value="Revenue grew eleven percent while churn fell to two percent."),
            ),
            get=get_page_resource,
        ),
    )


def test_long_pdf_reads_text_from_a_bounded_number_of_pages():
    pages: Final = tuple(_fake_pdf_page(content_bytes=1_000) for _ in range(40))

    _price_pdf_pages(pages, len)

    assert sum(page.extract_text.call_count for page in pages) <= EXTRAPOLATION_SAMPLES


def test_dense_pdf_page_is_priced_from_its_size_without_extracting_its_text():
    string_operand: Final = b"(Revenue grew eleven percent)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    decoded_operand_bytes: Final = len(string_operand) - 2
    content: Final = text_operator * (2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // decoded_operand_bytes + 1)
    page: Final = _fake_pdf_page(content_bytes=content)

    priced: Final = _price_pdf_pages((page,), len)

    page.extract_text.assert_not_called()
    assert priced >= 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES


def test_oversized_pdf_page_is_priced_from_its_string_operands_without_extracting_text():
    content: Final = b"BT /F1 9 Tf (Figure 3) Tj ET\n" + b"10 10 m 200 300 l S\n" * 10_000
    page: Final = _fake_pdf_page(content_bytes=content)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")
    blank_page.extract_text.return_value = ""

    priced: Final = _price_pdf_pages((page,), len) - _price_pdf_pages((blank_page,), len)

    page.extract_text.assert_not_called()
    assert priced <= len(b"(Figure 3)")


def test_oversized_pdf_form_xobject_is_priced_without_extracting_its_text():
    string_operand: Final = b"(form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    form_content: Final = text_operator * repetitions
    form_object: Final = _pdf_stream(
        form_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)

    assert priced is not None
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= (len(string_operand) - 2) * repetitions


def test_repeated_pdf_form_draws_are_priced_with_multiplicity():
    string_operand: Final = b"(repeated form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 1_024 // len(text_operator) + 1
    form_object: Final = _pdf_stream(
        text_operator * repetitions,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n" * 1_000,
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced is not None
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= (len(string_operand) - 2) * repetitions * 1_000


def test_oversized_nested_pdf_form_xobject_is_priced_without_extracting_its_text():
    string_operand: Final = b"(nested form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    nested_content: Final = text_operator * repetitions
    nested_form: Final = _pdf_stream(
        nested_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    form: Final = _pdf_stream(
        b"/Fm2 Do\n",
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] "
        b"/Resources << /Font << /F1 3 0 R >> /XObject << /Fm2 7 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form, nested_form),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)

    assert priced is not None
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= (len(string_operand) - 2) * repetitions


def test_exponential_pdf_form_draws_avoid_full_text_extraction():
    depth: Final = 20
    form_objects: Final = tuple(
        _pdf_stream(
            (f"/Fm{index + 1} Do\n" * 2).encode() if index < depth - 1 else b"BT /F1 12 Tf (leaf) Tj ET\n",
            (
                f"/Type /XObject /Subtype /Form /BBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> /XObject << /Fm{index + 1} {8 + index} 0 R >> >>"
            ).encode()
            if index < depth - 1
            else b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
        )
        for index in range(depth)
    )
    data_url: Final = _pdf_data_url_with_shared_xobjects((b"/Fm0 Do\n",), form_objects)
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced is not None
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= 2 ** (depth - 1)


@pytest.mark.timeout(10)
def test_cyclic_exponential_pdf_form_walk_is_memoized():
    depth: Final = 25

    def cyclic_form(index: int) -> bytes:
        next_index: Final = (index + 1) % depth
        next_form_content: Final = f"/Fm{next_index} Do\n/Fm{next_index} Do\n".encode()
        form_content: Final = b"BT /F1 12 Tf (x) Tj ET\n" + next_form_content
        form_dictionary: Final = (
            f"/Type /XObject /Subtype /Form /BBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> /XObject << /Fm{next_index} {7 + next_index} 0 R >> >>"
        ).encode()
        return _pdf_stream(form_content, form_dictionary)

    form_objects: Final = tuple(cyclic_form(index) for index in range(depth))
    data_url: Final = _pdf_data_url_with_shared_xobjects((b"/Fm0 Do\n",), form_objects)
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced is not None
    assert priced > 0
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )


def test_cyclic_pdf_form_xobject_is_priced():
    path_operator: Final = b"10 10 m 200 300 l S\n"
    form_content: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1) + b"/Fm1 Do\n"
    form: Final = _pdf_stream(
        form_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /XObject << /Fm1 6 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form,),
    )

    priced: Final = _count_inline_pdf_tokens(data_url, len)

    assert priced is not None


def test_pdf_image_xobject_does_not_trigger_the_content_limit():
    page_text: Final = b"BT /F1 12 Tf 20 760 Td (small page text) Tj ET\n"
    image_data: Final = b"\x00" * (1024 * 1024)
    image: Final = _pdf_stream(
        image_data,
        b"/Type /XObject /Subtype /Image /Width 1024 /Height 1024 /ColorSpace /DeviceRGB /BitsPerComponent 8",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        page_text,
        b"<< /Im1 6 0 R >>",
        (image,),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)

    assert priced is not None
    assert any("small page text" in call.args[0] for call in count_function.call_args_list)


def test_pdf_pages_with_shared_resources_price_only_the_form_they_draw():
    page_count: Final = 20
    string_operand: Final = b"(text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2_800 // len(string_operand)
    page_contents: Final = tuple(f"/Fm{index} Do\n".encode() for index in range(page_count))
    form_objects: Final = tuple(
        _pdf_stream(
            text_operator * repetitions,
            b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
        )
        for _ in range(page_count)
    )
    data_url: Final = _pdf_data_url_with_shared_xobjects(page_contents, form_objects)
    blank_data_url: Final = _pdf_data_url_with_shared_xobjects((b"",) * page_count, ())
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    rendering_tokens: Final = _count_inline_pdf_tokens(blank_data_url, len)
    string_operand_bytes: Final = page_count * (len(string_operand) - 2) * repetitions

    assert priced is not None
    assert rendering_tokens is not None
    assert priced <= string_operand_bytes * 3 // 2 + rendering_tokens


def test_pdf_form_without_resources_inherits_page_xobjects():
    string_operand: Final = b"(inherited form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    nested_form_content: Final = text_operator * repetitions
    inherited_form: Final = _pdf_stream(
        nested_form_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    unused_form: Final = _pdf_stream(
        nested_form_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    form_without_resources: Final = _pdf_stream(
        b"/Fm2 Do\n",
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792]",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n",
        b"<< /Fm1 6 0 R /Fm2 7 0 R /Fm3 8 0 R >>",
        (form_without_resources, inherited_form, unused_form),
    )
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    rendering_tokens: Final = _count_inline_pdf_tokens(blank_data_url, len)
    string_operand_bytes: Final = (len(string_operand) - 2) * repetitions

    assert priced is not None
    assert rendering_tokens is not None
    assert priced >= string_operand_bytes
    assert priced <= string_operand_bytes * 3 // 2 + rendering_tokens


def test_oversized_pdf_string_operand_counts_escaped_newline_continuations():
    escaped_newlines: Final = b"a\\\n" * 10_000
    literal_content: Final = b"BT /F1 12 Tf (" + escaped_newlines + b"a" * 50_000 + b") Tj ET\n"
    page: Final = _fake_pdf_page(content_bytes=literal_content)

    priced: Final = _price_pdf_pages((page,), len)

    assert priced >= 60_000


def test_nested_pdf_literal_string_keeps_its_full_operand():
    content: Final = b"BT /F1 12 Tf (a(b)" + b"x" * 100_000 + b") Tj ET\n"
    page: Final = _fake_pdf_page(content_bytes=content)

    priced: Final = _price_pdf_pages((page,), len)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced >= 100_000


def test_pdf_form_invocations_allow_comments_and_escaped_names():
    string_operand: Final = b"(named form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 1_024 // len(text_operator) + 1
    form_object: Final = _pdf_stream(
        text_operator * repetitions,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    path_operator: Final = b"10 10 m 20 20 l S\n"
    page_content: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1) + (
        b"/Fm1 %comment\n Do\n/F#6d1 Do\n"
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        page_content,
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced is not None
    assert priced >= 2 * (len(string_operand) - 2) * repetitions


def test_pdf_percent_inside_literal_does_not_hide_form_invocation():
    string_operand: Final = b"(form text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 100 * 1024 // len(text_operator) + 1
    form_object: Final = _pdf_stream(
        text_operator * repetitions,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_xobjects(
        b"BT /F1 12 Tf (50%) Tj ET /Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _count_inline_pdf_tokens(data_url, count_function)  # pyright: ignore[reportPrivateUsage]  # PDF regression test

    assert priced is not None
    assert priced >= (len(string_operand) - 2) * repetitions
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )


@pytest.mark.parametrize(
    "comments",
    [
        b"%comment\r\n" * 23,
        b"% a   \r\n" * 40,
    ],
    ids=["23-crlf-comments", "40-crlf-comments-with-trailing-spaces"],
)
def test_pdf_invocation_comment_separators_are_priceable(comments: bytes):
    content: Final = b"BT /F1 " + comments + b"12 Tf (hello) Tj ET"
    page: Final = _fake_pdf_page(content_bytes=content, include_xobjects=True)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")

    priced: Final = _price_pdf_pages((page,), len)
    expected: Final = _price_pdf_pages((blank_page,), len)

    assert priced == expected


@pytest.mark.parametrize("inline_image_in_form", [False, True], ids=["page-stream", "form-stream"])
def test_pdf_inline_image_bytes_do_not_push_content_over_the_limit(inline_image_in_form: bool):
    image_data: Final = random.Random(42).randbytes(270_000)
    inline_image: Final = (
        b"BI /W 270000 /H 1 /BPC 8 /CS /RGB ID " + image_data + b" EI\nBT /F1 9 Tf (Hello world) Tj ET\n"
    )
    form: Final = _pdf_stream(
        inline_image,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    page_content: Final = b"/Fm1 Do\n" if inline_image_in_form else inline_image
    page_xobjects: Final = b"<< /Fm1 6 0 R >>" if inline_image_in_form else b"<< >>"
    form_objects: Final = (form,) if inline_image_in_form else ()
    data_url: Final = _pdf_data_url_with_xobjects(page_content, page_xobjects, form_objects)
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    page: Final = _pdf_text_page(pdf_bytes)
    page_text: Final = page.extract_text() or ""

    def count_function(value: str) -> int:
        return len(value)

    expected: Final = count_function(page_text)
    count_calls: Final = MagicMock(side_effect=count_function)
    priced: Final = _count_inline_pdf_tokens(data_url, count_calls)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, count_function)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == expected
    count_calls.assert_called_once_with(page_text)


@pytest.mark.parametrize(
    "prefix",
    [b"(BI) Tj\n", b"/BI 1 Tf\n", b"/BIfoo 1 Tf\n"],
    ids=["literal-string", "inline-image-name", "longer-name"],
)
def test_pdf_inline_image_operator_is_not_found_inside_strings_or_names(prefix: bytes):
    string_operand: Final = b"(sample text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    page: Final = _fake_pdf_page(content_bytes=prefix + text_operator * repetitions)
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _price_pdf_pages((page,), count_function)

    page.extract_text.assert_not_called()
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= (len(string_operand) - 2) * repetitions


@pytest.mark.parametrize(
    "prefix",
    [b"BI /W 1 ", b"BI /W 1 ID xx "],
    ids=["missing-id", "missing-ei"],
)
def test_unterminated_pdf_inline_images_are_priced_from_sample(prefix: bytes):
    string_operand: Final = b"(sample text)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    page: Final = _fake_pdf_page(content_bytes=prefix + text_operator * repetitions)
    count_function: Final = MagicMock(side_effect=len)

    priced: Final = _price_pdf_pages((page,), count_function)

    page.extract_text.assert_not_called()
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert priced >= (len(string_operand) - 2) * repetitions


def test_pdf_comments_are_not_counted_as_string_operands():
    comment: Final = b"% (lots of text)\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(comment) + 1
    content: Final = comment * repetitions + b"BT /F1 12 Tf (x) Tj ET\n"
    page: Final = _fake_pdf_page(content_bytes=content)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")
    blank_page.extract_text.return_value = ""

    priced: Final = _price_pdf_pages((page,), len)
    blank_price: Final = _price_pdf_pages((blank_page,), len)

    assert priced - blank_price == len(b"x")


def test_oversized_hex_pdf_strings_price_like_literal_text():
    literal_operand: Final = b"(Hello world)"
    hex_operand: Final = b"<48656c6c6f20776f726c64>"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(b"BT /F1 12 Tf " + hex_operand + b" Tj ET\n") + 1
    literal_page: Final = _fake_pdf_page(content_bytes=(b"BT /F1 12 Tf " + literal_operand + b" Tj ET\n") * repetitions)
    hex_page: Final = _fake_pdf_page(content_bytes=(b"BT /F1 12 Tf " + hex_operand + b" Tj ET\n") * repetitions)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")

    literal_price: Final = _price_pdf_pages((literal_page,), len)
    hex_price: Final = _price_pdf_pages((hex_page,), len)
    blank_price: Final = _price_pdf_pages((blank_page,), len)
    literal_text_tokens: Final = literal_price - blank_price
    hex_text_tokens: Final = hex_price - blank_price

    assert hex_text_tokens * 5 <= literal_text_tokens * 6
    assert literal_text_tokens * 5 <= hex_text_tokens * 6


def test_oversized_pdf_octal_escapes_keep_sample_scaling_by_decoded_bytes():
    plain_count: Final = PDF_PAGE_MAX_EXACT_CONTENT_BYTES // 2 + 2
    octal_count: Final = PDF_PAGE_MAX_EXACT_CONTENT_BYTES // 2 + 2
    text_object: Final = b"BT /F1 12 Tf (" + b"A" * plain_count + b"\\101" * octal_count + b") Tj ET\n"
    page: Final = _fake_pdf_page(content_bytes=text_object)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")

    priced: Final = _price_pdf_pages((page,), len)
    blank_price: Final = _price_pdf_pages((blank_page,), len)
    expected: Final = plain_count + octal_count

    assert expected * 3 // 4 <= priced - blank_price <= expected * 5 // 4


def test_oversized_pdf_text_uses_a_sampled_token_estimate():
    string_operand: Final = b"(Revenue grew eleven percent)"
    text_operator: Final = b"BT /F1 12 Tf " + string_operand + b" Tj ET\n"
    repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_operator) + 1
    page: Final = _fake_pdf_page(content_bytes=text_operator * repetitions)
    blank_page: Final = _fake_pdf_page(content_bytes=b"")

    def four_chars_per_token(value: str) -> int:
        return len(value) // 4

    page_priced: Final = _price_pdf_pages((page,), four_chars_per_token)  # pyright: ignore[reportPrivateUsage]  # PDF regression test
    blank_page_priced: Final = _price_pdf_pages((blank_page,), four_chars_per_token)  # pyright: ignore[reportPrivateUsage]  # PDF regression test
    priced: Final = page_priced - blank_page_priced
    expected: Final = len(string_operand) * repetitions // 4

    assert expected * 4 // 5 <= priced <= expected * 6 // 5


def test_many_tiny_pdf_text_objects_price_within_sample_estimate():
    repetitions: Final = 200_000
    text_object: Final = b"BT /F1 12 Tf (x) Tj ET\n"
    page: Final = _fake_pdf_page(text_object * repetitions)
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)
    expected: Final = repetitions

    assert expected * 3 // 4 <= priced - blank_price <= expected * 5 // 4


def test_oversized_literal_cut_after_escape_stays_extractable():
    text_prefix: Final = b"BT /F1 12 Tf "
    suffix: Final = b" Tj ET"
    operand_prefix_length: Final = PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(text_prefix) - len(suffix)
    operand: Final = b"(" + b"x" * (operand_prefix_length - 2) + b"\\(" + b"x" * 128 + b")"
    page: Final = _fake_pdf_page(text_prefix + operand + suffix)
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)
    expected: Final = operand.count(b"x")

    assert expected * 3 // 4 <= priced - blank_price <= expected * 5 // 4


@pytest.mark.parametrize("backslash_count", [1, 2, 3])
def test_oversized_unclosed_literal_with_backslash_run_stays_extractable(backslash_count: int):
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) - 1)
    literal_prefix: Final = b"BT /F1 12 Tf (escaped \\(paren\\) " + b"x" * 512 + b"\\" * backslash_count
    closed_suffix: Final = b")" * (1 + backslash_count % 2) + b" Tj ET\n"
    closed_page: Final = _fake_pdf_page(path_padding + literal_prefix + closed_suffix)
    unclosed_page: Final = _fake_pdf_page(path_padding + literal_prefix)
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)
    closed_price: Final = _price_pdf_pages((closed_page,), count_x_tokens) - blank_price
    unclosed_price: Final = _price_pdf_pages((unclosed_page,), count_x_tokens) - blank_price
    expected: Final = literal_prefix.count(b"x")

    assert expected * 3 // 4 <= closed_price <= expected * 5 // 4
    assert expected * 3 // 4 <= unclosed_price <= expected * 5 // 4


def test_unclosed_end_of_stream_text_keeps_sampling_form_text():
    form_text: Final = b"BT /F1 12 Tf (form text) Tj ET\n"
    form_repetitions: Final = 2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(form_text) + 1
    form_object: Final = _pdf_stream(
        form_text * form_repetitions,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    unclosed_data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm Do\nBT /F1 12 Tf (hi) Tj",
        b"<< /Fm 6 0 R >>",
        (form_object,),
    )
    closed_data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm Do\nBT /F1 12 Tf (hi) Tj ET\n",
        b"<< /Fm 6 0 R >>",
        (form_object,),
    )

    def count_form_tokens(value: str) -> int:
        return value.count("form")

    unclosed_price: Final = _count_inline_pdf_tokens(unclosed_data_url, count_form_tokens)
    closed_price: Final = _count_inline_pdf_tokens(closed_data_url, count_form_tokens)

    assert unclosed_price is not None
    assert closed_price is not None
    assert closed_price * 3 // 4 <= unclosed_price <= closed_price * 5 // 4


def test_oversized_literal_cut_after_et_inside_string_stays_extractable():
    text_prefix: Final = b"BT /F1 12 Tf ("
    literal_prefix: Final = b"x" * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES - len(text_prefix) - len(b"ET")) + b"ET"
    literal_suffix: Final = b"x" * 256 + b") Tj ET"
    page: Final = _fake_pdf_page(text_prefix + literal_prefix + literal_suffix)
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)
    expected: Final = (literal_prefix + literal_suffix[: -len(b") Tj ET")]).count(b"x")

    assert expected * 3 // 4 <= priced - blank_price <= expected * 5 // 4


def test_unterminated_pdf_text_object_ending_with_et_inside_string_is_closed():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    page: Final = _fake_pdf_page(path_padding + b"BT /F1 12 Tf (x ET")
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)

    assert priced - blank_price == 1


def test_oversized_text_cut_inside_trailing_comment_stays_extractable():
    text_prefix: Final = b"BT /F1 12 Tf (x) Tj "
    comment: Final = b"%" + b"comment" * PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    page: Final = _fake_pdf_page(text_prefix + comment + b"\nET")
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)

    assert priced - blank_price == 1


def test_oversized_tj_array_cut_inside_comment_stays_extractable():
    repetitions: Final = 1_000
    content: Final = (
        b"BT /F1 12 Tf ["
        + b"(x) 0 " * repetitions
        + b"% comment with ["
        + b"comment" * PDF_PAGE_MAX_EXACT_CONTENT_BYTES
        + b"\n] TJ ET"
    )
    page: Final = _fake_pdf_page(content)
    blank_page: Final = _fake_pdf_page(b"")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _price_pdf_pages((page,), count_x_tokens)
    blank_price: Final = _price_pdf_pages((blank_page,), count_x_tokens)

    assert repetitions * 3 // 4 <= priced - blank_price <= repetitions * 5 // 4


def test_oversized_identity_h_pdf_uses_tounicode_for_two_byte_glyphs():
    repetitions: Final = 1_000
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    text_object: Final = b"BT /F1 12 Tf " + b"<0001> Tj " * repetitions + b"ET\n"
    data_url: Final = _pdf_data_url_with_identity_h_font(path_padding + text_object)
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    exact_text: Final = _pdf_extracted_text(pdf_bytes)
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert exact_text == "Hello world" * repetitions
    assert priced is not None
    assert blank_price is not None
    assert len(exact_text) * 4 // 5 <= priced - blank_price <= len(exact_text) * 6 // 5


def test_oversized_helvetica_pdf_text_stays_close_to_exact_extraction():
    repetitions: Final = 1_000
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    text_object: Final = b"BT /F1 12 Tf " + b"(Hello world) Tj " * repetitions + b"ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(path_padding + text_object, b"<< >>", ())
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    exact_text: Final = _pdf_extracted_text(pdf_bytes)
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert exact_text == "Hello world" * repetitions
    assert priced is not None
    assert blank_price is not None
    assert len(exact_text) * 4 // 5 <= priced - blank_price <= len(exact_text) * 6 // 5


def test_oversized_identity_h_pdf_counts_text_after_path_padding():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    text_object: Final = b"BT /F1 12 Tf <0001> Tj ET\n"
    data_url: Final = _pdf_data_url_with_identity_h_font(path_padding + text_object)
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == len("Hello world")


def test_depth_30_doubling_form_tree_prices_text_leaf_by_effective_draw_count():
    repetitions: Final = 1 << 30
    data_url: Final = _pdf_data_url_with_doubling_forms(31, b"BT /F1 12 Tf (x) Tj ET\n")

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    priced: Final = _count_inline_pdf_tokens(data_url, count_x_tokens)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, count_x_tokens)

    assert priced is not None
    assert blank_price is not None
    assert repetitions * 3 // 4 <= priced - blank_price <= repetitions * 5 // 4


def test_oversized_tj_array_truncation_keeps_its_text():
    content: Final = b"BT /F1 12 Tf [" + b"(x) 0 " * 20_000 + b"] TJ ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(content, b"<< >>", ())
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    exact_text: Final = _pdf_extracted_text(pdf_bytes)

    def count_x_tokens(value: str) -> int:
        return value.count("x")

    exact_tokens: Final = count_x_tokens(exact_text)
    priced: Final = _count_inline_pdf_tokens(data_url, count_x_tokens)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, count_x_tokens)

    assert priced is not None
    assert blank_price is not None
    assert exact_tokens > 10_000
    assert exact_tokens * 3 // 4 <= priced - blank_price <= exact_tokens * 5 // 4


def test_compact_bt_operator_keeps_oversized_page_text():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    content: Final = path_padding + b"BT/F1 12 Tf (Hello world) Tj ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(content, b"<< >>", ())
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == len("Hello world")


def test_nul_whitespace_after_bt_keeps_oversized_page_text():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    content: Final = path_padding + b"BT\x00/F1 12 Tf (Hello world) Tj ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(content, b"<< >>", ())
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == len("Hello world")


def test_compact_bi_operator_strips_inline_image_before_pricing_text():
    image_data: Final = random.Random(42).randbytes(270_000)
    content: Final = b"BI/W 270000/H 1/BPC 8/CS /RGB ID " + image_data + b" EI\nBT /F1 9 Tf (Hello world) Tj ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(content, b"<< >>", ())
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    page: Final = _pdf_text_page(pdf_bytes)
    exact_tokens: Final = len(page.extract_text() or "")
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == exact_tokens


def test_pdf_text_object_strings_containing_bt_and_et_do_not_split_sampling():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    text_object: Final = b"BT /F1 12 Tf (before ET) Tj (inside BT) Tj (after) Tj ET\n"
    data_url: Final = _pdf_data_url_with_xobjects(path_padding + text_object, b"<< >>", ())
    pdf_bytes: Final = base64.b64decode(data_url.removeprefix(PDF_DATA_URL_PREFIX))
    exact_text: Final = _pdf_extracted_text(pdf_bytes)
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert exact_text == "before ETinside BTafter"
    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == len(exact_text)


@pytest.mark.parametrize("pdf_pricing", ["claude", "openai"])
@pytest.mark.parametrize(("second_text", "second_count"), [(b"second", 7), (b"", 0)])
def test_shared_pdf_form_uses_each_parents_resources(
    pdf_pricing: Literal["claude", "openai"], second_text: bytes, second_count: int
):
    path: Final = b"10 10 m 20 20 l S\n"
    padding: Final = path * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path) + 1)
    attributes: Final = b"/Type /XObject /Subtype /Form /BBox [0 0 612 792]"
    fonts: Final = b" /Resources << /Font << /F1 3 0 R >> >>"

    def document(second_leaf: int) -> str:
        return _pdf_data_url_with_xobjects(
            padding + b"/First Do\n/Second Do\n",
            b"<< /First 6 0 R /Second 7 0 R >>",
            (
                _pdf_stream(
                    b"/Shared Do",
                    attributes + b" /Resources << /XObject << /Shared 8 0 R /Leaf 9 0 R >> >>",
                ),
                _pdf_stream(
                    b"/Shared Do",
                    attributes + f" /Resources << /XObject << /Shared 8 0 R /Leaf {second_leaf} 0 R >> >>".encode(),
                ),
                _pdf_stream(b"/Leaf Do", attributes),
                _pdf_stream(b"BT /F1 12 Tf (first) Tj ET", attributes + fonts),
                _pdf_stream(b"BT /F1 12 Tf (" + second_text + b") Tj ET", attributes + fonts),
            ),
        )

    def count_text(text: str) -> int:
        return text.count("first") + 7 * text.count("second")

    shared_price: Final = _count_inline_pdf_tokens(document(9), count_text, pdf_pricing)
    distinct_price: Final = _count_inline_pdf_tokens(document(10), count_text, pdf_pricing)
    assert shared_price is not None
    assert distinct_price is not None
    assert distinct_price - shared_price == second_count - 1


@pytest.mark.parametrize("context_count", [0, 2, 8])
def test_shared_pdf_stream_is_scanned_once_across_inherited_contexts(context_count: int):
    attributes: Final = b"/Type /XObject /Subtype /Form /BBox [0 0 612 792]"
    shared_number: Final = 6 + context_count
    shared_content: Final = b"/Leaf Do"
    parent_content: Final = b"/Shared Do"
    parents: Final = tuple(
        _pdf_stream(
            parent_content,
            attributes
            + f" /Resources << /XObject << /Shared {shared_number} 0 R /Leaf {shared_number + index + 1} 0 R >> >>".encode(),
        )
        for index in range(context_count)
    )
    leaves: Final = tuple(
        _pdf_stream(
            b"BT /F1 12 Tf (" + b"x" * (index + 1) + b") Tj ET",
            attributes + b" /Resources << /Font << /F1 3 0 R >> >>",
        )
        for index in range(context_count)
    )
    path: Final = b"10 10 m 20 20 l S\n"
    padding: Final = path * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path) + 1)
    data_url: Final = _pdf_data_url_with_xobjects(
        padding + b"".join(f"/Parent{index} Do\n".encode() for index in range(context_count)),
        b"<< " + b" ".join(f"/Parent{index} {6 + index} 0 R".encode() for index in range(context_count)) + b" >>",
        (*parents, _pdf_stream(shared_content, attributes), *leaves),
    )
    blank_price: Final = _count_inline_pdf_tokens(_pdf_data_url_with_xobjects(b"", b"<< >>", ()), len)
    with patch(
        "litellm.litellm_core_utils.token_counter._pdf_string_operand_bytes", wraps=_pdf_string_operand_bytes
    ) as scan:
        price: Final = _count_inline_pdf_tokens(data_url, len)
    assert price is not None
    assert blank_price is not None
    assert price - blank_price == sum(range(1, context_count + 1))
    assert sum(call.args[0] == shared_content for call in scan.call_args_list) == int(context_count > 0)
    assert sum(call.args[0] == parent_content for call in scan.call_args_list) == context_count


def test_repeated_identity_h_form_draws_count_each_text_sample():
    path_operator: Final = b"10 10 m 20 20 l S\n"
    path_padding: Final = path_operator * (PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(path_operator) + 1)
    form_content: Final = b"BT /F1 12 Tf <0001> Tj ET\n"
    form_object: Final = _pdf_stream(
        form_content,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    data_url: Final = _pdf_data_url_with_identity_h_font(
        path_padding + b"/Fm1 Do\n/Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    priced: Final = _count_inline_pdf_tokens(data_url, len)
    blank_data_url: Final = _pdf_data_url_with_xobjects(b"", b"<< >>", ())
    blank_price: Final = _count_inline_pdf_tokens(blank_data_url, len)

    assert priced is not None
    assert blank_price is not None
    assert priced - blank_price == 2 * len("Hello world")


def test_nul_whitespace_invokes_large_form_with_bounded_sample_pricing():
    text_object: Final = b"BT /F1 12 Tf (Hello world) Tj ET\n"
    repetitions: Final = 8 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(text_object) + 1
    form_object: Final = _pdf_stream(
        text_object * repetitions,
        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >>",
    )
    nul_data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1\x00Do\n",
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    space_data_url: Final = _pdf_data_url_with_xobjects(
        b"/Fm1 Do\n",
        b"<< /Fm1 6 0 R >>",
        (form_object,),
    )
    nul_count_function: Final = MagicMock(side_effect=len)
    space_count_function: Final = MagicMock(side_effect=len)

    nul_price: Final = _count_inline_pdf_tokens(nul_data_url, nul_count_function)
    space_price: Final = _count_inline_pdf_tokens(space_data_url, space_count_function)

    assert nul_price is not None
    assert space_price is not None
    assert nul_price == space_price
    assert nul_price > 0
    assert max((len(call.args[0]) for call in nul_count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )
    assert max((len(call.args[0]) for call in space_count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )


@pytest.mark.parametrize("pdf_pricing", ["claude", "openai"])
def test_dense_pdf_page_is_priced_without_extracting_its_full_text(pdf_pricing: Literal["claude", "openai"]):
    count_function: Final = MagicMock(side_effect=len)
    sentence: Final = "Revenue grew eleven percent while churn fell to two percent. "
    page_text: Final = sentence * (2 * PDF_PAGE_MAX_EXACT_CONTENT_BYTES // len(sentence))

    priced: Final = _count_inline_pdf_tokens(
        PDF_DATA_URL_PREFIX + _pdf_base64((page_text,)), count_function, pdf_pricing
    )

    assert priced is not None and priced >= len(page_text)
    assert max((len(call.args[0]) for call in count_function.call_args_list), default=0) <= (
        PDF_PAGE_MAX_EXACT_CONTENT_BYTES
    )


def test_long_pdf_of_identical_pages_prices_each_page_like_a_one_page_pdf():
    prompt: Final = {"type": "text", "text": "Summarize this file."}
    page_text: Final = "Revenue grew eleven percent while churn fell to two percent."
    page_count: Final = 3 * EXTRAPOLATION_SAMPLES + 1
    base: Final = _count_user_content([prompt])

    def pdf_cost(pages: int) -> int:
        document: Final = {"type": "document", "source": _pdf_source(_pdf_base64((page_text,) * pages))}
        return _count_user_content([prompt, document]) - base

    assert pdf_cost(page_count) == page_count * pdf_cost(1)


def test_pdf_document_without_pypdf_is_priced_like_an_image(monkeypatch: pytest.MonkeyPatch):
    prompt: Final = {"type": "text", "text": "Summarize this file."}
    source: Final = _pdf_source(_pdf_base64(("Revenue grew eleven percent while churn fell to two percent.",)))
    priced_by_page: Final = _count_user_content([prompt, {"type": "document", "source": source}])

    monkeypatch.setitem(sys.modules, "pypdf", None)

    priced_as_image: Final = _count_user_content([prompt, {"type": "document", "source": source}])
    assert priced_as_image == _count_user_content([prompt, {"type": "image", "source": source}])
    assert priced_as_image < priced_by_page


_REVENUE_LINE: Final = "Revenue grew eleven percent while churn fell to two percent."
_HEADCOUNT_LINE: Final = "Headcount is flat and the office lease was renewed for three years."
_ONE_PIXEL_IMAGE: Final = (
    b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 "
    b"/Length 1 >>\nstream\n\x00\nendstream"
)


_DRAW_X0: Final = b" q 10 0 0 10 0 0 cm /X0 Do Q"
_DRAW_X1: Final = b"/X1 Do"


def _form_xobject(resources_number: int, drawing: bytes = _DRAW_X1) -> bytes:
    return (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources {resources_number} 0 R "
        f"/Length {len(drawing)} >>\nstream\n".encode()
        + drawing
        + b"\nendstream"
    )


def _xobject_resources(xobject_number: int) -> bytes:
    return f"<< /XObject << /X1 {xobject_number} 0 R >> >>".encode()


def _as_pdf_document(pdf_base64: str) -> Mapping[str, object]:
    return {"type": "document", "source": _pdf_source(pdf_base64)}


def _as_pdf_file(pdf_base64: str) -> Mapping[str, object]:
    file_data: Final = "data:application/pdf;base64," + pdf_base64
    return {"type": "file", "file": {"filename": "report.pdf", "file_data": file_data}}


def _as_pdf_tool_result(pdf_base64: str) -> Mapping[str, object]:
    return {"type": "tool_result", "tool_use_id": "t1", "content": [_as_pdf_document(pdf_base64)]}


def _pdf_cost(model: str, pdf_base64: str, wrap: Callable[[str], Mapping[str, object]] = _as_pdf_document) -> int:
    return _count_user_content([wrap(pdf_base64)], model) - _count_user_content([wrap(_pdf_base64(()))], model)


def _text_cost(model: str, *lines: str) -> int:
    return sum(
        _count_user_content([{"type": "text", "text": line}], model) - _count_user_content([], model) for line in lines
    )


def _rendered_page_cost(width: int = 612, height: int = 792) -> int:
    return _pdf_cost("my-private-model", _pdf_base64(("",), width, height))


@pytest.mark.parametrize(
    "wrap", [_as_pdf_document, _as_pdf_file, _as_pdf_tool_result], ids=["document", "file", "tool_result"]
)
@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "azure/gpt-5.6-sol", "o3-2025-04-16", "azure/gpt-41", "eu/gpt-5.6-sol", "gpt-5.6-sol-2026-07-09"],
)
def test_openai_model_prices_each_pdf_text_page_as_its_text_plus_framing(
    model: str, wrap: Callable[[str], Mapping[str, object]]
):
    """OpenAI models bill a PDF page with text as its text plus 13 tokens and a blank page as nothing, with no page
    image. Measured 2026-10-20 on Azure chat completions gpt-5.6-sol (1, 2, and 10 pages of 15 text tokens billed
    +28, +56, +280, and 1, 2, or 10 blank pages billed the same as none) and on OpenAI /v1/responses/input_tokens
    for gpt-4o, gpt-4.1, gpt-5, gpt-5.5, o3, and o4-mini.

    Before the fix every model got Claude's pricing, so count_tokens answered about 1586 for a one-page text PDF
    that Azure gpt-5.6-sol billed at 61.
    """
    one_page: Final = _pdf_cost(model, _pdf_base64((_REVENUE_LINE,)), wrap)

    assert one_page == _text_cost(model, _REVENUE_LINE) + 13
    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE, _HEADCOUNT_LINE)), wrap) == (
        _text_cost(model, _REVENUE_LINE, _HEADCOUNT_LINE) + 2 * 13
    )
    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,) * 10), wrap) == 10 * one_page
    assert _pdf_cost(model, _pdf_base64(("",) * 5), wrap) == 0
    assert _pdf_cost(model, _pdf_base64(("", _REVENUE_LINE, "")), wrap) == one_page


@pytest.mark.parametrize(
    "xobjects",
    [(_ONE_PIXEL_IMAGE,), (_form_xobject(7), _xobject_resources(8), _ONE_PIXEL_IMAGE)],
    ids=["image_on_the_page", "image_inside_a_form"],
)
def test_openai_model_prices_a_page_that_draws_a_raster_image_as_its_text_plus_the_rendered_page(
    xobjects: tuple[bytes, ...],
):
    """A page embedding a raster image keeps the rendered-page estimate every model got before, so a reservation
    never undercounts it: on 2026-10-20 /v1/responses/input_tokens billed a page with one drawn image at 1620 on
    gpt-5.6-sol and gpt-6-sol, above OpenAI's own high-detail page image (765 for Letter), while a vector-drawn
    chart page billed no image.
    """
    model: Final = "gpt-4o"

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), xobjects=xobjects, drawing=_DRAW_X0)) == (
        _text_cost(model, _REVENUE_LINE) + _rendered_page_cost()
    )
    assert _pdf_cost(model, _pdf_base64(("",), xobjects=xobjects, drawing=_DRAW_X0)) == _rendered_page_cost()
    assert _pdf_cost(model, _pdf_base64(("",), 300, 300, xobjects=xobjects, drawing=_DRAW_X0)) == (
        _rendered_page_cost(300, 300)
    )


def test_openai_model_prices_no_page_image_for_an_image_the_page_lists_but_never_draws():
    """Pages sharing one /Resources all list its logo, but only the page that paints it bills a page image: measured
    2026-10-20 on Azure gpt-5.6-sol, a 3-page PDF drawing a logo on page 1 billed 403, not three page images."""
    model: Final = "gpt-4o"

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), xobjects=(_ONE_PIXEL_IMAGE,))) == (
        _text_cost(model, _REVENUE_LINE) + 13
    )
    form_listing_an_unused_image: Final = (_form_xobject(7, b""), _xobject_resources(8), _ONE_PIXEL_IMAGE)
    assert _pdf_cost(model, _pdf_base64(("",), xobjects=form_listing_an_unused_image, drawing=_DRAW_X0)) == 0


def test_openai_model_prices_no_image_for_forms_that_draw_none():
    """A vector-drawn form XObject bills no page image, and a form that lists itself cannot loop the walk."""
    model: Final = "gpt-4o"
    self_referencing_form: Final = (_form_xobject(7), _xobject_resources(6))

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), xobjects=self_referencing_form, drawing=_DRAW_X0)) == (
        _text_cost(model, _REVENUE_LINE) + 13
    )


_INLINE_IMAGE: Final = b" q 200 0 0 200 72 400 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x00 EI Q"


def test_openai_model_prices_a_page_that_draws_an_inline_image_like_an_image_xobject():
    """An image drawn inline in the content stream (BI ... ID ... EI) bills like an image XObject: measured
    2026-10-20, the same 200x200 pt picture billed 351 either way on Azure gpt-5.6-sol and 303 on gpt-4o."""
    model: Final = "gpt-4o"
    form_drawing_inline_image: Final = (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Length {len(_INLINE_IMAGE)} >>\nstream\n".encode()
        + _INLINE_IMAGE
        + b"\nendstream"
    )

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), drawing=_INLINE_IMAGE)) == (
        _text_cost(model, _REVENUE_LINE) + _rendered_page_cost()
    )
    assert _pdf_cost(model, _pdf_base64(("",), drawing=_INLINE_IMAGE, page_resources="<< >>")) == (
        _rendered_page_cost()
    )
    assert _pdf_cost(model, _pdf_base64(("",), xobjects=(form_drawing_inline_image,), drawing=_DRAW_X0)) == (
        _rendered_page_cost()
    )


def test_openai_model_keeps_the_text_of_a_page_whose_forms_nest_too_deep_to_walk():
    """A chain of 2,000 nested form XObjects used to exhaust the stack, which dropped the whole document's text and
    priced it as one image. The walk now stops at a fixed depth and prices the page as if it drew an image."""
    model: Final = "gpt-4o"
    depth: Final = 2000
    first: Final = 6
    chain_objects: Final = tuple(
        chain.from_iterable(
            (_form_xobject(first + 2 * level + 1), _xobject_resources(first + 2 * level + 2)) for level in range(depth)
        )
    )

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), xobjects=chain_objects, drawing=_DRAW_X0)) == (
        _text_cost(model, _REVENUE_LINE) + _rendered_page_cost()
    )


@pytest.mark.parametrize("drawing", [b" /A Do /B Do", b" /B Do /A Do"])
def test_openai_model_finds_an_image_a_shared_form_without_resources_draws_from_one_parent(drawing: bytes):
    """A form with no /Resources resolves names in the resources of whichever form draws it, so a child that two
    parents share reaches the image only through the parent whose resources list it."""
    model: Final = "gpt-4o"
    child_without_resources: Final = (
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Length 5 >>\nstream\n/Y Do\nendstream"
    )
    parents_and_child: Final = (
        _form_xobject(7, b"/C Do"),
        b"<< /XObject << /C 10 0 R >> >>",
        _form_xobject(9, b"/C Do"),
        b"<< /XObject << /C 10 0 R /Y 11 0 R >> >>",
        child_without_resources,
        _ONE_PIXEL_IMAGE,
    )

    assert _pdf_cost(
        model,
        _pdf_base64(
            ("",),
            xobjects=parents_and_child,
            drawing=drawing,
            page_resources="<< /XObject << /A 6 0 R /B 8 0 R >> >>",
        ),
    ) == _rendered_page_cost()


@pytest.mark.parametrize(
    ("drawing", "draws_image"),
    [(b" /X0 %scale it first\n Do", True), (b" /X0 % Do\n", False)],
    ids=["comment-before-do", "do-inside-a-comment"],
)
def test_openai_model_reads_a_comment_in_a_content_stream_as_whitespace(drawing: bytes, draws_image: bool):
    """A comment runs to the end of its line and separates tokens like whitespace, so it neither hides the Do after
    it nor draws anything itself."""
    model: Final = "gpt-4o"

    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,), xobjects=(_ONE_PIXEL_IMAGE,), drawing=drawing)) == (
        _text_cost(model, _REVENUE_LINE) + (_rendered_page_cost() if draws_image else 13)
    )


@pytest.mark.parametrize("escaped_name", ["/Im#231", "/#C4#E3", "/caf#C3#A9"], ids=["hex_escape", "gbk", "utf8"])
def test_openai_model_finds_an_image_drawn_by_an_escaped_name(escaped_name: str):
    """A name written with #xx escapes reads the same in the content stream and in /Resources, decoded the way pypdf
    decodes it, so an image drawn under such a name still prices the page as rendered."""
    model: Final = "gpt-4o"

    assert _pdf_cost(
        model,
        _pdf_base64(
            ("",),
            xobjects=(_ONE_PIXEL_IMAGE,),
            drawing=f" {escaped_name} Do".encode(),
            page_resources=f"<< /XObject << {escaped_name} 6 0 R >> >>",
        ),
    ) == _rendered_page_cost()


@pytest.mark.parametrize(("parent_count", "draws_too_many"), [(1, False), (2, True)])
def test_openai_model_prices_a_page_whose_forms_draw_too_many_xobjects_as_rendered(
    parent_count: int, draws_too_many: bool
):
    """A page prices as if it drew an image once its forms draw more than PDF_MAX_XOBJECT_DRAWS_PER_PAGE XObjects
    between them, so a page of forms nested to blow up the walk cannot stall the request."""
    model: Final = "gpt-4o"
    first: Final = 6
    child_number: Final = first + 2 * parent_count
    child_drawing: Final = b" ".join((b"/D Do",) * (PDF_MAX_XOBJECT_DRAWS_PER_PAGE // 2))
    child: Final = (
        f"<< /Type /XObject /Subtype /Form /BBox [0 0 1 1] /Length {len(child_drawing)} >>\nstream\n".encode()
        + child_drawing
        + b"\nendstream"
    )
    parents: Final = tuple(
        chain.from_iterable(
            (_form_xobject(first + 2 * index + 1, b"/C Do"), f"<< /XObject << /C {child_number} 0 R >> >>".encode())
            for index in range(parent_count)
        )
    )
    page_resources: Final = (
        "<< /Font << /F1 3 0 R >> /XObject << "
        + " ".join(f"/P{index} {first + 2 * index} 0 R" for index in range(parent_count))
        + " >> >>"
    )
    page_drawing: Final = b" " + b" ".join(f"/P{index} Do".encode() for index in range(parent_count))

    assert _pdf_cost(
        model,
        _pdf_base64(
            (_REVENUE_LINE,), xobjects=(*parents, child), drawing=page_drawing, page_resources=page_resources
        ),
    ) == _text_cost(model, _REVENUE_LINE) + (_rendered_page_cost() if draws_too_many else 13)


_NON_LATIN_SAMPLE: Final = "Привет, как дела? これは日本語のテキストです。 Ceci est un texte en français: éèàç. " * 8


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-sol",
        "gpt-5.4-mini",
        "gpt-6-sol",
        "azure/gpt-41",
        "azure/gpt-5.6-sol",
        "eu/gpt-5.6-sol",
        "gpt-5.6-sol-2026-07-09",
    ],
)
def test_openai_models_tiktoken_does_not_know_count_with_o200k(model: str):
    """OpenAI models newer than the installed tiktoken's map bill text with o200k_base: measured 2026-10-20 on
    /v1/responses/input_tokens, gpt-5, gpt-5.4-mini, gpt-5.5, gpt-5.6-sol, and gpt-6-sol all billed 255 for a
    non-Latin sample like this one that is 249 tokens in o200k_base and 297 in cl100k_base."""
    o200k: Final = tiktoken.get_encoding("o200k_base")

    assert token_counter_new(model=model, text=_NON_LATIN_SAMPLE) == len(o200k.encode(_NON_LATIN_SAMPLE))


@pytest.mark.parametrize(("deployment_name", "model"), [("azure/gpt-4", "gpt-4"), ("eu/gpt-35-turbo", "gpt-35-turbo")])
def test_prefixed_openai_model_names_count_text_like_the_model_they_name(deployment_name: str, model: str):
    assert token_counter_new(model=deployment_name, text=_NON_LATIN_SAMPLE) == token_counter_new(
        model=model, text=_NON_LATIN_SAMPLE
    )


@pytest.mark.parametrize(
    "model", ["azure/gpt-chat-latest", "azure/codex-mini", "azure/gpt-5.3-chat"]
)
def test_azure_openai_models_only_azure_lists_get_openai_pricing_and_o200k(model: str):
    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,))) == _text_cost(model, _REVENUE_LINE) + 13
    assert token_counter_new(model=model, text=_NON_LATIN_SAMPLE) == len(
        tiktoken.get_encoding("o200k_base").encode(_NON_LATIN_SAMPLE)
    )


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-4o",
        "github_copilot/gpt-4o",
        "azure_ai/gpt-4o",
        "azure/responses/gpt-5.6-sol",
        "azure/o_series/o3-mini",
    ],
)
def test_gpt_names_behind_any_provider_or_route_prefix_count_like_the_bare_name(model: str):
    """The proxy counts a deployment with its model name minus the provider prefix, so the SDK prices and encodes
    a prefixed GPT name the same way the bare name counts. OpenRouter gpt-4o billed a text PDF page at 237 on
    2026-10-20, closer to OpenAI's pricing than to Claude's 1534."""
    bare_name: Final = model.rpartition("/")[2]
    pdf_with_a_drawn_image: Final = _pdf_base64((_REVENUE_LINE,), xobjects=(_ONE_PIXEL_IMAGE,), drawing=_DRAW_X0)

    assert _pdf_cost(model, pdf_with_a_drawn_image) == _pdf_cost(bare_name, pdf_with_a_drawn_image)
    assert _pdf_cost(model, _pdf_base64(("",))) == 0
    assert token_counter_new(model=model, text=_NON_LATIN_SAMPLE) == len(
        tiktoken.get_encoding("o200k_base").encode(_NON_LATIN_SAMPLE)
    )


@pytest.mark.parametrize(
    "model",
    [
        "text-embedding-ada-002-v2",
        "tts-1",
        "whisper-1",
        "azure/whisper-1",
        "github_copilot/text-embedding-3-small-inference",
        "azure_text/gpt-35-turbo-instruct",
        "ft:davinci-002:my-org::abc123",
    ],
)
def test_openai_models_that_are_not_chat_models_keep_counting_with_cl100k_and_claude_pdf_pricing(model: str):
    assert token_counter_new(model=model, text=_NON_LATIN_SAMPLE) == len(
        tiktoken.get_encoding("cl100k_base").encode(_NON_LATIN_SAMPLE)
    )
    assert _pdf_cost(model, _pdf_base64(("",))) == 1534


@pytest.mark.parametrize(
    ("fine_tune", "base_model"),
    [
        ("ft:gpt-4o-mini-2024-07-18:my-org::abc123", "gpt-4o-mini-2024-07-18"),
        ("openai/ft:gpt-4.1-2025-04-14:my-org:support:def456", "gpt-4.1-2025-04-14"),
    ],
)
def test_openai_fine_tunes_count_like_their_base_model(fine_tune: str, base_model: str):
    pdf: Final = _pdf_base64((_REVENUE_LINE,))

    assert _pdf_cost(fine_tune, pdf) == _pdf_cost(base_model, pdf) == _text_cost(base_model, _REVENUE_LINE) + 13
    assert token_counter_new(model=fine_tune, text=_NON_LATIN_SAMPLE) == token_counter_new(
        model=base_model, text=_NON_LATIN_SAMPLE
    )


@pytest.mark.parametrize(
    "model",
    [
        "",
        "anthropic/claude-fable-5",
        "DeepSeek-V4-Flash-0731",
        "my-private-model",
        "azure/mistral-large-latest",
        "command-r-plus",
    ],
)
def test_non_openai_models_keep_pricing_pdf_pages_as_claude_renders_them(model: str):
    """Only known OpenAI and Azure OpenAI models switch to OpenAI's PDF billing, so an empty or unknown model keeps
    Claude's text plus rendered page image, 1534 tokens for a blank Letter page."""
    assert _pdf_cost(model, _pdf_base64(("",))) == 1534
    assert _pdf_cost(model, _pdf_base64((_REVENUE_LINE,))) == _text_cost(model, _REVENUE_LINE) + 1534


def test_messages_reach_token_count_prices_pdf_pages_like_token_counter():
    from litellm.litellm_core_utils.token_counter import messages_reach_token_count

    messages: Final = [{"role": "user", "content": [_as_pdf_document(_pdf_base64((_REVENUE_LINE, "")))]}]
    total: Final = token_counter_new(model="azure/gpt-5.6-sol", messages=messages)

    assert total == token_counter_new(model="azure/gpt-5.6-sol", messages=[{"role": "user", "content": []}]) + (
        _text_cost("azure/gpt-5.6-sol", _REVENUE_LINE) + 13
    )
    assert messages_reach_token_count(model="azure/gpt-5.6-sol", messages=messages, threshold=total) is True
    assert messages_reach_token_count(model="azure/gpt-5.6-sol", messages=messages, threshold=total + 1) is False


def _png_data_url(width: int, height: int) -> str:
    ihdr = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    return "data:image/png;base64," + base64.b64encode(ihdr + b"\x08\x06\x00\x00\x00").decode()


@pytest.mark.parametrize(
    ("width", "height", "expected"), [(256, 256, 262), (512, 512, 262), (768, 768, 772), (1024, 1024, 772)]
)
@pytest.mark.parametrize("detail", [None, "auto", "high"])
def test_inline_auto_images_count_high_detail_tiles(width: int, height: int, expected: int, detail: str | None) -> None:
    image_url = {"url": _png_data_url(width, height)}
    if detail is not None:
        image_url["detail"] = detail
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": image_url}]}]

    assert token_counter_new(model="gpt-4o", messages=messages) == expected


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "GIF", "WEBP"])
def test_inline_auto_images_read_supported_formats(image_format: str) -> None:
    output = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(output, format=image_format)
    url = f"data:image/{image_format.lower()};base64," + base64.b64encode(output.getvalue()).decode()

    assert calculate_img_tokens(url) == 765


@pytest.mark.parametrize(
    "data",
    [
        "https://example.com/image.png",
        "http://127.0.0.1/image.png",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "data:application/pdf;base64,JVBERi0xLjQK",
        "data:image/png;base64,%%%",
        "data:image/png;base64,iVBORw0KGgo=",
        "data:image/jpeg;base64,/9j/4AAQ",
        "data:image/png,not-base64",
        "",
        _png_data_url(0, 1024),
        _png_data_url(1024, 0),
    ],
)
def test_auto_image_fallback_does_not_fetch_urls_or_raise(data: str) -> None:
    # The unit suite disables sockets; a fetch must also not be attempted and swallowed.
    with patch("httpx.Client.get", side_effect=AssertionError("auto counting must not fetch URLs")) as get:
        assert calculate_img_tokens(data) == 85
    get.assert_not_called()


def test_auto_image_decode_respects_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = importlib.import_module("litellm.litellm_core_utils.token_counter")
    monkeypatch.setattr(counter, "MAX_IMAGE_URL_DOWNLOAD_SIZE_MB", 24 / (1024 * 1024))

    with patch("base64.b64decode", side_effect=AssertionError("oversized input must not be decoded")) as decode:
        assert calculate_img_tokens(_png_data_url(1024, 1024)) == 85
    decode.assert_not_called()


def test_low_and_default_image_counts_skip_inline_decoding() -> None:
    with patch("base64.b64decode", side_effect=AssertionError("fixed counts must not decode")) as decode:
        assert calculate_img_tokens(_png_data_url(1024, 1024), mode="low") == 85
        assert (
            calculate_img_tokens(_png_data_url(1024, 1024), use_default_image_token_count=True)
            == litellm.constants.DEFAULT_IMAGE_TOKEN_COUNT
        )
    decode.assert_not_called()


@pytest.mark.parametrize(("width", "height"), [(1, 1), (768, 768), (2000, 768), (768, 2000), (4096, 4096), (8000, 3072)])
def test_high_detail_image_token_upper_bound_covers_every_image_size(width: int, height: int) -> None:
    assert calculate_img_tokens(_png_data_url(width, height), mode="high") <= high_detail_image_token_upper_bound()


def test_high_detail_image_token_upper_bound_is_reached_by_the_largest_high_res_image() -> None:
    assert calculate_img_tokens(_png_data_url(2000, 768), mode="high") == high_detail_image_token_upper_bound()
    assert calculate_img_tokens(_png_data_url(1, 1), mode="high") < high_detail_image_token_upper_bound()


# Image sizing verified 2026-10-28: https://developers.openai.com/api/docs/guides/images-vision
# and https://platform.claude.com/docs/en/build-with-claude/vision-coordinates
@pytest.mark.parametrize(
    ("model", "width", "height", "detail", "expected"),
    [
        ("gpt-4o-mini", 1024, 1024, "low", 2833),
        ("gpt-4o-mini", 1024, 1024, "high", 25501),
        ("gpt-5.1", 512, 512, "auto", 210),
        ("gpt-6-astra", 2048, 2048, "high", 3000),
        ("gpt-6-astra", 4096, 512, "high", 2458),
        ("gpt-6-astra", 2048, 2048, "auto", 4916),
        ("gpt-5.6-sol", 2048, 2048, "low", 308),
        ("gpt-5.5", 6400, 6400, "original", 12000),
        ("gpt-5.4-mini", 2048, 2048, "low", 4916),
        ("gpt-5.4-nano", 2048, 2048, "auto", 3000),
        ("gpt-5.2", 4096, 512, "auto", 615),
        ("gpt-4.1-mini-2025-04-14", 1024, 1024, "low", 1659),
        ("claude-haiku-5-5", 200, 200, "auto", 64),
        ("anthropic/claude-haiku-5-5", 1024, 1024, "auto", 1369),
        ("claude-haiku-5-5", 1920, 1080, "low", 2691),
        ("claude-haiku-4-5", 1920, 1080, "auto", 1560),
        ("claude-haiku-4-5", 1075, 1520, "auto", 1551),
        ("claude-haiku-5-5", 3840, 2160, "auto", 4784),
        ("claude-haiku-5-5", 8000, 100, "auto", 184),
    ],
)
def test_image_estimate_uses_model_resolution_and_detail(
    model: str, width: int, height: int, detail: str, expected: int
) -> None:
    assert estimate_image_tokens(model, _png_data_url(width, height), detail) == expected


@pytest.mark.parametrize(
    ("models", "fallback", "expected_model"),
    [
        (("azure/deployment-prod",), "gpt-4o", "gpt-4o"),
        (("openai/gpt-4o",), "gpt-4o-mini", "gpt-4o"),
        (("azure/deployment-prod", "openai/gpt-4o"), "gpt-4o-mini", "gpt-4o-mini"),
        (("openai/gpt-4o", "openrouter/openai/gpt-4o-mini"), None, "gpt-4o-mini"),
        (("azure_ai/openai/gpt-6-astra",), None, "gpt-6-astra"),
        (("azure/gpt-41",), None, "gpt-4.1"),
        (("azure/eu/gpt-4o-mini",), None, "gpt-4o-mini"),
        (("unknown", "gpt-4o-mini"), None, "gpt-4o-mini"),
        ((), "gpt-6-astra", "gpt-6-astra"),
        (("other/gpt-4o-mini",), None, "unregistered-model"),
        (("azure/opaque/gpt-4o-mini",), None, "unregistered-model"),
        (("bedrock/anthropic.claude-haiku-5-5",), None, "claude-haiku-5-5"),
        (("bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",), None, "claude-haiku-4-5"),
        (("vertex_ai/claude-haiku-5-5@default",), None, "claude-haiku-5-5"),
        (("azure_ai/claude-haiku-5-5",), None, "claude-haiku-5-5"),
        (("anthropic/claude-haiku-5-5", "openai/gpt-4o"), "gpt-4o-mini", "claude-haiku-5-5"),
        (("other/claude-haiku-5-5",), None, "unregistered-model"),
    ],
)
def test_image_estimate_prefers_each_backend_policy_over_the_requested_model(
    models: tuple[str, ...], fallback: str | None, expected_model: str
) -> None:
    data: Final = _png_data_url(1024, 1024)
    assert estimate_image_tokens(models, data, "high", fallback_model=fallback) == estimate_image_tokens(
        expected_model, data, "high"
    )


@pytest.mark.parametrize(
    ("model", "detail"),
    [
        ("unknown", "high"),
        ("claude-unregistered", "auto"),
        ("gpt-4o", "original"),
        ("gpt-5.2", "original"),
        ("gpt-6-astra", "invalid"),
    ],
)
def test_image_estimate_preserves_default_when_policy_is_unavailable(model: str, detail: str) -> None:
    assert (
        estimate_image_tokens(model, _png_data_url(1024, 1024), detail) == litellm.constants.DEFAULT_IMAGE_TOKEN_COUNT
    )


@pytest.mark.parametrize(
    "data",
    [
        "https://image.invalid/photo.png",
        "data:image/png;base64,invalid!",
        "data:application/octet-stream;base64,bm90LWFuLWltYWdl",
        "data:image/jpeg;base64,/9j/4AAQ",
        "data:image/webp;base64,UklGRgAAAABXRUJQ",
        _png_data_url(0, 1024),
    ],
)
# Accepted patch limits verified 2026-10-28: https://developers.openai.com/api/docs/guides/images-vision
# and https://platform.claude.com/docs/en/build-with-claude/vision
@pytest.mark.parametrize(
    ("model", "detail", "bound"),
    [
        ("gpt-4o", "high", high_detail_image_token_upper_bound()),
        ("gpt-6-astra", "high", 3000),
        ("gpt-6-astra", "auto", 36000),
        ("claude-haiku-5-5", "auto", 4784),
        ("claude-haiku-4-5", "auto", 1568),
    ],
)
def test_image_estimate_uses_bounded_fallback_without_fetching_media(
    data: str, model: str, detail: str, bound: int
) -> None:
    with patch("httpx.Client.send") as send:
        estimate: Final = estimate_image_tokens(model, data, detail)
    send.assert_not_called()
    assert estimate == bound
    assert estimate > estimate_image_tokens(model, _png_data_url(1024, 1024), detail)


@pytest.mark.parametrize("container", ["PNG", "JPEG", "GIF", "WEBP"])
def test_image_estimate_depends_on_dimensions_and_not_compressed_payload_size(container: str) -> None:
    output: Final = io.BytesIO()
    Image.new("RGB", (1024, 1024)).save(output, format=container)
    data: Final = "data:application/octet-stream;base64," + base64.b64encode(output.getvalue()).decode()
    assert estimate_image_tokens("gpt-6-astra", data) == estimate_image_tokens("gpt-6-astra", _png_data_url(1024, 1024))


def test_image_estimate_bounds_header_decoding_for_large_inline_payloads() -> None:
    header: Final = base64.b64decode(_png_data_url(1024, 1024).split(",", 1)[1])
    data: Final = "data:image/png;base64," + base64.b64encode(header + b"\0" * 4_000_000).decode()
    with patch("base64.b64decode", wraps=base64.b64decode) as decode_header:
        estimate: Final = estimate_image_tokens("gpt-6-astra", data)
    assert estimate == estimate_image_tokens("gpt-6-astra", _png_data_url(1024, 1024))
    assert decode_header.call_count == 1
    assert len(decode_header.call_args.args[0]) <= 65_536


HUB_SETUP_SCRIPT: Final = """
import json
import sys
sys.path.insert(0, sys.argv[1])
import httpx
import huggingface_hub
import litellm
served = json.loads(sys.argv[2])
text = sys.argv[3]
requested = []
def handle(request):
    repo = request.url.path.lstrip("/").split("/resolve/")[0]
    if repo not in served or not request.url.path.endswith("/tokenizer.json"):
        return httpx.Response(404)
    requested.append(repo)
    payload = served[repo].encode()
    headers = {"content-length": str(len(payload)), "etag": '"fixture"', "x-repo-commit": "a" * 40}
    return httpx.Response(200, headers=headers, content=payload if request.method == "GET" else b"")
huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handle)))
"""

HUB_TOKENIZER_SCRIPT: Final = (
    HUB_SETUP_SCRIPT
    + """
litellm.cohere_models = {"command-r-v1"}
litellm.anthropic_models = {"claude-2"}
custom = litellm.create_pretrained_tokenizer("Xenova/llama-3-tokenizer")
print(json.dumps({
    "llama2": litellm.token_counter(model="meta-llama/Llama-2-7b-chat", text=text),
    "llama3": litellm.token_counter(model="meta-llama/llama-3-70b-instruct", text=text),
    "cohere": litellm.token_counter(model="command-r-v1", text=text),
    "anthropic": litellm.token_counter(model="claude-2", text=text),
    "custom": litellm.token_counter(custom_tokenizer=custom, text=text),
    "requested": sorted(set(requested)),
}))
"""
)

HUB_ROUND_TRIP_SCRIPT: Final = (
    HUB_SETUP_SCRIPT
    + """
tokens = litellm.encode(model="meta-llama/Llama-2-7b-chat", text=text)
print(json.dumps({"tokens": tokens, "decoded": litellm.decode(model="meta-llama/Llama-2-7b-chat", tokens=tokens), "requested": sorted(set(requested))}))
"""
)

HUB_RETRY_DURING_EXIT_SCRIPT: Final = (
    HUB_SETUP_SCRIPT
    + """
import threading
import time
from litellm.utils import _HuggingFaceTokenizerLoadOutcomes, _return_huggingface_tokenizer
retry_reached_hub = threading.Event()
def answer_once_the_process_is_exiting(request):
    retry_reached_hub.set()
    while threading.main_thread().is_alive():
        time.sleep(0.01)
    time.sleep(0.5)
    return handle(request)
now = [0.0]
outcomes = _HuggingFaceTokenizerLoadOutcomes(clock=lambda: now[0])
model = "meta-llama/llama-3-70b-instruct"
_return_huggingface_tokenizer(model, outcomes)
huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(answer_once_the_process_is_exiting)))
now[0] = 3600.0
_return_huggingface_tokenizer(model, outcomes)
print(json.dumps({"retry_reached_hub": retry_reached_hub.wait(timeout=60)}))
"""
)


def _word_level_tokenizer_json(
    pre_tokenizer: pre_tokenizers.PreTokenizer, vocab: Mapping[str, int] | None = None
) -> str:
    tokenizer: Final = Tokenizer(
        models.WordLevel(vocab=dict(vocab) if vocab is not None else {"[UNK]": 0}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer.to_str()


def _run_in_memory_hub(script: str, served: dict[str, str], text: str, tmp_path: Path) -> dict:
    result: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(litellm.__file__).parent.parent),
            json.dumps(served),
            text,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "HF_HOME": str(tmp_path / "home"),
            "HF_HUB_CACHE": str(tmp_path / "cache"),
            "HF_ENDPOINT": "http://127.0.0.1:9",
            "HF_HUB_OFFLINE": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_token_counter_uses_the_tokenizer_of_each_model_family_and_of_a_custom_tokenizer(tmp_path: Path) -> None:
    sample: Final = "Tokenizers disagree: anthropic, tiktoken; llama-2 & llama-3!"
    served: Final = {
        "hf-internal-testing/llama-tokenizer": _word_level_tokenizer_json(pre_tokenizers.WhitespaceSplit()),
        "Xenova/llama-3-tokenizer": _word_level_tokenizer_json(pre_tokenizers.Split(Regex("."), "isolated")),
        "Xenova/c4ai-command-r-v01-tokenizer": _word_level_tokenizer_json(pre_tokenizers.Whitespace()),
    }
    expected: Final = {repo: len(Tokenizer.from_str(payload).encode(sample).ids) for repo, payload in served.items()}
    anthropic_count: Final = len(Tokenizer.from_str(claude_json_str).encode(sample).ids)
    tiktoken_count: Final = litellm.token_counter(model="gpt-3.5-turbo", text=sample)
    assert len({*expected.values(), anthropic_count, tiktoken_count}) == len(expected) + 2

    counts: Final = _run_in_memory_hub(HUB_TOKENIZER_SCRIPT, served, sample, tmp_path)
    assert counts == {
        "llama2": expected["hf-internal-testing/llama-tokenizer"],
        "llama3": expected["Xenova/llama-3-tokenizer"],
        "cohere": expected["Xenova/c4ai-command-r-v01-tokenizer"],
        "anthropic": anthropic_count,
        "custom": expected["Xenova/llama-3-tokenizer"],
        "requested": sorted(served),
    }


def test_custom_openai_tokenizer_without_a_model_counts_with_the_default_encoding() -> None:
    sample: Final = "A custom tokenizer passed with no model still counts."

    assert litellm.token_counter(model=None, custom_tokenizer=litellm.utils.select_tokenizer("gpt-4o"), text=sample) == (
        len(tiktoken.get_encoding("cl100k_base").encode(sample))
    )


def test_process_exits_cleanly_while_a_background_tokenizer_retry_is_downloading(tmp_path: Path) -> None:
    assert _run_in_memory_hub(HUB_RETRY_DURING_EXIT_SCRIPT, {}, "", tmp_path) == {"retry_reached_hub": True}


def test_empty_custom_tokenizer_uses_model_tokenizer() -> None:
    text_value: Final = "A tokenizer fallback should preserve the model encoding."
    custom_count: Final = _get_exact_count_function("gpt-3.5-turbo", {})(text_value)
    model_count: Final = _get_exact_count_function("gpt-3.5-turbo", None)(text_value)

    assert custom_count == model_count


def _threshold_test_messages(turns: int) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "You are a terse assistant. " * 20}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"Question {index}: what is the capital of country number {index}?"})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Answer {index}: the capital is city number {index}."}],
            }
        )
    return messages


_THRESHOLD_TEST_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "lookup_capital",
            "description": "Look up the capital of a country",
            "parameters": {"type": "object", "properties": {"country": {"type": "string"}}},
        },
    }
]


@pytest.mark.parametrize(
    "model", ["claude-3-5-sonnet-20240620", "DeepSeek-V4-Flash-0731", "databricks-qwen35-122b-a10b"]
)
def test_messages_reach_token_count_agrees_with_token_counter_at_every_threshold(model: str) -> None:
    """The threshold check is the same arithmetic as token_counter(...) >= threshold, including the
    tools and system-message adjustments, so the boundary values must agree exactly."""
    from litellm.litellm_core_utils.token_counter import messages_reach_token_count

    messages = _threshold_test_messages(turns=12)
    total = token_counter_new(
        model=model,
        messages=messages,
        tools=_THRESHOLD_TEST_TOOLS,
        use_default_image_token_count=True,
    )
    assert total > 100
    for threshold in (0, 1, total - 1, total, total + 1, 10 * total):
        assert messages_reach_token_count(
            model=model,
            messages=messages,
            threshold=threshold,
            tools=_THRESHOLD_TEST_TOOLS,
            use_default_image_token_count=True,
        ) is (total >= threshold), threshold


_SHAPE_IMAGE: Final = "data:image/png;base64," + "iVBORw0KGgo=" * 4

_OPENAI_SHAPE_MESSAGES: Final = [
    {"role": "system", "content": "You are a careful assistant. " * 20},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this screenshot. " * 30},
            {"type": "image_url", "image_url": {"url": _SHAPE_IMAGE}},
        ],
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/a/b"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "file body line\n" * 40},
    {"role": "assistant", "content": "Here is what the file does. " * 20},
]

_ANTHROPIC_SHAPE_MESSAGES: Final = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this screenshot. " * 30, "cache_control": {"type": "ephemeral"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo=" * 4}},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/a/b"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "file body line\n" * 40}]}
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Here is what the file does. " * 20}]},
]

_RESPONSES_SHAPE_INPUT: Final = [
    {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe this screenshot. " * 30},
            {"type": "input_image", "image_url": _SHAPE_IMAGE},
        ],
    },
    {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"path": "/a/b"}'},
    {"type": "function_call_output", "call_id": "c1", "output": "file body line\n" * 40},
]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(_OPENAI_SHAPE_MESSAGES, id="openai_chat_shape"),
        pytest.param(_ANTHROPIC_SHAPE_MESSAGES, id="anthropic_messages_shape"),
    ],
)
def test_messages_reach_token_count_agrees_with_token_counter_per_message_shape(messages: list[dict]) -> None:
    """Content lists, images, tool calls, tool results and cache_control blocks in the OpenAI chat shape
    (/v1/chat/completions) and the Anthropic shape (/v1/messages) count the same bounded as in full."""
    from litellm.litellm_core_utils.token_counter import messages_reach_token_count

    total = token_counter_new(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        tools=_THRESHOLD_TEST_TOOLS,
        use_default_image_token_count=True,
    )
    assert total > 100
    for threshold in (0, 1, total - 1, total, total + 1, 10 * total):
        assert messages_reach_token_count(
            model="claude-3-5-sonnet-20240620",
            messages=messages,
            threshold=threshold,
            tools=_THRESHOLD_TEST_TOOLS,
            use_default_image_token_count=True,
        ) is (total >= threshold), threshold


def test_messages_reach_token_count_rejects_responses_items_exactly_like_token_counter() -> None:
    """Responses API input items are not chat messages; the full counter raises on them and the
    bounded counter raises the same error rather than silently returning a verdict."""
    from litellm.litellm_core_utils.token_counter import messages_reach_token_count

    with pytest.raises(ValueError, match="input_text") as full:
        token_counter_new(model="gpt-4o", messages=_RESPONSES_SHAPE_INPUT, use_default_image_token_count=True)
    with pytest.raises(ValueError, match="input_text") as bounded:
        messages_reach_token_count(
            model="gpt-4o", messages=_RESPONSES_SHAPE_INPUT, threshold=10**6, use_default_image_token_count=True
        )
    assert str(bounded.value) == str(full.value)


def test_messages_reach_token_count_stops_at_the_first_message_past_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the prompt-cache eligibility check used to tokenize every message of a 700k-token
    Claude Code conversation to compare against a 1024-token minimum, costing hundreds of
    milliseconds per request before routing. Counting must stop once the threshold is crossed."""
    import litellm.litellm_core_utils.token_counter as token_counter_module

    messages = _threshold_test_messages(turns=500)
    counted_batches: list[int] = []  # mutable-ok: recorder for the _count_messages double
    real_count_messages = token_counter_module._count_messages

    def counting(params, batch, use_default_image_token_count, default_token_count):
        counted_batches.append(len(batch))
        return real_count_messages(params, batch, use_default_image_token_count, default_token_count)

    monkeypatch.setattr(token_counter_module, "_count_messages", counting)
    assert token_counter_module.messages_reach_token_count(
        model="claude-3-5-sonnet-20240620", messages=messages, threshold=1024
    )
    assert all(size == 1 for size in counted_batches)
    bounded_calls: Final = len(counted_batches)
    assert bounded_calls < len(messages) // 4, bounded_calls

    assert not token_counter_module.messages_reach_token_count(
        model="claude-3-5-sonnet-20240620", messages=messages, threshold=10**9
    )
    assert len(counted_batches) - bounded_calls == len(messages)


def test_messages_reach_token_count_honours_disable_token_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the counter disabled token_counter reports 0, so only a non-positive threshold is reached."""
    from litellm.litellm_core_utils.token_counter import messages_reach_token_count

    monkeypatch.setattr(litellm, "disable_token_counter", True)
    messages = _threshold_test_messages(turns=3)
    assert messages_reach_token_count(model="gpt-4o", messages=messages, threshold=0) is True
    assert messages_reach_token_count(model="gpt-4o", messages=messages, threshold=1) is False


@pytest.mark.usefixtures("local_model_cost_map")
def test_get_modified_max_tokens_leaves_a_gemini_4_long_decode_budget_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "token_counter", lambda **_: 1_000)

    assert (
        get_modified_max_tokens(
            model="gemini/gemini-4-argon",
            base_model="gemini/gemini-4-argon",
            messages=[{"role": "user", "content": "Write a complete proof of the prime number theorem."}],
            user_max_tokens=200_000,
            buffer_perc=None,
            buffer_num=None,
        )
        == 200_000
    )


_MULTILINGUAL_SAMPLE: Final = "日本語のテキストとहिन्दी पाठ और русский текст. " * 20


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-sol",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "azure/gpt-41",
        "azure/gpt-5.1",
        "openai/gpt-5.1",
        "openai/gpt-4o",
        "gpt-41",
        "eu/gpt-6.1-sol",
        "azure/eu/gpt-6.1-sol",
        "azure/gpt-5.1-chat",
        "azure/gpt-6.1-sol-2026-09-29",
        "openai/gpt-6.1-sol-2026-09-29",
        "azure_ai/gpt-6.1-sol",
        "chatgpt/gpt-6.1-sol",
        "perplexity/openai/gpt-5",
    ],
)
def test_token_counter_counts_openai_models_newer_than_tiktokens_table_with_o200k(model: str):
    expected: Final = len(tiktoken.get_encoding("o200k_base").encode(_MULTILINGUAL_SAMPLE))

    assert expected != len(tiktoken.get_encoding("cl100k_base").encode(_MULTILINGUAL_SAMPLE))
    assert litellm.token_counter(model=model, text=_MULTILINGUAL_SAMPLE) == expected, (
        f"{model} should count with o200k_base. OpenAI's /v1/responses/input_tokens gave gpt-5, gpt-5.1, gpt-6-sol, "
        "gpt-6-luna, gpt-6-astra, gpt-6.1-sol and gpt-6.1-terra the same count on this sample, checked 2026-10-28"
    )


@pytest.mark.parametrize("model", ["openai/gpt-5.9", "chatgpt/gpt-5.3-instant", "azure/gpt-5.9-mini"])
def test_token_counter_counts_unlisted_openai_names_with_the_encoding_tiktoken_gives_them(model: str):
    assert litellm.token_counter(model=model, text=_MULTILINGUAL_SAMPLE) == len(
        tiktoken.encoding_for_model(model.rsplit("/", 1)[-1]).encode(_MULTILINGUAL_SAMPLE)
    )


@pytest.mark.parametrize("model", ["gpt-6.2", "openai/gpt-6.2", "azure/gpt-6.2-mini", "azure/eu/o9-mini", "gpt-10"])
def test_token_counter_counts_unlisted_names_of_newer_openai_releases_with_o200k(model: str):
    assert litellm.token_counter(model=model, text=_MULTILINGUAL_SAMPLE) == len(
        tiktoken.get_encoding("o200k_base").encode(_MULTILINGUAL_SAMPLE)
    )


@pytest.mark.parametrize(
    "model",
    [
        "groq/qwen3-32b",
        "openai/my-llama",
        "text-embedding-ada-002-v2",
        "azure/gpt-4",
        "azure/my-deployment",
        "azure/ada",
        "azure/mistral-large-latest",
        "azure_ai/mistral-large-latest",
    ],
)
def test_token_counter_keeps_cl100k_for_non_openai_models_and_variants_of_cl100k_models(model: str):
    assert litellm.token_counter(model=model, text=_MULTILINGUAL_SAMPLE) == len(
        tiktoken.get_encoding("cl100k_base").encode(_MULTILINGUAL_SAMPLE)
    )


def test_openai_model_check_holds_while_openai_models_are_registered_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(litellm.model_cost, "azure/azure-only-chat-model", {"litellm_provider": "azure", "mode": "chat"})
    registered: Final = tuple(f"concurrently-registered-model-{index}" for index in range(200))
    checked: Final = threading.Semaphore(0)

    def register() -> None:
        for name in registered:
            if not checked.acquire(timeout=10):
                return
            litellm.open_ai_chat_completion_models.add(name)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            registering: Final = pool.submit(register)
            while not registering.done():
                assert not is_openai_model("azure-only-chat-model")
                checked.release()
            registering.result()
    finally:
        litellm.open_ai_chat_completion_models.difference_update(registered)
