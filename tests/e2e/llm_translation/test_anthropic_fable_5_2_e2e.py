"""Claude Fable 5.2 over Anthropic direct. The row in model_prices_and_context_window.json
declares `supports_sampling_params: false`, `supports_forced_tool_use: false` and
`thinking_always_on: true`, and the proxy's translation has to follow those flags rather
than send the model a request Anthropic rejects. Prompt caching is checked end to end:
the usage the SDK sees, the cost header the proxy returns and the spend row it writes
all have to agree with the row's 5m, 1h and read rates."""

from __future__ import annotations

import json
from typing import Final, cast

import pytest
from anthropic.types import Message, TextBlockParam
from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import CostMapEntry, LiteLLMParamsBody
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, TypeAdapter
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND: Final = "anthropic/claude-fable-5-2"
COST_MAP_KEY: Final = ANTHROPIC_BACKEND.removeprefix("anthropic/")
CACHED_DOCUMENT: Final = "\n".join(
    f"Section {index}. This service agreement governs payment terms, renewal dates, data retention, "
    "uptime commitments and the obligations of both parties."
    for index in range(160)
)
WEATHER_TOOL: Final[ChatCompletionFunctionToolParam] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
PERSON_RESPONSE_FORMAT: Final[ResponseFormatJSONSchema] = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}


class _CacheUsage(BaseModel):
    """The usage litellm returns on a cached /chat/completions call. The two cache
    counters are litellm extensions to the OpenAI usage shape, so they are read
    through this model rather than the SDK's typed fields."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class _Person(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    age: int


def _register(
    proxy: ProxyClient, resources: ResourceManager, prefix: str, *, drop_params: bool | None = None
) -> tuple[str, str]:
    model: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY", drop_params=drop_params),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key(models=[model])


def _rates(proxy: ProxyClient) -> CostMapEntry:
    entry: Final = proxy.model_cost_map().get(COST_MAP_KEY)
    assert entry is not None, f"{COST_MAP_KEY} is missing from the proxy's cost map"
    assert (
        entry.input_cost_per_token is not None
        and entry.output_cost_per_token is not None
        and entry.cache_creation_input_token_cost is not None
        and entry.cache_creation_input_token_cost_above_1hr is not None
        and entry.cache_read_input_token_cost is not None
    ), f"{COST_MAP_KEY} declares no complete prompt-caching price: {entry!r}"
    return entry


def _expected_cost(rates: CostMapEntry, usage: _CacheUsage, *, one_hour_cache_tokens: int = 0) -> float:
    """litellm's Anthropic cost formula: prompt_tokens already counts the cached
    tokens, so the uncached remainder is billed at the input rate and each cache
    bucket at its own rate."""
    assert (
        rates.input_cost_per_token is not None
        and rates.output_cost_per_token is not None
        and rates.cache_creation_input_token_cost is not None
        and rates.cache_creation_input_token_cost_above_1hr is not None
        and rates.cache_read_input_token_cost is not None
    )
    uncached_prompt: Final = usage.prompt_tokens - usage.cache_creation_input_tokens - usage.cache_read_input_tokens
    five_minute_cache_tokens: Final = usage.cache_creation_input_tokens - one_hour_cache_tokens
    return (
        uncached_prompt * rates.input_cost_per_token
        + five_minute_cache_tokens * rates.cache_creation_input_token_cost
        + one_hour_cache_tokens * rates.cache_creation_input_token_cost_above_1hr
        + usage.cache_read_input_tokens * rates.cache_read_input_token_cost
        + usage.completion_tokens * rates.output_cost_per_token
    )


def _approx_equal(actual: float, expected: float) -> bool:
    """Spend math within one part per million, not exact float identity."""
    return abs(actual - expected) <= max(1e-12, abs(expected) * 1e-6)


def _cost_header(headers: object, response_id: str) -> float:
    header: Final = response_header(cast("dict[str, str]", headers), "x-litellm-response-cost")
    assert header is not None, f"{response_id}: the proxy returned no x-litellm-response-cost header"
    return float(header)


def _chat_usage(completion: ChatCompletion) -> _CacheUsage:
    assert completion.usage is not None, f"{completion.id}: response carried no usage"
    return TypeAdapter(_CacheUsage).validate_python(completion.usage.model_dump())


def _messages_usage(message: Message) -> _CacheUsage:
    return _CacheUsage(
        prompt_tokens=message.usage.input_tokens
        + (message.usage.cache_creation_input_tokens or 0)
        + (message.usage.cache_read_input_tokens or 0),
        completion_tokens=message.usage.output_tokens,
        cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
    )


def _cached_chat_messages(marker: str, question: str) -> list[ChatCompletionMessageParam]:
    return [
        cast(
            ChatCompletionMessageParam,
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{marker}\n{CACHED_DOCUMENT}",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ),
        {"role": "user", "content": question},
    ]


def _cached_system(marker: str, ttl: str) -> list[TextBlockParam]:
    return [
        cast(
            TextBlockParam,
            {
                "type": "text",
                "text": f"{marker}\n{CACHED_DOCUMENT}",
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            },
        )
    ]


def _assert_spend_row_matches(proxy: ProxyClient, response_id: str, expected_cost: float, usage: _CacheUsage) -> None:
    rows: Final = proxy.poll_logs_for_request_id(response_id, predicate=lambda rs: (rs[0].spend or 0) > 0)
    assert rows, f"{response_id}: no costed spend row landed before the deadline"
    row: Final = rows[0]
    assert row.spend is not None and _approx_equal(row.spend, expected_cost), (
        f"{response_id}: spend row {row.spend} != {expected_cost} computed from the cost-map rates and {usage!r}"
    )
    assert row.prompt_tokens == usage.prompt_tokens, (
        f"{response_id}: spend row prompt_tokens {row.prompt_tokens} != response prompt_tokens {usage.prompt_tokens}"
    )


class TestFable52CapabilityFlags:
    @pytest.mark.covers("llm.chat_completions.anthropic.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_drop_params_removes_sampling_params_the_model_rejects(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        assert rates.supports_sampling_params is False, f"{COST_MAP_KEY} no longer declares sampling params unsupported"
        model, key = _register(proxy, resources, "e2e-fable-sampling-dropped", drop_params=True)

        completion: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Reply with the single word OK. {unique_marker()}"}],
            max_tokens=200,
            temperature=0.2,
            top_p=0.5,
            extra_body=NO_PROXY_CACHE,
        )

        assert completion.choices and completion.choices[0].message.content, (
            f"temperature/top_p were not dropped for a model without sampling params: {completion!r}"
        )

    @pytest.mark.covers("llm.chat_completions.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.TOOL_CHOICE),
            mode=Mode.NONSTREAM,
        )
    )
    def test_drop_params_downgrades_forced_tool_choice_and_the_tool_is_still_called(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        assert rates.supports_forced_tool_use is False, f"{COST_MAP_KEY} no longer declares forced tool use unsupported"
        model, key = _register(proxy, resources, "e2e-fable-forced-tool", drop_params=True)

        completion: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"What is the weather in Oslo right now? {unique_marker()}"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=400,
            extra_body=NO_PROXY_CACHE,
        )

        tool_calls: Final = completion.choices[0].message.tool_calls if completion.choices else None
        first_call: Final = tool_calls[0] if tool_calls else None
        assert (
            isinstance(first_call, ChatCompletionMessageFunctionToolCall) and first_call.function.name == "get_weather"
        ), (
            f"forced tool_choice was not downgraded to a request the model accepts and answers with the tool: "
            f"{completion!r}"
        )

    @pytest.mark.covers("llm.chat_completions.anthropic.thinking.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_disabling_thinking_is_dropped_for_an_always_on_model(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        assert rates.thinking_always_on is True, f"{COST_MAP_KEY} no longer declares thinking always on"
        model, key = _register(proxy, resources, "e2e-fable-thinking-always-on")

        completion: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"What is 19 times 21? Answer with the number only. {unique_marker()}"}
            ],
            max_tokens=2000,
            extra_body={**NO_PROXY_CACHE, "thinking": {"type": "disabled"}},
        )

        assert completion.choices and completion.choices[0].finish_reason == "stop", (
            f"thinking.type=disabled was forwarded to a model that refuses it instead of being dropped: {completion!r}"
        )
        assert completion.choices[0].message.content, f"no answer: {completion!r}"

    @pytest.mark.covers("llm.chat_completions.anthropic.structured_output.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.RESPONSE_SCHEMA,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_json_schema_response_format_returns_only_the_schema(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, "e2e-fable-structured")

        completion: Final = sdk.openai(key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Carol turned 25 this year. {unique_marker()}"}],
            max_tokens=400,
            response_format=PERSON_RESPONSE_FORMAT,
            extra_body=NO_PROXY_CACHE,
        )

        content: Final = completion.choices[0].message.content if completion.choices else None
        assert content, f"structured output returned no content: {completion!r}"
        person: Final = TypeAdapter(_Person).validate_python(json.loads(content))
        assert person.name and person.age > 0, f"structured output did not fill the schema: {content!r}"


class TestFable52PromptCachingCost:
    @pytest.mark.covers(
        "llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works",
        "llm.chat_completions.anthropic.basic.nonstream.cost_logged",
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_cache_write_then_read_are_billed_at_the_cache_rates(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        model, key = _register(proxy, resources, "e2e-fable-cache-chat")
        client: Final = sdk.openai(key)
        marker: Final = unique_marker()

        first: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=_cached_chat_messages(marker, "How many sections are there? Number only."),
            max_tokens=200,
            extra_body=NO_PROXY_CACHE,
        )
        second: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=_cached_chat_messages(marker, "What does section 5 cover? One sentence."),
            max_tokens=200,
            extra_body=NO_PROXY_CACHE,
        )

        write: Final = first.parse()
        write_usage: Final = _chat_usage(write)
        assert write_usage.cache_creation_input_tokens > 0, f"first call wrote no cache: {write_usage!r}"
        write_cost: Final = _cost_header(first.headers, write.id)
        assert _approx_equal(write_cost, _expected_cost(rates, write_usage)), (
            f"cache write cost header {write_cost} != cost-map formula for {write_usage!r}"
        )

        read: Final = second.parse()
        read_usage: Final = _chat_usage(read)
        assert read_usage.cache_read_input_tokens > 0 and read_usage.cache_creation_input_tokens == 0, (
            f"second call did not read the cache the first one wrote: {read_usage!r}"
        )
        read_cost: Final = _cost_header(second.headers, read.id)
        assert _approx_equal(read_cost, _expected_cost(rates, read_usage)), (
            f"cache read cost header {read_cost} != cost-map formula for {read_usage!r}"
        )
        assert read_cost < write_cost, f"a cache read ({read_cost}) was not cheaper than the write ({write_cost})"

        _assert_spend_row_matches(proxy, write.id, write_cost, write_usage)
        _assert_spend_row_matches(proxy, read.id, read_cost, read_usage)

    @pytest.mark.covers("llm.chat_completions.anthropic.basic.stream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.STREAM,
        )
    )
    def test_streamed_cache_read_reports_usage_and_is_billed_at_the_read_rate(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        model, key = _register(proxy, resources, "e2e-fable-cache-stream")
        client: Final = sdk.openai(key)
        marker: Final = unique_marker()
        client.chat.completions.create(
            model=model,
            messages=_cached_chat_messages(marker, "How many sections are there? Number only."),
            max_tokens=200,
            extra_body=NO_PROXY_CACHE,
        )

        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=_cached_chat_messages(marker, "What does section 7 cover? One sentence."),
                max_tokens=200,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=NO_PROXY_CACHE,
            )
        )

        assert chunks, "the stream produced no chunks"
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
        assert text.strip(), f"the stream carried no text: {chunks!r}"
        usage_chunks: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
        assert usage_chunks, "stream_options.include_usage produced no usage chunk"
        usage: Final = TypeAdapter(_CacheUsage).validate_python(usage_chunks[-1].model_dump())
        assert usage.cache_read_input_tokens > 0, f"the streamed call read no cache: {usage!r}"
        _assert_spend_row_matches(proxy, chunks[0].id, _expected_cost(rates, usage), usage)

    @pytest.mark.covers(
        "llm.messages.anthropic.prompt_cache_1h.nonstream.works",
        "llm.messages.anthropic.basic.nonstream.cost_logged",
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.MESSAGES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_messages_one_hour_cache_write_is_billed_at_the_one_hour_rate(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        rates: Final = _rates(proxy)
        model, key = _register(proxy, resources, "e2e-fable-cache-1h")

        raw: Final = sdk.anthropic(key).messages.with_raw_response.create(
            model=model,
            max_tokens=200,
            system=_cached_system(unique_marker(), "1h"),
            messages=[{"role": "user", "content": "How many sections are there? Number only."}],
            extra_body=NO_PROXY_CACHE,
        )

        message: Final = raw.parse()
        one_hour_tokens: Final = (
            message.usage.cache_creation.ephemeral_1h_input_tokens if message.usage.cache_creation else 0
        )
        assert one_hour_tokens > 0, f"the 1h cache_control wrote no 1h cache: {message.usage!r}"
        usage: Final = _messages_usage(message)
        assert usage.cache_creation_input_tokens == one_hour_tokens, (
            f"5m and 1h cache counts disagree: {message.usage!r}"
        )
        cost: Final = _cost_header(raw.headers, message.id)
        assert _approx_equal(cost, _expected_cost(rates, usage, one_hour_cache_tokens=one_hour_tokens)), (
            f"1h cache write cost header {cost} != cost-map formula at the above-1hr rate for {message.usage!r}"
        )
        _assert_spend_row_matches(proxy, message.id, cost, usage)
