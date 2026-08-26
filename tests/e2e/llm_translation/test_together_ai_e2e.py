"""Live e2e: Together AI through the gateway on /chat/completions and /v1/messages.

The reasoning and tool-calling backend is the cheapest live ``together_ai/`` chat row
in the proxy's own cost map that carries both capability flags. Two backends are
pinned because the registry has no flag for what they prove: ``enable_thinking`` is a
Qwen chat-template contract, and MiniMax-M3 is the serverless model whose template
renders a replayed ``reasoning_content`` back into the prompt (Qwen and DeepSeek
silently drop it). MiniMax-M3 honors that replayed field on nearly every call, not
every call (one miss in dozens of otherwise identical calls), so the replay case asks
up to ``REPLAY_ATTEMPTS`` times and fails only when no answer carries the secret, which
a proxy that strips the field guarantees. Requires TOGETHER_API_KEY on the proxy; no
skip gate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicAssistantTurn,
    AnthropicContentBlock,
    AnthropicCustomTool,
    AnthropicMessagesBody,
    AnthropicToolResultBlock,
    AnthropicToolResultTurn,
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    CostMapEntry,
    JsonSchemaProperty,
    LiteLLMParamsBody,
    OutMessage,
    SpendLogRow,
    ToolCall,
    ToolInputSchema,
)
from passthrough_client import PassthroughClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

TEMPLATE_KWARGS_BACKEND = "together_ai/Qwen/Qwen3.5-9B"
REASONING_REPLAY_BACKEND = "together_ai/MiniMaxAI/MiniMax-M3"

SECRET_PROMPT = "Remember this for later and reply with just OK."
SECRET_REASONING = "The user told me their favorite color is chartreuse. I must remember it."
SECRET_QUESTION = "What is my favorite color? Answer with one word."
REPLAY_ATTEMPTS: Final = 3

ARITHMETIC_PROMPT = "What is 17 + 26? Answer with just the number."
WEATHER_PROMPT = "What is the weather in Paris? Use the tool."
WEATHER_REPORT = "Paris: 22 degrees Celsius, clear skies, wind from the northwest at 9 km/h"
COUNTING_PROMPT = "Count from 1 to 20, one number per line."

WEATHER_TOOL = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a location.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
)

MESSAGES_WEATHER_TOOL = AnthropicCustomTool(
    name="get_weather",
    description="Get the current weather for a location.",
    input_schema=ToolInputSchema(
        properties={"location": JsonSchemaProperty(type="string")},
        required=["location"],
    ),
)


@dataclass(frozen=True, slots=True)
class _Needs:
    function_calling: bool = False
    reasoning: bool = False


class _WeatherArgs(BaseModel):
    location: str


class _StreamToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class _StreamToolCall(BaseModel):
    function: _StreamToolCallFunction | None = None


class _StreamDelta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[_StreamToolCall] | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta | None = None


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice] = []


class _MessagesEventDelta(BaseModel):
    type: str | None = None
    text: str = ""


class _MessagesStreamEvent(BaseModel):
    type: str
    delta: _MessagesEventDelta | None = None


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def _cheapest_together_chat_model(registry: Mapping[str, CostMapEntry], needs: _Needs) -> str:
    today = date.today().isoformat()

    def qualifies(name: str, entry: CostMapEntry) -> bool:
        return (
            name.startswith("together_ai/")
            and entry.litellm_provider == "together_ai"
            and entry.mode == "chat"
            and (entry.deprecation_date is None or entry.deprecation_date > today)
            and (entry.input_cost_per_token or 0.0) > 0
            and (entry.output_cost_per_token or 0.0) > 0
            and (not needs.function_calling or bool(entry.supports_function_calling))
            and (not needs.reasoning or bool(entry.supports_reasoning))
        )

    candidates = sorted(
        (name for name, entry in registry.items() if qualifies(name, entry)),
        key=lambda name: (
            registry[name].input_cost_per_token or 0.0,
            registry[name].output_cost_per_token or 0.0,
            name,
        ),
    )
    assert candidates, f"no live together_ai chat model in the proxy's cost map satisfies {needs}"
    return candidates[0]


@pytest.fixture(scope="module")
def registry(client: PassthroughClient) -> dict[str, CostMapEntry]:
    return client.proxy.model_cost_map()


@pytest.fixture(scope="module")
def reasoning_tool_backend(registry: dict[str, CostMapEntry]) -> str:
    return _cheapest_together_chat_model(registry, _Needs(function_calling=True, reasoning=True))


def _register(client: PassthroughClient, resources: ResourceManager, backend: str) -> tuple[str, str]:
    model = f"e2e-together-{unique_marker()}"
    model_id = client.proxy.create_model(
        model, LiteLLMParamsBody(model=backend, api_key="os.environ/TOGETHER_API_KEY")
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model, resources.key()


def _message(response: ChatResponse) -> OutMessage:
    assert response.choices, f"Together returned no choices: {response}"
    message = response.choices[0].message
    assert message is not None, f"Together choice has no message: {response}"
    return message


def _carries_secret(answer: OutMessage) -> bool:
    return answer.content is not None and "chartreuse" in answer.content.lower()


def _answers_until_secret(client: PassthroughClient, key: str, body: ChatBody) -> Iterator[OutMessage]:
    answers: Final = (_message(unwrap(client.proxy.chat(key, body))) for _ in range(REPLAY_ATTEMPTS))
    for answer in answers:
        yield answer
        if _carries_secret(answer):
            return


def _deltas(result: StreamingResponse) -> list[_StreamDelta]:
    require_successful_call(result)
    assert result.is_streaming, f"response was not streamed: {result.headers}"
    assert not result.stream_error, f"stream errored: {result.stream_error}"
    assert result.stream_done, f"stream never reached [DONE]: {result.stream_events[-3:]}"
    return [
        choice.delta
        for event in result.stream_events
        for choice in _StreamChunk.model_validate_json(event).choices
        if choice.delta is not None
    ]


def _single_weather_call(message: OutMessage) -> ToolCall:
    assert message.tool_calls, f"Together dropped the tool call: {message}"
    assert len(message.tool_calls) == 1, f"expected one tool call, got {message.tool_calls}"
    call = message.tool_calls[0]
    assert call.id, f"tool call carries no id, so a tool result cannot answer it: {call}"
    assert call.function.name == "get_weather", f"wrong tool called: {call}"
    assert call.function.arguments, f"tool call carries no arguments: {call}"
    args = _WeatherArgs.model_validate_json(call.function.arguments)
    assert "paris" in args.location.lower(), f"tool arguments lost the location: {args}"
    return call


def _weather_call(client: PassthroughClient, key: str, model: str) -> OutMessage:
    return _message(
        unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=WEATHER_PROMPT)],
                    tools=[WEATHER_TOOL],
                    max_tokens=512,
                ),
            )
        )
    )


class TestTogetherChatCompletions:
    @pytest.mark.covers("llm.chat_completions.together_ai.thinking.nonstream.works")
    def test_reasoning_surfaces_as_reasoning_content(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)

        message = _message(
            unwrap(
                client.proxy.chat(
                    key,
                    ChatBody(
                        model=model,
                        messages=[ChatMessage(role="user", content=ARITHMETIC_PROMPT)],
                        max_tokens=1024,
                    ),
                )
            )
        )
        assert message.reasoning_content, (
            f"{reasoning_tool_backend} reasons, but no reasoning_content came back: {message}"
        )
        assert message.content and "43" in message.content, f"answer lost: {message}"

    @pytest.mark.covers("llm.chat_completions.together_ai.thinking.stream.works")
    def test_reasoning_streams_as_reasoning_content_deltas(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)

        deltas = _deltas(
            client.proxy.chat_stream(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=ARITHMETIC_PROMPT)],
                    max_tokens=1024,
                    stream=True,
                ),
            )
        )
        reasoning = "".join(delta.reasoning_content or "" for delta in deltas)
        content = "".join(delta.content or "" for delta in deltas)
        assert reasoning, f"stream carried no reasoning_content deltas: {deltas[:5]}"
        assert "43" in content, f"streamed answer lost: {content!r}"

    @pytest.mark.covers("llm.chat_completions.together_ai.tool_use.nonstream.works")
    def test_tool_call_is_returned(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)
        _single_weather_call(_weather_call(client, key, model))

    @pytest.mark.covers("llm.chat_completions.together_ai.tool_use.stream.works")
    def test_tool_call_is_streamed(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)

        deltas = _deltas(
            client.proxy.chat_stream(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=WEATHER_PROMPT)],
                    tools=[WEATHER_TOOL],
                    max_tokens=512,
                    stream=True,
                ),
            )
        )
        calls = [
            call.function
            for delta in deltas
            for call in delta.tool_calls or []
            if call.function is not None
        ]
        assert calls, f"stream carried no tool call deltas: {deltas[:5]}"
        names = {call.name for call in calls if call.name}
        assert names == {"get_weather"}, f"unexpected streamed tool names: {names}"
        arguments = "".join(call.arguments or "" for call in calls)
        args = _WeatherArgs.model_validate_json(arguments)
        assert "paris" in args.location.lower(), f"streamed tool arguments lost the location: {args}"

    @pytest.mark.covers("llm.chat_completions.together_ai.multi_turn.nonstream.works")
    def test_tool_result_round_trip(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)
        first = _weather_call(client, key, model)
        call = _single_weather_call(first)
        assert call.id is not None

        answer = _message(
            unwrap(
                client.proxy.chat(
                    key,
                    ChatBody(
                        model=model,
                        messages=[
                            ChatMessage(role="user", content=WEATHER_PROMPT),
                            ChatAssistantTurn(
                                content=first.content,
                                reasoning_content=first.reasoning_content,
                                tool_calls=first.tool_calls,
                            ),
                            ChatToolResultTurn(tool_call_id=call.id, content=WEATHER_REPORT),
                        ],
                        tools=[WEATHER_TOOL],
                        max_tokens=512,
                    ),
                )
            )
        )
        assert answer.content and "22" in answer.content, (
            f"the model never saw the tool result: {answer}"
        )

    @pytest.mark.covers("llm.chat_completions.together_ai.thinking.nonstream.template_kwargs_forwarded")
    def test_chat_template_kwargs_reach_together(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, TEMPLATE_KWARGS_BACKEND)

        def ask(chat_template_kwargs: dict[str, bool] | None) -> OutMessage:
            return _message(
                unwrap(
                    client.proxy.chat(
                        key,
                        ChatBody(
                            model=model,
                            messages=[ChatMessage(role="user", content=ARITHMETIC_PROMPT)],
                            max_tokens=1024,
                            chat_template_kwargs=chat_template_kwargs,
                        ),
                    )
                )
            )

        control = ask(None)
        assert control.reasoning_content, (
            f"control: {TEMPLATE_KWARGS_BACKEND} returned no reasoning_content by default, "
            f"so the disable assertion below cannot be trusted: {control}"
        )
        treatment = ask({"enable_thinking": False})
        assert not treatment.reasoning_content, (
            "chat_template_kwargs={'enable_thinking': False} did not reach Together: "
            f"reasoning_content is still present: {treatment}"
        )
        assert treatment.content and "43" in treatment.content, f"answer lost: {treatment}"

    @pytest.mark.covers("llm.chat_completions.together_ai.thinking.nonstream.replayed_reasoning_forwarded")
    def test_replayed_reasoning_content_reaches_together(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, REASONING_REPLAY_BACKEND)
        body: Final = ChatBody(
            model=model,
            messages=[
                ChatMessage(role="user", content=SECRET_PROMPT),
                ChatAssistantTurn(content="OK.", reasoning_content=SECRET_REASONING),
                ChatMessage(role="user", content=SECRET_QUESTION),
            ],
            max_tokens=512,
        )

        answers: Final = tuple(_answers_until_secret(client, key, body))
        assert any(_carries_secret(answer) for answer in answers), (
            f"the replayed reasoning_content never reached Together in {len(answers)} attempts: {answers}"
        )

    @pytest.mark.covers("llm.chat_completions.together_ai.basic.nonstream.cost_logged")
    def test_cost_header_and_spend_row_match_the_registry_price(
        self,
        client: PassthroughClient,
        resources: ResourceManager,
        registry: dict[str, CostMapEntry],
        reasoning_tool_backend: str,
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)

        result = client.proxy.transport.send(
            "/chat/completions",
            headers=client.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=f"{ARITHMETIC_PROMPT} {unique_marker()}")],
                max_tokens=1024,
            ),
        )
        require_successful_call(result)
        response = ChatResponse.model_validate_json(result.body)
        usage = response.usage
        assert usage is not None and usage.prompt_tokens and usage.completion_tokens, (
            f"response carries no usage, so the cost cannot be real: {result.body[:300]}"
        )
        header_cost = result.response_cost
        assert header_cost is not None and header_cost > 0, (
            f"x-litellm-response-cost header missing or non-positive: {result.headers}"
        )

        price = registry[reasoning_tool_backend]
        assert price.input_cost_per_token and price.output_cost_per_token
        cached = (usage.prompt_tokens_details.cached_tokens or 0) if usage.prompt_tokens_details else 0
        expected = (
            (usage.prompt_tokens - cached) * price.input_cost_per_token
            + cached * (price.cache_read_input_token_cost or 0.0)
            + usage.completion_tokens * price.output_cost_per_token
        )
        assert _approx_equal(header_cost, expected), (
            f"header cost {header_cost} disagrees with the registry price for "
            f"{reasoning_tool_backend} at {usage}: expected {expected}"
        )

        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(row.spend is not None and row.spend > 0 for row in rows)

        rows = client.proxy.poll_logs_for_key(key, predicate=_priced)
        priced = [row for row in rows if row.spend is not None and row.spend > 0]
        assert priced, f"no priced spend row landed for key {key}; got {rows}"
        row = priced[0]
        assert row.custom_llm_provider == "together_ai", f"spend row misattributed: {row}"
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"logged spend {row.spend} disagrees with the x-litellm-response-cost header {header_cost}"
        )


def _tool_use_blocks(content: list[AnthropicContentBlock] | None) -> list[AnthropicContentBlock]:
    assert content, f"/v1/messages returned no content blocks: {content}"
    return [block for block in content if block.type == "tool_use"]


def _messages_weather_call(
    client: PassthroughClient, key: str, model: str
) -> tuple[list[AnthropicContentBlock], AnthropicContentBlock]:
    response = unwrap(
        client.proxy.messages(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=512,
                tools=[MESSAGES_WEATHER_TOOL],
                messages=[ChatMessage(role="user", content=WEATHER_PROMPT)],
            ),
        )
    )
    tool_uses = _tool_use_blocks(response.content)
    assert len(tool_uses) == 1, f"expected one tool_use block, got {response.content}"
    block = tool_uses[0]
    assert block.name == "get_weather", f"wrong tool called: {block}"
    assert block.id, f"tool_use block carries no id, so a tool_result cannot answer it: {block}"
    assert response.content is not None
    return response.content, block


class TestTogetherMessages:
    @pytest.mark.covers("llm.messages.together_ai.tool_use.nonstream.works")
    def test_tool_use_block_is_returned(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)
        _messages_weather_call(client, key, model)

    @pytest.mark.covers("llm.messages.together_ai.multi_turn.nonstream.works")
    def test_tool_result_round_trip(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)
        first_content, block = _messages_weather_call(client, key, model)
        assert block.id is not None

        response = unwrap(
            client.proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=512,
                    tools=[MESSAGES_WEATHER_TOOL],
                    messages=[
                        ChatMessage(role="user", content=WEATHER_PROMPT),
                        AnthropicAssistantTurn(content=first_content),
                        AnthropicToolResultTurn(
                            content=[AnthropicToolResultBlock(tool_use_id=block.id, content=WEATHER_REPORT)]
                        ),
                    ],
                ),
            )
        )
        assert response.content, f"/v1/messages returned no content blocks: {response}"
        text = "".join(block.text or "" for block in response.content if block.type == "text")
        assert "22" in text, f"the model never saw the tool result: {response.content}"

    @pytest.mark.covers("llm.messages.together_ai.basic.stream.works")
    def test_streams_text_deltas(
        self, client: PassthroughClient, resources: ResourceManager, reasoning_tool_backend: str
    ) -> None:
        model, key = _register(client, resources, reasoning_tool_backend)

        result = client.proxy.messages_stream(
            key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=512,
                stream=True,
                messages=[ChatMessage(role="user", content=COUNTING_PROMPT)],
            ),
        )
        require_successful_call(result)
        assert result.is_streaming, f"response was not streamed: {result.headers}"
        assert not result.stream_error, f"stream errored: {result.stream_error}"
        events = [_MessagesStreamEvent.model_validate_json(event) for event in result.stream_events]
        types = [event.type for event in events]
        text_deltas = [
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta is not None and event.delta.text
        ]
        assert len(text_deltas) >= 2, f"stream was not incremental: {types}"
        assert "20" in "".join(text_deltas), f"streamed text lost the answer: {text_deltas}"
        assert "message_stop" in types, f"stream never reached message_stop: {types}"
