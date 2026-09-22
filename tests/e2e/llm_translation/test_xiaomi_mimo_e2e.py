"""Live e2e: Xiaomi MiMo v2.6 through the gateway on /chat/completions.

Both native ``xiaomi_mimo/`` v2.6 rows (pro and flash) are registered via
``/model/new`` and driven against Xiaomi's own endpoint. What the gateway owes
us is that the reasoning chain surfaces as ``reasoning_content``, tool calls
survive translation, and the cost header plus spend row follow the proxy's own
cost-map price for the row (read back from ``/model/info``, never pinned here).
Requires XIAOMI_MIMO_API_KEY on the proxy; no skip gate.
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    CostMapEntry,
    LiteLLMParamsBody,
    OutMessage,
    SpendLogRow,
)
from passthrough_client import PassthroughClient
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

BACKENDS: Final = ("xiaomi_mimo/mimo-v2.6-pro", "xiaomi_mimo/mimo-v2.6-flash")
ARITHMETIC_PROMPT = "What is 17 + 26? Answer with just the number."
WEATHER_PROMPT = "What is the weather in Paris? Use the tool."
COUNTING_PROMPT = "Count from 1 to 50, one number per line."

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


class _WeatherArgs(BaseModel):
    location: str


class _StreamDelta(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta | None = None


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice] = []


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


@pytest.fixture(scope="module")
def registry(client: PassthroughClient) -> dict[str, CostMapEntry]:
    return client.proxy.model_cost_map()


def _register(client: PassthroughClient, resources: ResourceManager, backend: str) -> tuple[str, str]:
    model = f"e2e-xiaomi-{unique_marker()}"
    model_id = client.proxy.create_model(
        model, LiteLLMParamsBody(model=backend, api_key="os.environ/XIAOMI_MIMO_API_KEY")
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model, resources.key()


def _message(response: ChatResponse) -> OutMessage:
    assert response.choices, f"Xiaomi returned no choices: {response}"
    message = response.choices[0].message
    assert message is not None, f"Xiaomi choice has no message: {response}"
    return message


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


@pytest.mark.parametrize("backend", BACKENDS)
class TestXiaomiMimoChatCompletions:
    @pytest.mark.covers("llm.chat_completions.xiaomi_mimo.basic.nonstream.cost_logged")
    def test_cost_header_and_spend_row_match_the_registry_price(
        self,
        client: PassthroughClient,
        resources: ResourceManager,
        registry: dict[str, CostMapEntry],
        backend: str,
    ) -> None:
        price = registry.get(backend)
        assert price is not None, f"{backend} has no row in the proxy's cost map, so native calls would bill $0"
        assert price.litellm_provider == "xiaomi_mimo", f"{backend} is filed under the wrong provider: {price}"
        assert price.input_cost_per_token and price.output_cost_per_token, f"{backend} carries no price: {price}"
        model, key = _register(client, resources, backend)

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
        message = _message(response)
        assert message.content and "43" in message.content, f"answer lost: {message}"
        assert message.reasoning_content, f"{backend} reasons, but no reasoning_content came back: {message}"

        usage = response.usage
        assert usage is not None and usage.prompt_tokens and usage.completion_tokens, (
            f"response carries no usage, so the cost cannot be real: {result.body[:300]}"
        )
        header_cost = result.response_cost
        assert header_cost is not None and header_cost > 0, (
            f"x-litellm-response-cost header missing or non-positive: {result.headers}"
        )
        cached = (usage.prompt_tokens_details.cached_tokens or 0) if usage.prompt_tokens_details else 0
        expected = (
            (usage.prompt_tokens - cached) * price.input_cost_per_token
            + cached * (price.cache_read_input_token_cost or 0.0)
            + usage.completion_tokens * price.output_cost_per_token
        )
        assert _approx_equal(header_cost, expected), (
            f"header cost {header_cost} disagrees with the registry price for {backend} at {usage}: expected {expected}"
        )

        def _priced(rows: list[SpendLogRow]) -> bool:
            return any(row.spend is not None and row.spend > 0 for row in rows)

        rows = client.proxy.poll_logs_for_key(key, predicate=_priced)
        priced = [row for row in rows if row.spend is not None and row.spend > 0]
        assert priced, f"no priced spend row landed for key {key}; got {rows}"
        row = priced[0]
        assert row.custom_llm_provider == "xiaomi_mimo", f"spend row misattributed: {row}"
        assert row.spend is not None and _approx_equal(row.spend, header_cost), (
            f"logged spend {row.spend} disagrees with the x-litellm-response-cost header {header_cost}"
        )

    @pytest.mark.covers("llm.chat_completions.xiaomi_mimo.thinking.stream.works")
    def test_reasoning_and_answer_stream_as_deltas(
        self, client: PassthroughClient, resources: ResourceManager, backend: str
    ) -> None:
        model, key = _register(client, resources, backend)

        deltas = _deltas(
            client.proxy.chat_stream(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=COUNTING_PROMPT)],
                    max_tokens=2048,
                    stream=True,
                ),
            )
        )
        reasoning = "".join(delta.reasoning_content or "" for delta in deltas)
        content = "".join(delta.content or "" for delta in deltas)
        assert reasoning, f"stream carried no reasoning_content deltas: {deltas[:5]}"
        assert "50" in content, f"streamed answer lost: {content[:300]!r}"

    @pytest.mark.covers("llm.chat_completions.xiaomi_mimo.tool_use.nonstream.works")
    def test_tool_call_is_returned(self, client: PassthroughClient, resources: ResourceManager, backend: str) -> None:
        model, key = _register(client, resources, backend)

        message = _message(
            unwrap(
                client.proxy.chat(
                    key,
                    ChatBody(
                        model=model,
                        messages=[ChatMessage(role="user", content=WEATHER_PROMPT)],
                        tools=[WEATHER_TOOL],
                        max_tokens=1024,
                    ),
                )
            )
        )
        assert message.tool_calls, f"{backend} dropped the tool call: {message}"
        call = message.tool_calls[0]
        assert call.id, f"tool call carries no id, so a tool result cannot answer it: {call}"
        assert call.function.name == "get_weather", f"wrong tool called: {call}"
        assert call.function.arguments, f"tool call carries no arguments: {call}"
        args = _WeatherArgs.model_validate_json(call.function.arguments)
        assert "paris" in args.location.lower(), f"tool arguments lost the location: {args}"
