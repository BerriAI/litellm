"""Live /chat/completions streaming through the Responses API bridge.

Responses-only models (gpt-5.3-codex here, the same shape as the GPT-5.6 models
customers reach over bedrock_mantle) cannot serve /chat/completions natively, so the
proxy translates the request to /v1/responses and translates each Responses event back
into a chat completion chunk. Two customer-visible contracts only hold on that path:

- every chunk of one stream carries the same ``id`` (#32854). The bridge builds a chunk
  per Responses event, so a regression there hands each chunk a fresh ``chatcmpl-<uuid>``
  and SDKs that accumulate by id (openai-go's ChatCompletionAccumulator) silently drop
  everything after the first chunk while the HTTP response still looks healthy
- the bridge always answers a streaming request with a real SSE stream (#33154). When it
  hands back an already-completed response instead, the proxy's SSE generator dies with
  "'async for' requires an object with __aiter__ method" mid-stream
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatTool, ChatToolFunction, LiteLLMParamsBody
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

RESPONSES_ONLY_BACKEND = "openai/gpt-5.3-codex"


class _BridgeToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class _BridgeToolCall(BaseModel):
    function: _BridgeToolCallFunction = _BridgeToolCallFunction()


class _BridgeDelta(BaseModel):
    content: str | None = None
    tool_calls: list[_BridgeToolCall] | None = None


class _BridgeChoice(BaseModel):
    delta: _BridgeDelta = _BridgeDelta()
    finish_reason: str | None = None


class _BridgeChunk(BaseModel):
    id: str
    choices: list[_BridgeChoice] = []


class _WeatherArgs(BaseModel):
    location: str


_WEATHER_TOOL = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
)


def _bridge_chunks(result: StreamingResponse) -> list[_BridgeChunk]:
    """Parse the SSE events of a bridged stream, failing loudly on a stream that never
    established, carried an error event, or delivered no chunks."""
    assert result.ok and result.is_streaming, f"bridged stream was not established: {result}"
    assert result.stream_error is None, f"bridged stream carried an error event: {result.stream_error}"
    chunks = [_BridgeChunk.model_validate_json(event) for event in result.stream_events]
    assert chunks, f"bridged stream delivered no chunks: {result.body[:500]}"
    return chunks


class TestResponsesBridgeChatCompletionsStreaming:
    @pytest.fixture
    def bridged_model(self, client: PassthroughClient, resources: ResourceManager) -> str:
        model = f"e2e-bridge-stream-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=RESPONSES_ONLY_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        return model

    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.stream.bridge_shares_chunk_id",
        exercised_on=["chat_completions"],
    )
    def test_bridged_stream_shares_one_chunk_id(
        self, client: PassthroughClient, resources: ResourceManager, bridged_model: str
    ) -> None:
        result = client.proxy.chat_stream(
            resources.key(),
            ChatBody(
                model=bridged_model,
                messages=[
                    ChatMessage(role="user", content=f"Count from 1 to 5, one number per line. {unique_marker()}")
                ],
                max_tokens=64,
                stream=True,
            ),
        )

        chunks = _bridge_chunks(result)
        ids = {chunk.id for chunk in chunks}
        assert len(ids) == 1, f"bridged stream used {len(ids)} different chunk ids: {sorted(ids)[:5]}"
        assert ids.pop().startswith("chatcmpl-"), f"bridged chunk id is not chat-completion shaped: {chunks[0].id}"

    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.stream.bridge_streams_sse",
        exercised_on=["chat_completions"],
    )
    def test_bridged_stream_delivers_content_finish_reason_and_done(
        self, client: PassthroughClient, resources: ResourceManager, bridged_model: str
    ) -> None:
        result = client.proxy.chat_stream(
            resources.key(),
            ChatBody(
                model=bridged_model,
                messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
                max_tokens=32,
                stream=True,
            ),
        )

        chunks = _bridge_chunks(result)
        content = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
        assert content.strip(), f"bridged stream completed with no content deltas: {result.stream_events[:3]}"
        assert any(choice.finish_reason for chunk in chunks for choice in chunk.choices), (
            f"bridged stream never emitted a finish_reason: {result.stream_events[-3:]}"
        )
        assert result.stream_done, f"bridged stream did not terminate with [DONE]: {result.stream_events[-2:]}"

    @pytest.mark.covers(
        "llm.chat_completions.openai.tool_use.stream.bridge_streams_tool_call",
        exercised_on=["chat_completions"],
    )
    def test_bridged_stream_reassembles_tool_call(
        self, client: PassthroughClient, resources: ResourceManager, bridged_model: str
    ) -> None:
        result = client.proxy.chat_stream(
            resources.key(),
            ChatBody(
                model=bridged_model,
                messages=[
                    ChatMessage(
                        role="user",
                        content="What is the weather in San Francisco? Use the get_weather tool.",
                    )
                ],
                tools=[_WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
                stream=True,
            ),
        )

        chunks = _bridge_chunks(result)
        calls = [call for chunk in chunks for choice in chunk.choices for call in (choice.delta.tool_calls or [])]
        assert calls, f"bridged stream returned no tool call for a tool-forced prompt: {result.stream_events[:5]}"
        name = "".join(call.function.name or "" for call in calls)
        arguments = "".join(call.function.arguments or "" for call in calls)
        assert name == "get_weather", f"bridged stream streamed the wrong tool name: {name!r}"
        args = _WeatherArgs.model_validate_json(arguments)
        assert args.location.strip(), f"bridged tool call arguments missing location: {arguments!r}"
