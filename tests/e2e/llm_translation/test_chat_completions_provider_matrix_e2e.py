"""Live /chat/completions coverage for the provider routes a customer reaches by
registering their own deployment: Anthropic's first-party API, Azure OpenAI,
Vertex AI Gemini, and Azure AI Foundry.

Each class is one route's translation contract. The gateway speaks the
OpenAI-compatible request shape to the caller and the provider's native shape
upstream, so every capability here (tool calls, vision, streaming, extended
thinking, prompt caching, schema-constrained output) is a translation that can
break per provider while the other routes keep passing. Each test registers the
deployment it needs through /model/new with `os.environ/...` credential
references the proxy resolves at call time, and deletes it on teardown.
"""

from __future__ import annotations

import time
from typing import Literal

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ImageContentPart,
    ImageUrl,
    LiteLLMParamsBody,
    TextContentPart,
    ThinkingParam,
)
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

ANTHROPIC_BACKEND = "anthropic/claude-haiku-4-5"
AZURE_OPENAI_BACKEND = "azure/gpt-5.6-sol-e2e"
VERTEX_BACKEND = "vertex_ai/gemini-2.5-flash"
AZURE_FOUNDRY_BACKEND = "azure_ai/claude-haiku-4-5"

CACHE_READ_DEADLINE_SECONDS = 30.0
CACHE_READ_INTERVAL_SECONDS = 3.0

SPLIT_COLOR_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAAAmklEQVR42u3QQQkAAAgEMJNcEvtjLFuIj8ES"
    "rCZ5JT2vlCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBB"
    "ggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBBggQJEiRIkCBB"
    "dxaTpa47LOh2vwAAAABJRU5ErkJggg=="
)


def _anthropic_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY")


def _azure_openai_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=AZURE_OPENAI_BACKEND,
        api_base="os.environ/AZURE_API_BASE",
        api_key="os.environ/AZURE_API_KEY",
    )


def _vertex_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=VERTEX_BACKEND,
        vertex_project="os.environ/VERTEXAI_PROJECT",
        vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
    )


def _azure_foundry_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=AZURE_FOUNDRY_BACKEND,
        api_base="os.environ/AZURE_AI_API_BASE",
        api_key="os.environ/AZURE_AI_API_KEY",
    )


def _register(
    client: PassthroughClient,
    resources: ResourceManager,
    prefix: str,
    params: LiteLLMParamsBody,
) -> tuple[str, str]:
    """Register a deployment for one test and hand back its model name plus a key
    that may call it. Both are torn down by the resources fixture."""
    model = f"{prefix}-{unique_marker()}"
    model_id = client.proxy.create_model(model, params)
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model, resources.key()


class _StreamToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class _StreamToolCall(BaseModel):
    function: _StreamToolCallFunction = _StreamToolCallFunction()


class _StreamDelta(BaseModel):
    content: str | None = None
    tool_calls: list[_StreamToolCall] | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta = _StreamDelta()


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice] = []


def _streamed_text(events: list[str]) -> str:
    """Concatenate the delta content across streamed chunks. Parsing every event as
    JSON also fails loudly on a truncated or garbled chunk, so an incomplete stream
    cannot pass as content."""
    chunks = [_StreamChunk.model_validate_json(event) for event in events]
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


def _streamed_tool_call(events: list[str]) -> tuple[str, str]:
    """Reassemble the tool call streamed across chunks: the name arrives once and the
    arguments arrive as fragments, so concatenating both and parsing the arguments as
    JSON catches a stream that never completes the call or splits its argument JSON."""
    chunks = [_StreamChunk.model_validate_json(event) for event in events]
    calls = [call for chunk in chunks for choice in chunk.choices for call in (choice.delta.tool_calls or [])]
    name = "".join(call.function.name or "" for call in calls)
    arguments = "".join(call.function.arguments or "" for call in calls)
    return name, arguments


def _assert_streamed_completion(result: StreamingResponse) -> None:
    """A streamed /chat/completions must deliver real content, not a clean-but-empty
    stream."""
    assert result.ok and result.is_streaming, f"stream was not established: {result}"
    assert result.stream_error is None, f"stream carried an error event: {result.stream_error}"
    assert len(result.stream_events) > 1, f"stream did not deliver multiple data events: {result}"
    assert _streamed_text(result.stream_events).strip(), (
        f"stream completed with no content deltas: {result.stream_events[:3]}"
    )


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

_WEATHER_PROMPT = "What is the weather in San Francisco? Use the get_weather tool."

_AZURE_HOSTED_PROMPT = "Reply with the single word pong."
"""Plain prose, with none of the random hex uniquifier the other routes append:
Azure runs Microsoft's prompt shields in front of both its OpenAI and its Foundry
deployments, and a random token in an otherwise trivial prompt intermittently
trips the jailbreak classifier into a 400. Each test registers its own uniquely
named deployment, which is what actually keeps runs from sharing a cached
response."""


def _assert_weather_tool_call(response: ChatResponse) -> None:
    """The model, forced to call the tool, must return a get_weather call whose
    arguments parse as JSON and carry a location. A translation that drops tool_calls
    or emits malformed argument JSON fails here rather than passing on a 200."""
    assert response.choices, f"chat returned no choices: {response}"
    message = response.choices[0].message
    calls = message.tool_calls if message else None
    assert calls, f"model returned no tool call for a tool-forced prompt: {response}"
    weather = next((call for call in calls if call.function.name == "get_weather"), None)
    assert weather is not None, f"expected a get_weather call, got {[c.function.name for c in calls]}"
    assert weather.function.arguments, f"get_weather call carried no arguments: {weather}"
    args = _WeatherArgs.model_validate_json(weather.function.arguments)
    assert args.location.strip(), f"get_weather arguments missing location: {weather.function.arguments}"


def _vision_messages() -> list[ChatMessage]:
    """A base64 data URI rather than a hosted image: Anthropic and Vertex fetch a
    remote image_url from their own side, and both are blocked by the usual public
    image hosts, so a link makes the test measure the host's crawler policy instead
    of the gateway's image translation."""
    return [
        ChatMessage(
            role="user",
            content=[
                TextContentPart(
                    text=(
                        "This image is split into two halves of different solid colors. "
                        "Name the two colors and nothing else."
                    )
                ),
                ImageContentPart(image_url=ImageUrl(url=SPLIT_COLOR_IMAGE)),
            ],
        )
    ]


def _assert_reads_split_colors(response: ChatResponse) -> None:
    assert response.choices, f"vision returned no choices: {response}"
    message = response.choices[0].message
    content = ((message.content if message else None) or "").lower()
    assert "red" in content and "blue" in content, (
        f"vision response did not read the two halves of the image: {content[:200]}"
    )


def _assert_answered(response: ChatResponse) -> None:
    assert response.model, f"response carried no model name: {response}"
    assert response.choices, f"response had no choices: {response}"
    message = response.choices[0].message
    assert message and message.content and message.content.strip(), (
        f"200 with an empty completion: {response}"
    )


class _Person(BaseModel):
    name: str
    age: int


_PERSON_SCHEMA: dict[str, object] = {
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


class _CacheControl(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"


class _CacheableTextPart(BaseModel):
    """A text content part carrying Anthropic's cache_control breakpoint. Anthropic
    caches only what a breakpoint marks, so the OpenAI-shaped body has to carry it
    through the translation; the shared TextContentPart has no such field."""

    type: Literal["text"] = "text"
    text: str
    cache_control: _CacheControl | None = None


class _CacheableMessage(BaseModel):
    role: str
    content: list[_CacheableTextPart]


class _CacheableChatBody(BaseModel):
    model: str
    messages: list[_CacheableMessage]
    max_tokens: int = 16


def _cacheable_prefix(marker: str) -> str:
    """A system prompt past Anthropic's 2048-token minimum cacheable size for Haiku,
    unique per run so no other run's cache entry can satisfy the read."""
    return " ".join(f"Standing instruction {index} for run {marker}: stay concise." for index in range(400))


def _cached_prefix_body(model: str, prefix: str, turn: str) -> _CacheableChatBody:
    return _CacheableChatBody(
        model=model,
        messages=[
            _CacheableMessage(role="system", content=[_CacheableTextPart(text=prefix, cache_control=_CacheControl())]),
            _CacheableMessage(role="user", content=[_CacheableTextPart(text=turn)]),
        ],
    )


class _ThinkingBlock(BaseModel):
    """One entry of `thinking_blocks`: Anthropic's own thinking representation as
    LiteLLM carries it onto the OpenAI-shaped response. `signature` is the token
    Anthropic issues for a genuine thinking block, so its presence is what separates
    a translated thinking block from prose the model happened to write."""

    type: str
    thinking: str | None = None
    signature: str | None = None


class _ThinkingMessage(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: list[_ThinkingBlock] | None = None


class _ThinkingChoice(BaseModel):
    message: _ThinkingMessage = _ThinkingMessage()


class _ThinkingUsageDetails(BaseModel):
    reasoning_tokens: int | None = None


class _ThinkingUsage(BaseModel):
    completion_tokens_details: _ThinkingUsageDetails = _ThinkingUsageDetails()


class _ThinkingChatResponse(BaseModel):
    """The slice of a /chat/completions answer that carries extended thinking: the
    answer text, Anthropic's reasoning content and thinking blocks, and the reasoning
    token accounting. The shared ChatResponse models neither thinking_blocks nor the
    reasoning-token count, and both are needed to prove thinking actually happened."""

    choices: list[_ThinkingChoice] = []
    usage: _ThinkingUsage = _ThinkingUsage()

    @property
    def message(self) -> _ThinkingMessage:
        assert self.choices, f"chat returned no choices: {self}"
        return self.choices[0].message

    @property
    def reasoning_tokens(self) -> int:
        return self.usage.completion_tokens_details.reasoning_tokens or 0

    @property
    def blocks(self) -> tuple[_ThinkingBlock, ...]:
        return tuple(self.message.thinking_blocks or ())


_THINKING_PROMPT = "What is 17 times 23? Think it through step by step."


class TestAnthropicChatCompletions:
    """Anthropic's first-party API through the OpenAI-compatible /chat/completions
    translation - the path a customer keeps when they point an OpenAI SDK at the
    gateway but bill Claude."""

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.basic.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_streams_real_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-anthropic-stream", _anthropic_params())

        result = client.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(role="user", content=f"Count from 1 to 5, one number per line. {unique_marker()}")
                ],
                max_tokens=64,
                stream=True,
            ),
        )
        _assert_streamed_completion(result)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_returns_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-anthropic-tool", _anthropic_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=_WEATHER_PROMPT)],
                    tools=[_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=256,
                ),
            )
        )
        _assert_weather_tool_call(response)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.tool_use.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_streams_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-anthropic-tool-stream", _anthropic_params())

        result = client.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=_WEATHER_PROMPT)],
                tools=[_WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
                stream=True,
            ),
        )
        assert result.ok and result.is_streaming, f"tool stream was not established: {result}"
        assert result.stream_error is None, f"tool stream carried an error event: {result.stream_error}"
        name, arguments = _streamed_tool_call(result.stream_events)
        assert name == "get_weather", f"streamed tool call named {name!r}: {result.stream_events[:5]}"
        args = _WeatherArgs.model_validate_json(arguments)
        assert args.location.strip(), f"streamed tool call arguments missing location: {arguments!r}"

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.vision.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_vision_describes_image(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-anthropic-vision", _anthropic_params())

        response = unwrap(client.proxy.chat(key, ChatBody(model=model, messages=_vision_messages(), max_tokens=32)))
        _assert_reads_split_colors(response)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_prompt_cache_hits_on_repeat(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        """A cache_control breakpoint on the system prefix must survive the
        translation, so the first call writes the prefix to Anthropic's cache and a
        later call reads it back instead of re-billing it at write price.

        Every call carries a fresh user turn after the breakpoint: that is both how a
        customer actually uses a cached prefix and what keeps the gateway's own
        response cache from replaying the first answer, which would hand the test
        turn one's write usage again and hide whether a read ever happened.
        """
        model, key = _register(client, resources, "e2e-anthropic-cache", _anthropic_params())
        prefix = _cacheable_prefix(unique_marker())

        def call() -> ChatResponse:
            return unwrap(
                client.proxy.transport.post(
                    "/chat/completions",
                    headers=client.proxy.transport.bearer(key),
                    json=_cached_prefix_body(
                        model, prefix, f"Reply with the single word pong. {unique_marker()}"
                    ),
                    response_type=ChatResponse,
                )
            )

        first = call()
        written = first.usage.cache_creation_input_tokens if first.usage else None
        assert written and written > 0, (
            f"a cache_control breakpoint must write the prefix to the cache, got usage={first.usage}"
        )

        deadline = time.monotonic() + CACHE_READ_DEADLINE_SECONDS
        while True:
            usage = call().usage
            read = usage.cache_read_input_tokens if usage else None
            if read and read >= written:
                return
            if time.monotonic() >= deadline:
                pytest.fail(
                    f"a later call must read the whole {written}-token cached prefix back, but the "
                    f"cache never became readable within {CACHE_READ_DEADLINE_SECONDS}s (last usage: {usage})"
                )
            time.sleep(CACHE_READ_INTERVAL_SECONDS)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.structured_output.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_structured_output_conforms_to_schema(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-anthropic-schema", _anthropic_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="Extract the person. John Doe is 42 years old.")],
                    response_format=_PERSON_SCHEMA,
                    max_tokens=256,
                ),
            )
        )
        assert response.choices, f"structured output returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content, f"structured output returned empty content: {response}"
        person = _Person.model_validate_json(content)
        assert person.name.strip() and person.age == 42, (
            f"schema-constrained extraction was wrong: {person}"
        )

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.thinking.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_anthropic_chat_thinking_changes_the_translated_response(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        """Enabling extended thinking has to change what comes back, so the same
        question is asked twice: once plain and once with a thinking budget.

        Asserting only that a thinking-enabled call returned some text would pass
        against a gateway that dropped the thinking parameter entirely, so each half
        of the contrast is pinned to Anthropic's own representation as translated
        onto the OpenAI-shaped response: the reasoning text, a thinking block
        carrying the signature Anthropic issues for it, and the reasoning-token
        accounting. The plain call must carry none of the three.
        """
        model, key = _register(client, resources, "e2e-anthropic-thinking", _anthropic_params())

        def ask(thinking: ThinkingParam | None) -> _ThinkingChatResponse:
            return unwrap(
                client.proxy.transport.post(
                    "/chat/completions",
                    headers=client.proxy.transport.bearer(key),
                    json=ChatBody(
                        model=model,
                        messages=[
                            ChatMessage(role="user", content=f"{_THINKING_PROMPT} {unique_marker()}")
                        ],
                        thinking=thinking,
                        max_tokens=2048,
                    ),
                    response_type=_ThinkingChatResponse,
                )
            )

        plain = ask(None)
        thought = ask(ThinkingParam(type="enabled", budget_tokens=1024))

        assert plain.message.content and plain.message.content.strip(), (
            f"the baseline call returned no answer: {plain}"
        )
        assert thought.message.content and thought.message.content.strip(), (
            f"the thinking call returned no answer: {thought}"
        )

        assert not (plain.message.reasoning_content or "").strip(), (
            f"thinking was never requested, so reasoning_content must be empty: {plain.message}"
        )
        assert not plain.blocks, f"thinking was never requested, so no thinking blocks may come back: {plain.blocks}"
        assert plain.reasoning_tokens == 0, (
            f"thinking was never requested, so no reasoning tokens may be billed: {plain.usage}"
        )

        assert (thought.message.reasoning_content or "").strip(), (
            "thinking was enabled but no reasoning_content came back on the Anthropic path"
        )
        signed = [block for block in thought.blocks if block.type == "thinking" and block.signature]
        assert signed, (
            f"thinking was enabled but no signed Anthropic thinking block survived the "
            f"translation: {thought.blocks}"
        )
        assert thought.reasoning_tokens > 0, (
            f"thinking was enabled but no reasoning tokens were reported: {thought.usage}"
        )


class TestAzureOpenAIChatCompletions:
    """Azure OpenAI, where the model lives behind a per-resource deployment name and
    the gateway authenticates with the resource's api_base plus api_key."""

    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_openai_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-azure-openai-chat", _azure_openai_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=_AZURE_HOSTED_PROMPT)],
                    max_tokens=512,
                ),
            )
        )
        _assert_answered(response)

    @pytest.mark.covers(
        "llm.chat_completions.azure_openai.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_openai_chat_returns_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-azure-openai-tool", _azure_openai_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=_WEATHER_PROMPT)],
                    tools=[_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=1024,
                    reasoning_effort="low",
                ),
            )
        )
        _assert_weather_tool_call(response)


class TestVertexChatCompletions:
    """Vertex AI Gemini, where the gateway mints a Google token from the project's
    service account instead of sending an API key."""

    @pytest.mark.covers(
        "llm.chat_completions.vertex.basic.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_vertex_chat_streams_real_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-vertex-stream", _vertex_params())

        result = client.proxy.chat_stream(
            key,
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(role="user", content=f"Count from 1 to 5, one number per line. {unique_marker()}")
                ],
                max_tokens=512,
                stream=True,
            ),
        )
        _assert_streamed_completion(result)

    @pytest.mark.covers(
        "llm.chat_completions.vertex.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_vertex_chat_returns_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-vertex-tool", _vertex_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=_WEATHER_PROMPT)],
                    tools=[_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=512,
                ),
            )
        )
        _assert_weather_tool_call(response)

    @pytest.mark.covers(
        "llm.chat_completions.vertex.vision.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_vertex_chat_vision_describes_image(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-vertex-vision", _vertex_params())

        response = unwrap(client.proxy.chat(key, ChatBody(model=model, messages=_vision_messages(), max_tokens=512)))
        _assert_reads_split_colors(response)


class TestAzureFoundryChatCompletions:
    """Azure AI Foundry (azure_ai), which serves Claude on Azure's own contract; the
    OpenAI-compatible route has to translate to it rather than to Anthropic's."""

    @pytest.mark.covers(
        "llm.chat_completions.azure_foundry.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_azure_foundry_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model, key = _register(client, resources, "e2e-azure-foundry-chat", _azure_foundry_params())

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=_AZURE_HOSTED_PROMPT)],
                    max_tokens=512,
                ),
            )
        )
        _assert_answered(response)
