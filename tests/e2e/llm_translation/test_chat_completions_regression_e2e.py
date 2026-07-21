"""Live /chat/completions coverage: the #28991 regression net plus per-provider
OpenAI-compatible translation.

GH #28991 broke /chat/completions (and /responses) for most models on some
releases: a clean 200 came back but with no real completion. A status check
alone would not have caught it, so TestChatCompletionsRegression asserts the
product promise - a non-empty assistant message and a real model name in the
body - across the three providers wired into the gateway config (OpenAI,
Anthropic, Gemini). A regression that empties the completion for any provider
fails that provider's row here.

The per-provider classes below cover the OpenAI-compatible /chat/completions
translation for providers customers reach by registering their own deployment
via /model/new (Cohere, Gemini, hosted_vllm), each deleted on teardown.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from e2e_config import require_env, unique_marker
from e2e_http import StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, ChatTool, ChatToolFunction, LiteLLMParamsBody, ThinkingParam
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

COHERE_BACKEND = "cohere/command-r-08-2024"
GEMINI_BACKEND = "gemini/gemini-2.5-flash"
OPENAI_BACKEND = "openai/gpt-5.6"
BEDROCK_CONVERSE_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class _StreamDelta(BaseModel):
    content: str | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta = _StreamDelta()


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice] = []


def _streamed_text(events: list[str]) -> str:
    """Concatenate the delta content across streamed chunks. Parsing every event as
    JSON also fails loudly on a truncated or garbled chunk (the vertex/gemini image
    streaming regression class), so an incomplete stream cannot pass as content."""
    chunks = [_StreamChunk.model_validate_json(event) for event in events]
    return "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)


def _assert_streamed_completion(result: StreamingResponse) -> None:
    """A streamed /chat/completions must deliver real content, not a clean-but-empty
    stream (the #28991 class on the streaming path)."""
    assert result.ok and result.is_streaming, f"stream was not established: {result}"
    assert result.stream_error is None, f"stream carried an error event: {result.stream_error}"
    assert result.chunks > 1, f"stream did not deliver multiple chunks: {result}"
    assert _streamed_text(result.stream_events).strip(), (
        f"stream completed with no content deltas: {result.stream_events[:3]}"
    )


def _bedrock_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=BEDROCK_CONVERSE_BACKEND,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
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


def _assert_weather_tool_call(response: ChatResponse) -> None:
    """The model, forced to call the tool, must return a get_weather call whose
    arguments parse as JSON and carry a location. A regression that drops tool_calls
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

CHAT_MODELS: tuple[tuple[str, str], ...] = (
    ("gpt-5.5", "openai"),
    ("claude-haiku-4-5", "anthropic"),
    ("gemini-2.5-flash", "gemini"),
)


class TestChatCompletionsRegression:
    @pytest.mark.parametrize(
        ("model", "route"),
        CHAT_MODELS,
        ids=[f"{model}-{route}" for model, route in CHAT_MODELS],
    )
    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.nonstream.works",
        "llm.chat_completions.anthropic.basic.nonstream.works",
        "llm.chat_completions.vertex.basic.nonstream.works",
        exercised_on=[],
    )
    def test_chat_returns_real_completion(
        self, client: PassthroughClient, scoped_key: str, model: str, route: str
    ) -> None:
        response = unwrap(
            client.proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"reply with one word {unique_marker()}",
                        )
                    ],
                    max_tokens=512,
                ),
            )
        )

        assert (
            response.model
        ), f"{model} ({route}): response carried no model name: {response}"
        assert (
            response.choices
        ), f"{model} ({route}): response had no choices: {response}"
        message = response.choices[0].message
        assert (
            message is not None and message.content and message.content.strip()
        ), f"{model} ({route}): 200 with an empty completion (#28991): {response}"


class TestCohereChat:
    """Cohere via the OpenAI-compatible /chat/completions path."""

    @pytest.mark.covers(
        "llm.chat_completions.cohere.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_cohere_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        (cohere_key,) = require_env("COHERE_API_KEY")
        model = f"e2e-cohere-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=COHERE_BACKEND, api_key=cohere_key),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"cohere chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"cohere empty content: {response}"


class TestGeminiChatCompletions:
    """Gemini via the OpenAI-compatible /chat/completions path, with cost logging.

    Complements the native /gemini passthrough suite by covering the translation
    path customers use when they keep the OpenAI SDK.
    """

    @pytest.mark.covers(
        "llm.chat_completions.gemini.basic.nonstream.works",
        "llm.chat_completions.gemini.basic.nonstream.cost_logged",
        exercised_on=["chat_completions"],
    )
    def test_gemini_chat_returns_content_and_logs_cost(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-gemini-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=GEMINI_BACKEND, api_key="os.environ/GEMINI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        tag = f"e2e-gemini-chat-{unique_marker()}"

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. marker={tag}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"gemini chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content, f"gemini chat returned empty content: {response}"

        rows = client.proxy.poll_logs_for_key(
            key,
            min_rows=1,
            predicate=lambda rs: any((r.spend or 0) > 0 for r in rs),
        )
        assert rows, f"no SpendLogs row for gemini chat on key ending ...{key[-6:]}"
        row = rows[0]
        assert (row.spend or 0) > 0, f"gemini chat was not costed: {row}"
        assert row.status == "success", f"gemini chat spend status={row.status!r}"


class TestHostedVllmChat:
    """hosted_vllm (self-hosted OpenAI-compatible server) via /chat/completions."""

    @pytest.mark.covers(
        "llm.chat_completions.hosted_vllm.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_hosted_vllm_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        (api_base,) = require_env("HOSTED_VLLM_API_BASE")
        api_key = (os.environ.get("HOSTED_VLLM_API_KEY") or "").strip() or None
        backend = (
            os.environ.get("HOSTED_VLLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct"
        ).strip()
        model = f"e2e-vllm-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=f"hosted_vllm/{backend}",
                api_base=api_base,
                api_key=api_key,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Reply with the single word pong. {unique_marker()}",
                        )
                    ],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"hosted_vllm chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"hosted_vllm empty content: {response}"


class TestOpenAIChatCompletions:
    """OpenAI /chat/completions, the SDK path the customer runs against the proxy.

    The streamed call must deliver real content deltas (a clean-but-empty stream is
    the regression), and a non-streamed call must be costed so per-request spend and
    the response-cost header stay accurate.
    """

    @pytest.mark.covers(
        "llm.chat_completions.openai.basic.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_openai_chat_streams_real_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        require_env("OPENAI_API_KEY")
        model = f"e2e-openai-chat-{unique_marker()}"
        model_id = client.proxy.create_model(
            model, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

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
        "llm.chat_completions.openai.basic.nonstream.cost_logged",
        exercised_on=["chat_completions"],
    )
    def test_openai_chat_logs_cost(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        require_env("OPENAI_API_KEY")
        model = f"e2e-openai-cost-{unique_marker()}"
        model_id = client.proxy.create_model(
            model, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
                    max_tokens=16,
                ),
            )
        )
        assert response.choices, f"openai chat returned no choices: {response}"

        rows = client.proxy.poll_logs_for_key(
            key, min_rows=1, predicate=lambda rs: any((r.spend or 0) > 0 for r in rs)
        )
        priced = [r for r in rows if (r.spend or 0) > 0]
        assert priced, f"openai chat was not costed on key ...{key[-6:]}: {rows}"
        assert priced[0].status == "success", f"openai chat spend status={priced[0].status!r}"

    @pytest.mark.covers(
        "llm.chat_completions.openai.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_openai_chat_returns_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        require_env("OPENAI_API_KEY")
        model = f"e2e-openai-tool-{unique_marker()}"
        model_id = client.proxy.create_model(
            model, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(role="user", content="What is the weather in San Francisco? Use the get_weather tool.")
                    ],
                    tools=[_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=128,
                ),
            )
        )
        _assert_weather_tool_call(response)

    @pytest.mark.covers(
        "llm.chat_completions.openai.structured_output.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_openai_chat_structured_output_conforms_to_schema(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        require_env("OPENAI_API_KEY")
        model = f"e2e-openai-schema-{unique_marker()}"
        model_id = client.proxy.create_model(
            model, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="Extract the person. John Doe is 42 years old.")],
                    response_format=_PERSON_SCHEMA,
                    max_tokens=128,
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
        "llm.chat_completions.openai.thinking.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_openai_chat_reasoning_reports_reasoning_tokens(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        require_env("OPENAI_API_KEY")
        model = f"e2e-openai-reasoning-{unique_marker()}"
        model_id = client.proxy.create_model(
            model, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user",
                            content="A train travels 60 miles in 1.5 hours. What is its average speed in mph?",
                        )
                    ],
                    reasoning_effort="low",
                    max_tokens=2048,
                ),
            )
        )
        assert response.choices, f"reasoning call returned no choices: {response}"
        message = response.choices[0].message
        assert message and message.content and message.content.strip(), f"reasoning call had no answer: {response}"
        details = response.usage.completion_tokens_details if response.usage else None
        assert details and details.reasoning_tokens and details.reasoning_tokens > 0, (
            f"a reasoning model must report reasoning tokens, got usage={response.usage}"
        )


class TestBedrockConverseChatCompletions:
    """Bedrock Converse via /chat/completions, the customer's AWS stack. A non-OpenAI
    provider must return real content on both the non-streamed and streamed paths.
    """

    def _register(self, client: PassthroughClient, resources: ResourceManager, prefix: str) -> str:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"{prefix}-{unique_marker()}"
        model_id = client.proxy.create_model(model, _bedrock_params())
        resources.defer(lambda: client.proxy.delete_model(model_id))
        return model

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.basic.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_converse_chat_returns_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = self._register(client, resources, "e2e-bedrock-chat")
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"Reply with the single word pong. {unique_marker()}")],
                    max_tokens=32,
                ),
            )
        )
        assert response.choices, f"bedrock converse chat returned no choices: {response}"
        content = response.choices[0].message.content if response.choices[0].message else None
        assert content and content.strip(), f"bedrock converse returned empty content: {response}"

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.basic.stream.works",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_converse_chat_streams_real_content(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = self._register(client, resources, "e2e-bedrock-stream")
        key = resources.key()

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
        "llm.chat_completions.bedrock_converse.tool_use.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_converse_chat_returns_tool_call(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = self._register(client, resources, "e2e-bedrock-tool")
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(role="user", content="What is the weather in San Francisco? Use the get_weather tool.")
                    ],
                    tools=[_WEATHER_TOOL],
                    tool_choice="required",
                    max_tokens=128,
                ),
            )
        )
        _assert_weather_tool_call(response)

    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.thinking.nonstream.works",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_converse_chat_returns_reasoning(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model = self._register(client, resources, "e2e-bedrock-thinking")
        key = resources.key()

        response = unwrap(
            client.proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="What is 17 times 23? Think it through step by step.")],
                    thinking=ThinkingParam(type="enabled", budget_tokens=1024),
                    max_tokens=2048,
                ),
            )
        )
        assert response.choices, f"bedrock thinking returned no choices: {response}"
        message = response.choices[0].message
        assert message and message.content and message.content.strip(), (
            f"bedrock thinking returned no answer content: {response}"
        )
        assert message.reasoning_content and message.reasoning_content.strip(), (
            "thinking was enabled but no reasoning_content came back on the Bedrock Converse path"
        )
