from __future__ import annotations

import base64
from itertools import chain
from types import MappingProxyType
from typing import Final

import pytest
from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from e2e_http import unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    LiteLLMParamsBody,
    OutMessage,
    ThinkingBlock,
    ThinkingParam,
)
from provider_edge import LiveEdge, start_provider_edge
from provider_edge_bedrock import ConverseBodyCapture, bedrock_signer
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

# Sonnet 4.6 is the newest Claude on Bedrock that still honors Anthropic's redacted-thinking test string
# (Sonnet 5.5 answers it with plain thinking, checked 2026-10-14). Source for the string:
# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#thinking-redaction
BACKEND: Final = "bedrock/converse/us.anthropic.claude-sonnet-4-6"
INVOKE_BACKEND: Final = "bedrock/invoke/us.anthropic.claude-sonnet-4-6"
REDACTED_THINKING_TRIGGER: Final = (
    "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
)
REGION: Final = "us-west-2"
EDGE_MOUNT: Final = f"bedrock/{REGION}"
THINKING: Final = ThinkingParam(type="enabled", budget_tokens=1024)
FOLLOW_UP: Final = "Reply with the single word ok."
WEATHER_TOOL: Final = ChatTool(
    function=ChatToolFunction(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _ReasoningContent(_Frozen):
    redactedContent: str | None = None


class _ConverseContentBlock(_Frozen):
    reasoningContent: _ReasoningContent | None = None


class _ConverseMessage(_Frozen):
    role: str
    content: tuple[_ConverseContentBlock, ...]


class _ConverseBody(_Frozen):
    messages: tuple[_ConverseMessage, ...]


class _Delta(_Frozen):
    content: str | None = None
    thinking_blocks: tuple[ThinkingBlock, ...] | None = None


class _StreamChoice(_Frozen):
    delta: _Delta


class _StreamChunk(_Frozen):
    choices: tuple[_StreamChoice, ...] = ()


def _replayed_redacted_content(capture: ConverseBodyCapture) -> tuple[str, ...]:
    """The redactedContent of every reasoning block in the last Converse request's assistant turns"""
    bodies: Final = capture.drain()
    assert bodies, "the proxy sent no Converse request through the edge"
    assistant: Final = (
        message for message in _ConverseBody.model_validate_json(bodies[-1]).messages if message.role == "assistant"
    )
    reasoning: Final = (block.reasoningContent for block in chain.from_iterable(m.content for m in assistant))
    return tuple(block.redactedContent for block in reasoning if block is not None and block.redactedContent)


def _edge(resources: ResourceManager) -> tuple[str, ConverseBodyCapture]:
    capture: Final = ConverseBodyCapture()
    edge: Final = start_provider_edge(
        LiveEdge(observe_request=capture.observe, sign=bedrock_signer(REGION)),
        mounts=MappingProxyType({EDGE_MOUNT: f"https://bedrock-runtime.{REGION}.amazonaws.com"}),
        bind_host=PROVIDER_EDGE_BIND_HOST,
        advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
    )
    resources.defer(edge.shutdown)
    return edge.edge.api_base(EDGE_MOUNT), capture


def _deploy(proxy: ProxyClient, resources: ResourceManager, api_base: str, backend: str) -> str:
    model: Final = f"e2e-bedrock-redacted-thinking-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=backend,
            api_base=api_base,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name=REGION,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str, ConverseBodyCapture]:
    api_base, capture = _edge(resources)
    return _deploy(proxy, resources, api_base, BACKEND), resources.key(), capture


def _chat(proxy: ProxyClient, key: str, body: ChatBody) -> OutMessage:
    response: Final = unwrap(proxy.chat(key, body))
    message: Final = response.choices[0].message if response.choices else None
    assert message is not None, f"chat returned no message: {response}"
    return message


def _chat_stream(proxy: ProxyClient, key: str, body: ChatBody) -> OutMessage:
    result: Final = proxy.chat_stream(key, body.model_copy(update={"stream": True}))
    assert result.ok and result.is_streaming, f"stream was not established: {result}"
    assert result.stream_error is None, f"stream carried an error event: {result.stream_error}"
    chunks: Final = tuple(_StreamChunk.model_validate_json(event) for event in result.stream_events)
    deltas: Final = tuple(choice.delta for choice in chain.from_iterable(chunk.choices for chunk in chunks))
    return OutMessage(
        content="".join(delta.content or "" for delta in deltas),
        thinking_blocks=list(chain.from_iterable(delta.thinking_blocks or () for delta in deltas)),
    )


def _redacted_data(message: OutMessage) -> tuple[str, ...]:
    data: Final = tuple(
        block.data or "" for block in message.thinking_blocks or () if block.type == "redacted_thinking"
    )
    assert data and all(data), f"turn 1 came back without redacted thinking: {message.thinking_blocks}"
    return data


class TestBedrockConverseRedactedThinkingReplay:
    @pytest.mark.provider_edge_host
    @pytest.mark.parametrize(
        "stream",
        [
            pytest.param(
                False,
                marks=meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.CHAT_COMPLETIONS,
                        providers=(Provider.BEDROCK,),
                        models=(BACKEND,),
                        capabilities=(Capability.REASONING,),
                        mode=Mode.NONSTREAM,
                    )
                ),
                id="nonstream",
            ),
            pytest.param(
                True,
                marks=meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.CHAT_COMPLETIONS,
                        providers=(Provider.BEDROCK,),
                        models=(BACKEND,),
                        capabilities=(Capability.REASONING,),
                        mode=Mode.STREAM,
                    )
                ),
                id="stream",
            ),
        ],
    )
    def test_redacted_thinking_is_sent_back_to_bedrock_on_the_next_turn(
        self, proxy: ProxyClient, resources: ResourceManager, stream: bool
    ) -> None:
        model, key, capture = _register(proxy, resources)
        send: Final = _chat_stream if stream else _chat
        question: Final = ChatMessage(role="user", content=REDACTED_THINKING_TRIGGER)

        first: Final = send(proxy, key, ChatBody(model=model, messages=[question], thinking=THINKING, max_tokens=2048))
        redacted: Final = _redacted_data(first)

        second: Final = send(
            proxy,
            key,
            ChatBody(
                model=model,
                thinking=THINKING,
                max_tokens=2048,
                messages=[
                    question,
                    ChatAssistantTurn(content=first.content, thinking_blocks=first.thinking_blocks),
                    ChatMessage(role="user", content=FOLLOW_UP),
                ],
            ),
        )

        assert (second.content or "").strip(), f"Bedrock answered the replayed turn with no text: {second}"
        assert _replayed_redacted_content(capture) == redacted

    @pytest.mark.provider_edge_host
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.REASONING),
            mode=Mode.NONSTREAM,
        )
    )
    def test_redacted_thinking_before_a_tool_call_is_sent_back_with_the_tool_result(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model, key, capture = _register(proxy, resources)
        question: Final = ChatMessage(
            role="user", content=f"{REDACTED_THINKING_TRIGGER}\n\nWhat is the weather in Paris? Use the tool."
        )

        first: Final = _chat(
            proxy,
            key,
            ChatBody(model=model, messages=[question], tools=[WEATHER_TOOL], thinking=THINKING, max_tokens=2048),
        )
        redacted: Final = _redacted_data(first)
        calls: Final = tuple(call for call in first.tool_calls or () if call.id)
        assert calls, f"turn 1 did not call get_weather: {first}"

        second: Final = _chat(
            proxy,
            key,
            ChatBody(
                model=model,
                tools=[WEATHER_TOOL],
                thinking=THINKING,
                max_tokens=2048,
                messages=[
                    question,
                    ChatAssistantTurn(
                        content=first.content, thinking_blocks=first.thinking_blocks, tool_calls=list(calls)
                    ),
                    *(
                        ChatToolResultTurn(tool_call_id=call.id or "", content="18 degrees C and sunny")
                        for call in calls
                    ),
                ],
            ),
        )

        assert (second.content or "").strip(), f"Bedrock answered the tool result with no text: {second}"
        assert _replayed_redacted_content(capture) == redacted

    @pytest.mark.provider_edge_host
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(INVOKE_BACKEND, BACKEND),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_redacted_thinking_from_bedrock_invoke_is_sent_to_converse_in_its_blob_encoding(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        """Invoke, like the Anthropic API, returns the bare ``data`` string. Converse needs it as a base64 blob, and
        Bedrock 400s with "Invalid data in redacted_thinking block" when a model group mixing the two routes
        replays it bare"""
        api_base, capture = _edge(resources)
        invoke_model: Final = _deploy(proxy, resources, api_base, INVOKE_BACKEND)
        converse_model: Final = _deploy(proxy, resources, api_base, BACKEND)
        key: Final = resources.key()
        question: Final = ChatMessage(role="user", content=REDACTED_THINKING_TRIGGER)

        first: Final = _chat(
            proxy, key, ChatBody(model=invoke_model, messages=[question], thinking=THINKING, max_tokens=2048)
        )
        redacted: Final = _redacted_data(first)

        second: Final = _chat(
            proxy,
            key,
            ChatBody(
                model=converse_model,
                thinking=THINKING,
                max_tokens=2048,
                messages=[
                    question,
                    ChatAssistantTurn(content=first.content, thinking_blocks=first.thinking_blocks),
                    ChatMessage(role="user", content=FOLLOW_UP),
                ],
            ),
        )

        assert (second.content or "").strip(), f"Bedrock answered the replayed turn with no text: {second}"
        assert _replayed_redacted_content(capture) == tuple(
            base64.b64encode(data.encode()).decode() for data in redacted
        )
