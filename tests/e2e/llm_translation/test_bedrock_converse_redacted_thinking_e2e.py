from __future__ import annotations

import base64
import queue
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import chain
from types import MappingProxyType
from typing import Final

import pytest
from _pytest.mark.structures import ParameterSet
from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from e2e_http import unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
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
from provider_cache import RequestSigner
from provider_edge import LiveEdge, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict

pytestmark = pytest.mark.e2e

# Sonnet 4.6 is the newest Claude on Bedrock that still honors Anthropic's redacted-thinking test string
# (Sonnet 5.5 answers it with plain thinking, checked 2026-10-14). Source for the string:
# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#thinking-redaction
BACKEND: Final = "bedrock/converse/us.anthropic.claude-sonnet-4-6"
INVOKE_BACKEND: Final = "bedrock/invoke/us.anthropic.claude-sonnet-4-6"
ANTHROPIC_BACKEND: Final = "anthropic/claude-sonnet-4-6"
OPENAI_BACKEND: Final = "openai/gpt-5.6"
ENCRYPTED_REASONING_PREFIX: Final = "litellm_encrypted_reasoning:"
REDACTED_THINKING_TRIGGER: Final = (
    "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
)
REGION: Final = "us-west-2"
EDGE_MOUNT: Final = f"bedrock/{REGION}"
ANTHROPIC_MOUNT: Final = "anthropic"
THINKING: Final = ThinkingParam(type="enabled", budget_tokens=1024)
OPENAI_THINKING: Final = ThinkingParam(type="enabled", budget_tokens=4096)
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


class _AnthropicWireBlock(_Frozen):
    type: str
    data: str | None = None


class _AnthropicWireMessage(_Frozen):
    role: str
    content: str | tuple[_AnthropicWireBlock, ...]


class _AnthropicWireBody(_Frozen):
    messages: tuple[_AnthropicWireMessage, ...]


class _Delta(_Frozen):
    content: str | None = None
    thinking_blocks: tuple[ThinkingBlock, ...] | None = None


class _StreamChoice(_Frozen):
    delta: _Delta


class _StreamChunk(_Frozen):
    choices: tuple[_StreamChoice, ...] = ()


@dataclass(frozen=True, slots=True)
class _WireCapture:
    """Every (url, body) the proxy sent upstream through the edges this test started"""

    _requests: queue.SimpleQueue[tuple[str, bytes]] = field(default_factory=queue.SimpleQueue)

    def observe(self, url: str, headers: Mapping[str, str], body: bytes | None) -> None:
        if body is not None:
            self._requests.put((url, body))

    def last_body(self, path_marker: str) -> bytes:
        sent: Final = tuple(self._requests.get_nowait() for _ in range(self._requests.qsize()))
        bodies: Final = tuple(body for url, body in sent if path_marker in url)
        assert bodies, f"the proxy sent no {path_marker} request through the edge, only {[url for url, _ in sent]}"
        return bodies[-1]


def _converse_reasoning(body: bytes) -> tuple[_ReasoningContent, ...]:
    assistant: Final = (
        message for message in _ConverseBody.model_validate_json(body).messages if message.role == "assistant"
    )
    blocks: Final = (block.reasoningContent for block in chain.from_iterable(m.content for m in assistant))
    return tuple(block for block in blocks if block is not None)


def _converse_redacted(body: bytes) -> tuple[str, ...]:
    return tuple(block.redactedContent for block in _converse_reasoning(body) if block.redactedContent)


def _anthropic_redacted(body: bytes) -> tuple[str, ...]:
    assistant: Final = (
        message.content
        for message in _AnthropicWireBody.model_validate_json(body).messages
        if message.role == "assistant" and not isinstance(message.content, str)
    )
    return tuple(block.data or "" for block in chain.from_iterable(assistant) if block.type == "redacted_thinking")


def _b64(data: str) -> str:
    return base64.b64encode(data.encode()).decode()


@dataclass(frozen=True, slots=True)
class _ClaudeRoute:
    """One way the proxy reaches Claude, with how that route carries a redacted_thinking block's ``data``"""

    backend: str
    mount: str
    path_marker: str
    provider: Provider
    replayed: Callable[[bytes], tuple[str, ...]]
    on_the_wire: Callable[[str], str]


CONVERSE: Final = _ClaudeRoute(BACKEND, EDGE_MOUNT, "/converse", Provider.BEDROCK, _converse_redacted, _b64)
INVOKE: Final = _ClaudeRoute(INVOKE_BACKEND, EDGE_MOUNT, "/invoke", Provider.BEDROCK, _anthropic_redacted, str)
ANTHROPIC: Final = _ClaudeRoute(
    ANTHROPIC_BACKEND, ANTHROPIC_MOUNT, "/v1/messages", Provider.ANTHROPIC, _anthropic_redacted, str
)
CLAUDE_ROUTES: Final = MappingProxyType({"converse": CONVERSE, "invoke": INVOKE, "anthropic": ANTHROPIC})


def _start_edge(
    resources: ResourceManager, capture: _WireCapture, mount: str, upstream: str, sign: RequestSigner | None
) -> str:
    edge: Final = start_provider_edge(
        LiveEdge(observe_request=capture.observe, sign=sign),
        mounts=MappingProxyType({mount: upstream}),
        bind_host=PROVIDER_EDGE_BIND_HOST,
        advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
    )
    resources.defer(edge.shutdown)
    return edge.edge.api_base(mount)


@dataclass(frozen=True, slots=True)
class _Edges:
    bedrock: str
    anthropic: str
    capture: _WireCapture


def _edges(resources: ResourceManager) -> _Edges:
    capture: Final = _WireCapture()
    return _Edges(
        bedrock=_start_edge(
            resources, capture, EDGE_MOUNT, f"https://bedrock-runtime.{REGION}.amazonaws.com", bedrock_signer(REGION)
        ),
        anthropic=_start_edge(resources, capture, ANTHROPIC_MOUNT, "https://api.anthropic.com", None),
        capture=capture,
    )


def _create(proxy: ProxyClient, resources: ResourceManager, params: LiteLLMParamsBody) -> str:
    model: Final = f"e2e-redacted-thinking-{unique_marker()}"
    model_id: Final = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _deploy(proxy: ProxyClient, resources: ResourceManager, edges: _Edges, route: _ClaudeRoute) -> str:
    if route.provider is Provider.ANTHROPIC:
        return _create(
            proxy,
            resources,
            LiteLLMParamsBody(model=route.backend, api_base=edges.anthropic, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
    return _create(
        proxy,
        resources,
        LiteLLMParamsBody(
            model=route.backend,
            api_base=edges.bedrock,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name=REGION,
        ),
    )


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


def _replay(model: str, question: ChatMessage, first: OutMessage) -> ChatBody:
    return ChatBody(
        model=model,
        thinking=THINKING,
        max_tokens=2048,
        messages=[
            question,
            ChatAssistantTurn(content=first.content, thinking_blocks=first.thinking_blocks),
            ChatMessage(role="user", content=FOLLOW_UP),
        ],
    )


def _route_pair(origin: str, destination: str) -> ParameterSet:
    routes: Final = (CLAUDE_ROUTES[origin], CLAUDE_ROUTES[destination])
    return pytest.param(
        origin,
        destination,
        marks=meta(
            Subject(
                domain=Domain.LLM_TRANSLATION,
                route=Route.CHAT_COMPLETIONS,
                providers=tuple(route.provider for route in routes),
                models=tuple(route.backend for route in routes),
                capabilities=(Capability.REASONING,),
                mode=Mode.NONSTREAM,
            )
        ),
        id=f"{origin}-to-{destination}",
    )


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
        edges: Final = _edges(resources)
        model: Final = _deploy(proxy, resources, edges, CONVERSE)
        key: Final = resources.key()
        send: Final = _chat_stream if stream else _chat
        question: Final = ChatMessage(role="user", content=REDACTED_THINKING_TRIGGER)

        first: Final = send(proxy, key, ChatBody(model=model, messages=[question], thinking=THINKING, max_tokens=2048))
        redacted: Final = _redacted_data(first)

        second: Final = send(proxy, key, _replay(model, question, first))

        assert (second.content or "").strip(), f"Bedrock answered the replayed turn with no text: {second}"
        assert _converse_redacted(edges.capture.last_body("/converse")) == tuple(_b64(data) for data in redacted)

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
        edges: Final = _edges(resources)
        model: Final = _deploy(proxy, resources, edges, CONVERSE)
        key: Final = resources.key()
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
        assert _converse_redacted(edges.capture.last_body("/converse")) == tuple(_b64(data) for data in redacted)

    @pytest.mark.provider_edge_host
    @pytest.mark.parametrize(
        "origin, destination",
        [
            _route_pair(origin, destination)
            for origin, destination in (
                ("anthropic", "anthropic"),
                ("anthropic", "invoke"),
                ("anthropic", "converse"),
                ("invoke", "anthropic"),
                ("invoke", "invoke"),
                ("invoke", "converse"),
                ("converse", "anthropic"),
                ("converse", "invoke"),
            )
        ],
    )
    def test_redacted_thinking_minted_on_one_claude_route_replays_on_another(
        self, proxy: ProxyClient, resources: ResourceManager, origin: str, destination: str
    ) -> None:
        """A model group can mix the Anthropic API, Bedrock Invoke and Bedrock Converse, so a turn minted on one
        is replayed on another. Converse carries ``data`` base64-wrapped as redactedContent, the others carry it
        as is, and Claude 400s with "Invalid data in redacted_thinking block" on any other encoding"""
        edges: Final = _edges(resources)
        target: Final = CLAUDE_ROUTES[destination]
        origin_model: Final = _deploy(proxy, resources, edges, CLAUDE_ROUTES[origin])
        destination_model: Final = _deploy(proxy, resources, edges, target)
        key: Final = resources.key()
        question: Final = ChatMessage(role="user", content=REDACTED_THINKING_TRIGGER)

        first: Final = _chat(
            proxy, key, ChatBody(model=origin_model, messages=[question], thinking=THINKING, max_tokens=2048)
        )
        redacted: Final = _redacted_data(first)

        second: Final = _chat(proxy, key, _replay(destination_model, question, first))

        assert (second.content or "").strip(), f"Claude answered the replayed turn with no text: {second}"
        assert target.replayed(edges.capture.last_body(target.path_marker)) == tuple(
            target.on_the_wire(data) for data in redacted
        )

    @pytest.mark.provider_edge_host
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK, Provider.OPENAI),
            models=(OPENAI_BACKEND, BACKEND),
            capabilities=(Capability.REASONING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_openai_encrypted_reasoning_from_messages_is_not_sent_to_claude_on_converse(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        """/v1/messages to an OpenAI model hands back its encrypted reasoning as thinking blocks carrying a
        LiteLLM blob only OpenAI can read. Claude rejects that blob, so replaying the turn to Claude leaves it out"""
        edges: Final = _edges(resources)
        gpt: Final = _create(
            proxy, resources, LiteLLMParamsBody(model=OPENAI_BACKEND, api_key="os.environ/OPENAI_API_KEY")
        )
        claude: Final = _deploy(proxy, resources, edges, CONVERSE)
        key: Final = resources.key()
        question: Final = ChatMessage(
            role="user",
            content="How many primes are there between 100 and 200? Think it through, then answer with just the number.",
        )

        first: Final = unwrap(
            proxy.messages(
                key, AnthropicMessagesBody(model=gpt, messages=[question], max_tokens=8192, thinking=OPENAI_THINKING)
            )
        )
        blocks: Final = tuple(
            ThinkingBlock.model_validate(block.model_dump())
            for block in first.content or ()
            if block.type in ("thinking", "redacted_thinking")
        )
        encrypted: Final = tuple(
            block for block in blocks if (block.data or block.signature or "").startswith(ENCRYPTED_REASONING_PREFIX)
        )
        assert encrypted, f"turn 1 came back without encrypted reasoning: {first.content}"
        text: Final = "".join(block.text or "" for block in first.content or () if block.type == "text")

        second: Final = _chat(
            proxy, key, _replay(claude, question, OutMessage(content=text, thinking_blocks=list(blocks)))
        )

        assert (second.content or "").strip(), f"Claude answered the replayed turn with no text: {second}"
        assert _converse_reasoning(edges.capture.last_body("/converse")) == ()
