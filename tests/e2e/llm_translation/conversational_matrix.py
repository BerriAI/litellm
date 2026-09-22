"""The endpoint x deployment x auth matrix behind test_conversational_matrix_e2e.py.

One conversation, three wire formats. Each `Surface` speaks its own API through
the customer SDK (chat completions and Responses through the OpenAI SDK, Messages
through the Anthropic SDK) and folds what came back into the surface-neutral
`Reply` / `StreamedReply`, so a single behavior test asserts the same contract on
every cell. A new model, from an existing or a new provider, is one `Deployment` row
in DEPLOYMENTS; a new way of handing the proxy a provider credential is one
`AuthMethod`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol

import anthropic
import openai
import pytest
from _pytest.mark.structures import ParameterSet
from anthropic.types import (
    MessageParam,
    RawMessageStreamEvent,
    TextBlock,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from e2e_config import provider_edge_base, unique_marker
from lifecycle import ResourceManager
from llm_translation.sdk_clients import NO_PROXY_CACHE, SdkClients, response_header
from models import CredentialCreateBody, LiteLLMParamsBody
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseStreamEvent,
    ToolChoiceFunctionParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from proxy_client import ProxyClient
from pydantic import BaseModel

SurfaceName = Literal["chat_completions", "messages", "responses"]
AuthMethod = Literal["env_ref", "stored_credential"]
Capability = Literal["basic", "tool_use", "multi_turn"]
Streaming = Literal["stream", "nonstream"]
Assertion = Literal["works", "cost_logged"]
ToolMode = Literal["none", "forced", "offered"]

SURFACES: Final[tuple[SurfaceName, ...]] = ("chat_completions", "messages", "responses")
AUTH_METHODS: Final[tuple[AuthMethod, ...]] = ("env_ref", "stored_credential")

MAX_OUTPUT_TOKENS: Final = 512
INSTRUCTIONS: Final = "You are a terse assistant. Answer in one short sentence."
GREETING_PROMPT: Final = "Say hello."
WEATHER_PROMPT: Final = "What is the weather in Paris right now? Use the get_weather tool."
WEATHER_REPORT: Final = "Paris: 22 degrees Celsius, clear skies"
WEATHER_TOOL_NAME: Final = "get_weather"
WEATHER_TOOL_DESCRIPTION: Final = "Current weather for a city"
WEATHER_TOOL_SCHEMA: Final[Mapping[str, object]] = MappingProxyType(
    {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    }
)


@dataclass(frozen=True, slots=True)
class Deployment:
    """One deployment target: the litellm backend string plus how to wire it."""

    route: Literal["openai", "anthropic"]
    label: str
    backend: str
    api_key_env: str
    edge_mount: str
    edge_suffix: str

    def api_base(self) -> str | None:
        base: Final = provider_edge_base(self.edge_mount)
        return None if base is None else f"{base}{self.edge_suffix}"

    def api_key(self) -> str:
        key: Final = os.environ.get(self.api_key_env, "")
        assert key, f"{self.api_key_env} is not set in the test process environment"
        return key


DEPLOYMENTS: Final[tuple[Deployment, ...]] = (
    Deployment(
        route="openai",
        label="gpt-4o-mini",
        backend="openai/gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        edge_mount="openai",
        edge_suffix="/v1",
    ),
    Deployment(
        route="openai",
        label="gpt-5.4-mini",
        backend="openai/gpt-5.4-mini",
        api_key_env="OPENAI_API_KEY",
        edge_mount="openai",
        edge_suffix="/v1",
    ),
    Deployment(
        route="anthropic",
        label="claude-haiku-4-5",
        backend="anthropic/claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        edge_mount="anthropic",
        edge_suffix="",
    ),
)


@dataclass(frozen=True, slots=True)
class Cell:
    surface: SurfaceName
    deployment: Deployment
    auth: AuthMethod

    @property
    def id(self) -> str:
        return f"{self.surface}-{self.deployment.label}-{self.auth}"

    def registry_id(self, capability: Capability, streaming: Streaming, assertion: Assertion) -> str:
        return f"llm.{self.surface}.{self.deployment.route}.{capability}.{streaming}.{assertion}"


CELLS: Final[tuple[Cell, ...]] = tuple(
    Cell(surface=surface, deployment=deployment, auth=auth)
    for surface in SURFACES
    for deployment in DEPLOYMENTS
    for auth in AUTH_METHODS
)


def cells_covering(capability: Capability, streaming: Streaming, assertion: Assertion) -> tuple[ParameterSet, ...]:
    """Every cell as a pytest param carrying the registry id its test proves."""
    return tuple(
        pytest.param(cell, id=cell.id, marks=pytest.mark.covers(cell.registry_id(capability, streaming, assertion)))
        for cell in CELLS
    )


DeploymentKey = tuple[str, AuthMethod]


@dataclass(frozen=True, slots=True)
class Deployments:
    """Model aliases registered on the proxy, one per (deployment, auth)."""

    aliases: Mapping[DeploymentKey, str]

    def alias(self, cell: Cell) -> str:
        return self.aliases[(cell.deployment.label, cell.auth)]


def _litellm_params(deployment: Deployment, auth: AuthMethod, credential_name: str) -> LiteLLMParamsBody:
    match auth:
        case "env_ref":
            return LiteLLMParamsBody(
                model=deployment.backend, api_key=f"os.environ/{deployment.api_key_env}", api_base=deployment.api_base()
            )
        case "stored_credential":
            return LiteLLMParamsBody(
                model=deployment.backend, litellm_credential_name=credential_name, api_base=deployment.api_base()
            )


def _register(proxy: ProxyClient, resources: ResourceManager, deployment: Deployment, auth: AuthMethod) -> str:
    marker: Final = unique_marker()
    credential_name: Final = f"e2e-matrix-{deployment.label}-{marker}"
    if auth == "stored_credential":
        proxy.create_credential(
            CredentialCreateBody(credential_name=credential_name, credential_values={"api_key": deployment.api_key()})
        )
        resources.defer(lambda: proxy.delete_credential(credential_name))
    alias: Final = f"e2e-matrix-{deployment.label}-{auth}-{marker}"
    model_id: Final = proxy.create_model(alias, _litellm_params(deployment, auth, credential_name))
    resources.defer(lambda: proxy.delete_model(model_id))
    return alias


def register_deployments(proxy: ProxyClient) -> Iterator[Deployments]:
    resources: Final = ResourceManager(client=proxy)
    try:
        yield Deployments(
            aliases=MappingProxyType(
                {
                    (deployment.label, auth): _register(proxy, resources, deployment, auth)
                    for deployment in DEPLOYMENTS
                    for auth in AUTH_METHODS
                }
            )
        )
    finally:
        resources.teardown()


class WeatherArgs(BaseModel):
    location: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str

    def parsed(self) -> WeatherArgs:
        return WeatherArgs.model_validate_json(self.arguments)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Reply:
    """What every surface owes the caller for one non-streamed turn."""

    response_id: str
    text: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage | None
    call_id_header: str | None
    cost_header: str | None


@dataclass(frozen=True, slots=True)
class StreamedReply:
    """The reassembled stream: its text, whether the surface's own terminal event
    arrived, and whether usage was reported anywhere in the stream."""

    text: str
    finished: bool
    usage_reported: bool
    event_count: int


class Surface(Protocol):
    @property
    def name(self) -> SurfaceName: ...

    def reply(self, key: str, model: str, prompt: str, *, with_tool: bool = False) -> Reply: ...

    def stream(self, key: str, model: str, prompt: str) -> StreamedReply: ...

    def reply_to_tool_result(self, key: str, model: str, prompt: str, call: ToolCall, result: str) -> Reply: ...


def _chat_tool() -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": WEATHER_TOOL_NAME,
            "description": WEATHER_TOOL_DESCRIPTION,
            "parameters": dict(WEATHER_TOOL_SCHEMA),
        },
    }


def _messages_tool() -> ToolParam:
    return {
        "name": WEATHER_TOOL_NAME,
        "description": WEATHER_TOOL_DESCRIPTION,
        "input_schema": dict(WEATHER_TOOL_SCHEMA),
    }


def _responses_tool() -> FunctionToolParam:
    return {
        "type": "function",
        "name": WEATHER_TOOL_NAME,
        "description": WEATHER_TOOL_DESCRIPTION,
        "parameters": dict(WEATHER_TOOL_SCHEMA),
        "strict": False,
    }


def _chat_tool_choice() -> ChatCompletionNamedToolChoiceParam:
    return {"type": "function", "function": {"name": WEATHER_TOOL_NAME}}


def _messages_tool_choice() -> ToolChoiceToolParam:
    return {"type": "tool", "name": WEATHER_TOOL_NAME, "disable_parallel_tool_use": True}


def _responses_tool_choice() -> ToolChoiceFunctionParam:
    return {"type": "function", "name": WEATHER_TOOL_NAME}


def _usage(input_tokens: int | None, output_tokens: int | None) -> Usage | None:
    if input_tokens is None or output_tokens is None:
        return None
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


@dataclass(frozen=True, slots=True)
class ChatCompletionsSurface:
    sdk: SdkClients
    name: SurfaceName = "chat_completions"

    def _turn(self, key: str, model: str, messages: Sequence[ChatCompletionMessageParam], tool: ToolMode) -> Reply:
        raw: Final = self.sdk.openai(key).chat.completions.with_raw_response.create(
            model=model,
            messages=list(messages),
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            tools=openai.omit if tool == "none" else [_chat_tool()],
            tool_choice=_chat_tool_choice() if tool == "forced" else openai.omit,
            parallel_tool_calls=False if tool == "forced" else openai.omit,
            extra_body=NO_PROXY_CACHE,
        )
        completion: Final = raw.parse()
        message: Final = completion.choices[0].message
        calls: Final = tuple(
            ToolCall(call_id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in message.tool_calls or ()
            if isinstance(call, ChatCompletionMessageFunctionToolCall)
        )
        return Reply(
            response_id=completion.id,
            text=message.content or "",
            tool_calls=calls,
            usage=None
            if completion.usage is None
            else _usage(completion.usage.prompt_tokens, completion.usage.completion_tokens),
            call_id_header=response_header(raw.headers, "x-litellm-call-id"),
            cost_header=response_header(raw.headers, "x-litellm-response-cost"),
        )

    def reply(self, key: str, model: str, prompt: str, *, with_tool: bool = False) -> Reply:
        return self._turn(key, model, _chat_history(prompt), "forced" if with_tool else "none")

    def stream(self, key: str, model: str, prompt: str) -> StreamedReply:
        chunks: Final[tuple[ChatCompletionChunk, ...]] = tuple(
            self.sdk.openai(key).chat.completions.create(
                model=model,
                messages=_chat_history(prompt),
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=NO_PROXY_CACHE,
            )
        )
        return StreamedReply(
            text="".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices),
            finished=any(chunk.choices[0].finish_reason is not None for chunk in chunks if chunk.choices),
            usage_reported=any(chunk.usage is not None for chunk in chunks),
            event_count=len(chunks),
        )

    def reply_to_tool_result(self, key: str, model: str, prompt: str, call: ToolCall, result: str) -> Reply:
        tool_call: Final[ChatCompletionMessageFunctionToolCallParam] = {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        assistant: Final[ChatCompletionAssistantMessageParam] = {"role": "assistant", "tool_calls": [tool_call]}
        tool_result: Final[ChatCompletionToolMessageParam] = {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": result,
        }
        return self._turn(key, model, (*_chat_history(prompt), assistant, tool_result), "offered")


def _chat_history(prompt: str) -> tuple[ChatCompletionMessageParam, ...]:
    return ({"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt})


@dataclass(frozen=True, slots=True)
class MessagesSurface:
    sdk: SdkClients
    name: SurfaceName = "messages"

    def _turn(self, key: str, model: str, messages: Sequence[MessageParam], tool: ToolMode) -> Reply:
        raw: Final = self.sdk.anthropic(key).messages.with_raw_response.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=INSTRUCTIONS,
            messages=list(messages),
            tools=anthropic.omit if tool == "none" else [_messages_tool()],
            tool_choice=_messages_tool_choice() if tool == "forced" else anthropic.omit,
            extra_body=NO_PROXY_CACHE,
        )
        message: Final = raw.parse()
        return Reply(
            response_id=message.id,
            text="".join(block.text for block in message.content if isinstance(block, TextBlock)),
            tool_calls=tuple(
                ToolCall(call_id=block.id, name=block.name, arguments=json.dumps(block.input))
                for block in message.content
                if isinstance(block, ToolUseBlock)
            ),
            usage=_usage(message.usage.input_tokens, message.usage.output_tokens),
            call_id_header=response_header(raw.headers, "x-litellm-call-id"),
            cost_header=response_header(raw.headers, "x-litellm-response-cost"),
        )

    def reply(self, key: str, model: str, prompt: str, *, with_tool: bool = False) -> Reply:
        return self._turn(key, model, ({"role": "user", "content": prompt},), "forced" if with_tool else "none")

    def stream(self, key: str, model: str, prompt: str) -> StreamedReply:
        events: Final[tuple[RawMessageStreamEvent, ...]] = tuple(
            self.sdk.anthropic(key).messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=INSTRUCTIONS,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        return StreamedReply(
            text="".join(
                event.delta.text
                for event in events
                if event.type == "content_block_delta" and event.delta.type == "text_delta"
            ),
            finished=any(event.type == "message_stop" for event in events),
            usage_reported=any(event.type == "message_delta" and event.usage.output_tokens > 0 for event in events),
            event_count=len(events),
        )

    def reply_to_tool_result(self, key: str, model: str, prompt: str, call: ToolCall, result: str) -> Reply:
        tool_use: Final[ToolUseBlockParam] = {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": call.parsed().model_dump(),
        }
        tool_result: Final[ToolResultBlockParam] = {
            "type": "tool_result",
            "tool_use_id": call.call_id,
            "content": result,
        }
        history: Final[tuple[MessageParam, ...]] = (
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [tool_result]},
        )
        return self._turn(key, model, history, "offered")


@dataclass(frozen=True, slots=True)
class ResponsesSurface:
    sdk: SdkClients
    name: SurfaceName = "responses"

    def _turn(self, key: str, model: str, history: ResponseInputParam, tool: ToolMode) -> Reply:
        raw: Final = self.sdk.openai(key).responses.with_raw_response.create(
            model=model,
            input=history,
            instructions=INSTRUCTIONS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=openai.omit if tool == "none" else [_responses_tool()],
            tool_choice=_responses_tool_choice() if tool == "forced" else openai.omit,
            parallel_tool_calls=False if tool == "forced" else openai.omit,
            extra_body=NO_PROXY_CACHE,
        )
        response: Final = raw.parse()
        return Reply(
            response_id=response.id,
            text=response.output_text,
            tool_calls=tuple(
                ToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
                for item in response.output
                if isinstance(item, ResponseFunctionToolCall)
            ),
            usage=None if response.usage is None else _usage(response.usage.input_tokens, response.usage.output_tokens),
            call_id_header=response_header(raw.headers, "x-litellm-call-id"),
            cost_header=response_header(raw.headers, "x-litellm-response-cost"),
        )

    def reply(self, key: str, model: str, prompt: str, *, with_tool: bool = False) -> Reply:
        return self._turn(key, model, [{"role": "user", "content": prompt}], "forced" if with_tool else "none")

    def stream(self, key: str, model: str, prompt: str) -> StreamedReply:
        events: Final[tuple[ResponseStreamEvent, ...]] = tuple(
            self.sdk.openai(key).responses.create(
                model=model,
                input=prompt,
                instructions=INSTRUCTIONS,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        return StreamedReply(
            text="".join(event.delta for event in events if event.type == "response.output_text.delta"),
            finished=bool(events) and events[-1].type == "response.completed",
            usage_reported=any(
                event.type == "response.completed" and event.response.usage is not None for event in events
            ),
            event_count=len(events),
        )

    def reply_to_tool_result(self, key: str, model: str, prompt: str, call: ToolCall, result: str) -> Reply:
        function_call: Final[ResponseFunctionToolCallParam] = {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }
        output: Final[FunctionCallOutput] = {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": result,
        }
        return self._turn(key, model, [{"role": "user", "content": prompt}, function_call, output], "offered")


def build_surfaces(sdk: SdkClients) -> Mapping[SurfaceName, Surface]:
    return MappingProxyType[SurfaceName, Surface](
        {
            "chat_completions": ChatCompletionsSurface(sdk),
            "messages": MessagesSurface(sdk),
            "responses": ResponsesSurface(sdk),
        }
    )
