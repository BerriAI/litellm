"""Ollama behind the proxy on /chat/completions, /v1/messages and /v1/responses.

Ollama has two litellm routes with different tool plumbing: `ollama_chat/` calls
/api/chat and forwards native tools, while `ollama/` calls /api/generate, which
has no tools field, so litellm prompts the model for a JSON function call and
turns that JSON back into a tool call. Each route runs the same conversation
contract as the conversational matrix on every surface, through the matrix's
SDK-backed surfaces, plus a streamed tool call on chat completions, the shape
coding agents such as OpenCode consume.

The deployments set drop_params because Ollama has no parallel_tool_calls,
which the matrix surfaces send alongside a forced tool_choice. Live only: Ollama
Cloud has no provider edge mount, and its requests are not priced in the cost
map, so there is no cost cell here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain, product
from types import MappingProxyType
from typing import Final, Literal, cast

import pytest
from _pytest.mark.structures import ParameterSet
from e2e_config import unique_marker
from e2e_metadata import Capability as SubjectCapability
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from llm_translation.conversational_matrix import (
    GREETING_PROMPT,
    INSTRUCTIONS,
    MAX_OUTPUT_TOKENS,
    SURFACES,
    WEATHER_PROMPT,
    WEATHER_REPORT,
    WEATHER_TOOL_DESCRIPTION,
    WEATHER_TOOL_NAME,
    WEATHER_TOOL_SCHEMA,
    Surface,
    SurfaceName,
    ToolCall,
    WeatherArgs,
    build_surfaces,
)
from llm_translation.sdk_clients import NO_PROXY_CACHE, SdkClients
from models import LiteLLMParamsBody
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

OllamaRoute = Literal["ollama_chat", "ollama"]
Capability = Literal["basic", "tool_use", "multi_turn"]
Streaming = Literal["stream", "nonstream"]

OLLAMA_API_BASE: Final = "https://ollama.com"
OLLAMA_MODEL: Final = "gemma4:31b"
ROUTES: Final[tuple[OllamaRoute, ...]] = ("ollama_chat", "ollama")


@dataclass(frozen=True, slots=True)
class Cell:
    surface: SurfaceName
    route: OllamaRoute

    @property
    def id(self) -> str:
        return f"{self.surface}-{self.route}"


def _cells(capability: Capability, streaming: Streaming) -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(
            Cell(surface=surface, route=route),
            id=f"{surface}-{route}",
            marks=pytest.mark.covers(f"llm.{surface}.{route}.{capability}.{streaming}.works"),
        )
        for surface, route in product(SURFACES, ROUTES)
    )


def _streamed_tool_cells() -> tuple[ParameterSet, ...]:
    return tuple(
        pytest.param(route, id=route, marks=pytest.mark.covers(f"llm.chat_completions.{route}.tool_use.stream.works"))
        for route in ROUTES
    )


def _register(proxy: ProxyClient, resources: ResourceManager, route: OllamaRoute) -> str:
    alias: Final = f"e2e-ollama-{route}-{unique_marker()}"
    model_id: Final = proxy.create_model(
        alias,
        LiteLLMParamsBody(
            model=f"{route}/{OLLAMA_MODEL}",
            api_base=OLLAMA_API_BASE,
            api_key="os.environ/OLLAMA_API_KEY",
            drop_params=True,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return alias


@pytest.fixture(scope="module")
def aliases(proxy: ProxyClient) -> Iterator[Mapping[OllamaRoute, str]]:
    resources: Final = ResourceManager(client=proxy)
    try:
        yield MappingProxyType({route: _register(proxy, resources, route) for route in ROUTES})
    finally:
        resources.teardown()


@pytest.fixture(scope="module")
def surfaces(sdk: SdkClients) -> Mapping[SurfaceName, Surface]:
    return build_surfaces(sdk)


def _weather_call(cell: Cell, surface: Surface, key: str, model: str) -> ToolCall:
    reply: Final = surface.reply(key, model, WEATHER_PROMPT, with_tool=True)
    assert len(reply.tool_calls) == 1, (
        f"{cell.id}: expected one {WEATHER_TOOL_NAME} call, got {reply.tool_calls} text={reply.text!r}"
    )
    call: Final = reply.tool_calls[0]
    assert call.name == WEATHER_TOOL_NAME, f"{cell.id}: called {call.name!r}, not {WEATHER_TOOL_NAME!r}"
    assert call.call_id, f"{cell.id}: tool call has no id, so the caller cannot answer it: {call}"
    assert "paris" in call.parsed().location.lower(), f"{cell.id}: tool arguments lost the location: {call}"
    return call


def _weather_tool() -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": WEATHER_TOOL_NAME,
            "description": WEATHER_TOOL_DESCRIPTION,
            "parameters": dict(WEATHER_TOOL_SCHEMA),
        },
    }


def _subject(mode: Mode, *, tools: bool, route: Route | None = None) -> Subject:
    return Subject(
        domain=Domain.LLM_TRANSLATION,
        route=route,
        providers=(Provider.OLLAMA,),
        models=(OLLAMA_MODEL,),
        capabilities=(SubjectCapability.FUNCTION_CALLING,) if tools else (),
        mode=mode,
    )


class TestOllamaConversation:
    @pytest.mark.parametrize("cell", _cells("basic", "nonstream"))
    @meta(_subject(Mode.NONSTREAM, tools=False))
    def test_reply_carries_assistant_text_and_usage(
        self,
        cell: Cell,
        aliases: Mapping[OllamaRoute, str],
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        reply: Final = surfaces[cell.surface].reply(resources.key(), aliases[cell.route], GREETING_PROMPT)

        assert reply.response_id, f"{cell.id}: response has no id"
        assert reply.text.strip(), f"{cell.id}: response carried no assistant text"
        assert reply.usage is not None and reply.usage.input_tokens > 0 and reply.usage.output_tokens > 0, (
            f"{cell.id}: usage missing or zero: {reply.usage}"
        )
        assert reply.call_id_header, f"{cell.id}: x-litellm-call-id header missing"

    @pytest.mark.parametrize("cell", _cells("basic", "stream"))
    @meta(_subject(Mode.STREAM, tools=False))
    def test_stream_delivers_text_usage_and_a_terminal_event(
        self,
        cell: Cell,
        aliases: Mapping[OllamaRoute, str],
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        streamed: Final = surfaces[cell.surface].stream(resources.key(), aliases[cell.route], GREETING_PROMPT)

        assert streamed.event_count > 1, f"{cell.id}: stream arrived as {streamed.event_count} event(s)"
        assert streamed.text.strip(), f"{cell.id}: stream carried no text deltas"
        assert streamed.finished, f"{cell.id}: stream never sent its terminal event"
        assert streamed.usage_reported, f"{cell.id}: stream never reported usage"

    @pytest.mark.parametrize("cell", _cells("tool_use", "nonstream"))
    @meta(_subject(Mode.NONSTREAM, tools=True))
    def test_tool_call_is_returned_named_and_addressable(
        self,
        cell: Cell,
        aliases: Mapping[OllamaRoute, str],
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        _ = _weather_call(cell, surfaces[cell.surface], resources.key(), aliases[cell.route])

    @pytest.mark.parametrize("cell", _cells("multi_turn", "nonstream"))
    @meta(_subject(Mode.NONSTREAM, tools=True))
    def test_tool_result_round_trip_reaches_the_model(
        self,
        cell: Cell,
        aliases: Mapping[OllamaRoute, str],
        surfaces: Mapping[SurfaceName, Surface],
        resources: ResourceManager,
    ) -> None:
        key: Final = resources.key()
        model: Final = aliases[cell.route]
        surface: Final = surfaces[cell.surface]
        call: Final = _weather_call(cell, surface, key, model)

        answer: Final = surface.reply_to_tool_result(key, model, WEATHER_PROMPT, call, WEATHER_REPORT)
        assert "22" in answer.text, f"{cell.id}: the model never saw the tool result: {answer.text!r}"


class TestOllamaStreamedToolCall:
    @pytest.mark.parametrize("route", _streamed_tool_cells())
    @meta(_subject(Mode.STREAM, tools=True, route=Route.CHAT_COMPLETIONS))
    def test_tool_call_streams_as_tool_call_deltas(
        self,
        route: OllamaRoute,
        aliases: Mapping[OllamaRoute, str],
        sdk: SdkClients,
        resources: ResourceManager,
    ) -> None:
        chunks: Final[tuple[ChatCompletionChunk, ...]] = tuple(
            sdk.openai(resources.key()).chat.completions.create(
                model=aliases[route],
                messages=[
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": WEATHER_PROMPT},
                ],
                tools=[_weather_tool()],
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
                extra_body=NO_PROXY_CACHE,
            )
        )
        choices: Final = tuple(chunk.choices[0] for chunk in chunks if chunk.choices)
        text: Final = "".join(choice.delta.content or "" for choice in choices)
        deltas: Final = tuple(chain.from_iterable(choice.delta.tool_calls or () for choice in choices))
        call_ids: Final = tuple(delta.id for delta in deltas if delta.id)
        indexes: Final = frozenset(delta.index for delta in deltas)
        functions: Final = tuple(delta.function for delta in deltas if delta.function is not None)
        names: Final = tuple(function.name for function in functions if function.name)
        arguments: Final = "".join(function.arguments or "" for function in functions)
        finish_reasons: Final = tuple(choice.finish_reason for choice in choices if choice.finish_reason is not None)

        assert names == (WEATHER_TOOL_NAME,), f"{route}: streamed tool names {names}, text={text!r}"
        assert len(call_ids) == 1, f"{route}: expected one streamed tool call id, got {call_ids}"
        assert indexes == {0}, f"{route}: streamed tool call deltas used indexes {sorted(indexes)}"
        assert WEATHER_TOOL_NAME not in text, f"{route}: the tool call leaked into assistant text: {text!r}"
        location: Final = WeatherArgs.model_validate(cast(object, json.loads(arguments))).location
        assert "paris" in location.lower(), f"{route}: streamed tool arguments lost the location: {arguments!r}"
        assert finish_reasons[-1:] == ("tool_calls",), f"{route}: stream finished with {finish_reasons}"
