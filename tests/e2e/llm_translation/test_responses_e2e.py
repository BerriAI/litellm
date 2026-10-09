"""Live e2e: POST /v1/responses returns a real completion.

Registers an OpenAI deployment at runtime and drives the Responses API through
the gateway with the real OpenAI SDK, the client customers actually use
(LIT-4577), asserting output text came back. Malformed bodies the SDK refuses
to build stay on the shared transport. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, cast

import openai
import pytest
from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from e2e_http import assert_client_error
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from openai.types.responses import (
    FunctionShellToolParam,
    FunctionToolParam,
    Response,
    ResponseCompletedEvent,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
)
from provider_edge import LiveEdge, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from proxy_client import ProxyClient
from pydantic import BaseModel, TypeAdapter
from responses_helpers import AZURE_OPENAI_BACKEND, azure_openai_params
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e


class _OptionalResponsesBody(BaseModel):
    model: str | None = None
    input: str | None = None
    max_output_tokens: int | None = None


OPENAI_MINI_BACKEND: Final = "openai/gpt-4o-mini"
OPENAI_VISION_BACKEND: Final = "openai/gpt-4o"
ANTHROPIC_BACKEND: Final = "anthropic/claude-haiku-4-5"
BEDROCK_CONVERSE_BACKEND: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
VERTEX_BACKEND: Final = "vertex_ai/gemini-2.5-flash"
GEMINI_BACKEND: Final = "gemini/gemini-2.5-flash"
OPENAI_RESPONSES_BACKEND: Final = "openai/gpt-5.5"
INSTRUCTIONS = "You are a helpful assistant"
CAT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
BEDROCK_EDGE_REGION: Final = "us-east-1"
BEDROCK_EDGE_MOUNT: Final = f"bedrock/{BEDROCK_EDGE_REGION}"


class ConverseRequestBody(BaseModel):
    additionalModelRequestFields: dict[str, str] | None = None


@dataclass(slots=True)
class ConverseRequestCapture:
    """The Converse bodies the proxy actually sent upstream, as seen by a live
    edge sitting between the proxy and Bedrock."""

    _bodies: list[ConverseRequestBody] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, url: str, headers: Mapping[str, str], body: bytes | None) -> None:
        if body is None or "/converse" not in url:
            return
        with self._lock:
            self._bodies.append(ConverseRequestBody.model_validate_json(body))

    @property
    def bodies(self) -> tuple[ConverseRequestBody, ...]:
        with self._lock:
            return tuple(self._bodies)


WEATHER_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather for a location",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    "strict": False,
}

LOCATIONS_TOOL: Final[FunctionToolParam] = {
    "type": "function",
    "name": "get_locations",
    "description": "Return locations that need weather information",
    "parameters": {
        "type": "object",
        "properties": {"locations": {"type": "array", "items": {"type": "string"}}},
        "required": ["locations"],
        "additionalProperties": False,
    },
    "strict": True,
}


class LocationsArguments(BaseModel):
    locations: list[str]


class ResponseUsageCost(BaseModel):
    cost: float | None = None


def _openai_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=OPENAI_MINI_BACKEND, api_key="os.environ/OPENAI_API_KEY")


def _openai_responses_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=OPENAI_RESPONSES_BACKEND, api_key="os.environ/OPENAI_API_KEY")


def _anthropic_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=ANTHROPIC_BACKEND, api_key="os.environ/ANTHROPIC_API_KEY")


def _bedrock_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=BEDROCK_CONVERSE_BACKEND,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
    )


def _vertex_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=VERTEX_BACKEND,
        vertex_project="os.environ/VERTEXAI_PROJECT",
        vertex_location="us-central1",
    )


def _gemini_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=GEMINI_BACKEND, api_key="os.environ/GEMINI_API_KEY")


def _register(
    proxy: ProxyClient, resources: ResourceManager, params: LiteLLMParamsBody, prefix: str = "e2e-responses"
) -> str:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _function_calls(response: Response) -> tuple[ResponseFunctionToolCall, ...]:
    return tuple(item for item in response.output if isinstance(item, ResponseFunctionToolCall))


def _assert_weather_call(response: Response) -> None:
    function_call = next((call for call in _function_calls(response) if call.name == "get_weather"), None)
    assert function_call is not None, f"no get_weather function call: {response.output!r}"
    raw_arguments = cast(object, json.loads(function_call.arguments))
    arguments = WeatherArguments.model_validate(raw_arguments)
    assert arguments.location, f"function call arguments missing location: {function_call.arguments}"


class WeatherArguments(BaseModel):
    location: str


class TestResponses:
    @pytest.mark.covers("llm.responses.openai.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), f"/responses returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.openai.basic.stream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
            mode=Mode.STREAM,
        )
    )
    def test_responses_streaming_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _openai_params())
        client: Final = sdk.openai(resources.key())

        stream: Final = client.responses.create(
            model=model,
            input="reply with one word",
            instructions=INSTRUCTIONS,
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
        events: Final = tuple(stream)
        assert events, "responses stream returned no events"
        deltas: Final = tuple(event.delta for event in events if event.type == "response.output_text.delta")
        assert any(delta for delta in deltas), "responses stream returned no text deltas"
        completed: Final = events[-1]
        assert isinstance(completed, ResponseCompletedEvent), (
            f"responses stream did not terminate with response.completed: {completed.type}"
        )
        usage: Final = completed.response.usage
        assert usage is not None, f"response.completed had no usage: {completed.response!r}"
        assert usage.input_tokens > 0, f"response.completed had no input tokens: {usage!r}"
        assert usage.output_tokens > 0, f"response.completed had no output tokens: {usage!r}"
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens, (
            f"response.completed token totals were inconsistent: {usage!r}"
        )
        usage_cost: Final = TypeAdapter(ResponseUsageCost).validate_python(
            cast(object, usage.model_extra if usage.model_extra is not None else {})
        )
        assert usage_cost.cost is not None, f"response.completed usage had no cost: {usage.model_extra!r}"
        assert usage_cost.cost > 0, f"response.completed cost was not positive: {usage_cost.cost}"

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.GEMINI,),
            models=(GEMINI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_gemini_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _gemini_params(), prefix="e2e-responses-gemini")
        client: Final = sdk.openai(resources.key())

        response: Final = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), f"/responses over gemini returned no output text: {response.output!r}"

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.GEMINI,),
            models=(GEMINI_BACKEND,),
            mode=Mode.STREAM,
        )
    )
    def test_responses_gemini_streaming_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _gemini_params(), prefix="e2e-responses-gemini")
        client: Final = sdk.openai(resources.key())

        stream: Final = client.responses.create(
            model=model,
            input="reply with one word",
            instructions=INSTRUCTIONS,
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
        events: Final = tuple(stream)
        deltas: Final = tuple(event.delta for event in events if event.type == "response.output_text.delta")
        assert any(deltas), "responses stream over gemini returned no text deltas"
        assert isinstance(events[-1], ResponseCompletedEvent), (
            f"responses stream over gemini did not end with response.completed: {events[-1].type}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.GEMINI,),
            models=(GEMINI_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_gemini_replays_legacy_function_call_output(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _gemini_params(), prefix="e2e-responses-gemini-tool")
        client: Final = sdk.openai(resources.key())
        function_call_id: Final = f"fc_{unique_marker()}"
        input_items: Final[ResponseInputParam] = [
            {
                "type": "message",
                "role": "user",
                "content": "What is the temperature in Paris today?",
            },
            {
                "type": "function_call",
                "arguments": '{"location": "Paris, France"}',
                "call_id": function_call_id,
                "name": "get_temperature",
                "id": function_call_id,
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": function_call_id,
                "output": "Temperature is exactly 31 Celsius.",
            },
        ]
        tools: Final[tuple[FunctionToolParam, ...]] = (
            {
                "type": "function",
                "name": "get_temperature",
                "description": "Get the current temperature for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        )

        response: Final = client.responses.create(
            model=model,
            input=input_items,
            tools=tools,
            store=False,
            extra_body=NO_PROXY_CACHE,
        )
        assert response.status == "completed", f"legacy tool replay was not completed: {response.status}"
        assert "31" in response.output_text, (
            f"legacy tool result was missing from output text: {response.output_text!r}"
        )

    @pytest.mark.covers("llm.responses.openai.basic.nonstream.cost_logged")
    @meta(
        Subject(
            domain=Domain.SPEND_BUDGETS,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_logs_cost(self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        raw = client.responses.with_raw_response.create(
            model=model,
            input=f"reply with one word {unique_marker()}",
            instructions=INSTRUCTIONS,
            extra_body=NO_PROXY_CACHE,
        )
        response = raw.parse()
        assert response.output_text.strip(), f"/responses returned no output text: {response.output!r}"
        assert raw.headers.get("x-litellm-call-id") and response.id, (
            f"missing response identifiers: id={response.id!r}, headers={dict(raw.headers)}"
        )

        rows = proxy.poll_logs_for_request_id(
            response.id,
            predicate=lambda logged_rows: any((row.spend or 0) > 0 for row in logged_rows),
        )
        row = next((logged_row for logged_row in rows if (logged_row.spend or 0) > 0), None)
        assert row is not None, f"no costed spend row for response id {response.id}"
        assert "gpt-4o-mini" in (row.model or ""), f"unexpected spend row model: {row.model}"

    @pytest.mark.covers("llm.responses.openai.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
            extra_body=NO_PROXY_CACHE,
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.openai.vision.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_VISION_BACKEND,),
            capabilities=(Capability.VISION,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_vision_describes_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(
            proxy,
            resources,
            LiteLLMParamsBody(model=OPENAI_VISION_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
        )
        client = sdk.openai(resources.key())

        vision_input: ResponseInputParam = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What animal is shown in this image? Answer in one word"},
                    {"type": "input_image", "image_url": CAT_IMAGE_URL, "detail": "auto"},
                ],
            }
        ]
        response = client.responses.create(
            model=model, input=vision_input, instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        text = response.output_text.strip().lower()
        assert text, f"/responses vision returned no output text: {response.output!r}"
        assert any(keyword in text for keyword in ("cat", "feline")), (
            f"vision response did not describe the image: {text[:300]}"
        )

    @pytest.mark.covers("llm.responses.anthropic.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_anthropic_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _anthropic_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), f"/responses returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_anthropic_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _anthropic_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
            extra_body=NO_PROXY_CACHE,
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.anthropic.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_anthropic_strict_array_schema_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _anthropic_params())
        client: Final = sdk.openai(resources.key())

        response: Final = client.responses.create(
            model=model,
            input="Find the weather locations for Tokyo and Paris using get_locations.",
            instructions=INSTRUCTIONS,
            tools=[LOCATIONS_TOOL],
            tool_choice="required",
            extra_body=NO_PROXY_CACHE,
        )
        function_call: Final = next(
            (call for call in _function_calls(response) if call.name == "get_locations"),
            None,
        )
        assert function_call is not None, f"response had no get_locations call: {response.output!r}"
        arguments: Final = LocationsArguments.model_validate_json(function_call.arguments)
        assert arguments.locations, f"get_locations returned no locations: {function_call.arguments}"

    @pytest.mark.covers("llm.responses.anthropic.multi_turn.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_anthropic_tool_output_continues_with_previous_response_id(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, _anthropic_params())
        client: Final = sdk.openai(resources.key())

        first: Final = client.responses.create(
            model=model,
            input="Find the weather locations for Tokyo and Paris using get_locations.",
            instructions=INSTRUCTIONS,
            tools=[LOCATIONS_TOOL],
            tool_choice="required",
            extra_body=NO_PROXY_CACHE,
        )
        function_call: Final = next(
            (call for call in _function_calls(first) if call.name == "get_locations"),
            None,
        )
        assert function_call is not None, f"response had no get_locations call: {first.output!r}"
        assert function_call.call_id, f"get_locations call had no call_id: {function_call!r}"
        arguments: Final = LocationsArguments.model_validate_json(function_call.arguments)
        assert arguments.locations, f"get_locations call had no locations: {function_call.arguments}"

        tool_result: Final = "Distinctive forecast: 47 degrees Celsius"
        follow_up_input: Final[ResponseInputParam] = [
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": tool_result,
            }
        ]
        second: Final = client.responses.create(
            model=model,
            previous_response_id=first.id,
            input=follow_up_input,
            instructions=INSTRUCTIONS,
            tools=[LOCATIONS_TOOL],
            extra_body=NO_PROXY_CACHE,
        )
        assert "47" in second.output_text, f"follow-up omitted tool result: {second.output_text!r}"

    @pytest.mark.covers("llm.responses.bedrock_converse.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_CONVERSE_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_bedrock_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _bedrock_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), f"/responses over bedrock returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.bedrock_converse.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_CONVERSE_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_bedrock_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _bedrock_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
            extra_body=NO_PROXY_CACHE,
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.vertex.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.VERTEX_AI,),
            models=(VERTEX_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_vertex_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _vertex_params(), prefix="e2e-responses-vertex")
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), f"/responses over vertex returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.vertex.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.VERTEX_AI,),
            models=(VERTEX_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_vertex_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _vertex_params(), prefix="e2e-responses-vertex-tool")
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
            tool_choice="required",
            extra_body=NO_PROXY_CACHE,
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.azure_openai.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.AZURE,),
            models=(AZURE_OPENAI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_azure_openai_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, azure_openai_params(), prefix="e2e-responses-azure-openai")
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, extra_body=NO_PROXY_CACHE
        )
        assert response.output_text.strip(), (
            f"/responses over azure openai returned no output text: {response.output!r}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.AZURE,),
            models=(AZURE_OPENAI_BACKEND,),
            mode=Mode.STREAM,
        )
    )
    def test_responses_azure_openai_streaming_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(proxy, resources, azure_openai_params(), prefix="e2e-responses-azure-stream")
        client: Final = sdk.openai(resources.key())

        stream: Final = client.responses.create(
            model=model,
            input="reply with one word",
            instructions=INSTRUCTIONS,
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
        events: Final = tuple(stream)
        deltas: Final = tuple(event.delta for event in events if event.type == "response.output_text.delta")
        assert any(deltas), "responses stream over azure openai returned no text deltas"
        assert isinstance(events[-1], ResponseCompletedEvent), (
            f"responses stream over azure openai did not end with response.completed: {events[-1].type}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.AZURE,),
            models=(AZURE_OPENAI_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_azure_openai_preview_api_version_accepts_truncation(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(
            proxy,
            resources,
            azure_openai_params(api_version="preview"),
            prefix="e2e-responses-azure-preview",
        )
        client: Final = sdk.openai(resources.key())

        response: Final = client.responses.create(
            model=model,
            input="reply with one word",
            instructions=INSTRUCTIONS,
            truncation="auto",
            extra_body=NO_PROXY_CACHE,
        )
        assert response.output_text.strip(), (
            f"/responses over azure openai preview returned no output text: {response.output!r}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_RESPONSES_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_compact_returns_compacted_conversation(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(
            proxy,
            resources,
            _openai_responses_params(),
            prefix="e2e-responses-compact",
        )
        client: Final = sdk.openai(resources.key())
        conversation: Final[ResponseInputParam] = [
            {"role": "user", "content": "Remember that my favorite color is blue."},
            {"role": "assistant", "content": "I will remember that your favorite color is blue."},
        ]

        compacted: Final = client.responses.compact(
            model=model,
            input=conversation,
            extra_body=NO_PROXY_CACHE,
        )
        assert compacted.id, f"/responses/compact returned no id: {compacted!r}"
        assert any(item.type == "compaction" for item in compacted.output), (
            f"/responses/compact returned no compaction item: {compacted.output!r}"
        )
        compacted_input: Final[ResponseInputParam] = TypeAdapter(ResponseInputParam).validate_python(
            [item.model_dump(exclude_none=True) for item in compacted.output]
            + [{"role": "user", "content": "What is my favorite color?"}]
        )

        response: Final = client.responses.create(
            model=model,
            input=compacted_input,
            extra_body=NO_PROXY_CACHE,
        )
        assert "blue" in response.output_text.lower(), (
            f"compacted conversation did not retain the favorite color: {response.output_text!r}"
        )

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_RESPONSES_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_context_management_compacts_server_side(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model: Final = _register(
            proxy,
            resources,
            _openai_responses_params(),
            prefix="e2e-responses-context-compaction",
        )
        client: Final = sdk.openai(resources.key())
        filler: Final = "The archive record has a blue marker beside every stored entry. " * 350
        conversation: Final[ResponseInputParam] = [
            {"role": "user", "content": filler},
            {"role": "assistant", "content": "I have read the archive and retained its details."},
            {"role": "user", "content": "Reply with one word to verify server-side compaction."},
        ]

        response: Final = client.responses.create(
            model=model,
            input=conversation,
            context_management=[{"type": "compaction", "compact_threshold": 1000}],
            extra_body=NO_PROXY_CACHE,
        )
        assert response.status == "completed", f"context management did not complete: {response.status}"
        assert any(item.type == "compaction" for item in response.output), (
            f"context management returned no compaction item: {response.output!r}"
        )

    @pytest.mark.covers("llm.responses.azure_openai.tool_use.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.AZURE,),
            models=(AZURE_OPENAI_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_responses_azure_openai_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, azure_openai_params(), prefix="e2e-responses-azure-openai-tool")
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
            tool_choice="required",
            extra_body=NO_PROXY_CACHE,
        )
        _assert_weather_call(response)

    @pytest.mark.provider_edge_host
    @pytest.mark.parametrize(
        "endpoint",
        [
            pytest.param(
                "/v1/responses",
                marks=meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.RESPONSES,
                        providers=(Provider.BEDROCK,),
                        models=(BEDROCK_CONVERSE_BACKEND,),
                        mode=Mode.NONSTREAM,
                    )
                ),
                id="/v1/responses",
            ),
            pytest.param(
                "/v1/chat/completions",
                marks=meta(
                    Subject(
                        domain=Domain.LLM_TRANSLATION,
                        route=Route.CHAT_COMPLETIONS,
                        providers=(Provider.BEDROCK,),
                        models=(BEDROCK_CONVERSE_BACKEND,),
                        mode=Mode.NONSTREAM,
                    )
                ),
                id="/v1/chat/completions",
            ),
        ],
    )
    def test_bedrock_forwards_allowed_safety_identifier_as_additional_model_request_field(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, endpoint: str
    ) -> None:
        """Judges the Converse bodies the edge captured, not the reply: Claude on
        Bedrock rejects the forwarded field with a 400, which the chat leg's
        ``Result`` carries as a value and the OpenAI SDK raises."""
        capture: Final = ConverseRequestCapture()
        edge: Final = start_provider_edge(
            LiveEdge(observe_request=capture.observe, sign=bedrock_signer(BEDROCK_EDGE_REGION)),
            mounts=MappingProxyType(
                {BEDROCK_EDGE_MOUNT: f"https://bedrock-runtime.{BEDROCK_EDGE_REGION}.amazonaws.com"}
            ),
            bind_host=PROVIDER_EDGE_BIND_HOST,
            advertise_host=PROVIDER_EDGE_ADVERTISE_HOST,
        )
        resources.defer(edge.shutdown)
        model: Final = f"e2e-responses-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=BEDROCK_CONVERSE_BACKEND,
                api_base=edge.edge.api_base(BEDROCK_EDGE_MOUNT),
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name=BEDROCK_EDGE_REGION,
                allowed_openai_params=["safety_identifier"],
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key: Final = resources.key()
        safety_identifier: Final = f"end-user-{unique_marker()}"

        if endpoint == "/v1/responses":
            with contextlib.suppress(openai.BadRequestError):
                sdk.openai(key).responses.create(
                    model=model,
                    input="reply with one word",
                    instructions=INSTRUCTIONS,
                    safety_identifier=safety_identifier,
                    extra_body=NO_PROXY_CACHE,
                )
        else:
            proxy.chat(
                key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content="reply with one word")],
                    safety_identifier=safety_identifier,
                ),
            )

        forwarded: Final = tuple(body.additionalModelRequestFields for body in capture.bodies)
        assert forwarded, f"{endpoint} produced no Bedrock Converse request"
        assert forwarded == ({"safety_identifier": safety_identifier},) * len(forwarded), (
            f"{endpoint} did not forward safety_identifier to Bedrock Converse on every attempt: {capture.bodies}"
        )

    @pytest.mark.skip(
        reason="stage red: product gap, /v1/responses 500s (aresponses TypeError) on missing input instead of 400"
    )
    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
        )
    )
    def test_missing_input_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = _register(proxy, resources, _openai_params(), prefix="e2e-responses-val")
        key = resources.key()
        result = proxy.transport.send(
            "/v1/responses",
            headers=proxy.transport.bearer(key),
            json=_OptionalResponsesBody(model=model),
        )
        assert_client_error(result, "responses missing input")

    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
        )
    )
    def test_missing_model_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key = resources.key()
        result = proxy.transport.send(
            "/v1/responses",
            headers=proxy.transport.bearer(key),
            json=_OptionalResponsesBody(input="ping"),
        )
        assert_client_error(result, "responses missing model")

    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MINI_BACKEND,),
        )
    )
    def test_empty_input_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = _register(proxy, resources, _openai_params(), prefix="e2e-responses-val")
        key = resources.key()
        result = proxy.transport.send(
            "/v1/responses",
            headers=proxy.transport.bearer(key),
            json=_OptionalResponsesBody(model=model, input=""),
        )
        assert_client_error(result, "responses empty input")


REASONING_BACKEND: Final = "openai/gpt-5.4-mini"
SHELL_BACKEND: Final = "openai/gpt-5.5"
TOOL_DATE: Final = "2025-01-15"
REASONING_ATTEMPTS: Final = 3

GET_TODAY_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "get_today",
    "description": "Return today's date",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    "strict": True,
}

TODAY_REPORT_FORMAT: ResponseFormatTextJSONSchemaConfigParam = {
    "type": "json_schema",
    "name": "today_report",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"today": {"type": "string"}, "number_of_r": {"type": "string"}},
        "required": ["today", "number_of_r"],
        "additionalProperties": False,
    },
}

SHELL_TOOL: FunctionShellToolParam = {"type": "shell", "environment": {"type": "container_auto"}}

_INPUT_ITEMS: Final = TypeAdapter(list[ResponseInputItemParam])


class TodayReport(BaseModel):
    today: str
    number_of_r: str


def _until_reasoning_emitted(create: Callable[[], Response]) -> Response:
    for _ in range(REASONING_ATTEMPTS - 1):
        response = create()
        if any(isinstance(item, ResponseReasoningItem) for item in response.output):
            return response
    return create()


class TestResponsesOpenAIHostedFeatures:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(REASONING_BACKEND,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.REASONING, Capability.RESPONSE_SCHEMA),
            mode=Mode.NONSTREAM,
        )
    )
    def test_reasoning_items_replay_into_structured_output_after_tool_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(
            proxy,
            resources,
            LiteLLMParamsBody(model=REASONING_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
            prefix="e2e-responses-reasoning",
        )
        client = sdk.openai(resources.key())
        question: ResponseInputItemParam = {
            "role": "user",
            "content": (
                "How many r are in strrawberrry? Call get_today first, then report today exactly as get_today "
                f"returned it and the count of r. {unique_marker()}"
            ),
        }

        first = _until_reasoning_emitted(
            lambda: client.responses.create(
                model=model,
                input=[question],
                tools=[GET_TODAY_TOOL],
                tool_choice={"type": "function", "name": "get_today"},
                reasoning={"effort": "medium", "summary": "auto"},
                text={"format": TODAY_REPORT_FORMAT},
                extra_body=NO_PROXY_CACHE,
            )
        )
        assert any(isinstance(item, ResponseReasoningItem) for item in first.output), (
            f"reasoning model returned no reasoning item: {first.output!r}"
        )
        call = next((call for call in _function_calls(first) if call.name == "get_today"), None)
        assert call is not None, f"forced get_today call missing: {first.output!r}"

        replayed = _INPUT_ITEMS.validate_python([item.model_dump(exclude_none=True) for item in first.output])
        tool_result: ResponseInputItemParam = {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": TOOL_DATE,
        }
        second = client.responses.create(
            model=model,
            input=[question, *replayed, tool_result],
            tools=[GET_TODAY_TOOL],
            reasoning={"effort": "medium", "summary": "auto"},
            text={"format": TODAY_REPORT_FORMAT},
            extra_body=NO_PROXY_CACHE,
        )
        assert second.status == "completed", f"second turn did not complete: {second.status} {second.output!r}"
        report = TodayReport.model_validate_json(second.output_text)
        assert TOOL_DATE in report.today, f"structured output ignored the tool result: {report!r}"

    @pytest.mark.provider_live
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(Provider.OPENAI,),
            models=(SHELL_BACKEND,),
            mode=Mode.STREAM,
        )
    )
    def test_shell_tool_stream_surfaces_shell_call_and_its_output(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(
            proxy,
            resources,
            LiteLLMParamsBody(model=SHELL_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
            prefix="e2e-responses-shell",
        )
        client = sdk.openai(resources.key())

        stream = client.responses.create(
            model=model,
            input="Run `python --version` in the shell and reply with what it printed.",
            tools=[SHELL_TOOL],
            tool_choice="required",
            max_output_tokens=1024,
            stream=True,
            extra_body=NO_PROXY_CACHE,
        )
        events = tuple(stream)
        completed = events[-1] if events else None
        assert isinstance(completed, ResponseCompletedEvent), (
            f"shell stream did not end with response.completed: {[event.type for event in events]}"
        )
        streamed_items = tuple(event.item for event in events if isinstance(event, ResponseOutputItemDoneEvent))
        assert any(isinstance(item, ResponseFunctionShellToolCall) for item in streamed_items), (
            f"no shell_call item reached the stream: {[item.type for item in streamed_items]}"
        )
        outputs = tuple(
            item for item in completed.response.output if isinstance(item, ResponseFunctionShellToolCallOutput)
        )
        assert outputs, f"completed response carries no shell_call_output: {completed.response.output!r}"
