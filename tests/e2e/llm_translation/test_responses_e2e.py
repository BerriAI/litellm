"""Live e2e: POST /v1/responses returns a real completion.

Registers an OpenAI deployment at runtime and drives the Responses API through
the gateway with the real OpenAI SDK, the client customers actually use
(LIT-4577), asserting output text came back. Malformed bodies the SDK refuses
to build stay on the shared transport. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, cast

import pytest
from e2e_config import PROVIDER_EDGE_ADVERTISE_HOST, PROVIDER_EDGE_BIND_HOST, unique_marker
from e2e_http import assert_client_error
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from provider_edge import LiveEdge, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


class _OptionalResponsesBody(BaseModel):
    model: str | None = None
    input: str | None = None
    max_output_tokens: int | None = None


BEDROCK_CONVERSE_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
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


def _openai_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY")


def _anthropic_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY")


def _bedrock_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=BEDROCK_CONVERSE_BACKEND,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
    )


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
    def test_responses_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(model=model, input="reply with one word", instructions=INSTRUCTIONS)
        assert response.output_text.strip(), f"/responses returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.openai.basic.stream.works")
    def test_responses_streaming_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        stream = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS, stream=True
        )
        events = tuple(stream)
        assert events, "responses stream returned no events"
        deltas = tuple(event.delta for event in events if event.type == "response.output_text.delta")
        assert any(delta for delta in deltas), "responses stream returned no text deltas"
        assert events[-1].type == "response.completed", (
            f"responses stream did not terminate with response.completed: {events[-1].type}"
        )

    @pytest.mark.covers("llm.responses.openai.basic.nonstream.cost_logged")
    def test_responses_logs_cost(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _openai_params())
        client = sdk.openai(resources.key())

        raw = client.responses.with_raw_response.create(
            model=model, input=f"reply with one word {unique_marker()}", instructions=INSTRUCTIONS
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
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.openai.vision.nonstream.works")
    def test_responses_vision_describes_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(
            proxy,
            resources,
            LiteLLMParamsBody(model="openai/gpt-4o", api_key="os.environ/OPENAI_API_KEY"),
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
        response = client.responses.create(model=model, input=vision_input, instructions=INSTRUCTIONS)
        text = response.output_text.strip().lower()
        assert text, f"/responses vision returned no output text: {response.output!r}"
        assert any(keyword in text for keyword in ("cat", "feline")), (
            f"vision response did not describe the image: {text[:300]}"
        )

    @pytest.mark.covers("llm.responses.anthropic.basic.nonstream.works")
    def test_responses_anthropic_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _anthropic_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(model=model, input="reply with one word", instructions=INSTRUCTIONS)
        assert response.output_text.strip(), f"/responses returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.anthropic.tool_use.nonstream.works")
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
        )
        _assert_weather_call(response)

    @pytest.mark.covers("llm.responses.bedrock_converse.basic.nonstream.works")
    def test_responses_bedrock_returns_completion(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, _bedrock_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(model=model, input="reply with one word", instructions=INSTRUCTIONS)
        assert response.output_text.strip(), f"/responses over bedrock returned no output text: {response.output!r}"

    @pytest.mark.covers("llm.responses.bedrock_converse.tool_use.nonstream.works")
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
        )
        _assert_weather_call(response)

    @pytest.mark.provider_edge_host
    @pytest.mark.parametrize("endpoint", ["/v1/responses", "/v1/chat/completions"])
    def test_bedrock_forwards_allowed_safety_identifier_as_additional_model_request_field(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, endpoint: str
    ) -> None:
        capture: Final = ConverseRequestCapture()
        edge: Final = start_provider_edge(
            LiveEdge(observe_request=capture.observe, sign=bedrock_signer(BEDROCK_EDGE_REGION)),
            mounts=MappingProxyType({BEDROCK_EDGE_MOUNT: f"https://bedrock-runtime.{BEDROCK_EDGE_REGION}.amazonaws.com"}),
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
            sdk.openai(key).responses.create(
                model=model,
                input="reply with one word",
                instructions=INSTRUCTIONS,
                safety_identifier=safety_identifier,
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

    @pytest.mark.skip(reason="stage red: product gap, /v1/responses 500s (aresponses TypeError) on missing input instead of 400")
    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
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
    def test_missing_model_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key = resources.key()
        result = proxy.transport.send(
            "/v1/responses",
            headers=proxy.transport.bearer(key),
            json=_OptionalResponsesBody(input="ping"),
        )
        assert_client_error(result, "responses missing model")

    @pytest.mark.covers("llm.responses.openai.input_validation.nonstream.works")
    def test_empty_input_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = _register(proxy, resources, _openai_params(), prefix="e2e-responses-val")
        key = resources.key()
        result = proxy.transport.send(
            "/v1/responses",
            headers=proxy.transport.bearer(key),
            json=_OptionalResponsesBody(model=model, input=""),
        )
        assert_client_error(result, "responses empty input")
