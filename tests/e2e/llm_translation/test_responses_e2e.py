"""Live e2e: POST /v1/responses returns a real completion.

Registers an OpenAI deployment at runtime and drives the Responses API through
the gateway with the real OpenAI SDK, the client customers actually use
(LIT-4577), asserting output text came back.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
from pydantic import BaseModel

from e2e_config import require_env, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_CONVERSE_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
INSTRUCTIONS = "You are a helpful assistant"
CAT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

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


def _register(proxy: ProxyClient, resources: ResourceManager, params: LiteLLMParamsBody) -> str:
    model = f"e2e-responses-{unique_marker()}"
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

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS
        )
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
        events = list(stream)
        assert events, "responses stream returned no events"
        deltas = [event.delta for event in events if event.type == "response.output_text.delta"]
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
                    {
                        "type": "input_text",
                        "text": "What animal is shown in this image? Answer in one word",
                    },
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

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS
        )
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
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = _register(proxy, resources, _bedrock_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model, input="reply with one word", instructions=INSTRUCTIONS
        )
        assert response.output_text.strip(), (
            f"/responses over bedrock returned no output text: {response.output!r}"
        )

    @pytest.mark.covers("llm.responses.bedrock_converse.tool_use.nonstream.works")
    def test_responses_bedrock_returns_function_call(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = _register(proxy, resources, _bedrock_params())
        client = sdk.openai(resources.key())

        response = client.responses.create(
            model=model,
            input="What is the weather in San Francisco? Use the get_weather tool.",
            instructions=INSTRUCTIONS,
            tools=[WEATHER_TOOL],
        )
        _assert_weather_call(response)
