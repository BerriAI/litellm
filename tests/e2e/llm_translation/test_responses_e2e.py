"""Live e2e: POST /v1/responses returns a real completion.

Registers an OpenAI deployment at runtime, drives the Responses API through the
gateway, and asserts output text came back. Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from endpoints_client import (
    EndpointsClient,
    FunctionParameterProperty,
    FunctionParameters,
    ResponsesFunctionTool,
    ResponsesOutputTextDeltaEvent,
    ResponsesResult,
    ResponsesStreamEventType,
)
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

BEDROCK_CONVERSE_BACKEND = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"

WEATHER_TOOL = ResponsesFunctionTool(
    name="get_weather",
    description="Get the weather for a location",
    parameters=FunctionParameters(
        properties={"location": FunctionParameterProperty(type="string")},
        required=["location"],
    ),
)


def _bedrock_params() -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=BEDROCK_CONVERSE_BACKEND,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
    )


class WeatherArguments(BaseModel):
    location: str


class TestResponses:
    @pytest.mark.covers("llm.responses.openai.basic.nonstream.works")
    def test_responses_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.openai.basic.stream.works")
    def test_responses_streaming_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word", stream=True)
        require_successful_call(result)
        delta_events = tuple(
            parsed
            for event in result.stream_events
            if (parsed := _parse_stream_event(event)) is not None
        )

        assert any(event.delta for event in delta_events), "responses stream returned no text deltas"
        assert result.stream_events, "responses stream returned no events"
        assert (
            ResponsesStreamEventType.model_validate_json(result.stream_events[-1]).type
            == "response.completed"
        ), "responses stream did not terminate with response.completed"

    @pytest.mark.covers("llm.responses.openai.basic.nonstream.cost_logged")
    def test_responses_logs_cost(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, f"reply with one word {unique_marker()}")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses returned no output text: {result.body[:300]}"
        assert result.call_id and parsed.id, f"missing response identifiers: {result.body[:300]}"

        rows = endpoints_client.proxy.poll_logs_for_request_id(
            parsed.id,
            predicate=lambda logged_rows: any((row.spend or 0) > 0 for row in logged_rows),
        )
        row = next((logged_row for logged_row in rows if (logged_row.spend or 0) > 0), None)
        assert row is not None, f"no costed spend row for response id {parsed.id}"
        assert "gpt-4o-mini" in (row.model or ""), f"unexpected spend row model: {row.model}"

    @pytest.mark.covers("llm.responses.openai.tool_use.nonstream.works")
    def test_responses_returns_function_call(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses_with_tools(
            key,
            model,
            "What is the weather in San Francisco? Use the get_weather tool.",
            [
                ResponsesFunctionTool(
                    name="get_weather",
                    description="Get the weather for a location",
                    parameters=FunctionParameters(
                        properties={"location": FunctionParameterProperty(type="string")},
                        required=["location"],
                    ),
                )
            ],
        )
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        function_call = next(
            (call for call in parsed.function_calls if call.name == "get_weather"),
            None,
        )
        assert function_call is not None, f"no get_weather function call: {result.body[:500]}"
        assert function_call.arguments is not None
        raw_arguments = cast(object, json.loads(function_call.arguments))
        arguments = WeatherArguments.model_validate(raw_arguments)
        assert arguments.location, f"function call arguments missing location: {function_call.arguments}"

    @pytest.mark.covers("llm.responses.openai.vision.nonstream.works")
    def test_responses_vision_describes_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses_vision(
            key,
            model,
            "What animal is shown in this image? Answer in one word",
            "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        )
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        text = parsed.text.strip().lower()
        assert text, f"/responses vision returned no output text: {result.body[:300]}"
        assert any(
            keyword in text
            for keyword in ("cat", "feline")
        ), f"vision response did not describe the image: {parsed.text[:300]}"

    @pytest.mark.covers("llm.responses.anthropic.basic.nonstream.works")
    def test_responses_anthropic_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.anthropic.tool_use.nonstream.works")
    def test_responses_anthropic_returns_function_call(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="anthropic/claude-haiku-4-5", api_key="os.environ/ANTHROPIC_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses_with_tools(
            key,
            model,
            "What is the weather in San Francisco? Use the get_weather tool.",
            [
                ResponsesFunctionTool(
                    name="get_weather",
                    description="Get the weather for a location",
                    parameters=FunctionParameters(
                        properties={"location": FunctionParameterProperty(type="string")},
                        required=["location"],
                    ),
                )
            ],
        )
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        function_call = next(
            (call for call in parsed.function_calls if call.name == "get_weather"),
            None,
        )
        assert function_call is not None, f"no get_weather function call: {result.body[:500]}"
        assert function_call.arguments is not None
        raw_arguments = cast(object, json.loads(function_call.arguments))
        arguments = WeatherArguments.model_validate(raw_arguments)
        assert arguments.location, f"function call arguments missing location: {function_call.arguments}"

    @pytest.mark.covers("llm.responses.bedrock_converse.basic.nonstream.works")
    def test_responses_bedrock_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(model, _bedrock_params())
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses(key, model, "reply with one word")
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses over bedrock returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.bedrock_converse.tool_use.nonstream.works")
    def test_responses_bedrock_returns_function_call(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"e2e-responses-{unique_marker()}"
        model_id = endpoints_client.create_model(model, _bedrock_params())
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.responses_with_tools(
            key, model, "What is the weather in San Francisco? Use the get_weather tool.", [WEATHER_TOOL]
        )
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        function_call = next((call for call in parsed.function_calls if call.name == "get_weather"), None)
        assert function_call is not None, f"no get_weather function call over bedrock: {result.body[:500]}"
        assert function_call.arguments is not None
        raw_arguments = cast(object, json.loads(function_call.arguments))
        arguments = WeatherArguments.model_validate(raw_arguments)
        assert arguments.location, f"function call arguments missing location: {function_call.arguments}"


def _parse_stream_event(
    event: str,
) -> ResponsesOutputTextDeltaEvent | None:
    try:
        return ResponsesOutputTextDeltaEvent.model_validate_json(event)
    except ValidationError:
        return None
