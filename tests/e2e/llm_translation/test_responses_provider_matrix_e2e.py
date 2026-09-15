"""Live /v1/responses coverage for the cloud providers that have no native
Responses API of their own: Azure OpenAI and Vertex AI Gemini.

The gateway accepts the OpenAI Responses request shape and translates it to
whatever the deployment's provider speaks, so a customer can keep one client
across clouds. Each test registers its deployment through /model/new with
`os.environ/...` credential references the proxy resolves at call time, drives
the endpoint, and deletes the deployment on teardown.
"""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import (
    EndpointsClient,
    FunctionParameterProperty,
    FunctionParameters,
    ResponsesFunctionTool,
    ResponsesResult,
)
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

AZURE_OPENAI_BACKEND = "azure/gpt-5.6-sol-e2e"
VERTEX_BACKEND = "vertex_ai/gemini-2.5-flash"

WEATHER_TOOL = ResponsesFunctionTool(
    name="get_weather",
    description="Get the weather for a location",
    parameters=FunctionParameters(
        properties={"location": FunctionParameterProperty(type="string")},
        required=["location"],
    ),
)

WEATHER_PROMPT = "What is the weather in San Francisco? Use the get_weather tool."

BASIC_PROMPT = "Reply with one word."


class WeatherArguments(BaseModel):
    location: str


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


def _register(
    endpoints_client: EndpointsClient,
    resources: ResourceManager,
    prefix: str,
    params: LiteLLMParamsBody,
) -> tuple[str, str]:
    model = f"{prefix}-{unique_marker()}"
    model_id = endpoints_client.create_model(model, params)
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model, resources.key()


def _assert_weather_function_call(body: str) -> None:
    parsed = ResponsesResult.model_validate_json(body)
    function_call = next((call for call in parsed.function_calls if call.name == "get_weather"), None)
    assert function_call is not None, f"no get_weather function call: {body[:500]}"
    assert function_call.arguments is not None
    raw_arguments = cast(object, json.loads(function_call.arguments))
    arguments = WeatherArguments.model_validate(raw_arguments)
    assert arguments.location, f"function call arguments missing location: {function_call.arguments}"


class TestAzureOpenAIResponses:
    @pytest.mark.covers("llm.responses.azure_openai.basic.nonstream.works")
    def test_azure_openai_responses_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources, "e2e-responses-azure", _azure_openai_params())

        result = endpoints_client.responses(key, model, BASIC_PROMPT)
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses over azure openai returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.azure_openai.tool_use.nonstream.works")
    def test_azure_openai_responses_returns_function_call(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources, "e2e-responses-azure-tool", _azure_openai_params())

        result = endpoints_client.responses_with_tools(key, model, WEATHER_PROMPT, [WEATHER_TOOL])
        require_successful_call(result)
        _assert_weather_function_call(result.body)


class TestVertexResponses:
    @pytest.mark.covers("llm.responses.vertex.basic.nonstream.works")
    def test_vertex_responses_returns_completion(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources, "e2e-responses-vertex", _vertex_params())

        result = endpoints_client.responses(key, model, BASIC_PROMPT)
        require_successful_call(result)
        parsed = ResponsesResult.model_validate_json(result.body)
        assert parsed.text.strip(), f"/responses over vertex returned no output text: {result.body[:300]}"

    @pytest.mark.covers("llm.responses.vertex.tool_use.nonstream.works")
    def test_vertex_responses_returns_function_call(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register(endpoints_client, resources, "e2e-responses-vertex-tool", _vertex_params())

        result = endpoints_client.responses_with_tools(key, model, WEATHER_PROMPT, [WEATHER_TOOL])
        require_successful_call(result)
        _assert_weather_function_call(result.body)
