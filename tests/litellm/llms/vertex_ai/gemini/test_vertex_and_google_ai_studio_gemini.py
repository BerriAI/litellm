import asyncio
from typing import List, cast
import json
from unittest.mock import MagicMock

import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm import ModelResponse
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.gemini.transformation import async_transform_request_body, sync_transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
    VertexLLM,
)
from litellm.types.utils import ChoiceLogprobs


def test_top_logprobs():
    non_default_params = {
        "top_logprobs": 2,
        "logprobs": True,
    }
    optional_params = {}
    model = "gemini"

    v = VertexGeminiConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert v["responseLogprobs"] is non_default_params["logprobs"]
    assert v["logprobs"] is non_default_params["top_logprobs"]


def test_get_model_for_vertex_ai_url():
    # Test case 1: Regular model name
    model = "gemini-pro"
    result = VertexGeminiConfig.get_model_for_vertex_ai_url(model)
    assert result == "gemini-pro"

    # Test case 2: Gemini spec model with UUID
    model = "gemini/ft-uuid-123"
    result = VertexGeminiConfig.get_model_for_vertex_ai_url(model)
    assert result == "ft-uuid-123"


def test_is_model_gemini_spec_model():
    # Test case 1: None input
    assert VertexGeminiConfig._is_model_gemini_spec_model(None) == False

    # Test case 2: Regular model name
    assert VertexGeminiConfig._is_model_gemini_spec_model("gemini-pro") == False

    # Test case 3: Gemini spec model
    assert VertexGeminiConfig._is_model_gemini_spec_model("gemini/custom-model") == True


def test_get_model_name_from_gemini_spec_model():
    # Test case 1: Regular model name
    model = "gemini-pro"
    result = VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
    assert result == "gemini-pro"

    # Test case 2: Gemini spec model
    model = "gemini/ft-uuid-123"
    result = VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
    assert result == "ft-uuid-123"


def test_vertex_ai_response_schema_dict():
    v = VertexGeminiConfig()
    transformed_request = v.map_openai_params(
        non_default_params={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "thought": {"type": "string"},
                                        "output": {"type": "string"},
                                    },
                                    "required": ["thought", "output"],
                                    "additionalProperties": False,
                                },
                            },
                            "final_answer": {"type": "string"},
                        },
                        "required": ["steps", "final_answer"],
                        "additionalProperties": False,
                    },
                    "strict": False,
                },
            },
        },
        optional_params={},
        model="gemini-2.0-flash-lite",
        drop_params=False,
    )

    schema = transformed_request["response_schema"]
    # should add propertyOrdering
    assert schema["propertyOrdering"] == ["steps", "final_answer"]
    # should add propertyOrdering (recursively, including array items)
    assert schema["properties"]["steps"]["items"]["propertyOrdering"] == [
        "thought",
        "output",
    ]
    # should strip strict and additionalProperties
    assert "strict" not in schema
    assert "additionalProperties" not in schema
    # validate the whole thing to catch regressions
    assert transformed_request["response_schema"] == {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "output": {"type": "string"},
                    },
                    "required": ["thought", "output"],
                    "propertyOrdering": ["thought", "output"],
                },
            },
            "final_answer": {"type": "string"},
        },
        "required": ["steps", "final_answer"],
        "propertyOrdering": ["steps", "final_answer"],
    }


class MathReasoning(BaseModel):
    steps: List["Step"]
    final_answer: str


class Step(BaseModel):
    thought: str
    output: str


def test_vertex_ai_response_schema_defs():
    v = VertexGeminiConfig()

    schema = cast(dict, v.get_json_schema_from_pydantic_object(MathReasoning))

    # pydantic conversion by default adds $defs to the schema, make sure this is still the case, otherwise this test isn't really testing anything
    assert "$defs" in schema["json_schema"]["schema"]

    transformed_request = v.map_openai_params(
        non_default_params={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "response_format": schema,
        },
        optional_params={},
        model="gemini-2.0-flash-lite",
        drop_params=False,
    )

    assert "$defs" not in transformed_request["response_schema"]
    assert transformed_request["response_schema"] == {
        "title": "MathReasoning",
        "type": "object",
        "properties": {
            "steps": {
                "title": "Steps",
                "type": "array",
                "items": {
                    "title": "Step",
                    "type": "object",
                    "properties": {
                        "thought": {"title": "Thought", "type": "string"},
                        "output": {"title": "Output", "type": "string"},
                    },
                    "required": ["thought", "output"],
                    "propertyOrdering": ["thought", "output"],
                },
            },
            "final_answer": {"title": "Final Answer", "type": "string"},
        },
        "required": ["steps", "final_answer"],
        "propertyOrdering": ["steps", "final_answer"],
    }


def test_vertex_ai_retain_property_ordering():
    v = VertexGeminiConfig()
    transformed_request = v.map_openai_params(
        non_default_params={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "math_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "output": {"type": "string"},
                            "thought": {"type": "string"},
                        },
                        "propertyOrdering": ["thought", "output"],
                    },
                },
            },
        },
        optional_params={},
        model="gemini-2.0-flash-lite",
        drop_params=False,
    )

    schema = transformed_request["response_schema"]
    # should leave existing value alone, despite dictionary ordering
    assert schema["propertyOrdering"] == ["thought", "output"]


def test_vertex_ai_thinking_output_part():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    v = VertexGeminiConfig()
    parts = [
        HttpxPartType(
            thought=True,
            text="I'm thinking...",
        ),
        HttpxPartType(text="Hello world"),
    ]
    content, reasoning_content = v.get_assistant_content_message(parts=parts)
    assert content == "Hello world"
    assert reasoning_content == "I'm thinking..."


def _mock_logging():
    mock_logging = MagicMock(spec=LiteLLMLoggingObj)
    mock_logging.pre_call = MagicMock(return_value=None)
    mock_logging.post_call = MagicMock(return_value=None)
    return mock_logging

def _mock_get_post_cached_content(api_key: str, respx_mock: respx.MockRouter) -> tuple[respx.MockRouter, respx.MockRouter]:
    get_mock = respx_mock.get(
        f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={api_key}"
    ).respond(
        json={
            "cachedContents": [],
            "nextPageToken": None,
        }
    )

    post_mock = respx_mock.post(
        f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={api_key}"
    ).respond(
        json={
            "name": "projects/fake_project/locations/fake_location/cachedContents/fake_cache_id",
            "model": "gemini-2.0-flash-001",
        }
    )
    return get_mock, post_mock

def test_google_ai_studio_gemini_message_caching_sync(
    # ideally this would unit test just a small transformation, but there's a lot going on with gemini/vertex
    # (hinges around branching for sync/async transformations).
    respx_mock: respx.MockRouter,
):
    mock_logging = _mock_logging()

    get_mock, post_mock = _mock_get_post_cached_content("fake_api_key", respx_mock)

    transformed_request = sync_transform_request_body(
        gemini_api_key="fake_api_key",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "you are a helpful assistant",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": "Hello, world!",
            },
        ],
        api_base=None,
        model="gemini-2.0-flash-001",
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params={},
        logging_obj=mock_logging,
        custom_llm_provider="vertex_ai",
        litellm_params={},
    )
    # Assert both GET and POST endpoints were called
    assert get_mock.calls.call_count == 1
    assert post_mock.calls.call_count == 1
    assert json.loads(post_mock.calls[0].request.content) == {
        "contents": [],
        "model": "models/gemini-2.0-flash-001",
        "displayName": "203ae753b6c793e1af13b13d0710de5863c486e610963ce243b07ee6830ce1d2",
        "tools": None,
        "toolConfig": None,
        "system_instruction": {"parts": [{"text": "you are a helpful assistant"}]},
    }

    assert transformed_request["contents"] == [
        {"parts": [{"text": "Hello, world!"}], "role": "user"}
    ]
    assert (
        transformed_request["cachedContent"]
        == "projects/fake_project/locations/fake_location/cachedContents/fake_cache_id"
    )

_GET_WEATHER_MESSAGES = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "you are a helpful assistant",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
    {
        "role": "user",
        "content": "What is the weather now?",
    },
]

_GET_WEATHER_TOOLS_OPTIONAL_PARAMS = {
    "tools": [
        {
            "functionDeclarations": [
                {"name": "get_weather", "description": "Get the current weather"}
            ],
        }
    ],
    "tool_choice": {
        "functionCallingConfig": {
            "mode": "ANY"
        }
    },
}

_EXPECTED_GET_WEATHER_CACHED_CONTENT_REQUEST_BODY = {
    "contents": [],
    "model": "models/gemini-2.0-flash-001",
    "displayName": "62398619ff33908a18561c1a342c580c3d876f169d103ec52128df38f04e03d1",
    "tools": [
        {
            "functionDeclarations": [
                {"name": "get_weather", "description": "Get the current weather"}
            ],
        }
    ],
    "toolConfig": {
        "functionCallingConfig": {
            "mode": "ANY"
        }
    },        
    "system_instruction": {"parts": [{"text": "you are a helpful assistant"}]},
}

def test_google_ai_studio_gemini_message_caching_with_tools_sync(
    respx_mock: respx.MockRouter,
):
    mock_logging = _mock_logging()

    get_mock, post_mock = _mock_get_post_cached_content("fake_api_key", respx_mock)

    transformed_request = sync_transform_request_body(
        gemini_api_key="fake_api_key",
        messages=_GET_WEATHER_MESSAGES,
        api_base=None,
        model="gemini-2.0-flash-001",
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params=_GET_WEATHER_TOOLS_OPTIONAL_PARAMS,
        logging_obj=mock_logging,
        custom_llm_provider="vertex_ai",
        litellm_params={},
    )
    # Assert both GET and POST endpoints were called
    assert get_mock.calls.call_count == 1
    assert post_mock.calls.call_count == 1
    assert json.loads(post_mock.calls[0].request.content) == _EXPECTED_GET_WEATHER_CACHED_CONTENT_REQUEST_BODY

    assert transformed_request["contents"] == [
        {"parts": [{"text": "What is the weather now?"}], "role": "user"}
    ]
    assert (
        transformed_request["cachedContent"]
        == "projects/fake_project/locations/fake_location/cachedContents/fake_cache_id"
    )
    assert transformed_request.get("tools") is None
    assert transformed_request.get("tool_choice") is None


@pytest.mark.asyncio
async def test_google_ai_studio_gemini_message_caching_with_tools_async(
    respx_mock: respx.MockRouter,
):
    mock_logging = _mock_logging()

    get_mock, post_mock = _mock_get_post_cached_content("fake_api_key", respx_mock)

    transformed_request = await async_transform_request_body(
        gemini_api_key="fake_api_key",
        messages=_GET_WEATHER_MESSAGES,
        api_base=None,
        model="gemini-2.0-flash-001",
        client=None,
        timeout=None,
        extra_headers=None,
        optional_params=_GET_WEATHER_TOOLS_OPTIONAL_PARAMS,
        logging_obj=mock_logging,
        custom_llm_provider="vertex_ai",
        litellm_params={},
    )
    # Assert both GET and POST endpoints were called
    assert get_mock.calls.call_count == 1
    assert post_mock.calls.call_count == 1
    assert json.loads(post_mock.calls[0].request.content) == _EXPECTED_GET_WEATHER_CACHED_CONTENT_REQUEST_BODY

    assert transformed_request["contents"] == [
        {"parts": [{"text": "What is the weather now?"}], "role": "user"}
    ]
    assert (
        transformed_request["cachedContent"]
        == "projects/fake_project/locations/fake_location/cachedContents/fake_cache_id"
    )
    assert transformed_request.get("tools") is None
    assert transformed_request.get("tool_choice") is None


def test_vertex_ai_empty_content():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    v = VertexGeminiConfig()
    parts = [
        HttpxPartType(
            functionCall={
                "name": "get_current_weather",
                "arguments": "{}",
            },
        ),
    ]
    content, reasoning_content = v.get_assistant_content_message(parts=parts)
    assert content is None
    assert reasoning_content is None
