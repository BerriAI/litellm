import asyncio
from copy import deepcopy
from typing import List, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import litellm
from litellm import ModelResponse
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import UsageMetadata
from litellm.types.utils import ChoiceLogprobs, Usage


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
    non_default_params = {
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
    }
    original_non_default_params = deepcopy(non_default_params)
    transformed_request = v.map_openai_params(
        non_default_params=non_default_params,
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
    # should not mutate the original non_default_params
    assert non_default_params == original_non_default_params


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


@pytest.mark.parametrize(
    "usage_metadata, inclusive, expected_usage",
    [
        (
            UsageMetadata(
                promptTokenCount=10,
                candidatesTokenCount=10,
                totalTokenCount=20,
                thoughtsTokenCount=5,
            ),
            True,
            Usage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                reasoning_tokens=5,
            ),
        ),
        (
            UsageMetadata(
                promptTokenCount=10,
                candidatesTokenCount=5,
                totalTokenCount=20,
                thoughtsTokenCount=5,
            ),
            False,
            Usage(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                reasoning_tokens=5,
            ),
        ),
    ],
)
def test_vertex_ai_candidate_token_count_inclusive(
    usage_metadata, inclusive, expected_usage
):
    """
    Test that the candidate token count is inclusive of the thinking token count
    """
    v = VertexGeminiConfig()
    assert v.is_candidate_token_count_inclusive(usage_metadata) is inclusive

    usage = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    assert usage.prompt_tokens == expected_usage.prompt_tokens
    assert usage.completion_tokens == expected_usage.completion_tokens
    assert usage.total_tokens == expected_usage.total_tokens


def test_streaming_chunk_includes_reasoning_tokens():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    # Simulate a streaming chunk as would be received from Gemini
    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 7,
            "totalTokenCount": 12,
            "thoughtsTokenCount": 3,
        },
    }
    iterator = ModelResponseIterator(streaming_response=[], sync_stream=True)
    streaming_chunk = iterator.chunk_parser(chunk)
    assert streaming_chunk["usage"] is not None
    assert streaming_chunk["usage"]["prompt_tokens"] == 5
    assert streaming_chunk["usage"]["completion_tokens"] == 7
    assert streaming_chunk["usage"]["total_tokens"] == 12
    assert (
        streaming_chunk["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3
    )


def test_check_finish_reason():
    config = VertexGeminiConfig()
    finish_reason_mappings = config.get_finish_reason_mapping()
    for k, v in finish_reason_mappings.items():
        assert (
            config._check_finish_reason(chat_completion_message=None, finish_reason=k)
            == v
        )


def test_vertex_ai_usage_metadata_response_token_count():
    """For Gemini Live API"""
    from litellm.types.utils import PromptTokensDetailsWrapper

    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "responseTokenCount": 74,
        "totalTokenCount": 131,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 57}],
        "responseTokensDetails": [{"modality": "TEXT", "tokenCount": 74}],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    print("result", result)
    assert result.prompt_tokens == 57
    assert result.completion_tokens == 74
    assert result.total_tokens == 131
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.audio_tokens is None
    assert result.prompt_tokens_details.cached_tokens is None
    assert result.completion_tokens_details.text_tokens == 74
