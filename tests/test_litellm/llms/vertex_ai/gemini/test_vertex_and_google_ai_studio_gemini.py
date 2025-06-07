import asyncio
import json
from copy import deepcopy
from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import litellm
from litellm import ModelResponse, completion
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
    assert (
        VertexGeminiConfig.is_candidate_token_count_inclusive(usage_metadata)
        is inclusive
    )

    usage = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    assert usage.prompt_tokens == expected_usage.prompt_tokens
    assert usage.completion_tokens == expected_usage.completion_tokens
    assert usage.total_tokens == expected_usage.total_tokens


def test_streaming_chunk_includes_reasoning_tokens():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    litellm_logging = MagicMock()

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
    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=litellm_logging
    )
    streaming_chunk = iterator.chunk_parser(chunk)
    assert streaming_chunk.usage is not None
    assert streaming_chunk.usage.prompt_tokens == 5
    assert streaming_chunk.usage.completion_tokens == 7
    assert streaming_chunk.usage.total_tokens == 12
    assert streaming_chunk.usage.completion_tokens_details.reasoning_tokens == 3


def test_streaming_chunk_includes_reasoning_content():
    """
    Ensure that when Gemini returns a chunk with `thought=True`, the parser maps it to `reasoning_content`.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    litellm_logging = MagicMock()

    # Simulate a streaming chunk from Gemini which contains reasoning (thought) content
    chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "I'm thinking through the problem...",
                            "thought": True,
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {},
    }

    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=litellm_logging
    )
    streaming_chunk = iterator.chunk_parser(chunk)

    # The text content should be empty and reasoning_content should be populated
    assert streaming_chunk.choices[0].delta.content is None
    assert (
        streaming_chunk.choices[0].delta.reasoning_content
        == "I'm thinking through the problem..."
    )


def test_check_finish_reason():
    finish_reason_mappings = VertexGeminiConfig.get_finish_reason_mapping()
    for k, v in finish_reason_mappings.items():
        assert (
            VertexGeminiConfig._check_finish_reason(
                chat_completion_message=None, finish_reason=k
            )
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


def test_vertex_ai_map_thinking_param_with_budget_tokens_0():
    """
    If budget_tokens is 0, do not set includeThoughts to True
    """
    from litellm.types.llms.anthropic import AnthropicThinkingParam

    v = VertexGeminiConfig()
    thinking_param: AnthropicThinkingParam = {"type": "enabled", "budget_tokens": 0}
    assert "includeThoughts" not in v._map_thinking_param(thinking_param=thinking_param)

    thinking_param: AnthropicThinkingParam = {"type": "enabled", "budget_tokens": 100}
    assert v._map_thinking_param(thinking_param=thinking_param) == {
        "includeThoughts": True,
        "thinkingBudget": 100,
    }


def test_vertex_ai_map_tools():
    v = VertexGeminiConfig()
    tools = v._map_function(value=[{"code_execution": {}}])
    assert len(tools) == 1
    assert tools[0]["code_execution"] == {}
    print(tools)

    new_tools = v._map_function(value=[{"codeExecution": {}}])
    assert len(new_tools) == 1
    print("new_tools", new_tools)
    assert new_tools[0]["code_execution"] == {}
    print(new_tools)

    assert tools == new_tools


def test_vertex_ai_map_tool_with_anyof():
    """
    Related issue: https://github.com/BerriAI/litellm/issues/11164

    Ensure if anyof is present, only the anyof field and its contents are kept - otherwise VertexAI will throw an error - https://github.com/BerriAI/litellm/issues/11164
    """
    v = VertexGeminiConfig()
    value = [
        {
            "type": "function",
            "function": {
                "name": "git_create_branch",
                "description": "Creates a new branch from an optional base branch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {"title": "Repo Path", "type": "string"},
                        "branch_name": {"title": "Branch Name", "type": "string"},
                        "base_branch": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Base Branch",
                        },
                    },
                    "required": ["repo_path", "branch_name"],
                    "title": "GitCreateBranch",
                },
            },
        }
    ]
    tools = v._map_function(value=value)

    assert tools[0]["function_declarations"][0]["parameters"]["properties"][
        "base_branch"
    ] == {
        "anyOf": [{"type": "string", "nullable": True, "title": "Base Branch"}]
    }, f"Expected only anyOf field and its contents to be kept, but got {tools[0]['function_declarations'][0]['parameters']['properties']['base_branch']}"


def test_vertex_ai_streaming_usage_calculation():
    """
    Ensure streaming usage calculation uses same function as non-streaming usage calculation
    """
    from unittest.mock import patch

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "candidatesTokenCount": 10,
        "totalTokenCount": 67,
    }

    # Test streaming chunk parsing
    with patch.object(VertexGeminiConfig, "_calculate_usage") as mock_calculate_usage:
        # Create a streaming chunk
        chunk = {
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": usage_metadata,
        }

        # Create iterator and parse chunk
        iterator = ModelResponseIterator(
            streaming_response=[], sync_stream=True, logging_obj=MagicMock()
        )
        iterator.chunk_parser(chunk)

        # Verify _calculate_usage was called with correct parameters
        mock_calculate_usage.assert_called_once_with(completion_response=chunk)

    # Test non-streaming response parsing
    with patch.object(VertexGeminiConfig, "_calculate_usage") as mock_calculate_usage:
        # Create a completion response
        completion_response = {
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": usage_metadata,
        }

        # Parse completion response
        v.transform_response(
            model="gemini-pro",
            raw_response=MagicMock(json=lambda: completion_response),
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        # Verify _calculate_usage was called with correct parameters
        mock_calculate_usage.assert_called_once_with(
            completion_response=completion_response,
        )


def test_vertex_ai_streaming_usage_web_search_calculation():
    """
    Ensure streaming usage calculation uses same function as non-streaming usage calculation
    """
    from unittest.mock import patch

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "candidatesTokenCount": 10,
        "totalTokenCount": 67,
    }

    # Create a streaming chunk
    chunk = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}]},
                "groundingMetadata": [
                    {"webSearchQueries": ["What is the capital of France?"]}
                ],
            }
        ],
        "usageMetadata": usage_metadata,
    }

    # Create iterator and parse chunk
    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=MagicMock()
    )
    completed_response = iterator.chunk_parser(chunk)

    usage: Usage = completed_response.usage
    assert usage.prompt_tokens_details.web_search_requests is not None
    assert usage.prompt_tokens_details.web_search_requests == 1
