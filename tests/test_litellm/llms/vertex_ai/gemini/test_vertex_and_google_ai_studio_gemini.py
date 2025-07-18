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


def test_vertex_ai_transform_parts():
    """
    Test the _transform_parts method for converting Vertex AI function calls
    to OpenAI-compatible tool calls and function calls.

    Tests both:
    1. Multiple tool calls within a single message
    2. Multiple tool calls across different messages (cumulative indexing)
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    # Test case 1: Function call mode (is_function_call=True)
    parts_with_function = [
        HttpxPartType(
            functionCall={
                "name": "get_current_weather",
                "args": {"location": "Boston", "unit": "celsius"},
            }
        ),
        HttpxPartType(text="Some text content"),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=parts_with_function, cumulative_tool_call_idx=0, is_function_call=True
    )

    # Should return function, no tools, and incremented index
    assert function is not None
    assert function["name"] == "get_current_weather"
    assert function["arguments"] == '{"location": "Boston", "unit": "celsius"}'
    assert tools is None
    assert updated_idx == 1  # Should be incremented from 0 to 1

    # Test case 2: Tool call mode (is_function_call=False) - Single message with multiple tool calls
    parts_with_multiple_functions = [
        HttpxPartType(
            functionCall={"name": "get_current_weather", "args": {"location": "Boston"}}
        ),
        HttpxPartType(
            functionCall={
                "name": "get_forecast",
                "args": {"location": "New York", "days": 3},
            }
        ),
        HttpxPartType(text="Some text content"),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=parts_with_multiple_functions,
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    # Should return multiple tools with correct indices
    assert function is None
    assert tools is not None
    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "get_current_weather"
    assert tools[0]["index"] == 0  # First tool call should have index 0
    assert tools[1]["function"]["name"] == "get_forecast"
    assert tools[1]["index"] == 1  # Second tool call should have index 1
    assert tools[1]["function"]["arguments"] == '{"location": "New York", "days": 3}'
    assert updated_idx == 2  # Should be incremented from 0 to 2 (two function calls)

    # Test case 3: Simulating multiple messages - cumulative indexing across messages
    # First message with 2 tool calls (starting from index 0)
    first_message_parts = [
        HttpxPartType(
            functionCall={"name": "get_weather", "args": {"location": "Boston"}}
        ),
        HttpxPartType(functionCall={"name": "get_time", "args": {"timezone": "EST"}}),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=first_message_parts, cumulative_tool_call_idx=0, is_function_call=False
    )

    assert function is None
    assert tools is not None
    assert len(tools) == 2
    assert tools[0]["index"] == 0
    assert tools[1]["index"] == 1
    assert updated_idx == 2

    # Second message with 1 tool call (continuing from previous index)
    second_message_parts = [
        HttpxPartType(
            functionCall={"name": "send_email", "args": {"to": "user@example.com"}}
        ),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=second_message_parts,
        cumulative_tool_call_idx=updated_idx,
        is_function_call=False,
    )

    assert function is None
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["index"] == 2  # Should continue from previous cumulative index
    assert tools[0]["function"]["name"] == "send_email"
    assert updated_idx == 3  # Should be incremented to 3

    # Third message with 2 more tool calls (continuing from previous index)
    third_message_parts = [
        HttpxPartType(
            functionCall={"name": "create_calendar_event", "args": {"title": "Meeting"}}
        ),
        HttpxPartType(functionCall={"name": "set_reminder", "args": {"time": "10:00"}}),
    ]

    function, tools, final_idx = VertexGeminiConfig._transform_parts(
        parts=third_message_parts,
        cumulative_tool_call_idx=updated_idx,
        is_function_call=False,
    )

    assert function is None
    assert tools is not None
    assert len(tools) == 2
    assert tools[0]["index"] == 3  # Should continue from previous cumulative index
    assert tools[1]["index"] == 4  # Should be incremented
    assert tools[0]["function"]["name"] == "create_calendar_event"
    assert tools[1]["function"]["name"] == "set_reminder"
    assert final_idx == 5  # Should be incremented to 5

    # Test case 4: No function calls
    parts_without_functions = [
        HttpxPartType(text="Just some text content"),
        HttpxPartType(text="More text"),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=parts_without_functions,
        cumulative_tool_call_idx=5,
        is_function_call=False,
    )

    # Should return nothing and preserve the cumulative index
    assert function is None
    assert tools is None
    assert updated_idx == 5  # Index should remain unchanged when no function calls

    # Test case 5: Empty parts list
    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=[], cumulative_tool_call_idx=10, is_function_call=False
    )

    # Should return nothing and preserve the cumulative index
    assert function is None
    assert tools is None
    assert updated_idx == 10  # Index should remain unchanged

    # Test case 6: Function call with empty args
    parts_with_empty_args = [
        HttpxPartType(functionCall={"name": "simple_function", "args": {}})
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=parts_with_empty_args, cumulative_tool_call_idx=0, is_function_call=True
    )

    # Should handle empty args correctly
    assert function is not None
    assert function["name"] == "simple_function"
    assert function["arguments"] == "{}"
    assert tools is None
    assert updated_idx == 1

    # Test case 7: Mixed content with function calls - ensuring tool call IDs are unique
    mixed_parts = [
        HttpxPartType(text="Before function call"),
        HttpxPartType(
            functionCall={"name": "function_a", "args": {"param": "value_a"}}
        ),
        HttpxPartType(text="Between function calls"),
        HttpxPartType(
            functionCall={"name": "function_b", "args": {"param": "value_b"}}
        ),
        HttpxPartType(text="After function calls"),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=mixed_parts, cumulative_tool_call_idx=100, is_function_call=False
    )

    assert function is None
    assert tools is not None
    assert len(tools) == 2
    assert tools[0]["index"] == 100
    assert tools[1]["index"] == 101
    # Verify that tool call IDs are unique
    assert tools[0]["id"] != tools[1]["id"]
    assert tools[0]["id"].startswith("call_")
    assert tools[1]["id"].startswith("call_")
    assert updated_idx == 102


def test_vertex_ai_usage_metadata_missing_token_count():
    """Test that missing tokenCount in responseTokensDetails defaults to 0"""
    from litellm.types.utils import PromptTokensDetailsWrapper

    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 57,
        "responseTokenCount": 74,
        "totalTokenCount": 131,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 57}],
        "responseTokensDetails": [
            {"modality": "TEXT"},  # Missing tokenCount
            {"modality": "AUDIO"},  # Missing tokenCount
        ],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})

    # Should not crash and should default missing tokenCount to 0
    assert result.prompt_tokens == 57
    assert result.completion_tokens == 74
    assert result.total_tokens == 131
    assert (
        result.completion_tokens_details.text_tokens == 0
    )  # Default value for missing tokenCount
    assert (
        result.completion_tokens_details.audio_tokens == 0
    )  # Default value for missing tokenCount


def test_vertex_ai_process_candidates_with_grounding_metadata():
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    result = v._process_candidates(
        _candidates=[
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "text": " rain during the day and a 20% chance at night. The temperature will be between 56°F (13°C) and 62°F (17°C)."
                        }
                    ],
                },
                "finishReason": "STOP",
                "groundingMetadata": {
                    "webSearchQueries": ["weather San Francisco today"],
                    "searchEntryPoint": {
                        "renderedContent": '<style>\n.container {\n  align-items: center;\n  border-radius: 8px;\n  display: flex;\n  font-family: Google Sans, Roboto, sans-serif;\n  font-size: 14px;\n  line-height: 20px;\n  padding: 8px 12px;\n}\n.chip {\n  display: inline-block;\n  border: solid 1px;\n  border-radius: 16px;\n  min-width: 14px;\n  padding: 5px 16px;\n  text-align: center;\n  user-select: none;\n  margin: 0 8px;\n  -webkit-tap-highlight-color: transparent;\n}\n.carousel {\n  overflow: auto;\n  scrollbar-width: none;\n  white-space: nowrap;\n  margin-right: -12px;\n}\n.headline {\n  display: flex;\n  margin-right: 4px;\n}\n.gradient-container {\n  position: relative;\n}\n.gradient {\n  position: absolute;\n  transform: translate(3px, -9px);\n  height: 36px;\n  width: 9px;\n}\n@media (prefers-color-scheme: light) {\n  .container {\n    background-color: #fafafa;\n    box-shadow: 0 0 0 1px #0000000f;\n  }\n  .headline-label {\n    color: #1f1f1f;\n  }\n  .chip {\n    background-color: #ffffff;\n    border-color: #d2d2d2;\n    color: #5e5e5e;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #f2f2f2;\n  }\n  .chip:focus {\n    background-color: #f2f2f2;\n  }\n  .chip:active {\n    background-color: #d8d8d8;\n    border-color: #b6b6b6;\n  }\n  .logo-dark {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);\n  }\n}\n@media (prefers-color-scheme: dark) {\n  .container {\n    background-color: #1f1f1f;\n    box-shadow: 0 0 0 1px #ffffff26;\n  }\n  .headline-label {\n    color: #fff;\n  }\n  .chip {\n    background-color: #2c2c2c;\n    border-color: #3c4043;\n    color: #fff;\n    text-decoration: none;\n  }\n  .chip:hover {\n    background-color: #353536;\n  }\n  .chip:focus {\n    background-color: #353536;\n  }\n  .chip:active {\n    background-color: #464849;\n    border-color: #53575b;\n  }\n  .logo-light {\n    display: none;\n  }\n  .gradient {\n    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);\n  }\n}\n</style>\n<div class="container">\n  <div class="headline">\n    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">\n      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>\n      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>\n      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>\n      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>\n    </svg>\n    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">\n      <circle cx="24" cy="23" fill="#FFF" r="22"/>\n      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>\n      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>\n      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>\n      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>\n    </svg>\n    <div class="gradient-container"><div class="gradient"></div></div>\n  </div>\n  <div class="carousel">\n    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGP8AFqMh5rK5GIcrn7IPR_roruBYkwl9apbI0dvDTV_9j_phIsditU7pvQPOy8mLpi6OEE_HuEGRG7LCpVT1-X4fCFmuyQwy_iyVtgZ5KgAZOW4SA1bjxDefc91H9Y8_1ehEXg02lXvwHk1CfcrrJETI2ErIqn-WGqsYpwysNLPKj_KWPzetU14WOAu5vnVIeaqwRVRTHESUahHtW1NA==">weather San Francisco today</a>\n  </div>\n</div>\n'
                    },
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://www.google.com/search?q=weather+in+San Francisco,+CA",
                                "title": "Weather information for locality: San Francisco, administrative_area: CA",
                                "domain": "google.com",
                            }
                        }
                    ],
                    "groundingSupports": [
                        {
                            "segment": {
                                "endIndex": 90,
                                "text": "The weather in San Francisco, California today, Wednesday, July 16, 2025, is mostly cloudy",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.6749268],
                        },
                        {
                            "segment": {
                                "startIndex": 92,
                                "endIndex": 157,
                                "text": "The temperature is 61°F (16°C), but it feels like 58°F (15°C)",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.96708393],
                        },
                        {
                            "segment": {
                                "startIndex": 159,
                                "endIndex": 219,
                                "text": "The humidity is around 77%, and there is a 0% chance of rain",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.8192763],
                        },
                        {
                            "segment": {
                                "startIndex": 221,
                                "endIndex": 360,
                                "text": "The forecast for today is cloudy during the day and light rain at night, with a 10% chance of rain during the day and a 20% chance at night",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.875334],
                        },
                        {
                            "segment": {
                                "startIndex": 362,
                                "endIndex": 425,
                                "text": "The temperature will be between 56°F (13°C) and 62°F (17°C)",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.8203865],
                        },
                    ],
                    "retrievalMetadata": {},
                },
            }
        ],
        model_response=ModelResponse(),
        standard_optional_params={},
    )

    print(result)
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
