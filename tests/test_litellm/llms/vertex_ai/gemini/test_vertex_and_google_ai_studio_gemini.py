import asyncio
import json
import re
from copy import deepcopy
from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import litellm
from litellm import ModelResponse, completion
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import UsageMetadata
from litellm.types.utils import ChoiceLogprobs, Usage
from litellm.utils import CustomStreamWrapper


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
    """
    Test that older Gemini models (1.5) use responseSchema (OpenAPI format).
    responseSchema requires propertyOrdering and doesn't support additionalProperties.
    """
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
        model="gemini-1.5-flash",  # Old model uses responseSchema (OpenAPI format)
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
    """
    Test that $defs are unpacked for older Gemini models using responseSchema.
    """
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
        model="gemini-1.5-flash",  # Old model uses responseSchema (OpenAPI format)
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


def test_vertex_ai_response_json_schema_for_gemini_2():
    """
    Test that Gemini 2.0+ models automatically use responseJsonSchema.

    responseJsonSchema uses standard JSON Schema format:
    - lowercase types (string, object, etc.)
    - no propertyOrdering required
    - supports additionalProperties
    """
    v = VertexGeminiConfig()

    transformed_request = v.map_openai_params(
        non_default_params={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        optional_params={},
        model="gemini-2.0-flash",  # Gemini 2.0+ automatically uses responseJsonSchema
        drop_params=False,
    )

    # Should use response_json_schema, not response_schema
    assert "response_json_schema" in transformed_request
    assert "response_schema" not in transformed_request

    # Types should be lowercase (standard JSON Schema format)
    assert transformed_request["response_json_schema"]["type"] == "object"
    assert transformed_request["response_json_schema"]["properties"]["name"]["type"] == "string"
    assert transformed_request["response_json_schema"]["properties"]["age"]["type"] == "integer"

    # Should NOT have propertyOrdering (not needed for responseJsonSchema)
    assert "propertyOrdering" not in transformed_request["response_json_schema"]

    # additionalProperties should be preserved (supported by responseJsonSchema)
    assert transformed_request["response_json_schema"].get("additionalProperties") == False


def test_vertex_ai_response_schema_for_old_models():
    """
    Test that older models (Gemini 1.5) automatically use responseSchema.
    """
    v = VertexGeminiConfig()

    transformed_request = v.map_openai_params(
        non_default_params={
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        },
        optional_params={},
        model="gemini-1.5-flash",  # Old model automatically uses responseSchema
        drop_params=False,
    )

    # Should use response_schema for older models
    assert "response_schema" in transformed_request
    assert "response_json_schema" not in transformed_request


def test_vertex_ai_retain_property_ordering():
    """
    Test that existing propertyOrdering is preserved for older models using responseSchema.
    """
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
        model="gemini-1.5-flash",  # Old model uses responseSchema which needs propertyOrdering
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


def test_streaming_chunk_with_tool_calls_and_thought_includes_reasoning_content():
    """
    Test that when Gemini returns a streaming chunk with both thought: true parts
    AND tool calls, the reasoning_content is correctly extracted from the thought parts.

    Per Google's docs: thought: true indicates reasoning content, NOT thoughtSignature.
    thoughtSignature is just a token for multi-turn context preservation.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    litellm_logging = MagicMock()

    chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "Let me think about how to get the time...",
                            "thought": True,  # This indicates reasoning content
                        },
                        {
                            "functionCall": {
                                "name": "get_current_time",
                                "args": {"timezone": "America/New_York"},
                            },
                            "thoughtSignature": "EsEDCr4DAdHtim...",  # Just a token, not reasoning
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 68,
            "candidatesTokenCount": 120,
            "totalTokenCount": 188,
        },
    }

    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=litellm_logging
    )
    streaming_chunk = iterator.chunk_parser(chunk)

    # Verify reasoning_content comes from the thought: true part
    assert streaming_chunk.choices[0].delta.reasoning_content == "Let me think about how to get the time..."

    # Verify tool calls are also present
    assert streaming_chunk.choices[0].delta.tool_calls is not None
    assert len(streaming_chunk.choices[0].delta.tool_calls) == 1
    assert streaming_chunk.choices[0].delta.tool_calls[0].function.name == "get_current_time"


def test_streaming_chunk_with_tool_calls_no_thought_no_reasoning_content():
    """
    Test that when Gemini returns tool calls with thoughtSignature but WITHOUT
    thought: true, there is NO reasoning_content.

    This is a regression test for the bug where functionCall data was incorrectly
    being placed into reasoning_content when thoughtSignature was present.
    Per Google's docs: thoughtSignature is just a token for multi-turn, not reasoning.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    litellm_logging = MagicMock()

    chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_current_time",
                                "args": {"timezone": "America/New_York"},
                            },
                            "thoughtSignature": "EsEDCr4DAdHtim...",  # Just a token, NOT thought: true
                        }
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 68,
            "candidatesTokenCount": 120,
            "totalTokenCount": 188,
        },
    }

    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=litellm_logging
    )
    streaming_chunk = iterator.chunk_parser(chunk)

    # reasoning_content should be None - thoughtSignature alone does NOT mean reasoning
    assert getattr(streaming_chunk.choices[0].delta, 'reasoning_content', None) is None

    # Tool calls should still work
    assert streaming_chunk.choices[0].delta.tool_calls is not None
    assert len(streaming_chunk.choices[0].delta.tool_calls) == 1
    assert streaming_chunk.choices[0].delta.tool_calls[0].function.name == "get_current_time"


def test_check_finish_reason():
    finish_reason_mappings = VertexGeminiConfig.get_finish_reason_mapping()
    for k, v in finish_reason_mappings.items():
        assert (
            VertexGeminiConfig._check_finish_reason(
                chat_completion_message=None, finish_reason=k
            )
            == v
        )


def test_finish_reason_unspecified_and_malformed_function_call():
    """
    Test that FINISH_REASON_UNSPECIFIED and MALFORMED_FUNCTION_CALL 
    return their lowercase values instead of being mapped to 'stop'
    since we don't have good mappings for these.
    """
    finish_reason_mappings = VertexGeminiConfig.get_finish_reason_mapping()
    
    # Test FINISH_REASON_UNSPECIFIED returns lowercase version
    assert finish_reason_mappings["FINISH_REASON_UNSPECIFIED"] == "finish_reason_unspecified"
    assert (
        VertexGeminiConfig._check_finish_reason(
            chat_completion_message=None, finish_reason="FINISH_REASON_UNSPECIFIED"
        )
        == "finish_reason_unspecified"
    )
    
    # Test MALFORMED_FUNCTION_CALL returns lowercase version
    assert finish_reason_mappings["MALFORMED_FUNCTION_CALL"] == "malformed_function_call"
    assert (
        VertexGeminiConfig._check_finish_reason(
            chat_completion_message=None, finish_reason="MALFORMED_FUNCTION_CALL"
        )
        == "malformed_function_call"
    )
    
    # Ensure these values are in the OpenAI finish reasons constant
    from litellm import OPENAI_FINISH_REASONS
    assert "finish_reason_unspecified" in OPENAI_FINISH_REASONS
    assert "malformed_function_call" in OPENAI_FINISH_REASONS


def test_vertex_ai_usage_metadata_response_token_count():
    """For Gemini Live API"""
    from litellm.types.utils import PromptTokensDetailsWrapper

    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 66,
        "responseTokenCount": 74,
        "totalTokenCount": 131,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 57}, {"modality": "IMAGE", "tokenCount": 9}],
        "responseTokensDetails": [{"modality": "TEXT", "tokenCount": 74}],
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    print("result", result)
    assert result.prompt_tokens == 66
    assert result.completion_tokens == 74
    assert result.total_tokens == 131
    assert result.prompt_tokens_details.text_tokens == 57
    assert result.prompt_tokens_details.image_tokens == 9
    assert result.prompt_tokens_details.audio_tokens is None
    assert result.prompt_tokens_details.cached_tokens is None
    assert result.completion_tokens_details.text_tokens == 74


def test_vertex_ai_usage_metadata_with_image_tokens():
    """Test candidatesTokensDetails with IMAGE modality (e.g., Imagen models)

    This test simulates the case where candidatesTokenCount is EXCLUSIVE of thoughtsTokenCount.
    Gemini API returns: totalTokenCount = promptTokenCount + candidatesTokenCount + thoughtsTokenCount
    """
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 14,
        "candidatesTokenCount": 1442,  # Does NOT include thoughtsTokenCount
        "totalTokenCount": 1614,  # 14 + 1442 + 158
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 14}],
        "candidatesTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 1120},
            {"modality": "TEXT", "tokenCount": 322}  # 1442 - 1120 = 322
        ],
        "thoughtsTokenCount": 158
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    print("result", result)

    # Verify basic token counts
    assert result.prompt_tokens == 14
    # completion_tokens = candidatesTokenCount + thoughtsTokenCount (when exclusive)
    assert result.completion_tokens == 1600  # 1442 + 158
    assert result.total_tokens == 1614

    # Verify detailed token breakdown
    assert result.completion_tokens_details.image_tokens == 1120
    assert result.completion_tokens_details.text_tokens == 322
    assert result.completion_tokens_details.reasoning_tokens == 158

    # Verify the math: completion_tokens = image + text + reasoning
    # 1600 = 1120 (image) + 322 (text) + 158 (reasoning)
    assert (
        result.completion_tokens_details.image_tokens
        + result.completion_tokens_details.text_tokens
        + result.completion_tokens_details.reasoning_tokens
        == result.completion_tokens
    )


def test_vertex_ai_usage_metadata_with_image_tokens_auto_calculated_text():
    """Test that text_tokens is auto-calculated when only IMAGE modality is provided

    This test verifies the auto-calculation logic at line 1367-1372 in vertex_and_google_ai_studio_gemini.py
    """
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 14,
        "candidatesTokenCount": 1442,
        "totalTokenCount": 1614,  # 14 + 1442 + 158 (exclusive)
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 14}],
        "candidatesTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 1120}
            # TEXT modality omitted - should be auto-calculated
        ],
        "thoughtsTokenCount": 158
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    print("result", result)

    # Verify basic token counts
    assert result.prompt_tokens == 14
    assert result.completion_tokens == 1600  # 1442 + 158
    assert result.total_tokens == 1614

    # Verify image_tokens is set
    assert result.completion_tokens_details.image_tokens == 1120

    # Verify text_tokens is auto-calculated: candidatesTokenCount - image_tokens
    # Note: reasoning_tokens is NOT subtracted here because candidatesTokenCount is exclusive
    # 1442 - 1120 = 322
    expected_text_tokens = 1442 - 1120
    assert result.completion_tokens_details.text_tokens == expected_text_tokens
    assert result.completion_tokens_details.reasoning_tokens == 158


def test_vertex_ai_usage_metadata_with_image_tokens_in_prompt():
    """Test promptTokensDetails with IMAGE modality for multimodal inputs
    
    This test verifies the fix for issue #18182 where image_tokens were missing
    from prompt_tokens_details when calling Gemini models with image inputs.
    
    Example scenario: User sends a text prompt + image, and Gemini generates an image response.
    The promptTokensDetails should include both TEXT and IMAGE token counts.
    
    In this test case, candidatesTokenCount is INCLUSIVE of thoughtsTokenCount because:
    promptTokenCount (533) + candidatesTokenCount (1337) = totalTokenCount (1870)
    """
    v = VertexGeminiConfig()
    usage_metadata = {
        "promptTokenCount": 533,
        "candidatesTokenCount": 1337,  # INCLUSIVE of thoughtsTokenCount
        "totalTokenCount": 1870,
        "promptTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 527},
            {"modality": "TEXT", "tokenCount": 6}
        ],
        "candidatesTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 1120}
        ],
        "thoughtsTokenCount": 217
    }
    usage_metadata = UsageMetadata(**usage_metadata)
    result = v._calculate_usage(completion_response={"usageMetadata": usage_metadata})
    print("result", result)
    
    # Verify basic token counts
    assert result.prompt_tokens == 533
    # candidatesTokenCount is INCLUSIVE, so completion_tokens = candidatesTokenCount
    assert result.completion_tokens == 1337
    assert result.total_tokens == 1870
    
    # Verify prompt_tokens_details includes both text and image tokens
    assert result.prompt_tokens_details.text_tokens == 6
    assert result.prompt_tokens_details.image_tokens == 527
    
    # Verify completion_tokens_details
    assert result.completion_tokens_details.image_tokens == 1120
    assert result.completion_tokens_details.reasoning_tokens == 217
    
    # Verify the math: prompt_tokens = text + image
    # 533 = 6 (text) + 527 (image)
    assert (
        result.prompt_tokens_details.text_tokens
        + result.prompt_tokens_details.image_tokens
        == result.prompt_tokens
    )


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
    optional_params = {}
    tools = v._map_function(value=[{"code_execution": {}}], optional_params=optional_params)
    assert len(tools) == 1
    assert tools[0]["code_execution"] == {}
    print(tools)

    new_optional_params = {}
    new_tools = v._map_function(value=[{"codeExecution": {}}], optional_params=new_optional_params)
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
    optional_params = {}
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
    tools = v._map_function(value=value, optional_params=optional_params)

    assert tools[0]["function_declarations"][0]["parameters"]["properties"][
        "base_branch"
    ] == {
        "anyOf": [{"type": "string", "nullable": True, "title": "Base Branch"}]
    }, f"Expected only anyOf field and its contents to be kept, but got {tools[0]['function_declarations'][0]['parameters']['properties']['base_branch']}"

    new_optional_params = {}
    new_value = [
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
                        },
                    },
                    "required": ["repo_path", "branch_name"],
                    "title": "GitCreateBranch",
                },
            },
        }
    ]
    new_tools = v._map_function(value=new_value, optional_params=new_optional_params)

    assert new_tools[0]["function_declarations"][0]["parameters"]["properties"][
        "base_branch"
    ] == {
        "anyOf": [{"type": "string", "nullable": True}]
    }, f"Expected only anyOf field and its contents to be kept, but got {new_tools[0]['function_declarations'][0]['parameters']['properties']['base_branch']}"


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
                    {"webSearchQueries": ["", "What is the capital of France?", "Capital of France"]}
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
    assert usage.prompt_tokens_details.web_search_requests == 2


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


def test_vertex_ai_tool_call_id_format():
    """
    Test that tool call IDs have the correct format and length.
    
    The ID should be in format 'call_' + 28 hex characters (total 33 characters).
    This test verifies the fix for keeping the code line under 40 characters.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.llms.vertex_ai import HttpxPartType

    # Create parts with function calls
    parts_with_functions = [
        HttpxPartType(
            functionCall={
                "name": "get_weather",
                "args": {"location": "San Francisco", "unit": "celsius"},
            }
        ),
        HttpxPartType(
            functionCall={
                "name": "get_time", 
                "args": {"timezone": "PST"}
            }
        ),
    ]

    function, tools, updated_idx = VertexGeminiConfig._transform_parts(
        parts=parts_with_functions, cumulative_tool_call_idx=0, is_function_call=False
    )

    # Verify tools were created
    assert function is None
    assert tools is not None
    assert len(tools) == 2

    # Test ID format for both tool calls
    for tool in tools:
        tool_id = tool["id"]
        
        # Should start with 'call_'
        assert tool_id.startswith("call_"), f"ID should start with 'call_', got: {tool_id}"
        
        # Should have exactly 33 total characters (call_ + 28 hex chars)
        assert len(tool_id) == 33, f"ID should be 33 characters long, got {len(tool_id)}: {tool_id}"
        
        # The part after 'call_' should be 28 hex characters
        hex_part = tool_id[5:]  # Remove 'call_' prefix
        assert len(hex_part) == 28, f"Hex part should be 28 characters, got {len(hex_part)}: {hex_part}"
        
        # Should only contain valid hex characters
        assert re.match(r'^[0-9a-f]{28}$', hex_part), f"Should contain only lowercase hex chars, got: {hex_part}"

    # Verify IDs are unique
    assert tools[0]["id"] != tools[1]["id"], "Tool call IDs should be unique"

    # Test with multiple generations to ensure uniqueness
    ids_generated = set()
    for _ in range(10):
        _, test_tools, _ = VertexGeminiConfig._transform_parts(
            parts=[HttpxPartType(functionCall={"name": "test", "args": {}})],
            cumulative_tool_call_idx=0,
            is_function_call=False,
        )
        if test_tools:
            ids_generated.add(test_tools[0]["id"])
    
    # All generated IDs should be unique
    assert len(ids_generated) == 10, f"All 10 IDs should be unique, got {len(ids_generated)} unique IDs"


def test_vertex_ai_code_line_length():
    """
    Test that the specific code line generating tool call IDs is within character limit.
    
    This is a meta-test to ensure the code change meets the 40-character requirement.
    """
    import inspect

    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Get the source code of the _transform_parts method
    source_lines = inspect.getsource(VertexGeminiConfig._transform_parts).split('\n')
    
    # Find the line that generates the ID
    id_line = None
    for line in source_lines:
        if '"id": f"call_' in line and 'uuid.uuid4().hex[:28]' in line:
            id_line = line.strip()  # Remove indentation for length check
            break
    
    assert id_line is not None, "Could not find the ID generation line in source code"
    
    # Check that the line is 40 characters or less (excluding indentation)
    line_length = len(id_line)
    assert line_length <= 40, f"ID generation line is {line_length} characters, should be ≤40: {id_line}"
    
    # Verify it contains the expected UUID format
    assert 'uuid.uuid4().hex[:28]' in id_line, f"Line should contain shortened UUID format: {id_line}"


def test_vertex_ai_map_google_maps_tool_simple():
    """
    Test googleMaps tool transformation without location data.
    
    Input:
        value=[{"googleMaps": {"enableWidget": "ENABLE_WIDGET"}}]
        optional_params={}
    
    Expected Output:
        tools=[{"googleMaps": {"enableWidget": "ENABLE_WIDGET"}}]
        optional_params={} (unchanged)
    """
    v = VertexGeminiConfig()
    optional_params = {}
    
    tools = v._map_function(
        value=[{"googleMaps": {"enableWidget": "ENABLE_WIDGET"}}],
        optional_params=optional_params
    )
    
    assert len(tools) == 1
    assert "googleMaps" in tools[0]
    assert tools[0]["googleMaps"]["enableWidget"] == "ENABLE_WIDGET"
    assert "toolConfig" not in optional_params


def test_vertex_ai_map_google_maps_tool_with_location():
    """
    Test googleMaps tool transformation with location data.
    Verifies latitude/longitude/languageCode are extracted to toolConfig.retrievalConfig.
    
    Input:
        value=[{
            "googleMaps": {
                "enableWidget": "ENABLE_WIDGET",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "languageCode": "en_US"
            }
        }]
        optional_params={}
    
    Expected Output:
        tools=[{
            "googleMaps": {"enableWidget": "ENABLE_WIDGET"}
        }]
        optional_params={
            "toolConfig": {
                "retrievalConfig": {
                    "latLng": {
                        "latitude": 37.7749,
                        "longitude": -122.4194
                    },
                    "languageCode": "en_US"
                }
            }
        }
    """
    v = VertexGeminiConfig()
    optional_params = {}
    
    tools = v._map_function(
        value=[{
            "googleMaps": {
                "enableWidget": "ENABLE_WIDGET",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "languageCode": "en_US"
            }
        }],
        optional_params=optional_params
    )
    
    assert len(tools) == 1
    assert "googleMaps" in tools[0]
    
    google_maps_tool = tools[0]["googleMaps"]
    assert google_maps_tool["enableWidget"] == "ENABLE_WIDGET"
    assert "latitude" not in google_maps_tool
    assert "longitude" not in google_maps_tool
    assert "languageCode" not in google_maps_tool
    
    assert "toolConfig" in optional_params
    assert "retrievalConfig" in optional_params["toolConfig"]
    
    retrieval_config = optional_params["toolConfig"]["retrievalConfig"]
    assert retrieval_config["latLng"]["latitude"] == 37.7749
    assert retrieval_config["latLng"]["longitude"] == -122.4194
    assert retrieval_config["languageCode"] == "en_US"

def test_vertex_ai_penalty_parameters_validation():
    """
    Test that penalty parameters are properly validated for different Gemini models.
    
    This test ensures that:
    1. Models that don't support penalty parameters (like preview models) filter them out
    2. Models that support penalty parameters include them in the request
    3. Appropriate warnings are logged for unsupported models
    """
    v = VertexGeminiConfig()

    # Test cases: (model_name, should_support_penalty_params)
    test_cases = [
        ("gemini-2.5-pro-preview-06-05", False),  # Preview model - should not support
    ]

    for model, should_support in test_cases:
        # Test _supports_penalty_parameters method
        assert v._supports_penalty_parameters(model) == should_support, \
            f"Model {model} penalty support should be {should_support}"

        # Test get_supported_openai_params method
        supported_params = v.get_supported_openai_params(model)
        has_penalty_params = "frequency_penalty" in supported_params and "presence_penalty" in supported_params
        assert has_penalty_params == should_support, \
            f"Model {model} should {'include' if should_support else 'exclude'} penalty params in supported list"

    # Test parameter mapping for unsupported model
    model = "gemini-2.5-pro-preview-06-05"
    non_default_params = {
        "temperature": 0.7,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "max_tokens": 100
    }

    optional_params = {}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False
    )

    # Penalty parameters should be filtered out for unsupported models
    assert "frequency_penalty" not in result, "frequency_penalty should be filtered out for unsupported model"
    assert "presence_penalty" not in result, "presence_penalty should be filtered out for unsupported model"

    # Other parameters should still be included
    assert "temperature" in result, "temperature should still be included"
    assert "max_output_tokens" in result, "max_output_tokens should still be included"
    assert result["temperature"] == 0.7
    assert result["max_output_tokens"] == 100


def test_vertex_ai_gemini_3_penalty_parameters_unsupported():
    """
    Test that penalty parameters are not supported for Gemini 3 models.
    
    This test ensures that:
    1. Gemini 3 models do not support penalty parameters
    2. Penalty parameters are excluded from supported params list for Gemini 3 models
    3. Penalty parameters are filtered out when mapping params for Gemini 3 models
    """
    v = VertexGeminiConfig()

    # Test Gemini 3 models
    gemini_3_models = [
        "gemini-3-pro-preview",
        "vertex_ai/gemini-3-pro-preview",
        "gemini/gemini-3-pro-preview",
    ]

    for model in gemini_3_models:
        # Test _supports_penalty_parameters method
        assert v._supports_penalty_parameters(model) == False, \
            f"Gemini 3 model {model} should not support penalty parameters"

        # Test get_supported_openai_params method
        supported_params = v.get_supported_openai_params(model)
        assert "frequency_penalty" not in supported_params, \
            f"frequency_penalty should not be in supported params for {model}"
        assert "presence_penalty" not in supported_params, \
            f"presence_penalty should not be in supported params for {model}"

        # Test parameter mapping - penalty params should be filtered out
        non_default_params = {
            "temperature": 0.7,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "max_tokens": 100
        }

        optional_params = {}
        result = v.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False
        )

        # Penalty parameters should be filtered out for Gemini 3 models
        assert "frequency_penalty" not in result, \
            f"frequency_penalty should be filtered out for Gemini 3 model {model}"
        assert "presence_penalty" not in result, \
            f"presence_penalty should be filtered out for Gemini 3 model {model}"

        # Other parameters should still be included
        assert "temperature" in result, \
            f"temperature should still be included for Gemini 3 model {model}"
        assert "max_output_tokens" in result, \
            f"max_output_tokens should still be included for Gemini 3 model {model}"
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100

    # Test that non-Gemini 3 models still support penalty parameters (if they're not in the unsupported list)
    non_gemini_3_model = "gemini-2.5-pro"
    assert v._supports_penalty_parameters(non_gemini_3_model) == True, \
        f"Non-Gemini 3 model {non_gemini_3_model} should support penalty parameters"
    
    supported_params = v.get_supported_openai_params(non_gemini_3_model)
    assert "frequency_penalty" in supported_params, \
        f"frequency_penalty should be in supported params for {non_gemini_3_model}"
    assert "presence_penalty" in supported_params, \
        f"presence_penalty should be in supported params for {non_gemini_3_model}"


def test_vertex_ai_annotation_streaming_events():
    """
    Test that annotation events are properly emitted during streaming for Vertex AI Gemini.
    
    This test verifies:
    1. Grounding metadata is converted to annotations in streaming chunks
    2. Annotations are included in the delta of streaming chunks
    3. Multiple annotations are handled correctly
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )
    from litellm.types.llms.openai import ChatCompletionAnnotation

    litellm_logging = MagicMock()

    # Simulate a streaming chunk with grounding metadata (as Gemini sends it)
    chunk_with_annotations = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "The weather in San Francisco today is clear."}]
                },
                "groundingMetadata": {
                    "webSearchQueries": ["weather San Francisco today"],
                    "searchEntryPoint": {
                        "renderedContent": '<div>Search results</div>'
                    },
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://www.google.com/search?q=weather+in+San+Francisco,+CA",
                                "title": "Weather information for San Francisco, CA",
                                "domain": "google.com",
                            }
                        }
                    ],
                    "groundingSupports": [
                        {
                            "segment": {
                                "startIndex": 0,
                                "endIndex": 50,
                                "text": "The weather in San Francisco today is clear.",
                            },
                            "groundingChunkIndices": [0],
                            "confidenceScores": [0.95],
                        }
                    ],
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 15,
            "totalTokenCount": 25,
        },
    }

    # Create iterator and parse chunk
    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=litellm_logging
    )
    streaming_chunk = iterator.chunk_parser(chunk_with_annotations)

    # Verify the chunk was parsed correctly
    assert streaming_chunk.choices is not None
    assert len(streaming_chunk.choices) == 1
    
    # Check that annotations are present in the delta
    delta = streaming_chunk.choices[0].delta
    assert hasattr(delta, "annotations")
    assert delta.annotations is not None
    assert len(delta.annotations) > 0

    # Verify annotation structure
    annotation = delta.annotations[0]
    assert isinstance(annotation, dict)  # ChatCompletionAnnotation is a TypedDict
    assert annotation["type"] == "url_citation"
    assert annotation["url_citation"]["start_index"] == 0
    assert annotation["url_citation"]["end_index"] == 50
    assert "google.com" in annotation["url_citation"]["url"]
    assert "Weather information" in annotation["url_citation"]["title"]


@pytest.mark.asyncio
async def test_vertex_ai_streaming_bad_request_is_not_wrapped():
    class DummyLogging:
        def __init__(self):
            self.model_call_details = {"litellm_params": {}}
            self.optional_params = {}
            self.messages = []
            self.completion_start_time = None
            self.stream_options = None

        def failure_handler(self, *args, **kwargs):
            return None

        async def async_failure_handler(self, *args, **kwargs):
            return None

    async def failing_make_call(client=None, **kwargs):
        raise VertexAIError(status_code=400, message="bad input", headers={})

    stream = CustomStreamWrapper(
        completion_stream=None,
        make_call=failing_make_call,
        model="gemini-3-pro-preview",
        logging_obj=DummyLogging(),
        custom_llm_provider="vertex_ai_beta",
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        await stream.__anext__()

    assert getattr(exc_info.value, "status_code", None) == 400


def test_vertex_ai_annotation_conversion():
    """
    Test the conversion of Vertex AI grounding metadata to OpenAI annotations.
    
    This test verifies the _convert_grounding_metadata_to_annotations method
    correctly transforms grounding metadata into the expected format.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Sample grounding metadata as returned by Vertex AI
    grounding_metadata = {
        "webSearchQueries": ["weather San Francisco", "current time San Francisco"],
        "searchEntryPoint": {
            "renderedContent": '<div>Search interface</div>'
        },
        "groundingChunks": [
            {
                "web": {
                    "uri": "https://www.google.com/search?q=weather+in+San+Francisco,+CA",
                    "title": "Weather information for San Francisco, CA",
                    "domain": "google.com",
                }
            },
            {
                "web": {
                    "uri": "https://www.google.com/search?q=time+in+San+Francisco,+CA",
                    "title": "Current time in San Francisco, CA",
                    "domain": "google.com",
                }
            }
        ],
        "groundingSupports": [
            {
                "segment": {
                    "startIndex": 0,
                    "endIndex": 30,
                    "text": "The weather in San Francisco",
                },
                "groundingChunkIndices": [0],
                "confidenceScores": [0.95],
            },
            {
                "segment": {
                    "startIndex": 32,
                    "endIndex": 60,
                    "text": "is currently 72°F",
                },
                "groundingChunkIndices": [0],
                "confidenceScores": [0.88],
            },
            {
                "segment": {
                    "startIndex": 62,
                    "endIndex": 85,
                    "text": "and the time is 2:30 PM",
                },
                "groundingChunkIndices": [1],
                "confidenceScores": [0.92],
            }
        ],
    }

    # Convert grounding metadata to annotations
    content_text = "The weather in San Francisco is currently 72°F and the time is 2:30 PM"
    annotations = VertexGeminiConfig._convert_grounding_metadata_to_annotations(
        [grounding_metadata], content_text
    )

    # Verify annotations were created
    assert len(annotations) == 3  # One for each grounding support

    # Check first annotation (weather)
    weather_annotation = annotations[0]
    assert weather_annotation["type"] == "url_citation"
    assert weather_annotation["url_citation"]["start_index"] == 0
    assert weather_annotation["url_citation"]["end_index"] == 30
    assert "google.com" in weather_annotation["url_citation"]["url"]
    assert "Weather information" in weather_annotation["url_citation"]["title"]

    # Check second annotation (weather continuation)
    weather_cont_annotation = annotations[1]
    assert weather_cont_annotation["type"] == "url_citation"
    assert weather_cont_annotation["url_citation"]["start_index"] == 32
    assert weather_cont_annotation["url_citation"]["end_index"] == 60
    assert "google.com" in weather_cont_annotation["url_citation"]["url"]
    assert "Weather information" in weather_cont_annotation["url_citation"]["title"]

    # Check third annotation (time)
    time_annotation = annotations[2]
    assert time_annotation["type"] == "url_citation"
    assert time_annotation["url_citation"]["start_index"] == 62
    assert time_annotation["url_citation"]["end_index"] == 85
    assert "google.com" in time_annotation["url_citation"]["url"]
    assert "Current time" in time_annotation["url_citation"]["title"]


def test_vertex_ai_annotation_empty_grounding_metadata():
    """
    Test handling of empty or missing grounding metadata.
    
    This test ensures the annotation conversion handles edge cases gracefully.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Test with empty grounding metadata
    empty_metadata = {}
    annotations = VertexGeminiConfig._convert_grounding_metadata_to_annotations(
        [empty_metadata], "test content"
    )
    assert len(annotations) == 0

    # Test with missing groundingSupports
    metadata_no_supports = {
        "webSearchQueries": ["test query"],
        "groundingChunks": [{"web": {"uri": "https://example.com", "title": "Test"}}],
    }
    annotations = VertexGeminiConfig._convert_grounding_metadata_to_annotations(
        [metadata_no_supports], "test content"
    )
    assert len(annotations) == 0

    # Test with empty groundingSupports
    metadata_empty_supports = {
        "webSearchQueries": ["test query"],
        "groundingChunks": [{"web": {"uri": "https://example.com", "title": "Test"}}],
        "groundingSupports": [],
    }
    annotations = VertexGeminiConfig._convert_grounding_metadata_to_annotations(
        [metadata_empty_supports], "test content"
    )
    assert len(annotations) == 0


# ==================== Gemini 3 Pro Preview Tests ====================

def test_is_gemini_3_or_newer():
    """Test the _is_gemini_3_or_newer method for version detection"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Gemini 3 models
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-3-pro-preview") == True
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-3-flash") == True
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-3-pro") == True
    assert VertexGeminiConfig._is_gemini_3_or_newer("vertex_ai/gemini-3-pro-preview") == True
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini/gemini-3-pro-preview") == True

    # Gemini 2.5 and older models
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-2.5-pro") == False
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-2.5-flash") == False
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-2.0-flash") == False
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-1.5-pro") == False
    assert VertexGeminiConfig._is_gemini_3_or_newer("gemini-pro") == False

    # Edge cases
    assert VertexGeminiConfig._is_gemini_3_or_newer("") == False


def test_reasoning_effort_maps_to_thinking_level_gemini_3():
    """Test that reasoning_effort maps to thinking_level AND includeThoughts for Gemini 3+ models"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    model = "gemini-3-pro-preview"
    optional_params = {}

    # Test minimal -> low + includeThoughts=True
    non_default_params = {"reasoning_effort": "minimal"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test low -> low + includeThoughts=True
    optional_params = {}
    non_default_params = {"reasoning_effort": "low"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test medium -> high + includeThoughts=True (medium not available yet)
    optional_params = {}
    non_default_params = {"reasoning_effort": "medium"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "high"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test high -> high + includeThoughts=True
    optional_params = {}
    non_default_params = {"reasoning_effort": "high"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "high"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test disable -> low + includeThoughts=False (cannot fully disable in Gemini 3)
    optional_params = {}
    non_default_params = {"reasoning_effort": "disable"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["thinkingConfig"]["includeThoughts"] is False

    # Test none -> low + includeThoughts=False (cannot fully disable in Gemini 3)
    optional_params = {}
    non_default_params = {"reasoning_effort": "none"}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["thinkingConfig"]["includeThoughts"] is False


def test_reasoning_effort_dict_format_gemini_3():
    """
    Test that reasoning_effort works when passed as dict format from OpenAI Agents SDK.

    The OpenAI Agents SDK passes reasoning_effort as {"effort": "high", "summary": "auto"}
    instead of just a string. This test verifies that we correctly extract the effort value.

    Related issue: https://github.com/BerriAI/litellm/issues/19411
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    model = "gemini-3-pro-preview"

    # Test dict format with effort="high" (OpenAI Agents SDK format)
    optional_params = {}
    non_default_params = {"reasoning_effort": {"effort": "high", "summary": "auto"}}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "high"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test dict format with effort="low"
    optional_params = {}
    non_default_params = {"reasoning_effort": {"effort": "low"}}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test dict format with effort="medium"
    optional_params = {}
    non_default_params = {"reasoning_effort": {"effort": "medium"}}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    assert result["thinkingConfig"]["thinkingLevel"] == "high"
    assert result["thinkingConfig"]["includeThoughts"] is True

    # Test dict format without effort key - should fall back to Gemini 3 default (low)
    optional_params = {}
    non_default_params = {"reasoning_effort": {"summary": "auto"}}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )
    # Gemini 3 defaults to thinkingLevel="low" when no explicit effort is set
    assert result["thinkingConfig"]["thinkingLevel"] == "low"


def test_temperature_default_for_gemini_3():
    """Test that temperature defaults to 1.0 for Gemini 3+ models when not specified"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()
    model = "gemini-3-pro-preview"
    optional_params = {}

    # No temperature specified
    non_default_params = {}
    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )

    # Should default to 1.0
    assert "temperature" in result
    assert result["temperature"] == 1.0


def test_media_resolution_from_detail_parameter():
    """Test that OpenAI's detail parameter is correctly mapped to media_resolution"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _convert_detail_to_media_resolution_enum,
        _gemini_convert_messages_with_history,
    )

    # Test detail -> media_resolution enum mapping
    assert _convert_detail_to_media_resolution_enum("low") == {"level": "MEDIA_RESOLUTION_LOW"}
    assert _convert_detail_to_media_resolution_enum("high") == {"level": "MEDIA_RESOLUTION_HIGH"}
    assert _convert_detail_to_media_resolution_enum("auto") is None
    assert _convert_detail_to_media_resolution_enum(None) is None

    # Test with actual message transformation using base64 image
    # Using a minimal valid base64-encoded 1x1 PNG
    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )
    
    # Verify media_resolution is set at the Part level (not inside inline_data)
    assert len(contents) == 1
    assert len(contents[0]["parts"]) >= 1
    # Find the part with inline_data
    image_part = None
    for part in contents[0]["parts"]:
        if "inline_data" in part or "inlineData" in part:
            image_part = part
            break
    assert image_part is not None
    # media_resolution should be at the Part level, not inside inline_data
    assert "media_resolution" in image_part
    media_res = image_part.get("media_resolution")
    assert media_res == {"level": "MEDIA_RESOLUTION_HIGH"}


def test_media_resolution_low_detail():
    """Test that detail='low' maps to media_resolution enum with MEDIA_RESOLUTION_LOW"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    # Using a minimal valid base64-encoded 1x1 PNG
    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "low"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )
    
    # Find the part with inline_data
    image_part = None
    for part in contents[0]["parts"]:
        if "inline_data" in part:
            image_part = part
            break
    assert image_part is not None
    assert "inline_data" in image_part
    # media_resolution should be at the Part level, not inside inline_data
    assert "media_resolution" in image_part
    assert image_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_LOW"}


def test_media_resolution_auto_detail():
    """Test that detail='auto' or None doesn't set media_resolution"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    # Using a minimal valid base64-encoded 1x1 PNG
    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Test with auto
    messages_auto = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "auto"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(messages=messages_auto)
    # Find the part with inline_data
    image_part = None
    for part in contents[0]["parts"]:
        if "inline_data" in part:
            image_part = part
            break
    assert image_part is not None
    assert "inline_data" in image_part
    # media_resolution should not be set for auto (check Part level, not inline_data)
    assert "media_resolution" not in image_part

    # Test with None
    messages_none = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(messages=messages_none)
    # Find the part with inline_data
    image_part = None
    for part in contents[0]["parts"]:
        if "inline_data" in part:
            image_part = part
            break
    assert image_part is not None
    assert "inline_data" in image_part
    # media_resolution should not be set (check Part level, not inline_data)
    assert "media_resolution" not in image_part


def test_media_resolution_per_part():
    """Test that different images can have different media_resolution values"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    # Using minimal valid base64-encoded 1x1 PNGs
    base64_image1 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    base64_image2 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image1,
                        "detail": "low"
                    }
                },
                {
                    "type": "text",
                    "text": "Compare these images"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image2,
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )
    
    # Should have one content with multiple parts
    assert len(contents) == 1
    assert len(contents[0]["parts"]) == 3  # image1, text, image2
    
    # First image should have low resolution (first part is the image)
    image1_part = contents[0]["parts"][0]
    assert "inline_data" in image1_part
    # media_resolution should be at the Part level, not inside inline_data
    assert "media_resolution" in image1_part
    assert image1_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_LOW"}
    
    # Second image should have high resolution (third part is the second image)
    image2_part = contents[0]["parts"][2]
    assert "inline_data" in image2_part
    # media_resolution should be at the Part level, not inside inline_data
    assert "media_resolution" in image2_part
    assert image2_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_HIGH"}


def test_media_resolution_only_for_gemini_3_models():
    """Ensure media_resolution is not added for non-Gemini 3 models."""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image,
                        "detail": "high",
                    },
                }
            ],
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-2.5-pro"
    )
    image_part = None
    for part in contents[0]["parts"]:
        if "inline_data" in part:
            image_part = part
            break
    assert image_part is not None
    assert "inline_data" in image_part
    # media_resolution should not be at the Part level for non-Gemini 3 models
    assert "media_resolution" not in image_part
    assert "mediaResolution" not in image_part


def test_gemini_3_image_models_no_thinking_config():
    """
    Test that Gemini 3 image models do NOT receive automatic thinkingConfig.

    Related issue: https://github.com/BerriAI/litellm/issues/17013
    gemini-3-pro-image-preview does not support thinking_level parameter
    and returns BadRequestError: "Thinking level is not supported for this model"
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()

    # Test gemini-3-pro-image-preview (the specific model from the bug report)
    model = "gemini-3-pro-image-preview"
    optional_params = {}
    non_default_params = {}

    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )

    # Should NOT have thinkingConfig automatically added
    assert "thinkingConfig" not in result
    # But should still get temperature=1.0 for Gemini 3
    assert result["temperature"] == 1.0


def test_gemini_3_text_models_get_thinking_config():
    """
    Test that Gemini 3 text models DO receive automatic thinkingConfig.
    This ensures we didn't break the existing behavior for non-image models.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()

    # Test gemini-3-pro-preview (text model, should get thinking)
    model = "gemini-3-pro-preview"
    optional_params = {}
    non_default_params = {}

    result = v.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model=model,
        drop_params=False,
    )

    # Should have thinkingConfig automatically added
    assert "thinkingConfig" in result
    assert result["thinkingConfig"]["thinkingLevel"] == "low"
    assert result["temperature"] == 1.0


def test_gemini_image_models_excluded_from_thinking():
    """
    Test that any Gemini model with 'image' in the name is excluded from thinking config.
    This covers current and future image models.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    v = VertexGeminiConfig()

    # Test various image model patterns
    image_models = [
        "gemini-3-pro-image-preview",
        "gemini-3-pro-image-generation",
        "gemini-3-flash-image-preview",
        "gemini/gemini-3-image-edit",
    ]

    for model in image_models:
        optional_params = {}
        non_default_params = {}

        result = v.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # None of these should have thinkingConfig
        assert "thinkingConfig" not in result, f"Model {model} should not have thinkingConfig"


def test_partial_json_chunk_after_first_chunk():
    """
    Test that partial JSON chunks are handled correctly even AFTER the first chunk.

    This tests the fix for:
    - https://github.com/BerriAI/litellm/issues/16562
    - https://github.com/BerriAI/litellm/issues/16037
    - https://github.com/BerriAI/litellm/issues/14747
    - https://github.com/BerriAI/litellm/issues/10410
    - https://github.com/BerriAI/litellm/issues/5650

    The bug was that accumulation mode only activated on the first chunk.
    If chunk 1 was valid and chunk 5 arrived partial, it would crash.
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    iterator = ModelResponseIterator(
        streaming_response=MagicMock(),
        sync_stream=True,
        logging_obj=MagicMock(),
    )

    # First chunk arrives COMPLETE - this sets sent_first_chunk = True
    first_chunk = '{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}'
    result1 = iterator.handle_valid_json_chunk(first_chunk)
    assert result1 is not None, "First complete chunk should parse OK"
    assert iterator.sent_first_chunk is True, "sent_first_chunk should be True after first chunk"

    # Later chunk arrives PARTIAL (simulating network fragmentation)
    partial_chunk = '{"candidates": [{"content":'
    result2 = iterator.handle_valid_json_chunk(partial_chunk)

    # Should switch to accumulation mode instead of crashing
    assert result2 is None, "Partial chunk should return None while accumulating"
    assert iterator.chunk_type == "accumulated_json", "Should switch to accumulated_json mode"


def test_partial_json_chunk_on_first_chunk():
    """Test that first chunk being partial still works (existing behavior)."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    iterator = ModelResponseIterator(
        streaming_response=MagicMock(),
        sync_stream=True,
        logging_obj=MagicMock(),
    )

    # First chunk is partial
    partial = '{"candidates": [{"content":'
    result = iterator.handle_valid_json_chunk(partial)

    assert result is None, "Partial first chunk should return None"
    assert iterator.chunk_type == "accumulated_json", "Should switch to accumulated_json mode"



def test_google_ai_studio_presence_penalty_supported():
    """
    Test that presence_penalty is supported for Google AI Studio Gemini.

    Regression test for https://github.com/BerriAI/litellm/issues/14753
    """
    config = GoogleAIStudioGeminiConfig()
    supported_params = config.get_supported_openai_params(model="gemini-2.0-flash")

    assert "presence_penalty" in supported_params
# ==================== Tool Type Separation Tests ====================
# These tests verify that each Tool object contains exactly one type per Vertex AI API spec
# Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/Tool


def test_vertex_ai_multiple_tool_types_separate_objects():
    """
    Test that multiple tool types are placed in separate Tool objects.

    This is required by Vertex AI API spec:
    "A Tool object should contain exactly one type of Tool"

    Related error without this fix:
    "tools[0].tool_type: one_of 'tool_type' has more than one initialized field:
    enterprise_web_search, url_context"

    Input:
        value=[
            {"enterpriseWebSearch": {}},
            {"url_context": {}},
        ]

    Expected Output:
        tools=[
            {"enterpriseWebSearch": {}},  # First Tool object
            {"url_context": {}},          # Second Tool object (separate!)
        ]

    NOT (incorrect - causes API error):
        tools=[
            {"enterpriseWebSearch": {}, "url_context": {}}  # Multiple types in one object
        ]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    tools = v._map_function(
        value=[
            {"enterpriseWebSearch": {}},
            {"url_context": {}},
        ],
        optional_params=optional_params
    )

    # Should have 2 separate Tool objects
    assert len(tools) == 2, f"Expected 2 separate Tool objects, got {len(tools)}"

    # Each Tool object should contain exactly ONE type
    tool_types_in_first = [k for k in tools[0].keys()]
    tool_types_in_second = [k for k in tools[1].keys()]

    assert len(tool_types_in_first) == 1, f"First Tool should have exactly 1 type, got {tool_types_in_first}"
    assert len(tool_types_in_second) == 1, f"Second Tool should have exactly 1 type, got {tool_types_in_second}"

    # Verify the correct tool types are present
    assert "enterpriseWebSearch" in tools[0], "First Tool should contain enterpriseWebSearch"
    assert "url_context" in tools[1], "Second Tool should contain url_context"


def test_vertex_ai_function_declarations_with_other_tools_separate():
    """
    Test that function declarations and other tool types are in separate Tool objects.

    This ensures that when using both function calling AND special tools like
    google_search or code_execution, they are properly separated per API spec.

    Input:
        value=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
            {"googleSearch": {}},
            {"code_execution": {}},
        ]

    Expected Output:
        tools=[
            {"function_declarations": [{"name": "get_weather", "description": "Get weather"}]},
            {"googleSearch": {}},
            {"code_execution": {}},
        ]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    tools = v._map_function(
        value=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
            {"googleSearch": {}},
            {"code_execution": {}},
        ],
        optional_params=optional_params
    )

    # Should have 3 separate Tool objects
    assert len(tools) == 3, f"Expected 3 separate Tool objects, got {len(tools)}"

    # Find each tool type
    func_tool = None
    search_tool = None
    code_tool = None

    for tool in tools:
        if "function_declarations" in tool:
            func_tool = tool
        elif "googleSearch" in tool:
            search_tool = tool
        elif "code_execution" in tool:
            code_tool = tool

    # Verify all tools are present and separate
    assert func_tool is not None, "function_declarations Tool should be present"
    assert search_tool is not None, "googleSearch Tool should be present"
    assert code_tool is not None, "code_execution Tool should be present"

    # Verify each Tool has exactly one type
    assert len(func_tool.keys()) == 1, "function_declarations Tool should have only one key"
    assert len(search_tool.keys()) == 1, "googleSearch Tool should have only one key"
    assert len(code_tool.keys()) == 1, "code_execution Tool should have only one key"

    # Verify function declaration content
    assert func_tool["function_declarations"][0]["name"] == "get_weather"


def test_vertex_ai_single_tool_type_still_works():
    """
    Test that single tool type usage still works correctly (backward compatibility).

    Input:
        value=[{"code_execution": {}}]

    Expected Output:
        tools=[{"code_execution": {}}]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    tools = v._map_function(
        value=[{"code_execution": {}}],
        optional_params=optional_params
    )

    assert len(tools) == 1
    assert "code_execution" in tools[0]
    assert tools[0]["code_execution"] == {}


def test_vertex_ai_openai_web_search_tool_transformation():
    """
    Test that OpenAI-style web_search and web_search_preview tools are transformed to googleSearch.

    This fixes the issue where passing OpenAI-style web search tools like:
        {"type": "web_search"} or {"type": "web_search_preview"}
    would be silently ignored (the request succeeds but grounding is not applied).

    The fix transforms these to Gemini's googleSearch tool.

    Input:
        value=[{"type": "web_search"}]

    Expected Output:
        tools=[{"googleSearch": {}}]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    # Test web_search transformation
    tools = v._map_function(
        value=[{"type": "web_search"}],
        optional_params=optional_params
    )

    assert len(tools) == 1, f"Expected 1 Tool object, got {len(tools)}"
    assert "googleSearch" in tools[0], f"Expected googleSearch in tool, got {tools[0].keys()}"
    assert tools[0]["googleSearch"] == {}, f"Expected empty googleSearch config, got {tools[0]['googleSearch']}"


def test_vertex_ai_openai_web_search_preview_tool_transformation():
    """
    Test that OpenAI-style web_search_preview tool is transformed to googleSearch.

    Input:
        value=[{"type": "web_search_preview"}]

    Expected Output:
        tools=[{"googleSearch": {}}]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    # Test web_search_preview transformation
    tools = v._map_function(
        value=[{"type": "web_search_preview"}],
        optional_params=optional_params
    )

    assert len(tools) == 1, f"Expected 1 Tool object, got {len(tools)}"
    assert "googleSearch" in tools[0], f"Expected googleSearch in tool, got {tools[0].keys()}"
    assert tools[0]["googleSearch"] == {}, f"Expected empty googleSearch config, got {tools[0]['googleSearch']}"


def test_vertex_ai_openai_web_search_with_function_tools():
    """
    Test that OpenAI-style web_search tool works alongside function tools.

    Input:
        value=[
            {"type": "web_search"},
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
        ]

    Expected Output:
        tools=[
            {"googleSearch": {}},
            {"function_declarations": [{"name": "get_weather", "description": "Get weather"}]},
        ]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    tools = v._map_function(
        value=[
            {"type": "web_search"},
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather"}},
        ],
        optional_params=optional_params
    )

    # Should have 2 separate Tool objects
    assert len(tools) == 2, f"Expected 2 Tool objects, got {len(tools)}"

    # Find each tool type
    search_tool = None
    func_tool = None

    for tool in tools:
        if "googleSearch" in tool:
            search_tool = tool
        elif "function_declarations" in tool:
            func_tool = tool

    # Verify both tools are present
    assert search_tool is not None, "googleSearch Tool should be present"
    assert func_tool is not None, "function_declarations Tool should be present"

    # Verify googleSearch is empty config
    assert search_tool["googleSearch"] == {}

    # Verify function declaration content
    assert func_tool["function_declarations"][0]["name"] == "get_weather"


def test_vertex_ai_multiple_function_declarations_grouped():
    """
    Test that multiple function declarations are grouped in ONE Tool object.

    Function declarations are the exception - they CAN be grouped together
    in a single Tool object (up to 512 declarations).

    Input:
        value=[
            {"type": "function", "function": {"name": "func1", "description": "First function"}},
            {"type": "function", "function": {"name": "func2", "description": "Second function"}},
        ]

    Expected Output:
        tools=[
            {
                "function_declarations": [
                    {"name": "func1", "description": "First function"},
                    {"name": "func2", "description": "Second function"},
                ]
            }
        ]
    """
    v = VertexGeminiConfig()
    optional_params = {}

    tools = v._map_function(
        value=[
            {"type": "function", "function": {"name": "func1", "description": "First function"}},
            {"type": "function", "function": {"name": "func2", "description": "Second function"}},
        ],
        optional_params=optional_params
    )

    # Should have only 1 Tool object (function declarations grouped)
    assert len(tools) == 1, f"Expected 1 Tool object for grouped functions, got {len(tools)}"

    # Should contain function_declarations with 2 functions
    assert "function_declarations" in tools[0]
    assert len(tools[0]["function_declarations"]) == 2

    # Verify function names
    func_names = [f["name"] for f in tools[0]["function_declarations"]]
    assert "func1" in func_names
    assert "func2" in func_names


def test_gemini_3_flash_preview_token_usage_fallback():
    """Test fallback logic when candidatesTokensDetails is missing (e.g. Gemini 3 Flash Preview)."""
    v = VertexGeminiConfig()

    usage_metadata_dict = {
        "promptTokenCount": 2145,
        "candidatesTokenCount": 509,
        "totalTokenCount": 2654,
        # candidatesTokensDetails intentionally omitted
    }

    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)

    assert result.completion_tokens == 509
    assert result.prompt_tokens == 2145
    assert result.total_tokens == 2654

    # Text tokens should be derived from candidatesTokenCount
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.text_tokens == 509
    assert result.completion_tokens_details.image_tokens is None
    assert result.completion_tokens_details.audio_tokens is None


def test_gemini_no_reasoning_fallback():
    """Test fallback when reasoning_effort is absent and details are missing."""
    v = VertexGeminiConfig()

    usage_metadata_dict = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 264,
        "totalTokenCount": 364,
    }

    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)

    assert result.completion_tokens == 264
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.text_tokens == 264
    assert (
        result.completion_tokens_details.reasoning_tokens is None
        or result.completion_tokens_details.reasoning_tokens == 0
    )


def test_gemini_token_usage_standard_response():
    """Verify that standard responses with details are computed correctly and not overwritten."""
    v = VertexGeminiConfig()

    usage_metadata_dict = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "totalTokenCount": 150,
        "candidatesTokensDetails": [
            {"modality": "TEXT", "tokenCount": 40},
            {"modality": "IMAGE", "tokenCount": 10},
        ],
    }

    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)

    assert result.completion_tokens == 50
    assert result.completion_tokens_details.text_tokens == 40
    assert result.completion_tokens_details.image_tokens == 10


def test_gemini_image_gen_usage_metadata_prompt_vs_completion_separation():
    """
    Test that image generation models correctly separate prompt and completion token details.
    
    This is a regression test for the bug where prompt_tokens_details.image_tokens
    was incorrectly set to the completion's image token count instead of 0.
    
    Scenario: Text-only prompt generates an image response
    - Input: Text prompt (no images)
    - Output: Generated image + text description
    
    Expected behavior:
    - prompt_tokens_details.image_tokens should be 0 (text-only input)
    - completion_tokens_details.image_tokens should be 1290 (generated image)
    
    Bug behavior (before fix):
    - prompt_tokens_details.image_tokens was 1290 (incorrect!)
    - completion_tokens_details.image_tokens was 1290 (correct)
    
    The bug was caused by reusing the same variables (image_tokens, audio_tokens, text_tokens)
    for both prompt and completion token details.
    """
    v = VertexGeminiConfig()
    
    # Simulate Gemini image generation model response metadata
    # User sends text-only prompt, model generates image + text
    usage_metadata_dict = {
        "promptTokenCount": 101,
        "candidatesTokenCount": 1290,
        "totalTokenCount": 1391,
        # Prompt is text-only (no image tokens in input)
        "promptTokensDetails": [
            {"modality": "TEXT", "tokenCount": 101}
        ],
        # Response contains generated image + text
        "candidatesTokensDetails": [
            {"modality": "IMAGE", "tokenCount": 1290}
        ],
    }
    
    completion_response = {"usageMetadata": usage_metadata_dict}
    result = v._calculate_usage(completion_response=completion_response)
    
    # Verify basic token counts
    assert result.prompt_tokens == 101
    assert result.completion_tokens == 1290
    assert result.total_tokens == 1391
    
    # CRITICAL: Prompt tokens details should show NO image tokens (text-only input)
    assert result.prompt_tokens_details.text_tokens == 101, \
        "Prompt text tokens should be 101"
    assert result.prompt_tokens_details.image_tokens is None, \
        "Prompt image tokens should be None (text-only input, no images in prompt)"
    assert result.prompt_tokens_details.audio_tokens is None, \
        "Prompt audio tokens should be None"
    
    # Completion tokens details should show the generated image tokens
    assert result.completion_tokens_details.image_tokens == 1290, \
        "Completion image tokens should be 1290 (generated image)"
    
    # Verify text_tokens is auto-calculated for completion
    # candidatesTokenCount (1290) - image_tokens (1290) = 0
    assert result.completion_tokens_details.text_tokens == 0, \
        "Completion text tokens should be 0 (image-only response)"


def test_file_object_detail_parameter():
    """Test that detail parameter works for type: file objects (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this video?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "https://example.com/video.mp4",
                        "format": "video/mp4",
                        "detail": "low"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    # Verify media_resolution is set for file objects
    assert len(contents) == 1
    assert len(contents[0]["parts"]) == 2  # text + file

    # Find the file part
    file_part = None
    for part in contents[0]["parts"]:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None, "File part should exist"
    assert "media_resolution" in file_part, "media_resolution should be set for file objects"
    assert file_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_LOW"}


def test_video_metadata_fps():
    """Test fps parameter in video_metadata (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "gs://bucket/video.mp4",
                        "format": "video/mp4",
                        "video_metadata": {"fps": 5}
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    # Find the file part
    file_part = None
    for part in contents[0]["parts"]:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None
    assert "video_metadata" in file_part, "video_metadata should be present"
    assert file_part["video_metadata"]["fps"] == 5


def test_video_metadata_complete():
    """Test all video_metadata fields: fps, start_offset, end_offset (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video clip"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "gs://bucket/video.mp4",
                        "format": "video/mp4",
                        "video_metadata": {
                            "start_offset": "10s",
                            "end_offset": "60s",
                            "fps": 5
                        }
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    # Find the file part
    file_part = None
    for part in contents[0]["parts"]:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None
    assert "video_metadata" in file_part

    # Verify field name conversion: snake_case -> camelCase
    vm = file_part["video_metadata"]
    assert vm["startOffset"] == "10s", "start_offset should be converted to startOffset"
    assert vm["endOffset"] == "60s", "end_offset should be converted to endOffset"
    assert vm["fps"] == 5, "fps should remain unchanged"


def test_detail_and_video_metadata_combined():
    """Test using both detail and video_metadata together (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze video"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "https://example.com/video.mp4",
                        "format": "video/mp4",
                        "detail": "high",
                        "video_metadata": {"fps": 10}
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    # Find the file part
    file_part = None
    for part in contents[0]["parts"]:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None
    assert "media_resolution" in file_part
    assert file_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_HIGH"}
    assert "video_metadata" in file_part
    assert file_part["video_metadata"]["fps"] == 10


def test_new_detail_levels():
    """Test new detail levels: medium and ultra_high (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _convert_detail_to_media_resolution_enum,
        _gemini_convert_messages_with_history,
    )

    # Test mapping function
    assert _convert_detail_to_media_resolution_enum("low") == {"level": "MEDIA_RESOLUTION_LOW"}
    assert _convert_detail_to_media_resolution_enum("medium") == {"level": "MEDIA_RESOLUTION_MEDIUM"}
    assert _convert_detail_to_media_resolution_enum("high") == {"level": "MEDIA_RESOLUTION_HIGH"}
    assert _convert_detail_to_media_resolution_enum("ultra_high") == {"level": "MEDIA_RESOLUTION_ULTRA_HIGH"}

    # Test with actual message transformation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": "https://example.com/video.mp4",
                        "format": "video/mp4",
                        "detail": "medium"
                    }
                }
            ]
        }
    ]

    contents = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    file_part = None
    for part in contents[0]["parts"]:
        if "file_data" in part:
            file_part = part
            break

    assert file_part is not None
    assert file_part["media_resolution"] == {"level": "MEDIA_RESOLUTION_MEDIUM"}


def test_video_metadata_only_for_gemini_3():
    """Test that video_metadata is only applied for Gemini 3+ models (Issue #19026)"""
    from litellm.llms.vertex_ai.gemini.transformation import (
        _gemini_convert_messages_with_history,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_id": "https://example.com/video.mp4",
                        "format": "video/mp4",
                        "detail": "high",
                        "video_metadata": {"fps": 5}
                    }
                }
            ]
        }
    ]

    # Test with Gemini 1.5 (should not have video_metadata or media_resolution)
    contents_1_5 = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-1.5-pro"
    )

    file_part_1_5 = None
    for part in contents_1_5[0]["parts"]:
        if "file_data" in part:
            file_part_1_5 = part
            break

    assert file_part_1_5 is not None
    assert "media_resolution" not in file_part_1_5, "Gemini 1.5 should not have media_resolution"
    assert "video_metadata" not in file_part_1_5, "Gemini 1.5 should not have video_metadata"

    # Test with Gemini 3 (should have both)
    contents_3 = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-3-pro-preview"
    )

    file_part_3 = None
    for part in contents_3[0]["parts"]:
        if "file_data" in part:
            file_part_3 = part
            break

    assert file_part_3 is not None
    assert "media_resolution" in file_part_3, "Gemini 3 should have media_resolution"
    assert "video_metadata" in file_part_3, "Gemini 3 should have video_metadata"



def test_chunk_parser_handles_prompt_feedback_block():
    """Test chunk_parser correctly handles promptFeedback.blockReason"""
    from unittest.mock import Mock
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    # Arrange - mock a blocked response
    blocked_chunk = {
        "promptFeedback": {
            "blockReason": "PROHIBITED_CONTENT",
            "blockReasonMessage": "The prompt is blocked due to prohibited contents"
        },
        "responseId": "test_response_id",
        "modelVersion": "gemini-3-pro-preview"
    }

    logging_obj = Mock()
    logging_obj.optional_params = {}

    streaming_obj = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj
    )

    # Act
    result = streaming_obj.chunk_parser(blocked_chunk)

    # Assert
    assert result is not None, "Result should not be None"
    assert len(result.choices) == 1, "Should have exactly one choice"
    assert result.choices[0].finish_reason == "content_filter", f"finish_reason should be content_filter, got {result.choices[0].finish_reason}"
    assert result.choices[0].delta.content is None, "content should be None"


def test_chunk_parser_handles_prompt_feedback_safety_block():
    """Test chunk_parser handles different blockReason types (SAFETY)"""
    from unittest.mock import Mock
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    # Arrange - mock a SAFETY blocked response
    blocked_chunk = {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "The prompt is blocked due to safety concerns"
        },
        "responseId": "test_safety_response_id",
    }

    logging_obj = Mock()
    logging_obj.optional_params = {}

    streaming_obj = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj
    )

    # Act
    result = streaming_obj.chunk_parser(blocked_chunk)

    # Assert
    assert result is not None
    assert len(result.choices) == 1
    assert result.choices[0].finish_reason == "content_filter"


def test_chunk_parser_handles_prompt_feedback_block_with_usage():
    """Test chunk_parser correctly extracts usageMetadata when promptFeedback.blockReason is present"""
    from unittest.mock import Mock
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    # Arrange - 模拟一个包含 usageMetadata 的 blocked response
    blocked_chunk = {
        "promptFeedback": {
            "blockReason": "PROHIBITED_CONTENT",
            "blockReasonMessage": "The prompt is blocked due to prohibited contents"
        },
        "responseId": "test_response_id_with_usage",
        "modelVersion": "gemini-3-pro-preview",
        "usageMetadata": {
            "promptTokenCount": 8175,
            "candidatesTokenCount": 0,
            "totalTokenCount": 8175
        }
    }

    logging_obj = Mock()
    logging_obj.optional_params = {}

    streaming_obj = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        logging_obj=logging_obj
    )

    # Act
    result = streaming_obj.chunk_parser(blocked_chunk)

    # Assert - 验证 content_filter 响应和 usage 都被正确处理
    assert result is not None, "Result should not be None"
    assert len(result.choices) == 1, "Should have exactly one choice"
    assert result.choices[0].finish_reason == "content_filter", f"finish_reason should be content_filter, got {result.choices[0].finish_reason}"
    assert result.choices[0].delta.content is None, "content should be None"

    # 验证 usage 信息被正确提取
    assert hasattr(result, "usage"), "result should have usage attribute"
    assert result.usage is not None, "usage should not be None"
    assert result.usage.prompt_tokens == 8175, f"prompt_tokens should be 8175, got {result.usage.prompt_tokens}"
    assert result.usage.completion_tokens == 0, f"completion_tokens should be 0, got {result.usage.completion_tokens}"
    assert result.usage.total_tokens == 8175, f"total_tokens should be 8175, got {result.usage.total_tokens}"


def test_vertex_ai_traffic_type_preserved_in_hidden_params_streaming():
    """Test trafficType is preserved in _hidden_params for streaming."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        ModelResponseIterator,
    )

    chunk = {
        "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 200,
            "totalTokenCount": 300,
            "trafficType": "ON_DEMAND",
        },
    }

    iterator = ModelResponseIterator(
        streaming_response=[], sync_stream=True, logging_obj=MagicMock()
    )
    result = iterator.chunk_parser(chunk)

    assert result._hidden_params["provider_specific_fields"]["traffic_type"] == "ON_DEMAND"


def test_vertex_ai_traffic_type_preserved_in_hidden_params_non_streaming():
    """Test trafficType is preserved in _hidden_params for non-streaming."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    completion_response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 100,
            "totalTokenCount": 150,
            "trafficType": "PROVISIONED_THROUGHPUT",
        },
    }

    raw_response = MagicMock()
    raw_response.json.return_value = completion_response

    result = VertexGeminiConfig().transform_response(
        model="gemini-pro",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result._hidden_params["provider_specific_fields"]["traffic_type"] == "PROVISIONED_THROUGHPUT"


def test_vertex_ai_traffic_type_surfaced_in_responses_api():
    """Test trafficType is surfaced as provider_specific_fields in ResponsesAPIResponse."""
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    # Create a ModelResponse with provider_specific_fields in _hidden_params
    from litellm.types.utils import Choices, Message

    model_response = ModelResponse()
    model_response._hidden_params["provider_specific_fields"] = {"traffic_type": "ON_DEMAND"}
    model_response.choices = [
        Choices(
            message=Message(content="Hello", role="assistant"),
            finish_reason="stop",
            index=0,
        )
    ]

    responses_api_response = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="test",
        chat_completion_response=model_response,
        responses_api_request={},
    )

    assert responses_api_response.provider_specific_fields["traffic_type"] == "ON_DEMAND"

