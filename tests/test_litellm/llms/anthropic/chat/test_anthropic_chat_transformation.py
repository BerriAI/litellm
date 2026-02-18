import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.utils import PromptTokensDetailsWrapper, ServerToolUse


def test_response_format_transformation_unit_test():
    config = AnthropicConfig()

    response_format_json_schema = {
        "description": 'Progress report for the thinking process\n\nThis model represents a snapshot of the agent\'s current progress during\nthe thinking process, providing a brief description of the current activity.\n\nAttributes:\n    agent_doing: Brief description of what the agent is currently doing.\n                Should be kept under 10 words. Example: "Learning about home automation"',
        "properties": {"agent_doing": {"title": "Agent Doing", "type": "string"}},
        "required": ["agent_doing"],
        "title": "ThinkingStep",
        "type": "object",
        "additionalProperties": False,
    }

    result = config._create_json_tool_call_for_response_format(
        json_schema=response_format_json_schema
    )

    assert result["input_schema"]["properties"] == {
        "agent_doing": {"title": "Agent Doing", "type": "string"}
    }
    print(result)


def test_calculate_usage():
    """
    Do not include cache_creation_input_tokens in the prompt_tokens

    Fixes https://github.com/BerriAI/litellm/issues/9812
    """
    config = AnthropicConfig()

    usage_object = {
        "input_tokens": 3,
        "cache_creation_input_tokens": 12304,
        "cache_read_input_tokens": 0,
        "output_tokens": 550,
    }
    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    assert usage.prompt_tokens == 12307
    assert usage.completion_tokens == 550
    assert usage.total_tokens == 12307 + 550
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage.prompt_tokens_details.cache_creation_tokens == 12304
    assert usage._cache_creation_input_tokens == 12304
    assert usage._cache_read_input_tokens == 0


@pytest.mark.parametrize(
    "usage_object,expected_usage",
    [
        [
            {
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "input_tokens": None,
                "output_tokens": 43,
                "server_tool_use": None,
            },
            {
                "prompt_tokens": 0,
                "completion_tokens": 43,
                "total_tokens": 43,
                "_cache_creation_input_tokens": 0,
                "_cache_read_input_tokens": 0,
            },
        ],
        [
            {
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
                "input_tokens": 1,
                "output_tokens": None,
                "server_tool_use": None,
            },
            {
                "prompt_tokens": 1 + 200 + 100,
                "completion_tokens": 0,
                "total_tokens": 1 + 200 + 100,
                "_cache_creation_input_tokens": 100,
                "_cache_read_input_tokens": 200,
            },
        ],
        [
            {"server_tool_use": {"web_search_requests": 10}},
            {"server_tool_use": ServerToolUse(web_search_requests=10)},
        ],
    ],
)
def test_calculate_usage_nulls(usage_object, expected_usage):
    """
    Correctly deal with null values in usage object

    Fixes https://github.com/BerriAI/litellm/issues/11920
    """
    config = AnthropicConfig()

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    for k, v in expected_usage.items():
        assert hasattr(usage, k)
        assert getattr(usage, k) == v


@pytest.mark.parametrize(
    "usage_object",
    [{"server_tool_use": {"web_search_requests": None}}, {"server_tool_use": None}],
)
def test_calculate_usage_server_tool_null(usage_object):
    """
    Correctly deal with null values in usage object

    Fixes https://github.com/BerriAI/litellm/issues/11920
    """
    config = AnthropicConfig()

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    assert not hasattr(usage, "server_tool_use")


def test_extract_response_content_with_citations():
    config = AnthropicConfig()

    completion_response = {
        "id": "msg_01XrAv7gc5tQNDuoADra7vB4",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {"type": "text", "text": "According to the documents, "},
            {
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "The grass is green. ",
                        "document_index": 0,
                        "document_title": "My Document",
                        "start_char_index": 0,
                        "end_char_index": 20,
                    }
                ],
                "type": "text",
                "text": "the grass is green",
            },
            {"type": "text", "text": " and "},
            {
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "The sky is blue.",
                        "document_index": 0,
                        "document_title": "My Document",
                        "start_char_index": 20,
                        "end_char_index": 36,
                    }
                ],
                "type": "text",
                "text": "the sky is blue",
            },
            {"type": "text", "text": "."},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 610,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 51,
        },
    }

    _, citations, _, _, _, _, _, _ = config.extract_response_content(completion_response)
    assert citations == [
        [
            {
                "type": "char_location",
                "cited_text": "The grass is green. ",
                "document_index": 0,
                "document_title": "My Document",
                "start_char_index": 0,
                "end_char_index": 20,
                "supported_text": "the grass is green",
            },
        ],
        [
            {
                "type": "char_location",
                "cited_text": "The sky is blue.",
                "document_index": 0,
                "document_title": "My Document",
                "start_char_index": 20,
                "end_char_index": 36,
                "supported_text": "the sky is blue",
            },
        ],
    ]


def test_map_tool_helper():
    config = AnthropicConfig()

    tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}

    result, _ = config._map_tool_helper(tool)
    assert result is not None
    assert result["name"] == "web_search"
    assert result["max_uses"] == 5


def test_server_tool_use_usage():
    config = AnthropicConfig()

    usage_object = {
        "input_tokens": 15956,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 567,
        "server_tool_use": {"web_search_requests": 1},
    }
    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    assert usage.server_tool_use.web_search_requests == 1


def test_web_search_tool_transformation():
    from litellm.types.llms.openai import OpenAIWebSearchOptions

    config = AnthropicConfig()

    openai_web_search_options = OpenAIWebSearchOptions(
        user_location={
            "type": "approximate",
            "approximate": {
                "city": "San Francisco",
            },
        }
    )

    anthropic_web_search_tool = config.map_web_search_tool(openai_web_search_options)
    assert anthropic_web_search_tool is not None
    assert anthropic_web_search_tool["user_location"] is not None
    assert anthropic_web_search_tool["user_location"]["type"] == "approximate"
    assert anthropic_web_search_tool["user_location"]["city"] == "San Francisco"


@pytest.mark.parametrize(
    "search_context_size, expected_max_uses", [("low", 1), ("medium", 5), ("high", 10)]
)
def test_web_search_tool_transformation_with_search_context_size(
    search_context_size, expected_max_uses
):
    from litellm.types.llms.openai import OpenAIWebSearchOptions

    config = AnthropicConfig()

    openai_web_search_options = OpenAIWebSearchOptions(
        user_location={
            "type": "approximate",
            "approximate": {
                "city": "San Francisco",
            },
        },
        search_context_size=search_context_size,
    )

    anthropic_web_search_tool = config.map_web_search_tool(openai_web_search_options)
    assert anthropic_web_search_tool is not None
    assert anthropic_web_search_tool["user_location"] is not None
    assert anthropic_web_search_tool["user_location"]["type"] == "approximate"
    assert anthropic_web_search_tool["user_location"]["city"] == "San Francisco"
    assert anthropic_web_search_tool["max_uses"] == expected_max_uses


def test_web_search_tool_result_extraction():
    """
    Test that web_search_tool_result blocks are correctly extracted and preserved.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    - web_search_tool_result was being dropped entirely from the response
    - This caused multi-turn conversations to fail because the web search results
      were not available for reconstruction
    """
    config = AnthropicConfig()

    # Simulating actual Anthropic API response with web search
    completion_response = {
        "id": "msg_web_search_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "web_search",
                "input": {"query": "average weight african elephant kg"}
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01ABC123",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/elephants",
                        "title": "African Elephant Facts",
                        "encrypted_content": "encrypted_data_here",
                        "page_age": "2024-01-15",
                        "snippet": "Adult African elephants weigh between 4,000-6,000 kg..."
                    }
                ]
            },
            {
                "type": "text",
                "text": "Based on my search, African elephants weigh around 5,000 kg."
            },
            {
                "type": "tool_use",
                "id": "toolu_01XYZ789",
                "name": "add_numbers",
                "input": {"a": 5000, "b": 100}
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": 1}
        }
    }

    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )

    # Verify text extraction
    assert "Based on my search" in text
    assert "5,000 kg" in text

    # Verify tool calls (should have both server_tool_use and tool_use)
    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] == "srvtoolu_01ABC123"
    assert tool_calls[0]["function"]["name"] == "web_search"
    assert tool_calls[1]["id"] == "toolu_01XYZ789"
    assert tool_calls[1]["function"]["name"] == "add_numbers"

    # Verify web_search_results is extracted (THIS WAS THE BUG - it was None before the fix)
    assert web_search_results is not None
    assert len(web_search_results) == 1
    assert web_search_results[0]["type"] == "web_search_tool_result"
    assert web_search_results[0]["tool_use_id"] == "srvtoolu_01ABC123"
    assert len(web_search_results[0]["content"]) == 1
    assert web_search_results[0]["content"][0]["url"] == "https://example.com/elephants"
    assert web_search_results[0]["content"][0]["title"] == "African Elephant Facts"


def test_web_search_tool_result_in_provider_specific_fields():
    """
    Test that web_search_results is included in provider_specific_fields.

    This ensures users can access the web search results via:
    response.choices[0].message.provider_specific_fields["web_search_results"]
    """
    import httpx

    from litellm.types.utils import ModelResponse

    config = AnthropicConfig()

    completion_response = {
        "id": "msg_web_search_provider_fields",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_provider_test",
                "name": "web_search",
                "input": {"query": "test query"}
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_provider_test",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/test",
                        "title": "Test Result",
                        "snippet": "Test snippet content"
                    }
                ]
            },
            {
                "type": "text",
                "text": "Here is the result."
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 25,
            "server_tool_use": {"web_search_requests": 1}
        }
    }

    raw_response = httpx.Response(status_code=200, headers={})
    model_response = ModelResponse()

    result = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )

    # Verify web_search_results is in provider_specific_fields
    provider_fields = result.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert "web_search_results" in provider_fields
    assert len(provider_fields["web_search_results"]) == 1
    assert provider_fields["web_search_results"][0]["type"] == "web_search_tool_result"
    assert provider_fields["web_search_results"][0]["tool_use_id"] == "srvtoolu_provider_test"


def test_multiple_web_search_tool_results():
    """
    Test that multiple web_search_tool_result blocks are all extracted.
    """
    config = AnthropicConfig()

    completion_response = {
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_search1",
                "name": "web_search",
                "input": {"query": "african elephant weight"}
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_search1",
                "content": [{"type": "web_search_result", "url": "https://example1.com", "title": "Result 1", "snippet": "First result"}]
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_search2",
                "name": "web_search",
                "input": {"query": "asian elephant weight"}
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_search2",
                "content": [{"type": "web_search_result", "url": "https://example2.com", "title": "Result 2", "snippet": "Second result"}]
            },
            {
                "type": "text",
                "text": "Found information about both elephants."
            }
        ]
    }

    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )

    # Verify both web_search_tool_results are extracted
    assert web_search_results is not None
    assert len(web_search_results) == 2
    assert web_search_results[0]["tool_use_id"] == "srvtoolu_search1"
    assert web_search_results[1]["tool_use_id"] == "srvtoolu_search2"


def test_add_code_execution_tool():
    config = AnthropicConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this sheet?"},
                {
                    "type": "container_upload",
                    "file_id": "file_011CPd1KVEsbD8MjfZSwBd1u",
                },
            ],
        }
    ]
    tools = []
    tools = config.add_code_execution_tool(messages=messages, tools=tools)
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["type"] == "code_execution_20250522"


def test_map_tool_choice():
    config = AnthropicConfig()

    tool_choice = "none"
    result = config._map_tool_choice(tool_choice=tool_choice, parallel_tool_use=True)
    assert result is not None
    assert result["type"] == "none"
    print(result)


def test_map_tool_choice_string_auto():
    """Test that string 'auto' maps to Anthropic type='auto'"""
    config = AnthropicConfig()
    result = config._map_tool_choice(tool_choice="auto", parallel_tool_use=None)
    assert result is not None
    assert result["type"] == "auto"


def test_map_tool_choice_string_required():
    """Test that string 'required' maps to Anthropic type='any'"""
    config = AnthropicConfig()
    result = config._map_tool_choice(tool_choice="required", parallel_tool_use=None)
    assert result is not None
    assert result["type"] == "any"


def test_map_tool_choice_dict_type_function_with_name():
    """
    Test that dict {"type": "function", "function": {"name": "my_tool"}}
    (OpenAI format) maps to Anthropic type='tool' with name.
    """
    config = AnthropicConfig()
    result = config._map_tool_choice(
        tool_choice={"type": "function", "function": {"name": "my_tool"}},
        parallel_tool_use=None,
    )
    assert result is not None
    assert result["type"] == "tool"
    assert result["name"] == "my_tool"


def test_map_tool_choice_dict_type_auto():
    """
    Test that dict {"type": "auto"} maps to Anthropic type='auto'.
    This handles Cursor's format for tool_choice.
    """
    config = AnthropicConfig()
    result = config._map_tool_choice(
        tool_choice={"type": "auto"},
        parallel_tool_use=None,
    )
    assert result is not None
    assert result["type"] == "auto"


def test_map_tool_choice_dict_type_required():
    """
    Test that dict {"type": "required"} maps to Anthropic type='any'.
    """
    config = AnthropicConfig()
    result = config._map_tool_choice(
        tool_choice={"type": "required"},
        parallel_tool_use=None,
    )
    assert result is not None
    assert result["type"] == "any"


def test_map_tool_choice_dict_type_none():
    """
    Test that dict {"type": "none"} maps to Anthropic type='none'.
    """
    config = AnthropicConfig()
    result = config._map_tool_choice(
        tool_choice={"type": "none"},
        parallel_tool_use=None,
    )
    assert result is not None
    assert result["type"] == "none"


def test_map_tool_choice_dict_type_function_without_name():
    """
    Test that dict {"type": "function"} without name is handled gracefully.
    Should return None since there's no valid tool name.
    """
    config = AnthropicConfig()
    result = config._map_tool_choice(
        tool_choice={"type": "function"},
        parallel_tool_use=None,
    )
    assert result is None


def test_transform_response_with_prefix_prompt():
    import httpx

    from litellm.types.utils import ModelResponse

    config = AnthropicConfig()

    completion_response = {
        "id": "msg_01XrAv7gc5tQNDuoADra7vB4",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": " The grass is green."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 610,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 51,
        },
    }

    raw_response = httpx.Response(
        status_code=200,
        headers={},
    )

    model_response = ModelResponse()

    result = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt="You are a helpful assistant.",
    )

    assert result is not None
    assert (
        result.choices[0].message.content
        == "You are a helpful assistant. The grass is green."
    )


def test_get_supported_params_thinking():
    config = AnthropicConfig()
    params = config.get_supported_openai_params(model="claude-sonnet-4-20250514")
    assert "thinking" in params


def test_anthropic_memory_tool_auto_adds_beta_header():
    """
    Tests that LiteLLM automatically adds the required 'anthropic-beta' header
    when the memory tool is present, and the user has NOT provided a beta header.
    """

    config = AnthropicConfig()
    memory_tool = [{"type": "memory_20250818", "name": "memory"}]
    messages = [{"role": "user", "content": "Remember this."}]

    headers = {}
    optional_params = {"tools": memory_tool}

    config.transform_request(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers=headers,
    )

    assert "anthropic-beta" in headers
    assert headers["anthropic-beta"] == "context-management-2025-06-27"


def _sample_context_management_payload():
    return {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 5000},
                "exclude_tools": ["web_search"],
                "clear_tool_inputs": False,
            }
        ]
    }


def test_anthropic_messages_validate_adds_beta_header():
    config = AnthropicMessagesConfig()
    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        optional_params={"context_management": _sample_context_management_payload()},
        litellm_params={},
    )
    assert headers["anthropic-beta"] == "context-management-2025-06-27"


def test_anthropic_messages_transform_includes_context_management():
    config = AnthropicMessagesConfig()
    payload = _sample_context_management_payload()
    headers = {
        "x-api-key": "test",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        anthropic_messages_optional_request_params={
            "max_tokens": 512,
            "context_management": payload,
        },
        litellm_params={},
        headers=headers,
    )
    assert result["context_management"] == payload


def test_anthropic_chat_headers_add_context_management_beta():
    config = AnthropicConfig()
    headers = config.update_headers_with_optional_anthropic_beta(
        headers={},
        optional_params={"context_management": _sample_context_management_payload()},
    )
    assert headers["anthropic-beta"] == "context-management-2025-06-27"


def test_anthropic_beta_header_merging_with_output_format():
    """
    Test that anthropic-beta headers from extra_headers are merged with
    output_format beta headers instead of being overridden.
    
    This is a regression test for: https://github.com/BerriAI/litellm/issues/...
    When using response_format with a Pydantic model AND extra_headers with
    anthropic-beta (e.g., for context-1m extension), both beta headers should
    be present in the final request.
    """
    config = AnthropicConfig()
    
    # Simulate headers that already have the context-1m beta header from extra_headers
    headers = {"anthropic-beta": "context-1m-2025-08-07"}
    
    # Simulate output_format being set (happens when using response_format with Sonnet 4.5)
    optional_params = {
        "output_format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {}}
        }
    }
    
    result_headers = config.update_headers_with_optional_anthropic_beta(
        headers, optional_params
    )
    
    # Both beta headers should be present
    beta_value = result_headers["anthropic-beta"]
    assert "context-1m-2025-08-07" in beta_value, \
        f"User's context-1m beta header missing from: {beta_value}"
    assert "structured-outputs-2025-11-13" in beta_value, \
        f"Structured output beta header missing from: {beta_value}"


def test_anthropic_beta_header_merging_with_multiple_features():
    """
    Test that multiple beta headers can be merged when using multiple features.
    """
    config = AnthropicConfig()
    
    # Start with a user-provided beta header
    headers = {"anthropic-beta": "context-1m-2025-08-07"}
    
    # Use multiple features that require beta headers
    optional_params = {
        "output_format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {}}
        },
        "context_management": _sample_context_management_payload(),
        "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]
    }
    
    result_headers = config.update_headers_with_optional_anthropic_beta(
        headers, optional_params
    )
    
    beta_value = result_headers["anthropic-beta"]
    
    # All beta headers should be present
    assert "context-1m-2025-08-07" in beta_value
    assert "structured-outputs-2025-11-13" in beta_value
    assert "context-management-2025-06-27" in beta_value
    assert "web-fetch-2025-09-10" in beta_value


def test_anthropic_chat_transform_request_includes_context_management():
    config = AnthropicConfig()
    headers = {}
    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "context_management": _sample_context_management_payload(),
            "max_tokens": 256,
        },
        litellm_params={},
        headers=headers,
    )
    assert result["context_management"] == _sample_context_management_payload()


def test_anthropic_structured_output_beta_header():
    from litellm.types.utils import CallTypes
    from litellm.utils import return_raw_request

    response = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "final_output",
                    "strict": True,
                    "schema": {
                        "description": 'Progress report for the thinking process\n\nThis model represents a snapshot of the agent\'s current progress during\nthe thinking process, providing a brief description of the current activity.\n\nAttributes:\n    agent_doing: Brief description of what the agent is currently doing.\n                Should be kept under 10 words. Example: "Learning about home automation"',
                        "properties": {
                            "agent_doing": {"title": "Agent Doing", "type": "string"}
                        },
                        "required": ["agent_doing"],
                        "title": "ThinkingStep",
                        "type": "object",
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response is not None
    print(f"response: {response}")
    print(f"raw_request_headers: {response['raw_request_headers']}")
    assert (
        "structured-outputs-2025-11-13"
        in response["raw_request_headers"]["anthropic-beta"]
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-opus-4-6-20250918",
        "claude-opus-4.6-20250918",
        "claude-opus-4-5-20251101",
        "claude-opus-4.5-20251101",
    ],
)
def test_opus_uses_native_structured_output(model_name):
    """
    Test that Opus 4.5 and 4.6 models use native Anthropic structured outputs
    (output_format) rather than the tool-based workaround.
    """
    config = AnthropicConfig()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    }

    optional_params = config.map_openai_params(
        non_default_params={"response_format": response_format},
        optional_params={},
        model=model_name,
        drop_params=False,
    )

    # Should use output_format (native structured outputs)
    assert "output_format" in optional_params
    assert optional_params["output_format"]["type"] == "json_schema"

    # Should NOT create a tool-based workaround
    assert "tools" not in optional_params
    assert "tool_choice" not in optional_params

    # Should set json_mode
    assert optional_params.get("json_mode") is True


def test_non_structured_output_model_uses_tool_workaround():
    """
    Test that models NOT in the native structured output list still use the
    tool-based workaround for response_format.
    """
    config = AnthropicConfig()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    }

    optional_params = config.map_openai_params(
        non_default_params={"response_format": response_format},
        optional_params={},
        model="claude-3-5-sonnet-20241022",
        drop_params=False,
    )

    # Should NOT use output_format
    assert "output_format" not in optional_params

    # Should use tool-based workaround
    assert "tools" in optional_params
    assert "tool_choice" in optional_params


# ============ Tool Search Tests ============


def test_tool_search_regex_detection():
    """Test that tool search regex tools are properly detected"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    config = AnthropicModelInfo()
    
    # Test with tool search regex tool
    tools = [
        {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex"
        }
    ]
    assert config.is_tool_search_used(tools) is True
    
    # Test without tool search
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather"}
        }
    ]
    assert config.is_tool_search_used(tools) is False


def test_tool_search_bm25_detection():
    """Test that tool search BM25 tools are properly detected"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    config = AnthropicModelInfo()
    
    # Test with tool search BM25 tool
    tools = [
        {
            "type": "tool_search_tool_bm25_20251119",
            "name": "tool_search_tool_bm25"
        }
    ]
    assert config.is_tool_search_used(tools) is True


def test_tool_search_beta_header():
    """Test that tool search beta header is automatically added"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    config = AnthropicModelInfo()
    
    headers = config.get_anthropic_headers(
        api_key="test-key",
        tool_search_used=True,
    )
    
    assert "anthropic-beta" in headers
    assert "advanced-tool-use-2025-11-20" in headers["anthropic-beta"]


def test_tool_search_regex_mapping():
    """Test that tool search regex tools are properly mapped"""
    config = AnthropicConfig()
    
    tool = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex"
    }
    
    mapped_tool, mcp_server = config._map_tool_helper(tool)
    
    assert mapped_tool is not None
    assert mapped_tool["type"] == "tool_search_tool_regex_20251119"
    assert mapped_tool["name"] == "tool_search_tool_regex"
    assert mcp_server is None


def test_tool_search_bm25_mapping():
    """Test that tool search BM25 tools are properly mapped"""
    config = AnthropicConfig()
    
    tool = {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25"
    }
    
    mapped_tool, mcp_server = config._map_tool_helper(tool)
    
    assert mapped_tool is not None
    assert mapped_tool["type"] == "tool_search_tool_bm25_20251119"
    assert mapped_tool["name"] == "tool_search_tool_bm25"
    assert mcp_server is None


def test_deferred_tools_separation():
    """Test that deferred and non-deferred tools are properly separated"""
    config = AnthropicConfig()
    
    tools = [
        {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex"
        },
        {
            "type": "function",
            "function": {"name": "get_weather"},
            "defer_loading": True
        },
        {
            "type": "function",
            "function": {"name": "search_files"},
            "defer_loading": False
        }
    ]
    
    non_deferred, deferred = config._separate_deferred_tools(tools)
    
    assert len(non_deferred) == 2  # tool_search and search_files
    assert len(deferred) == 1  # get_weather


def test_server_tool_use_in_response():
    """Test that server_tool_use blocks are parsed correctly"""
    config = AnthropicConfig()
    
    completion_response = {
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC123",
                "name": "tool_search_tool_regex",
                "input": {"query": "weather"}
            }
        ]
    }
    
    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "srvtoolu_01ABC123"
    assert tool_calls[0]["function"]["name"] == "tool_search_tool_regex"
    assert web_search_results is None


def test_tool_search_usage_tracking():
    """Test that tool_search_requests are tracked in usage"""
    config = AnthropicConfig()
    
    usage_object = {
        "input_tokens": 100,
        "output_tokens": 50,
        "server_tool_use": {
            "tool_search_requests": 2
        }
    }
    
    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    
    assert usage.server_tool_use is not None
    assert usage.server_tool_use.tool_search_requests == 2


def test_tool_reference_expansion():
    """Test that tool_reference blocks are expanded correctly"""
    config = AnthropicConfig()
    
    deferred_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather"
            }
        }
    ]
    
    content = [
        {"type": "text", "text": "I'll search for tools"},
        {"type": "tool_reference", "tool_name": "get_weather"}
    ]
    
    expanded = config._expand_tool_references(content, deferred_tools)
    
    assert len(expanded) == 2
    assert expanded[0]["type"] == "text"
    assert expanded[1]["type"] == "function"
    assert expanded[1]["function"]["name"] == "get_weather"


def test_defer_loading_preserved_in_transformation():
    """Test that defer_loading parameter is preserved when transforming tools"""
    config = AnthropicConfig()
    
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        "defer_loading": True
    }
    
    mapped_tool, mcp_server = config._map_tool_helper(tool)
    
    assert mapped_tool is not None
    assert mapped_tool.get("defer_loading") is True
    assert mapped_tool["name"] == "get_weather"
    assert mcp_server is None


def test_tool_search_complete_response_parsing():
    """Test parsing a complete tool search response with server_tool_use and tool_search_tool_result blocks"""
    config = AnthropicConfig()
    
    # Simulating actual Anthropic API response with tool search
    completion_response = {
        "content": [
            {
                "type": "text",
                "text": "I'll search for weather-related tools that can help you."
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_015i6aVA2niwzv4RG4DtnxDJ",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "weather", "limit": 5},
                "caller": {"type": "direct"}
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_015i6aVA2niwzv4RG4DtnxDJ",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}]
                }
            },
            {
                "type": "text",
                "text": "Great! I found a weather tool."
            },
            {
                "type": "tool_use",
                "id": "toolu_01CrCNx4ntSaeeV9iArT4JfQ",
                "name": "get_weather",
                "input": {"location": "San Francisco"}
            }
        ],
        "usage": {
            "input_tokens": 1639,
            "output_tokens": 170,
            "server_tool_use": {"web_search_requests": 0}
        }
    }
    
    # Extract content
    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )

    # Verify text extraction (should concatenate both text blocks)
    assert "I'll search for weather-related tools" in text
    assert "Great! I found a weather tool" in text

    # Verify tool calls (should have both server_tool_use and tool_use)
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "tool_search_tool_regex"
    assert tool_calls[1]["function"]["name"] == "get_weather"

    # Verify web_search_results is None (this response has tool_search, not web_search)
    assert web_search_results is None
    
    # Verify usage calculation counts tool_search_requests from content
    usage = config.calculate_usage(
        usage_object=completion_response["usage"],
        reasoning_content=None,
        completion_response=completion_response
    )
    
    assert usage.server_tool_use is not None
    assert usage.server_tool_use.web_search_requests == 0
    assert usage.server_tool_use.tool_search_requests == 1  # Counted from server_tool_use blocks


def test_allowed_callers_field_preservation():
    """Test that allowed_callers field is preserved during tool transformation."""
    config = AnthropicConfig()
    
    # Test with top-level allowed_callers
    tool_with_allowed_callers = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        },
        "allowed_callers": ["code_execution_20250825"]
    }
    
    transformed_tool, _ = config._map_tool_helper(tool_with_allowed_callers)
    assert transformed_tool is not None
    assert "allowed_callers" in transformed_tool
    assert transformed_tool["allowed_callers"] == ["code_execution_20250825"]


def test_programmatic_tool_calling_beta_header():
    """Test that beta header is automatically added when programmatic tool calling is detected."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    model_info = AnthropicModelInfo()
    
    # Test detection with allowed_callers
    tools = [
        {
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute a SQL query",
                "parameters": {"type": "object", "properties": {}}
            },
            "allowed_callers": ["code_execution_20250825"]
        }
    ]
    
    is_programmatic = model_info.is_programmatic_tool_calling_used(tools)
    assert is_programmatic is True
    
    # Test header generation
    headers = model_info.get_anthropic_headers(
        api_key="test-key",
        programmatic_tool_calling_used=True
    )
    
    assert "anthropic-beta" in headers
    assert "advanced-tool-use-2025-11-20" in headers["anthropic-beta"]


def test_caller_field_in_response():
    """Test that caller field is correctly parsed from tool_use blocks."""
    config = AnthropicConfig()
    
    # Mock response with programmatic tool call
    completion_response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll query the database."
            },
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "query_database",
                "input": {"sql": "SELECT * FROM users"},
                "caller": {
                    "type": "code_execution_20250825",
                    "tool_id": "srvtoolu_abc"
                }
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    }
    
    text, citations, thinking, reasoning, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(completion_response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "toolu_123"
    assert tool_calls[0]["function"]["name"] == "query_database"
    assert "caller" in tool_calls[0]
    assert tool_calls[0]["caller"]["type"] == "code_execution_20250825"
    assert tool_calls[0]["caller"]["tool_id"] == "srvtoolu_abc"
    assert web_search_results is None


def test_code_execution_20250825_tool_type():
    """Test that code_execution_20250825 tool type is handled correctly."""
    config = AnthropicConfig()
    
    tool = {
        "type": "code_execution_20250825",
        "name": "code_execution"
    }
    
    transformed_tool, _ = config._map_tool_helper(tool)
    assert transformed_tool is not None
    assert transformed_tool["type"] == "code_execution_20250825"
    assert transformed_tool["name"] == "code_execution"


def test_allowed_callers_in_function_field():
    """Test that allowed_callers in function field is also preserved."""
    config = AnthropicConfig()
    
    # Test with function.allowed_callers
    tool = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            },
            "allowed_callers": ["code_execution_20250825"]
        }
    }
    
    transformed_tool, _ = config._map_tool_helper(tool)
    assert transformed_tool is not None
    assert "allowed_callers" in transformed_tool
    assert transformed_tool["allowed_callers"] == ["code_execution_20250825"]


def test_input_examples_field_preservation():
    """Test that input_examples field is preserved during tool transformation."""
    config = AnthropicConfig()
    
    # Test with top-level input_examples
    tool_with_examples = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        },
        "input_examples": [
            {"location": "San Francisco, CA", "unit": "fahrenheit"},
            {"location": "Tokyo, Japan", "unit": "celsius"}
        ]
    }
    
    transformed_tool, _ = config._map_tool_helper(tool_with_examples)
    assert transformed_tool is not None
    assert "input_examples" in transformed_tool
    assert len(transformed_tool["input_examples"]) == 2
    assert transformed_tool["input_examples"][0]["location"] == "San Francisco, CA"


def test_input_examples_beta_header():
    """Test that beta header is automatically added when input_examples is detected."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    model_info = AnthropicModelInfo()
    
    # Test detection with input_examples
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {"type": "object", "properties": {}}
            },
            "input_examples": [
                {"location": "San Francisco, CA"}
            ]
        }
    ]
    
    is_examples_used = model_info.is_input_examples_used(tools)
    assert is_examples_used is True
    
    # Test header generation
    headers = model_info.get_anthropic_headers(
        api_key="test-key",
        input_examples_used=True
    )
    
    assert "anthropic-beta" in headers
    assert "advanced-tool-use-2025-11-20" in headers["anthropic-beta"]


def test_input_examples_in_function_field():
    """Test that input_examples in function field is also preserved."""
    config = AnthropicConfig()
    
    # Test with function.input_examples
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            },
            "input_examples": [
                {"location": "Paris, France"},
                {"location": "London, UK"}
            ]
        }
    }
    
    transformed_tool, _ = config._map_tool_helper(tool)
    assert transformed_tool is not None
    assert "input_examples" in transformed_tool
    assert len(transformed_tool["input_examples"]) == 2


def test_input_examples_with_other_features():
    """Test that input_examples works alongside other tool features."""
    config = AnthropicConfig()
    
    # Tool with input_examples, defer_loading, and allowed_callers
    tool = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        },
        "input_examples": [
            {"sql": "SELECT * FROM users WHERE id = 1"}
        ],
        "defer_loading": True,
        "allowed_callers": ["code_execution_20250825"]
    }
    
    transformed_tool, _ = config._map_tool_helper(tool)
    assert transformed_tool is not None
    assert "input_examples" in transformed_tool
    assert "defer_loading" in transformed_tool
    assert "allowed_callers" in transformed_tool
    assert transformed_tool["defer_loading"] is True
    assert transformed_tool["allowed_callers"] == ["code_execution_20250825"]


def test_input_examples_empty_list_not_added():
    """Test that empty input_examples list is not added to transformed tool."""
    config = AnthropicConfig()
    
    # Tool with empty input_examples
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        "input_examples": []
    }
    
    transformed_tool, _ = config._map_tool_helper(tool)
    assert transformed_tool is not None
    # Empty list should not be added
    assert "input_examples" not in transformed_tool or len(transformed_tool.get("input_examples", [])) == 0


# ============ Effort Parameter Tests ============


def test_effort_output_config_preservation():
    """Test that output_config with effort is preserved in transformation."""
    config = AnthropicConfig()
    
    messages = [{"role": "user", "content": "Analyze this code"}]
    optional_params = {
        "output_config": {
            "effort": "medium"
        }
    }
    
    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    assert "output_config" in result
    assert result["output_config"]["effort"] == "medium"


def test_effort_beta_header_injection():
    """Test that effort beta header is automatically added when output_config is detected."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    
    model_info = AnthropicModelInfo()
    
    # Test with effort parameter
    optional_params = {
        "output_config": {
            "effort": "low"
        }
    }
    
    effort_used = model_info.is_effort_used(optional_params=optional_params)
    assert effort_used is True
    
    headers = model_info.get_anthropic_headers(
        api_key="test-key",
        effort_used=effort_used
    )
    
    assert "anthropic-beta" in headers
    assert "effort-2025-11-24" in headers["anthropic-beta"]


def test_effort_validation():
    """Test that only valid effort values are accepted."""
    config = AnthropicConfig()
    
    messages = [{"role": "user", "content": "Test"}]
    
    # Valid values should work
    for effort in ["high", "medium", "low"]:
        optional_params = {"output_config": {"effort": effort}}
        result = config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        assert result["output_config"]["effort"] == effort
    
    # Invalid value should raise error
    with pytest.raises(ValueError, match="Invalid effort value"):
        optional_params = {"output_config": {"effort": "invalid"}}
        config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )


def test_effort_with_claude_opus_45():
    """Test effort parameter works with Claude Opus 4.5 model."""
    config = AnthropicConfig()
    
    messages = [{"role": "user", "content": "Complex analysis task"}]
    optional_params = {
        "output_config": {
            "effort": "high"
        }
    }
    
    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    assert "output_config" in result
    assert result["output_config"]["effort"] == "high"
    assert result["model"] == "claude-opus-4-5-20251101"


def test_effort_validation_with_opus_46():
    """Test that all four effort levels are accepted for Claude Opus 4.6."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Test"}]

    for effort in ["high", "medium", "low", "max"]:
        optional_params = {"output_config": {"effort": effort}}
        result = config.transform_request(
            model="claude-opus-4-6-20260205",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        assert result["output_config"]["effort"] == effort


def test_max_effort_rejected_for_opus_45():
    """Test that effort='max' is rejected when using Claude Opus 4.5."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(ValueError, match="effort='max' is only supported by Claude Opus 4.6"):
        optional_params = {"output_config": {"effort": "max"}}
        config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )


def test_effort_with_other_features():
    """Test effort works alongside other features (thinking, tools)."""
    config = AnthropicConfig()
    
    messages = [{"role": "user", "content": "Use tools efficiently"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_data",
                "description": "Get data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    optional_params = {
        "output_config": {
            "effort": "low"
        },
        "tools": tools,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1000
        }
    }
    
    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )

    # Verify all features are present
    assert "output_config" in result
    assert result["output_config"]["effort"] == "low"
    assert "tools" in result
    assert len(result["tools"]) > 0
    assert "thinking" in result


def test_translate_system_message_skips_empty_string_content():
    """
    Test that translate_system_message skips system messages with empty string content.

    Fixes: Vertex AI Anthropic API error "messages: text content blocks must be non-empty"
    """
    config = AnthropicConfig()

    # Test empty string content - should not produce any anthropic system message content
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello"},
    ]

    result = config.translate_system_message(messages)

    # Empty system message should produce no anthropic content blocks
    assert len(result) == 0


def test_translate_system_message_skips_empty_list_content():
    """
    Test that translate_system_message skips empty text blocks in list content.

    Fixes: Vertex AI Anthropic API error "messages: text content blocks must be non-empty"
    """
    config = AnthropicConfig()

    # Test list content with empty text block
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Valid content"},
            {"type": "text", "text": ""},
        ]},
        {"role": "user", "content": "Hello"},
    ]

    result = config.translate_system_message(messages)

    # Only non-empty text blocks should be included
    assert len(result) == 1
    assert result[0]["text"] == "Valid content"


def test_translate_system_message_preserves_valid_content():
    """
    Test that translate_system_message preserves valid system message content.
    """
    config = AnthropicConfig()

    # Test valid string content
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    result = config.translate_system_message(messages)

    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "You are a helpful assistant."


def test_translate_system_message_preserves_cache_control():
    """
    Test that translate_system_message preserves cache_control on valid content.
    """
    config = AnthropicConfig()

    # Test list content with cache_control
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "Cached content", "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": "Hello"},
    ]

    result = config.translate_system_message(messages)

    assert len(result) == 1
    assert result[0]["text"] == "Cached content"
    assert result[0]["cache_control"] == {"type": "ephemeral"}


# ============ Dynamic max_tokens Tests ============


def test_get_max_tokens_for_model_claude_3():
    """
    Test that get_max_tokens_for_model returns correct value for Claude 3 models.
    Claude 3 models have max_output_tokens of 4096.
    """
    config = AnthropicConfig()

    # Claude 3 Sonnet should return 4096
    max_tokens = config.get_max_tokens_for_model("claude-3-sonnet-20240229")
    assert max_tokens == 4096


def test_get_max_tokens_for_model_claude_35():
    """
    Test that get_max_tokens_for_model returns correct value for Claude 3.5 models.
    Claude 3.5 models have max_output_tokens of 8192.

    Fixes: https://github.com/BerriAI/litellm/issues/8835
    """
    config = AnthropicConfig()

    # Claude 3.5 Sonnet should return 8192
    max_tokens = config.get_max_tokens_for_model("claude-3-5-sonnet-20241022")
    assert max_tokens == 8192


def test_get_max_tokens_for_model_claude_37():
    """
    Test that get_max_tokens_for_model returns correct value for Claude 3.7 models.
    Claude 3.7 Sonnet has max_output_tokens of 64000 by default.
    128K output requires the beta header 'output-128k-2025-02-19'.

    Fixes: https://github.com/BerriAI/litellm/issues/8835
    """
    config = AnthropicConfig()

    # Claude 3.7 Sonnet should return 64000 (64K default, 128K requires beta header)
    max_tokens = config.get_max_tokens_for_model("claude-3-7-sonnet-20250219")
    assert max_tokens == 64000


def test_get_max_tokens_for_model_unknown():
    """
    Test that get_max_tokens_for_model returns 4096 fallback for unknown models.
    """
    config = AnthropicConfig()

    # Unknown model should return 4096 as fallback
    max_tokens = config.get_max_tokens_for_model("unknown-model-xyz")
    assert max_tokens == 4096


def test_get_max_tokens_for_model_none():
    """
    Test that get_max_tokens_for_model returns 4096 fallback when model is None.
    """
    config = AnthropicConfig()

    # None model should return 4096 as fallback
    max_tokens = config.get_max_tokens_for_model(None)
    assert max_tokens == 4096


def test_get_config_with_model_uses_dynamic_max_tokens():
    """
    Test that get_config returns dynamic max_tokens based on model.

    Fixes: https://github.com/BerriAI/litellm/issues/8835
    """
    # Claude 3 model should get 4096
    config_claude3 = AnthropicConfig.get_config(model="claude-3-sonnet-20240229")
    assert config_claude3["max_tokens"] == 4096

    # Claude 3.5 model should get 8192
    config_claude35 = AnthropicConfig.get_config(model="claude-3-5-sonnet-20241022")
    assert config_claude35["max_tokens"] == 8192

    # Claude 3.7 model should get 64000 (64K default, 128K requires beta header)
    config_claude37 = AnthropicConfig.get_config(model="claude-3-7-sonnet-20250219")
    assert config_claude37["max_tokens"] == 64000


def test_get_config_without_model_uses_fallback():
    """
    Test that get_config without model parameter uses 4096 fallback.
    """
    config = AnthropicConfig.get_config()
    assert config["max_tokens"] == 4096


def test_transform_request_uses_dynamic_max_tokens():
    """
    Test that transform_request uses dynamic max_tokens based on model
    when max_tokens is not explicitly provided.

    Fixes: https://github.com/BerriAI/litellm/issues/8835
    """
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]

    # Claude 3.5 model should get 8192 as default max_tokens
    result = config.transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        optional_params={},  # No max_tokens provided
        litellm_params={},
        headers={}
    )

    assert result["max_tokens"] == 8192


def test_transform_request_respects_user_max_tokens():
    """
    Test that transform_request respects user-provided max_tokens
    and doesn't override it with dynamic value.
    """
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]

    # User provides explicit max_tokens=1000, should not be overridden
    result = config.transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        optional_params={"max_tokens": 1000},
        litellm_params={},
        headers={}
    )

    assert result["max_tokens"] == 1000


def test_calculate_usage_completion_tokens_details_always_populated():
    """
    Test that completion_tokens_details is always populated in Usage object,
    not just when there's reasoning_content.
    
    Fixes: https://github.com/BerriAI/litellm/issues/18772
    Bug: completion_tokens_details was None for regular Claude responses without reasoning
    """
    config = AnthropicConfig()

    # Test without reasoning_content - completion_tokens_details should still be populated
    usage_object = {
        "input_tokens": 37,
        "output_tokens": 248,
    }
    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    
    # completion_tokens_details should NOT be None
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens is 0
    assert usage.completion_tokens_details.text_tokens == 248
    assert usage.completion_tokens == 248
    assert usage.prompt_tokens == 37
    assert usage.total_tokens == 285


def test_calculate_usage_completion_tokens_details_with_reasoning():
    """
    Test that completion_tokens_details correctly splits text_tokens and reasoning_tokens
    when reasoning_content is present.
    
    Fixes: https://github.com/BerriAI/litellm/issues/18772
    """
    config = AnthropicConfig()

    # Test with reasoning_content - should split tokens correctly
    usage_object = {
        "input_tokens": 100,
        "output_tokens": 500,
    }
    # Simulating reasoning content that would count as ~50 tokens
    reasoning_content = "Let me think about this step by step. " * 10  # Roughly 50 tokens
    
    usage = config.calculate_usage(
        usage_object=usage_object, 
        reasoning_content=reasoning_content
    )
    
    # completion_tokens_details should be populated with both reasoning and text tokens
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens is not None
    assert usage.completion_tokens_details.reasoning_tokens > 0
    # text_tokens should be total minus reasoning
    expected_text_tokens = 500 - usage.completion_tokens_details.reasoning_tokens
    assert usage.completion_tokens_details.text_tokens == expected_text_tokens
    assert usage.completion_tokens == 500


# ============ Reasoning Effort Tests ============


def test_reasoning_effort_maps_to_adaptive_thinking_for_opus_4_6():
    """
    Test that reasoning_effort maps to adaptive thinking type for Claude Opus 4.6.
    
    For Claude Opus 4.6, reasoning_effort should map to {"type": "adaptive"} 
    regardless of the effort level specified.
    """
    config = AnthropicConfig()
    
    # Test with different reasoning_effort values - all should map to adaptive
    for effort in ["low", "medium", "high", "minimal"]:
        non_default_params = {"reasoning_effort": effort}
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-opus-4-6-20250514",
            drop_params=False
        )
        
        # Should map to adaptive thinking type
        assert "thinking" in result
        assert result["thinking"]["type"] == "adaptive"
        # Should not have budget_tokens for adaptive type
        assert "budget_tokens" not in result["thinking"]
        # reasoning_effort should not be in the result (it's transformed to thinking)
        assert "reasoning_effort" not in result


def test_reasoning_effort_maps_to_budget_thinking_for_non_opus_4_6():
    """
    Test that reasoning_effort maps to budget-based thinking config for non-Opus 4.6 models.
    
    For models other than Claude Opus 4.6, reasoning_effort should map to 
    thinking config with budget_tokens based on the effort level.
    """
    config = AnthropicConfig()
    
    # Test with Claude Sonnet 4.5 (non-Opus 4.6 model)
    test_cases = [
        ("low", 1024),      # DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET
        ("medium", 2048),   # DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET
        ("high", 4096),     # DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET
        ("minimal", 128),   # DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET
    ]
    
    for effort, expected_budget in test_cases:
        non_default_params = {"reasoning_effort": effort}
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-sonnet-4-5-20250929",
            drop_params=False
        )
        
        # Should map to enabled thinking type with budget_tokens
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == expected_budget
        # reasoning_effort should not be in the result (it's transformed to thinking)
        assert "reasoning_effort" not in result


def test_code_execution_tool_results_extraction():
    """
    Test that code execution tool results (bash_code_execution_tool_result, 
    text_editor_code_execution_tool_result) are properly extracted and exposed 
    in provider_specific_fields.
    
    Related to: https://github.com/BerriAI/litellm/issues/xxxxx
    """
    import httpx

    from litellm.types.utils import ModelResponse
    
    config = AnthropicConfig()
    
    # Mock Anthropic response with code execution tool results
    mock_anthropic_response = {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "text",
                "text": "I'll calculate that for you."
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC",
                "name": "bash_code_execution",
                "input": {
                    "command": "python3 << 'EOF'\nprint(2 + 2)\nEOF\n"
                }
            },
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01ABC",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "4\n",
                    "stderr": "",
                    "return_code": 0
                }
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01DEF",
                "name": "text_editor_code_execution",
                "input": {
                    "command": "create",
                    "path": "test.txt",
                    "file_text": "Hello"
                }
            },
            {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01DEF",
                "content": {
                    "type": "text_editor_code_execution_result",
                    "is_file_update": False
                }
            },
            {
                "type": "text",
                "text": "Done!"
            }
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    
    # Create mock HTTP response
    mock_raw_response = MagicMock(spec=httpx.Response)
    mock_raw_response.json.return_value = mock_anthropic_response
    mock_raw_response.status_code = 200
    mock_raw_response.headers = {}
    
    model_response = ModelResponse()
    
    transformed_response = config.transform_parsed_response(
        completion_response=mock_anthropic_response,
        raw_response=mock_raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )
    
    # Verify tool calls are present
    assert transformed_response.choices[0].message.tool_calls is not None
    assert len(transformed_response.choices[0].message.tool_calls) == 2
    
    # Verify first tool call
    assert transformed_response.choices[0].message.tool_calls[0].id == "srvtoolu_01ABC"
    assert transformed_response.choices[0].message.tool_calls[0].function.name == "bash_code_execution"
    
    # Verify second tool call
    assert transformed_response.choices[0].message.tool_calls[1].id == "srvtoolu_01DEF"
    assert transformed_response.choices[0].message.tool_calls[1].function.name == "text_editor_code_execution"
    
    # Verify tool results are in provider_specific_fields
    provider_fields = transformed_response.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert "tool_results" in provider_fields
    assert provider_fields["tool_results"] is not None
    assert len(provider_fields["tool_results"]) == 2
    
    # Verify bash_code_execution_tool_result
    bash_result = provider_fields["tool_results"][0]
    assert bash_result["type"] == "bash_code_execution_tool_result"
    assert bash_result["tool_use_id"] == "srvtoolu_01ABC"
    assert bash_result["content"]["stdout"] == "4\n"
    assert bash_result["content"]["return_code"] == 0
    
    # Verify text_editor_code_execution_tool_result
    editor_result = provider_fields["tool_results"][1]
    assert editor_result["type"] == "text_editor_code_execution_tool_result"
    assert editor_result["tool_use_id"] == "srvtoolu_01DEF"
    assert editor_result["content"]["is_file_update"] is False
    
    # Verify text content is properly concatenated
    assert "I'll calculate that for you." in transformed_response.choices[0].message.content
    assert "Done!" in transformed_response.choices[0].message.content


def test_tool_search_tool_result_not_in_tool_results():
    """
    Test that tool_search_tool_result is NOT included in tool_results
    since it's internal metadata, not actual tool execution results.
    """
    import httpx

    from litellm.types.utils import ModelResponse
    
    config = AnthropicConfig()
    
    mock_anthropic_response = {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "text",
                "text": "Found tools."
            },
            {
                "type": "tool_search_tool_result",
                "tool_references": ["tool1", "tool2"]
            }
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    
    mock_raw_response = MagicMock(spec=httpx.Response)
    mock_raw_response.json.return_value = mock_anthropic_response
    mock_raw_response.status_code = 200
    mock_raw_response.headers = {}
    
    model_response = ModelResponse()
    
    transformed_response = config.transform_parsed_response(
        completion_response=mock_anthropic_response,
        raw_response=mock_raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )
    
    # Verify tool_search_tool_result is NOT in tool_results
    provider_fields = transformed_response.choices[0].message.provider_specific_fields
    assert provider_fields.get("tool_results") is None


def test_web_search_tool_result_backwards_compatibility():
    """
    Test that web_search_tool_result continues to be stored in web_search_results
    for backwards compatibility, not in tool_results.
    """
    import httpx

    from litellm.types.utils import ModelResponse
    
    config = AnthropicConfig()
    
    mock_anthropic_response = {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "text",
                "text": "Here are the results."
            },
            {
                "type": "web_search_tool_result",
                "search_query": "test query",
                "results": [{"title": "Result 1", "url": "https://example.com"}]
            }
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    
    mock_raw_response = MagicMock(spec=httpx.Response)
    mock_raw_response.json.return_value = mock_anthropic_response
    mock_raw_response.status_code = 200
    mock_raw_response.headers = {}
    
    model_response = ModelResponse()
    
    transformed_response = config.transform_parsed_response(
        completion_response=mock_anthropic_response,
        raw_response=mock_raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )
    
    # Verify web_search_tool_result is in web_search_results (not tool_results)
    provider_fields = transformed_response.choices[0].message.provider_specific_fields
    assert "web_search_results" in provider_fields
    assert provider_fields["web_search_results"] is not None
    assert len(provider_fields["web_search_results"]) == 1
    assert provider_fields["web_search_results"][0]["type"] == "web_search_tool_result"
    
    # Should NOT be in tool_results
    assert provider_fields.get("tool_results") is None


# ============ Compaction Tests ============


def test_compaction_block_extraction():
    """
    Test that compaction blocks are correctly extracted from Anthropic response.
    """
    config = AnthropicConfig()
    
    completion_response = {
        "id": "msg_compaction_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [
            {
                "type": "compaction",
                "content": "Summary of the conversation: The user requested help building a web scraper..."
            },
            {
                "type": "text",
                "text": "I don't have access to real-time data, so I can't provide the current weather in San Francisco."
            }
        ],
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 86,
            "output_tokens": 100
        }
    }
    
    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )
    
    # Verify compaction blocks are extracted
    assert compaction_blocks is not None
    assert len(compaction_blocks) == 1
    assert compaction_blocks[0]["type"] == "compaction"
    assert "Summary of the conversation" in compaction_blocks[0]["content"]
    
    # Verify text content is extracted
    assert "I don't have access to real-time data" in text


def test_compaction_block_in_provider_specific_fields():
    """
    Test that compaction blocks are included in provider_specific_fields.
    """
    import httpx

    from litellm.types.utils import ModelResponse
    
    config = AnthropicConfig()
    
    completion_response = {
        "id": "msg_compaction_provider_fields",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [
            {
                "type": "compaction",
                "content": "Summary of the conversation: The user requested help building a web scraper..."
            },
            {
                "type": "text",
                "text": "Here is the response."
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 25
        }
    }
    
    raw_response = httpx.Response(status_code=200, headers={})
    model_response = ModelResponse()
    
    result = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )
    
    # Verify compaction_blocks is in provider_specific_fields
    provider_fields = result.choices[0].message.provider_specific_fields
    assert provider_fields is not None
    assert "compaction_blocks" in provider_fields
    assert len(provider_fields["compaction_blocks"]) == 1
    assert provider_fields["compaction_blocks"][0]["type"] == "compaction"
    assert "Summary of the conversation" in provider_fields["compaction_blocks"][0]["content"]


def test_multiple_compaction_blocks():
    """
    Test that multiple compaction blocks are all extracted.
    """
    config = AnthropicConfig()
    
    completion_response = {
        "content": [
            {
                "type": "compaction",
                "content": "First summary..."
            },
            {
                "type": "text",
                "text": "Some text."
            },
            {
                "type": "compaction",
                "content": "Second summary..."
            }
        ]
    }
    
    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )
    
    # Verify both compaction blocks are extracted
    assert compaction_blocks is not None
    assert len(compaction_blocks) == 2
    assert compaction_blocks[0]["content"] == "First summary..."
    assert compaction_blocks[1]["content"] == "Second summary..."


def test_compaction_block_request_transformation():
    """
    Test that compaction blocks from provider_specific_fields are correctly
    transformed back to Anthropic format in requests.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )
    
    messages = [
        {
            "role": "user",
            "content": "What is the weather in San Francisco?"
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I don't have access to real-time data."
                }
            ],
            "provider_specific_fields": {
                "compaction_blocks": [
                    {
                        "type": "compaction",
                        "content": "Summary of the conversation: The user requested help building a web scraper..."
                    }
                ]
            }
        },
        {
            "role": "user",
            "content": "What about New York?"
        }
    ]
    
    result = anthropic_messages_pt(
        messages=messages,
        model="claude-opus-4-6",
        llm_provider="anthropic"
    )
    
    # Find the assistant message
    assistant_message = None
    for msg in result:
        if msg["role"] == "assistant":
            assistant_message = msg
            break
    
    assert assistant_message is not None
    assert "content" in assistant_message
    assert isinstance(assistant_message["content"], list)
    
    # Verify compaction block is at the beginning
    assert assistant_message["content"][0]["type"] == "compaction"
    assert "Summary of the conversation" in assistant_message["content"][0]["content"]
    
    # Verify text content follows
    text_blocks = [c for c in assistant_message["content"] if c.get("type") == "text"]
    assert len(text_blocks) > 0
    assert "I don't have access to real-time data" in text_blocks[0]["text"]


def test_compaction_with_context_management():
    """
    Test that compaction works with context_management parameter.
    """
    config = AnthropicConfig()
    
    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "context_management": {
            "edits": [
                {
                    "type": "compact_20260112"
                }
            ]
        },
        "max_tokens": 100
    }
    
    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    # Verify context_management is included
    assert "context_management" in result
    assert result["context_management"]["edits"][0]["type"] == "compact_20260112"


def test_compaction_block_with_other_content_types():
    """
    Test that compaction blocks work alongside other content types like thinking blocks and tool calls.
    """
    config = AnthropicConfig()
    
    completion_response = {
        "content": [
            {
                "type": "compaction",
                "content": "Summary of previous conversation..."
            },
            {
                "type": "thinking",
                "thinking": "Let me think about this..."
            },
            {
                "type": "text",
                "text": "Based on my analysis..."
            },
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco"}
            }
        ]
    }
    
    text, citations, thinking_blocks, reasoning_content, tool_calls, web_search_results, tool_results, compaction_blocks = config.extract_response_content(
        completion_response
    )
    
    # Verify all content types are extracted
    assert compaction_blocks is not None
    assert len(compaction_blocks) == 1
    assert thinking_blocks is not None
    assert len(thinking_blocks) == 1
    assert "Based on my analysis" in text
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_compaction_block_empty_list_not_added():
    """
    Test that empty compaction_blocks list is not added to provider_specific_fields.
    """
    import httpx

    from litellm.types.utils import ModelResponse
    
    config = AnthropicConfig()
    
    # Response without compaction blocks
    completion_response = {
        "id": "msg_no_compaction",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [
            {
                "type": "text",
                "text": "Just a regular response."
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    }
    
    raw_response = httpx.Response(status_code=200, headers={})
    model_response = ModelResponse()
    
    result = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        json_mode=False,
        prefix_prompt=None,
    )
    
    # Verify compaction_blocks is not in provider_specific_fields when there are none
    provider_fields = result.choices[0].message.provider_specific_fields
    if provider_fields:
        assert "compaction_blocks" not in provider_fields or provider_fields.get("compaction_blocks") is None


def test_fast_mode_beta_header():
    """
    Test that fast mode correctly adds the fast-mode-2026-02-01 beta header.
    """
    config = AnthropicConfig()
    
    headers = {}
    optional_params = {"speed": "fast"}
    
    result_headers = config.update_headers_with_optional_anthropic_beta(
        headers=headers,
        optional_params=optional_params
    )
    
    assert "anthropic-beta" in result_headers
    assert "fast-mode-2026-02-01" in result_headers["anthropic-beta"]


def test_fast_mode_with_other_beta_headers():
    """
    Test that fast mode beta header is combined with other beta headers.
    """
    config = AnthropicConfig()
    
    headers = {}
    optional_params = {
        "speed": "fast",
        "output_format": {"type": "json_object"}
    }
    
    result_headers = config.update_headers_with_optional_anthropic_beta(
        headers=headers,
        optional_params=optional_params
    )
    
    assert "anthropic-beta" in result_headers
    assert "fast-mode-2026-02-01" in result_headers["anthropic-beta"]
    assert "structured-outputs-2025-11-13" in result_headers["anthropic-beta"]


def test_fast_mode_usage_calculation():
    """
    Test that fast mode speed parameter is passed through to usage object.
    """
    config = AnthropicConfig()
    
    usage_object = {
        "input_tokens": 1000,
        "output_tokens": 500,
    }
    
    usage = config.calculate_usage(
        usage_object=usage_object,
        reasoning_content=None,
        speed="fast"
    )
    
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert hasattr(usage, "speed")
    assert usage.speed == "fast"


def test_fast_mode_cost_calculation():
    """
    Test that fast mode correctly prepends 'fast/' to model name for pricing lookup.
    """
    from unittest.mock import patch

    from litellm.llms.anthropic.cost_calculation import cost_per_token
    from litellm.types.utils import Usage

    # Mock the generic_cost_per_token to verify correct model name is passed
    with patch('litellm.llms.anthropic.cost_calculation.generic_cost_per_token') as mock_cost:
        mock_cost.return_value = (0.03, 0.15)  # $30 and $150 per MTok
        
        # Test fast mode
        usage_fast = Usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            speed="fast"
        )
        
        prompt_cost, completion_cost = cost_per_token(
            model="claude-opus-4-6",
            usage=usage_fast
        )
        
        # Verify that generic_cost_per_token was called with "fast/claude-opus-4-6"
        mock_cost.assert_called_once()
        call_args = mock_cost.call_args
        assert call_args[1]['model'] == "fast/claude-opus-4-6"
        assert call_args[1]['custom_llm_provider'] == "anthropic"


def test_fast_mode_with_inference_geo():
    """
    Test that fast mode works correctly with inference_geo prefix.
    Expected format: fast/us/claude-opus-4-6
    """
    from unittest.mock import patch

    from litellm.llms.anthropic.cost_calculation import cost_per_token
    from litellm.types.utils import Usage

    # Mock the generic_cost_per_token to verify correct model name is passed
    with patch('litellm.llms.anthropic.cost_calculation.generic_cost_per_token') as mock_cost:
        mock_cost.return_value = (0.03, 0.15)
        
        # Test with both speed and inference_geo
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            speed="fast",
            inference_geo="us"
        )
        
        # This should look up "fast/us/claude-opus-4-6" in pricing
        prompt_cost, completion_cost = cost_per_token(
            model="claude-opus-4-6",
            usage=usage
        )
        
        # Verify that generic_cost_per_token was called with "fast/us/claude-opus-4-6"
        mock_cost.assert_called_once()
        call_args = mock_cost.call_args
        assert call_args[1]['model'] == "fast/us/claude-opus-4-6"
        assert call_args[1]['custom_llm_provider'] == "anthropic"


def test_fast_mode_parameter_in_supported_params():
    """
    Test that 'speed' is in the list of supported OpenAI params.
    """
    config = AnthropicConfig()
    
    supported_params = config.get_supported_openai_params(model="claude-opus-4-6")
    
    assert "speed" in supported_params


def test_fast_mode_parameter_mapping():
    """
    Test that speed parameter is correctly mapped in map_openai_params.
    """
    config = AnthropicConfig()
    
    non_default_params = {"speed": "fast"}
    optional_params = {}
    
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-opus-4-6",
        drop_params=False
    )
    
    assert "speed" in result
    assert result["speed"] == "fast"
