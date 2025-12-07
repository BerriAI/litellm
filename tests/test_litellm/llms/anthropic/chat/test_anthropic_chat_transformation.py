import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

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

    _, citations, _, _, _ = config.extract_response_content(completion_response)
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


def test_transform_parsed_response_includes_context_management_metadata():
    import httpx

    from litellm.types.utils import ModelResponse

    config = AnthropicConfig()
    context_management_payload = {
        "applied_edits": [
            {
                "type": "clear_tool_uses_20250919",
                "cleared_tool_uses": 2,
                "cleared_input_tokens": 5000,
            }
        ]
    }
    completion_response = {
        "id": "msg_context_management_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Done."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 5,
        },
        "context_management": context_management_payload,
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
        prefix_prompt=None,
    )

    assert result.__dict__.get("context_management") == context_management_payload
    provider_fields = result.choices[0].message.provider_specific_fields
    assert (
        provider_fields
        and provider_fields["context_management"] == context_management_payload
    )


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
    
    text, citations, thinking_blocks, reasoning_content, tool_calls = config.extract_response_content(
        completion_response
    )
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "srvtoolu_01ABC123"
    assert tool_calls[0]["function"]["name"] == "tool_search_tool_regex"


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
    text, citations, thinking_blocks, reasoning_content, tool_calls = config.extract_response_content(
        completion_response
    )
    
    # Verify text extraction (should concatenate both text blocks)
    assert "I'll search for weather-related tools" in text
    assert "Great! I found a weather tool" in text
    
    # Verify tool calls (should have both server_tool_use and tool_use)
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "tool_search_tool_regex"
    assert tool_calls[1]["function"]["name"] == "get_weather"
    
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
    
    text, citations, thinking, reasoning, tool_calls = config.extract_response_content(completion_response)
    
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "toolu_123"
    assert tool_calls[0]["function"]["name"] == "query_database"
    assert "caller" in tool_calls[0]
    assert tool_calls[0]["caller"]["type"] == "code_execution_20250825"
    assert tool_calls[0]["caller"]["tool_id"] == "srvtoolu_abc"


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
