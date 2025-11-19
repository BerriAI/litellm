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
    assert provider_fields and provider_fields["context_management"] == context_management_payload
