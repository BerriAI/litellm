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
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 550
    assert usage.total_tokens == 3 + 550
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage._cache_creation_input_tokens == 12304
    assert usage._cache_read_input_tokens == 0

@pytest.mark.parametrize("usage_object,expected_usage", [
    [
        {
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "input_tokens": None,
            "output_tokens": 43,
            "server_tool_use": None
        },
        {
            "prompt_tokens": 0,
            "completion_tokens": 43,
            "total_tokens": 43,
            "_cache_creation_input_tokens": 0,
            "_cache_read_input_tokens": 0
        }
    ],
    [
        {
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 200,
            "input_tokens": 1,
            "output_tokens": None,
            "server_tool_use": None
        },
        {
            "prompt_tokens": 1 + 200,
            "completion_tokens": 0,
            "total_tokens": 1 + 200,
            "_cache_creation_input_tokens": 100,
            "_cache_read_input_tokens": 200,
        }
    ],
    [
        {
            "server_tool_use": {
                "web_search_requests": 10
            }
        },
        {
            "server_tool_use": ServerToolUse(web_search_requests=10)
        }
    ]
])
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

@pytest.mark.parametrize("usage_object", [
    {
        "server_tool_use": {
            "web_search_requests": None
        }
    },
    {
        "server_tool_use": None
    }
])
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
        "model": "claude-3-5-sonnet-20241022",
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
    assert citations is not None


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
        "model": "claude-3-5-sonnet-20241022",
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


def test_get_max_tokens_fallback_request(monkeypatch):
    """
    Directly test _get_max_tokens_fallback: request param should override config and env
    """
    monkeypatch.setenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", "9999")
    config = AnthropicConfig()
    config_dict = {"max_tokens": 5555}
    optional_params = {"max_tokens": 1234}
    result = config._get_max_tokens_fallback(optional_params, config_dict)
    assert result == 1234

def test_get_max_tokens_fallback_config(monkeypatch):
    """
    Directly test _get_max_tokens_fallback: config should override env if request not set
    """
    monkeypatch.setenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", "9999")
    config = AnthropicConfig()
    config_dict = {"max_tokens": 5555}
    optional_params = {}
    result = config._get_max_tokens_fallback(optional_params, config_dict)
    assert result == 5555

def test_get_max_tokens_fallback_env(monkeypatch):
    """
    Directly test _get_max_tokens_fallback: env variable used if neither request nor config is set
    """
    from litellm.constants import DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS
    config = AnthropicConfig()
    config_dict = {}
    optional_params = {}
    # Should fallback to DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS constant
    result = config._get_max_tokens_fallback(optional_params, config_dict)
    assert result == DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS

    def test_max_tokens_fallback_request_overrides(monkeypatch):
        """
        Request max_tokens should override config and env
        """
        monkeypatch.setenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", "9999")
        class DummyConfig:
            @classmethod
            def get_config(cls):
                return {"max_tokens": 5555}
        config = AnthropicConfig()
        config.get_config = DummyConfig.get_config
        non_default_params = {"max_tokens": 1234}
        optional_params = {}
        result = config.map_openai_params(non_default_params, optional_params, "model", False)
        assert result["max_tokens"] == 1234

    def test_max_tokens_fallback_config_overrides_env(monkeypatch):
        """
        Config max_tokens should override env if request not set
        """
        monkeypatch.setenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", "9999")
        class DummyConfig:
            @classmethod
            def get_config(cls):
                return {"max_tokens": 5555}
        config = AnthropicConfig()
        config.get_config = DummyConfig.get_config
        non_default_params = {}
        optional_params = {}
        result = config.map_openai_params(non_default_params, optional_params, "model", False)
        assert result["max_tokens"] == 5555

    def test_max_tokens_fallback_env(monkeypatch):
        """
        Env variable is used if neither request nor config is set
        """
        monkeypatch.setenv("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS", "8888")
        class DummyConfig:
            @classmethod
            def get_config(cls):
                return {}
        config = AnthropicConfig()
        config.get_config = DummyConfig.get_config
        non_default_params = {}
        optional_params = {}
        result = config.map_openai_params(non_default_params, optional_params, "model", False)
        assert result["max_tokens"] == 8888
