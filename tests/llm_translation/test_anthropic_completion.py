# What is this?
## Unit tests for Anthropic Adapter

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import (
    AnthropicConfig,
    Router,
    adapter_completion,
    AnthropicExperimentalPassThroughConfig,
)
from litellm.adapters.anthropic_adapter import anthropic_adapter
from litellm.types.llms.anthropic import AnthropicResponse
from litellm.types.utils import GenericStreamingChunk, ChatCompletionToolCallChunk
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.llms.anthropic.common_utils import process_anthropic_headers
from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest


def test_anthropic_completion_messages_translation():
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    translated_messages = AnthropicExperimentalPassThroughConfig().translate_anthropic_messages_to_openai(messages=messages)  # type: ignore

    assert translated_messages == [{"role": "user", "content": "Hey, how's it going?"}]


def test_anthropic_completion_input_translation():
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
    }
    translated_input = anthropic_adapter.translate_completion_input_params(kwargs=data)

    assert translated_input is not None

    assert translated_input["model"] == "gpt-3.5-turbo"
    assert translated_input["messages"] == [
        {"role": "user", "content": "Hey, how's it going?"}
    ]


def test_anthropic_completion_input_translation_with_metadata():
    """
    Tests that cost tracking works as expected with LiteLLM Proxy

    LiteLLM Proxy will insert litellm_metadata for anthropic endpoints to track user_api_key and user_api_key_team_id

    This test ensures that the `litellm_metadata` is not present in the translated input
    It ensures that `litellm.acompletion()` will receieve metadata which is a litellm specific param
    """
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
        "litellm_metadata": {
            "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "user_api_key_alias": None,
            "user_api_end_user_max_budget": None,
            "litellm_api_version": "1.40.19",
            "global_max_parallel_requests": None,
            "user_api_key_user_id": "default_user_id",
            "user_api_key_org_id": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_team_max_budget": None,
            "user_api_key_team_spend": None,
            "user_api_key_spend": 0.0,
            "user_api_key_max_budget": None,
            "user_api_key_metadata": {},
        },
    }
    translated_input = anthropic_adapter.translate_completion_input_params(kwargs=data)

    assert "litellm_metadata" not in translated_input
    assert "metadata" in translated_input
    assert translated_input["metadata"] == data["litellm_metadata"]


def streaming_format_tests(chunk: dict, idx: int):
    """
    1st chunk -  chunk.get("type") == "message_start"
    2nd chunk - chunk.get("type") == "content_block_start"
    3rd chunk - chunk.get("type") == "content_block_delta"
    """
    if idx == 0:
        assert chunk.get("type") == "message_start"
    elif idx == 1:
        assert chunk.get("type") == "content_block_start"
    elif idx == 2:
        assert chunk.get("type") == "content_block_delta"


@pytest.mark.parametrize("stream", [True])  # False
def test_anthropic_completion_e2e(stream):
    litellm.set_verbose = True

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = adapter_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
        stream=stream,
    )

    print("Response: {}".format(response))

    assert response is not None

    if stream is False:
        assert isinstance(response, AnthropicResponse)
    else:
        """
        - ensure finish reason is returned
        - assert content block is started and stopped
        - ensure last chunk is 'message_stop'
        """
        assert isinstance(response, litellm.types.utils.AdapterCompletionStreamWrapper)
        finish_reason: Optional[str] = None
        message_stop_received = False
        content_block_started = False
        content_block_finished = False
        for idx, chunk in enumerate(response):
            print(chunk)
            streaming_format_tests(chunk=chunk, idx=idx)
            if chunk.get("delta", {}).get("stop_reason") is not None:
                finish_reason = chunk.get("delta", {}).get("stop_reason")
            if chunk.get("type") == "message_stop":
                message_stop_received = True
            if chunk.get("type") == "content_block_stop":
                content_block_finished = True
            if chunk.get("type") == "content_block_start":
                content_block_started = True
        assert content_block_started and content_block_finished
        assert finish_reason is not None
        assert message_stop_received is True


anthropic_chunk_list = [
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "To"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " answer"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " your question about the weather"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " in Boston and Los"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " Angeles today, I'll"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " need to"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " use"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " the"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " get_current_weather"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " function"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " for"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " both"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " cities"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": ". Let"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " me fetch"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " that"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " information"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " for"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " you."},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_12345",
            "name": "get_current_weather",
            "input": {},
        },
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": ""},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"locat'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": 'ion": "Bos'},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": 'ton, MA"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_023423423",
            "name": "get_current_weather",
            "input": {},
        },
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": ""},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"l'},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": "oca"},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": "tio"},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": 'n": "Lo'},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": "s Angel"},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": 'es, CA"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 137},
    },
    {"type": "message_stop"},
]


def test_anthropic_tool_streaming():
    """
    OpenAI starts tool_use indexes at 0 for the first tool, regardless of preceding text.

    Anthropic gives tool_use indexes starting at the first chunk, meaning they often start at 1
    when they should start at 0
    """
    litellm.set_verbose = True
    response_iter = ModelResponseIterator([], False)

    # First index is 0, we'll start earlier because incrementing is easier
    correct_tool_index = -1
    for chunk in anthropic_chunk_list:
        parsed_chunk = response_iter.chunk_parser(chunk)
        if tool_use := parsed_chunk.get("tool_use"):

            # We only increment when a new block starts
            if tool_use.get("id") is not None:
                correct_tool_index += 1
            assert tool_use["index"] == correct_tool_index


def test_anthropic_tool_calling_translation():
    kwargs = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Would development of a software platform be under ASC 350-40 or ASC 985?",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "37d6f703-cbcc-497d-95a1-2aa24a114adc",
                        "name": "TaskPlanningTool",
                        "input": {
                            "completed_steps": [],
                            "next_steps": [
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 350-40 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 985 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development.",
                                },
                            ],
                            "learnings": [],
                            "potential_issues": [
                                "The distinction between the two standards might not be clear-cut for all types of software development.",
                                "There might be specific circumstances or details about the software platform that could affect which standard applies.",
                            ],
                            "missing_info": [
                                "Specific details about the type of software platform being developed (e.g., for internal use or for sale).",
                                "Whether the entity developing the software is also the end-user or if it's being developed for external customers.",
                            ],
                            "done": False,
                            "required_formatting": None,
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "eb7023b1-5ee8-43b8-b90f-ac5a23d37c31",
                        "content": {
                            "completed_steps": [],
                            "next_steps": [
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 350-40 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 985 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development.",
                                },
                            ],
                            "formatting_step": None,
                        },
                    }
                ],
            },
        ],
    }

    from litellm.adapters.anthropic_adapter import anthropic_adapter

    translated_params = anthropic_adapter.translate_completion_input_params(
        kwargs=kwargs
    )

    print(translated_params["messages"])

    assert len(translated_params["messages"]) > 0
    assert translated_params["messages"][0]["role"] == "user"


def test_process_anthropic_headers_empty():
    result = process_anthropic_headers({})
    assert result == {}, "Expected empty dictionary for no input"


def test_process_anthropic_headers_with_all_headers():
    input_headers = Headers(
        {
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "90",
            "anthropic-ratelimit-tokens-limit": "10000",
            "anthropic-ratelimit-tokens-remaining": "9000",
            "other-header": "value",
        }
    )

    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "90",
        "x-ratelimit-limit-tokens": "10000",
        "x-ratelimit-remaining-tokens": "9000",
        "llm_provider-anthropic-ratelimit-requests-limit": "100",
        "llm_provider-anthropic-ratelimit-requests-remaining": "90",
        "llm_provider-anthropic-ratelimit-tokens-limit": "10000",
        "llm_provider-anthropic-ratelimit-tokens-remaining": "9000",
        "llm_provider-other-header": "value",
    }

    result = process_anthropic_headers(input_headers)
    assert result == expected_output, "Unexpected output for all Anthropic headers"


def test_process_anthropic_headers_with_partial_headers():
    input_headers = Headers(
        {
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-tokens-remaining": "9000",
            "other-header": "value",
        }
    )

    expected_output = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-tokens": "9000",
        "llm_provider-anthropic-ratelimit-requests-limit": "100",
        "llm_provider-anthropic-ratelimit-tokens-remaining": "9000",
        "llm_provider-other-header": "value",
    }

    result = process_anthropic_headers(input_headers)
    assert result == expected_output, "Unexpected output for partial Anthropic headers"


def test_process_anthropic_headers_with_no_matching_headers():
    input_headers = Headers(
        {"unrelated-header-1": "value1", "unrelated-header-2": "value2"}
    )

    expected_output = {
        "llm_provider-unrelated-header-1": "value1",
        "llm_provider-unrelated-header-2": "value2",
    }

    result = process_anthropic_headers(input_headers)
    assert result == expected_output, "Unexpected output for non-matching headers"


def test_anthropic_computer_tool_use():
    from litellm import completion

    tools = [
        {
            "type": "computer_20241022",
            "function": {
                "name": "computer",
                "parameters": {
                    "display_height_px": 100,
                    "display_width_px": 100,
                    "display_number": 1,
                },
            },
        }
    ]
    model = "claude-3-5-sonnet-20241022"
    messages = [{"role": "user", "content": "Save a picture of a cat to my desktop."}]

    try:
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
            # headers={"anthropic-beta": "computer-use-2024-10-22"},
        )
        print(resp)
    except litellm.InternalServerError:
        pass


@pytest.mark.parametrize(
    "computer_tool_used, prompt_caching_set, expected_beta_header",
    [
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_anthropic_beta_header(
    computer_tool_used, prompt_caching_set, expected_beta_header
):
    headers = litellm.AnthropicConfig().get_anthropic_headers(
        api_key="fake-api-key",
        computer_tool_used=computer_tool_used,
        prompt_caching_set=prompt_caching_set,
    )

    if expected_beta_header:
        assert "anthropic-beta" in headers
    else:
        assert "anthropic-beta" not in headers


@pytest.mark.parametrize(
    "cache_control_location",
    [
        "inside_function",
        "outside_function",
    ],
)
def test_anthropic_tool_helper(cache_control_location):
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    tool = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }

    if cache_control_location == "inside_function":
        tool["function"]["cache_control"] = {"type": "ephemeral"}
    else:
        tool["cache_control"] = {"type": "ephemeral"}

    tool = AnthropicConfig()._map_tool_helper(tool=tool)

    assert tool["cache_control"] == {"type": "ephemeral"}


def test_create_json_tool_call_for_response_format():
    """
    tests using response_format=json with anthropic

    A tool call to anthropic is made when response_format=json is used.

    """
    # Initialize AnthropicConfig
    config = AnthropicConfig()

    # Test case 1: No schema provided
    # See Anthropics Example 5 on how to handle cases when no schema is provided https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
    tool = config._create_json_tool_call_for_response_format()
    assert tool["name"] == "json_tool_call"
    _input_schema = tool.get("input_schema")
    assert _input_schema is not None
    assert _input_schema.get("type") == "object"
    assert _input_schema.get("additionalProperties") is True
    assert _input_schema.get("properties") == {}

    # Test case 2: With custom schema
    # reference: https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
    custom_schema = {"name": {"type": "string"}, "age": {"type": "integer"}}
    tool = config._create_json_tool_call_for_response_format(json_schema=custom_schema)
    assert tool["name"] == "json_tool_call"
    _input_schema = tool.get("input_schema")
    assert _input_schema is not None
    assert _input_schema.get("type") == "object"
    assert _input_schema.get("properties") == custom_schema
    assert "additionalProperties" not in _input_schema


from litellm import completion


class TestAnthropicCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "claude-3-haiku-20240307"}

    def test_pdf_handling(self, pdf_messages):
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.types.llms.anthropic import AnthropicMessagesDocumentParam
        import json

        client = HTTPHandler()

        with patch.object(client, "post", new=MagicMock()) as mock_client:
            response = completion(
                model="claude-3-5-sonnet-20241022",
                messages=pdf_messages,
                client=client,
            )

            mock_client.assert_called_once()

            json_data = json.loads(mock_client.call_args.kwargs["data"])
            headers = mock_client.call_args.kwargs["headers"]

            assert headers["anthropic-beta"] == "pdfs-2024-09-25"

            json_data["messages"][0]["role"] == "user"
            _document_validation = AnthropicMessagesDocumentParam(
                **json_data["messages"][0]["content"][1]
            )
            assert _document_validation["type"] == "document"
            assert _document_validation["source"]["media_type"] == "application/pdf"
            assert _document_validation["source"]["type"] == "base64"


def test_convert_tool_response_to_message_with_values():
    """Test converting a tool response with 'values' key to a message"""
    tool_calls = [
        ChatCompletionToolCallChunk(
            id="test_id",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name="json_tool_call",
                arguments='{"values": {"name": "John", "age": 30}}',
            ),
            index=0,
        )
    ]

    message = AnthropicChatCompletion._convert_tool_response_to_message(
        tool_calls=tool_calls
    )

    assert message is not None
    assert message.content == '{"name": "John", "age": 30}'


def test_convert_tool_response_to_message_without_values():
    """
    Test converting a tool response without 'values' key to a message

    Anthropic API returns the JSON schema in the tool call, OpenAI Spec expects it in the message. This test ensures that the tool call is converted to a message correctly.

    Relevant issue: https://github.com/BerriAI/litellm/issues/6741
    """
    tool_calls = [
        ChatCompletionToolCallChunk(
            id="test_id",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name="json_tool_call", arguments='{"name": "John", "age": 30}'
            ),
            index=0,
        )
    ]

    message = AnthropicChatCompletion._convert_tool_response_to_message(
        tool_calls=tool_calls
    )

    assert message is not None
    assert message.content == '{"name": "John", "age": 30}'


def test_convert_tool_response_to_message_invalid_json():
    """Test converting a tool response with invalid JSON"""
    tool_calls = [
        ChatCompletionToolCallChunk(
            id="test_id",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name="json_tool_call", arguments="invalid json"
            ),
            index=0,
        )
    ]

    message = AnthropicChatCompletion._convert_tool_response_to_message(
        tool_calls=tool_calls
    )

    assert message is not None
    assert message.content == "invalid json"


def test_convert_tool_response_to_message_no_arguments():
    """Test converting a tool response with no arguments"""
    tool_calls = [
        ChatCompletionToolCallChunk(
            id="test_id",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(name="json_tool_call"),
            index=0,
        )
    ]

    message = AnthropicChatCompletion._convert_tool_response_to_message(
        tool_calls=tool_calls
    )

    assert message is None
