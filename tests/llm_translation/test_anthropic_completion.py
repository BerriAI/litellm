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
)
from litellm.types.llms.anthropic import AnthropicResponse
from litellm.types.utils import GenericStreamingChunk, ChatCompletionToolCallChunk
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.llms.anthropic.common_utils import process_anthropic_headers
from litellm.llms.anthropic.chat.handler import AnthropicChatCompletion
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest


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


@pytest.mark.parametrize(
    "tool_type, tool_config, message_content",
    [
        (
            "computer_20241022",
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
            },
            "Save a picture of a cat to my desktop.",
        ),
        (
            "web_fetch_20250910",
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "max_uses": 5,
            },
            "Please analyze the content at https://example.com/article",
        ),
    ],
)
def test_anthropic_tool_use(tool_type, tool_config, message_content):
    """Test Anthropic tool use with computer use and web fetch tools."""
    from litellm import completion

    litellm._turn_on_debug()

    tools = [tool_config]
    model = "claude-3-5-sonnet-20241022"
    messages = [{"role": "user", "content": message_content}]

    try:
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
        )
        print(f"Tool type: {tool_type}")
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

    tool, _ = AnthropicConfig()._map_tool_helper(tool=tool)

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
    assert _input_schema.get("name") == custom_schema["name"]
    assert _input_schema.get("age") == custom_schema["age"]
    assert "additionalProperties" not in _input_schema


from litellm import completion


class TestAnthropicCompletion(BaseLLMChatTest, BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "anthropic/claude-3-5-sonnet-latest"}

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "anthropic/claude-3-7-sonnet-latest",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_anthropic_tool_invoke,
        )

        result = convert_to_anthropic_tool_invoke([tool_call_no_arguments])
        print(result)

    def test_tool_call_and_json_response_format(self):
        """
        Test that the tool call and JSON response format is supported by the LLM API
        """
        litellm.set_verbose = True
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        class RFormat(BaseModel):
            question: str
            answer: str

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip("Model does not support response schema")

        try:
            res = litellm.completion(
                **base_completion_call_args,
                messages=[
                    {
                        "role": "system",
                        "content": "response user question with JSON object",
                    },
                    {"role": "user", "content": "Hey! What's the weather in NewYork?"},
                ],
                tool_choice="required",
                response_format=RFormat,
                tools=[
                    {
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
                ],
            )
            assert res is not None

            assert res.choices[0].message.tool_calls is not None
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")


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

    message = AnthropicConfig._convert_tool_response_to_message(tool_calls=tool_calls)

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

    message = AnthropicConfig._convert_tool_response_to_message(tool_calls=tool_calls)

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

    message = AnthropicConfig._convert_tool_response_to_message(tool_calls=tool_calls)

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

    message = AnthropicConfig._convert_tool_response_to_message(tool_calls=tool_calls)

    assert message is None


def test_anthropic_tool_with_image():
    from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
    import json

    b64_data = "iVBORw0KGgoAAAANSUhEu6U3//C9t/fKv5wDgpP1r5796XwC4zyH1D565bHGDqbY85AMb0nIQe+u3J390Xbtb9XgXxcK0/aqRXpdYcwgARbCN03FJk"
    image_url = f"data:image/png;base64,{b64_data}"
    messages = [
        {
            "content": [
                {"type": "text", "text": "go to github ryanhoangt by browser"},
                {
                    "type": "text",
                    "text": '<extra_info>\nThe following information has been included based on a keyword match for "github". It may or may not be relevant to the user\'s request.\n\nYou have access to an environment variable, `GITHUB_TOKEN`, which allows you to interact with\nthe GitHub API.\n\nYou can use `curl` with the `GITHUB_TOKEN` to interact with GitHub\'s API.\nALWAYS use the GitHub API for operations instead of a web browser.\n\nHere are some instructions for pushing, but ONLY do this if the user asks you to:\n* NEVER push directly to the `main` or `master` branch\n* Git config (username and email) is pre-set. Do not modify.\n* You may already be on a branch called `openhands-workspace`. Create a new branch with a better name before pushing.\n* Use the GitHub API to create a pull request, if you haven\'t already\n* Use the main branch as the base branch, unless the user requests otherwise\n* After opening or updating a pull request, send the user a short message with a link to the pull request.\n* Do all of the above in as few steps as possible. E.g. you could open a PR with one step by running the following bash commands:\n```bash\ngit remote -v && git branch # to find the current org, repo and branch\ngit checkout -b create-widget && git add . && git commit -m "Create widget" && git push -u origin create-widget\ncurl -X POST "https://api.github.com/repos/$ORG_NAME/$REPO_NAME/pulls" \\\n    -H "Authorization: Bearer $GITHUB_TOKEN" \\\n    -d \'{"title":"Create widget","head":"create-widget","base":"openhands-workspace"}\'\n```\n</extra_info>',
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "role": "user",
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you navigate to the GitHub profile of ryanhoangt using the browser.",
                }
            ],
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 1,
                    "function": {
                        "arguments": '{"code": "goto(\'https://github.com/ryanhoangt\')"}',
                        "name": "browser",
                    },
                    "id": "tooluse_UxfOQT6jRq-SvoQ9La_1sA",
                    "type": "function",
                }
            ],
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": "[Current URL: https://github.com/ryanhoangt]\n[Focused element bid: 119]\n\n[Action executed successfully.]\n============== BEGIN accessibility tree ==============\nRootWebArea 'ryanhoangt (Ryan H. Tran) · GitHub', focused\n\t[119] generic\n\t\t[120] generic\n\t\t\t[121] generic\n\t\t\t\t[122] link 'Skip to content', clickable\n\t\t\t\t[123] generic\n\t\t\t\t\t[124] generic\n\t\t\t\t[135] generic\n\t\t\t\t\t[137] generic, clickable\n\t\t\t\t[142] banner ''\n\t\t\t\t\t[143] heading 'Navigation Menu'\n\t\t\t\t\t[146] generic\n\t\t\t\t\t\t[147] generic\n\t\t\t\t\t\t\t[148] generic\n\t\t\t\t\t\t\t[155] link 'Homepage', clickable\n\t\t\t\t\t\t\t[158] generic\n\t\t\t\t\t\t[160] generic\n\t\t\t\t\t\t\t[161] generic\n\t\t\t\t\t\t\t\t[162] navigation 'Global'\n\t\t\t\t\t\t\t\t\t[163] list ''\n\t\t\t\t\t\t\t\t\t\t[164] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[165] button 'Product', expanded=False\n\t\t\t\t\t\t\t\t\t\t[244] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[245] button 'Solutions', expanded=False\n\t\t\t\t\t\t\t\t\t\t[288] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[289] button 'Resources', expanded=False\n\t\t\t\t\t\t\t\t\t\t[325] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[326] button 'Open Source', expanded=False\n\t\t\t\t\t\t\t\t\t\t[352] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[353] button 'Enterprise', expanded=False\n\t\t\t\t\t\t\t\t\t\t[392] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t[393] link 'Pricing', clickable\n\t\t\t\t\t\t\t\t[394] generic\n\t\t\t\t\t\t\t\t\t[395] generic\n\t\t\t\t\t\t\t\t\t\t[396] generic, clickable\n\t\t\t\t\t\t\t\t\t\t\t[397] button 'Search or jump to…', clickable, hasPopup='dialog'\n\t\t\t\t\t\t\t\t\t\t\t\t[398] generic\n\t\t\t\t\t\t\t\t\t\t[477] generic\n\t\t\t\t\t\t\t\t\t\t\t[478] generic\n\t\t\t\t\t\t\t\t\t\t\t[499] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[500] generic\n\t\t\t\t\t\t\t\t\t[534] generic\n\t\t\t\t\t\t\t\t\t\t[535] link 'Sign in', clickable\n\t\t\t\t\t\t\t\t\t[536] link 'Sign up', clickable\n\t\t\t[553] generic\n\t\t\t[554] generic\n\t\t\t[556] generic\n\t\t\t\t[557] main ''\n\t\t\t\t\t[558] generic\n\t\t\t\t\t[566] generic\n\t\t\t\t\t\t[567] generic\n\t\t\t\t\t\t\t[568] generic\n\t\t\t\t\t\t\t\t[569] generic\n\t\t\t\t\t\t\t\t\t[570] generic\n\t\t\t\t\t\t\t\t\t\t[571] LayoutTable ''\n\t\t\t\t\t\t\t\t\t\t\t[572] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[573] image '@ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t\t[574] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[575] strong ''\n\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t\t\t[576] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[577] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[578] link 'Follow', clickable\n\t\t\t\t\t\t\t\t[579] generic\n\t\t\t\t\t\t\t\t\t[580] generic\n\t\t\t\t\t\t\t\t\t\t[581] navigation 'User profile'\n\t\t\t\t\t\t\t\t\t\t\t[582] link 'Overview', clickable\n\t\t\t\t\t\t\t\t\t\t\t[585] link 'Repositories 136', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t[588] generic '136'\n\t\t\t\t\t\t\t\t\t\t\t[589] link 'Projects', clickable\n\t\t\t\t\t\t\t\t\t\t\t[593] link 'Packages', clickable\n\t\t\t\t\t\t\t\t\t\t\t[597] link 'Stars 311', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t[600] generic '311'\n\t\t\t\t\t[621] generic\n\t\t\t\t\t\t[622] generic\n\t\t\t\t\t\t\t[623] generic\n\t\t\t\t\t\t\t\t[624] generic\n\t\t\t\t\t\t\t\t\t[625] generic\n\t\t\t\t\t\t\t\t\t\t[626] LayoutTable ''\n\t\t\t\t\t\t\t\t\t\t\t[627] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[628] image '@ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t\t[629] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[630] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[631] strong ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t\t[632] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[633] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[634] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[635] link 'Follow', clickable\n\t\t\t\t\t\t\t\t\t[636] generic\n\t\t\t\t\t\t\t\t\t\t[637] generic\n\t\t\t\t\t\t\t\t\t\t\t[638] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[639] link \"View ryanhoangt's full-sized avatar\", clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[640] image \"View ryanhoangt's full-sized avatar\"\n\t\t\t\t\t\t\t\t\t\t\t\t[641] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[642] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[643] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[644] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[645] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[646] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '🎯'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[647] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[648] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Focusing'\n\t\t\t\t\t\t\t\t\t\t\t[649] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[650] heading 'Ryan H. Tran ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[651] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Ryan H. Tran'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[652] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'ryanhoangt'\n\t\t\t\t\t\t\t\t\t\t[660] generic\n\t\t\t\t\t\t\t\t\t\t\t[661] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[662] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[663] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[665] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[666] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[667] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[668] link 'Follow', clickable\n\t\t\t\t\t\t\t\t\t\t\t[669] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[670] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[671] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText \"Working with Attention. It's all we need\"\n\t\t\t\t\t\t\t\t\t\t\t\t[672] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[673] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[674] link '11 followers', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[677] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '11'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '·'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[678] link '30 following', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[679] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '30'\n\t\t\t\t\t\t\t\t\t\t\t\t[680] list ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[681] listitem 'Home location: Earth'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[684] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Earth'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[685] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[688] link 'hoangt.dev', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[689] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[692] link 'https://orcid.org/0009-0000-3619-0932', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[693] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[694] image 'X'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[696] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[697] link '@ryanhoangt', clickable\n\t\t\t\t\t\t\t\t\t\t[698] generic\n\t\t\t\t\t\t\t\t\t\t\t[699] heading 'Achievements'\n\t\t\t\t\t\t\t\t\t\t\t\t[700] link 'Achievements', clickable\n\t\t\t\t\t\t\t\t\t\t\t[701] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[702] link 'Achievement: Pair Extraordinaire', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[703] image 'Achievement: Pair Extraordinaire'\n\t\t\t\t\t\t\t\t\t\t\t\t[704] link 'Achievement: Pull Shark x2', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[705] image 'Achievement: Pull Shark'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[706] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'x2'\n\t\t\t\t\t\t\t\t\t\t\t\t[707] link 'Achievement: YOLO', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[708] image 'Achievement: YOLO'\n\t\t\t\t\t\t\t\t\t\t[720] generic\n\t\t\t\t\t\t\t\t\t\t\t[721] heading 'Highlights'\n\t\t\t\t\t\t\t\t\t\t\t[722] list ''\n\t\t\t\t\t\t\t\t\t\t\t\t[723] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[724] link 'Developer Program Member', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t[727] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[730] generic 'Label: Pro'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'PRO'\n\t\t\t\t\t\t\t\t\t\t[731] button 'Block or Report'\n\t\t\t\t\t\t\t\t\t\t\t[732] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[733] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Block or Report'\n\t\t\t\t\t\t\t\t\t\t[734] generic\n\t\t\t\t\t\t\t[775] generic\n\t\t\t\t\t\t\t\t[817] generic, clickable\n\t\t\t\t\t\t\t\t\t[818] generic\n\t\t\t\t\t\t\t\t\t\t[819] generic\n\t\t\t\t\t\t\t\t\t\t\t[820] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[821] heading 'PinnedLoading'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[822] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[826] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Loading'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[827] status '', live='polite', atomic, relevant='additions text'\n\t\t\t\t\t\t\t\t\t\t\t\t[828] list '', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t[829] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[830] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[831] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[832] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[833] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[836] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[837] link 'All-Hands-AI/OpenHands', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[838] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[839] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'OpenHands'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[843] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[844] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[845] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '🙌 OpenHands: Code Less, Make More'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[846] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[847] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[848] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[849] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[850] link 'stars 37.5k', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[851] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[852] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[853] link 'forks 4.2k', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[854] image 'forks'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[855] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[856] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[857] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[858] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[859] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[860] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[863] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[864] link 'nus-apr/auto-code-rover', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[865] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'nus-apr/'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[866] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'auto-code-rover'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[870] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[871] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[872] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'A project structure aware autonomous software engineer aiming for autonomous program improvement. Resolved 37.3% tasks (pass@1) in SWE-bench lite and 46.2% tasks (pass@1) in SWE-bench verified with…'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[873] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[874] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[875] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[876] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[877] link 'stars 2.7k', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[878] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[879] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[880] link 'forks 288', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[881] image 'forks'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[882] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[883] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[884] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[885] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[886] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[887] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[890] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[891] link 'TransformerLensOrg/TransformerLens', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[892] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'TransformerLensOrg/'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[893] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'TransformerLens'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[897] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[898] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[899] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'A library for mechanistic interpretability of GPT-style language models'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[900] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[901] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[902] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[903] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[904] link 'stars 1.6k', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[905] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[906] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[907] link 'forks 308', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[908] image 'forks'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[909] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[910] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[911] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[912] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[913] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[914] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[917] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[918] link 'danbraunai/simple_stories_train', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[919] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'danbraunai/'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[920] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'simple_stories_train'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[924] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[925] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[926] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Trains small LMs. Designed for training on SimpleStories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[927] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[928] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[929] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[930] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[931] link 'stars 3', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[932] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[933] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[934] link 'fork 1', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[935] image 'fork'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[936] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[937] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[938] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[939] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[940] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[941] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[944] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[945] link 'locify', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[946] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'locify'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[950] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[951] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[952] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'A library for LLM-based agents to navigate large codebases efficiently.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[953] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[954] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[955] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[956] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[957] link 'stars 6', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[958] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[959] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t[960] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[961] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[962] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[963] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[964] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[967] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[968] link 'iDunno', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[969] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'iDunno'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[973] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[974] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Public'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[975] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'A Distributed ML Cluster'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[976] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[977] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[978] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[979] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Java'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[980] link 'stars 3', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[981] image 'stars'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[982] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t[983] generic\n\t\t\t\t\t\t\t\t\t\t\t[984] generic\n\t\t\t\t\t\t\t\t\t\t\t\t[985] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[986] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[987] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[988] heading '481 contributions in the last year'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[989] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[990] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[991] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2099] grid 'Contribution Graph', clickable, multiselectable=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2100] caption ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Contribution Graph'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2101] rowgroup ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2102] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2103] gridcell 'Day of Week'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2104] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Day of Week'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2105] gridcell 'December'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2106] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'December'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2108] gridcell 'January'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2109] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'January'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2111] gridcell 'February'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2112] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'February'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2114] gridcell 'March'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2115] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'March'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2117] gridcell 'April'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2118] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'April'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2120] gridcell 'May'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2121] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'May'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2123] gridcell 'June'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2124] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'June'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2126] gridcell 'July'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2127] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'July'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2129] gridcell 'August'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2130] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'August'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2132] gridcell 'September'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2133] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'September'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2135] gridcell 'October'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2136] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'October'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2138] gridcell 'November'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2139] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'November'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2141] rowgroup ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2142] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2143] gridcell 'Sunday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2144] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Sunday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2146] gridcell '14 contributions on November 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-4'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2147] gridcell '3 contributions on December 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2148] gridcell '5 contributions on December 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2149] gridcell 'No contributions on December 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2150] gridcell '5 contributions on December 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2151] gridcell 'No contributions on December 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2152] gridcell '1 contribution on January 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2153] gridcell '2 contributions on January 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2154] gridcell '2 contributions on January 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2155] gridcell '2 contributions on January 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2156] gridcell 'No contributions on February 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2157] gridcell '1 contribution on February 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2158] gridcell 'No contributions on February 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2159] gridcell 'No contributions on February 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2160] gridcell 'No contributions on March 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2161] gridcell 'No contributions on March 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2162] gridcell 'No contributions on March 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2163] gridcell '2 contributions on March 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2164] gridcell '3 contributions on March 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2165] gridcell 'No contributions on April 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2166] gridcell '5 contributions on April 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2167] gridcell '2 contributions on April 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2168] gridcell 'No contributions on April 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2169] gridcell 'No contributions on May 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2170] gridcell 'No contributions on May 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2171] gridcell '1 contribution on May 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2172] gridcell '1 contribution on May 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2173] gridcell '2 contributions on June 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2174] gridcell '5 contributions on June 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2175] gridcell '1 contribution on June 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2176] gridcell 'No contributions on June 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2177] gridcell 'No contributions on June 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2178] gridcell 'No contributions on July 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2179] gridcell 'No contributions on July 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2180] gridcell '5 contributions on July 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2181] gridcell 'No contributions on July 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2182] gridcell '3 contributions on August 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2183] gridcell '1 contribution on August 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2184] gridcell '1 contribution on August 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2185] gridcell '1 contribution on August 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2186] gridcell '1 contribution on September 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2187] gridcell 'No contributions on September 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2188] gridcell '1 contribution on September 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2189] gridcell '2 contributions on September 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2190] gridcell '1 contribution on September 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2191] gridcell '2 contributions on October 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2192] gridcell '2 contributions on October 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2193] gridcell '4 contributions on October 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2194] gridcell '1 contribution on October 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2195] gridcell '14 contributions on November 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-4'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2196] gridcell '10 contributions on November 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-3'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2197] gridcell '2 contributions on November 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2198] gridcell '1 contribution on November 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2199] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2200] gridcell 'Monday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2201] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Monday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2203] gridcell 'No contributions on November 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2204] gridcell 'No contributions on December 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2205] gridcell '2 contributions on December 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2206] gridcell '2 contributions on December 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2207] gridcell '3 contributions on December 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2208] gridcell '2 contributions on January 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2209] gridcell '1 contribution on January 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2210] gridcell 'No contributions on January 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2211] gridcell '3 contributions on January 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2212] gridcell '3 contributions on January 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2213] gridcell 'No contributions on February 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2214] gridcell '2 contributions on February 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2215] gridcell '1 contribution on February 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2216] gridcell 'No contributions on February 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2217] gridcell 'No contributions on March 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2218] gridcell '1 contribution on March 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2219] gridcell '1 contribution on March 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2220] gridcell 'No contributions on March 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2221] gridcell '1 contribution on April 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2222] gridcell '1 contribution on April 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2223] gridcell '1 contribution on April 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2224] gridcell '1 contribution on April 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2225] gridcell '1 contribution on April 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2226] gridcell '2 contributions on May 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2227] gridcell 'No contributions on May 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2228] gridcell 'No contributions on May 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2229] gridcell '1 contribution on May 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2230] gridcell 'No contributions on June 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2231] gridcell '3 contributions on June 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2232] gridcell 'No contributions on June 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2233] gridcell 'No contributions on June 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2234] gridcell '1 contribution on July 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2235] gridcell 'No contributions on July 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2236] gridcell 'No contributions on July 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2237] gridcell 'No contributions on July 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2238] gridcell '1 contribution on July 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2239] gridcell '1 contribution on August 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2240] gridcell 'No contributions on August 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2241] gridcell '2 contributions on August 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2242] gridcell '1 contribution on August 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2243] gridcell 'No contributions on September 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2244] gridcell 'No contributions on September 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2245] gridcell '1 contribution on September 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2246] gridcell '2 contributions on September 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2247] gridcell '1 contribution on September 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2248] gridcell '1 contribution on October 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2249] gridcell '1 contribution on October 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2250] gridcell '7 contributions on October 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2251] gridcell '1 contribution on October 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2252] gridcell '4 contributions on November 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2253] gridcell '2 contributions on November 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2254] gridcell '1 contribution on November 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2255] gridcell '1 contribution on November 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2256] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2257] gridcell 'Tuesday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2258] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Tuesday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2260] gridcell 'No contributions on November 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2261] gridcell '3 contributions on December 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2262] gridcell '1 contribution on December 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2263] gridcell 'No contributions on December 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2264] gridcell '2 contributions on December 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2265] gridcell '2 contributions on January 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2266] gridcell 'No contributions on January 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2267] gridcell 'No contributions on January 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2268] gridcell 'No contributions on January 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2269] gridcell 'No contributions on January 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2270] gridcell 'No contributions on February 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2271] gridcell 'No contributions on February 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2272] gridcell 'No contributions on February 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2273] gridcell 'No contributions on February 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2274] gridcell 'No contributions on March 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2275] gridcell 'No contributions on March 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2276] gridcell 'No contributions on March 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2277] gridcell 'No contributions on March 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2278] gridcell '1 contribution on April 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2279] gridcell '1 contribution on April 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2280] gridcell '1 contribution on April 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2281] gridcell '2 contributions on April 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2282] gridcell '1 contribution on April 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2283] gridcell 'No contributions on May 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2284] gridcell '1 contribution on May 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2285] gridcell '2 contributions on May 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2286] gridcell '2 contributions on May 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2287] gridcell '1 contribution on June 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2288] gridcell '1 contribution on June 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2289] gridcell 'No contributions on June 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2290] gridcell 'No contributions on June 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2291] gridcell '1 contribution on July 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2292] gridcell '1 contribution on July 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2293] gridcell '1 contribution on July 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2294] gridcell '1 contribution on July 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2295] gridcell 'No contributions on July 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2296] gridcell 'No contributions on August 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2297] gridcell 'No contributions on August 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2298] gridcell 'No contributions on August 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2299] gridcell 'No contributions on August 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2300] gridcell '1 contribution on September 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2301] gridcell 'No contributions on September 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2302] gridcell 'No contributions on September 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2303] gridcell '2 contributions on September 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2304] gridcell '1 contribution on October 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2305] gridcell '1 contribution on October 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2306] gridcell '1 contribution on October 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2307] gridcell '3 contributions on October 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2308] gridcell '2 contributions on October 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2309] gridcell '3 contributions on November 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2310] gridcell '3 contributions on November 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2311] gridcell '2 contributions on November 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2312] gridcell 'No contributions on November 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2313] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2314] gridcell 'Wednesday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2315] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Wednesday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2317] gridcell '1 contribution on November 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2318] gridcell '3 contributions on December 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2319] gridcell '1 contribution on December 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2320] gridcell '4 contributions on December 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2321] gridcell '2 contributions on December 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2322] gridcell '1 contribution on January 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2323] gridcell 'No contributions on January 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2324] gridcell 'No contributions on January 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2325] gridcell 'No contributions on January 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2326] gridcell 'No contributions on January 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2327] gridcell 'No contributions on February 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2328] gridcell '1 contribution on February 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2329] gridcell '1 contribution on February 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2330] gridcell '1 contribution on February 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2331] gridcell 'No contributions on March 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2332] gridcell 'No contributions on March 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2333] gridcell 'No contributions on March 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2334] gridcell 'No contributions on March 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2335] gridcell '3 contributions on April 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2336] gridcell 'No contributions on April 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2337] gridcell '1 contribution on April 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2338] gridcell 'No contributions on April 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2339] gridcell 'No contributions on May 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2340] gridcell '1 contribution on May 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2341] gridcell '2 contributions on May 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2342] gridcell '1 contribution on May 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2343] gridcell 'No contributions on May 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2344] gridcell '3 contributions on June 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2345] gridcell '1 contribution on June 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2346] gridcell '1 contribution on June 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2347] gridcell '1 contribution on June 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2348] gridcell 'No contributions on July 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2349] gridcell '1 contribution on July 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2350] gridcell 'No contributions on July 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2351] gridcell '1 contribution on July 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2352] gridcell '2 contributions on July 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2353] gridcell '1 contribution on August 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2354] gridcell '1 contribution on August 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2355] gridcell '2 contributions on August 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2356] gridcell '1 contribution on August 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2357] gridcell 'No contributions on September 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2358] gridcell 'No contributions on September 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2359] gridcell '1 contribution on September 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2360] gridcell '1 contribution on September 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2361] gridcell '1 contribution on October 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2362] gridcell '1 contribution on October 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2363] gridcell '3 contributions on October 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2364] gridcell '4 contributions on October 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2365] gridcell '1 contribution on October 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2366] gridcell '2 contributions on November 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2367] gridcell '1 contribution on November 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2368] gridcell 'No contributions on November 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2369] gridcell '1 contribution on November 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2370] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2371] gridcell 'Thursday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2372] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Thursday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2374] gridcell 'No contributions on November 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2375] gridcell 'No contributions on December 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2376] gridcell '2 contributions on December 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2377] gridcell '3 contributions on December 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2378] gridcell 'No contributions on December 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2379] gridcell 'No contributions on January 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2380] gridcell 'No contributions on January 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2381] gridcell 'No contributions on January 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2382] gridcell '1 contribution on January 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2383] gridcell 'No contributions on February 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2384] gridcell 'No contributions on February 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2385] gridcell 'No contributions on February 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2386] gridcell '1 contribution on February 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2387] gridcell '1 contribution on February 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2388] gridcell '6 contributions on March 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2389] gridcell 'No contributions on March 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2390] gridcell 'No contributions on March 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2391] gridcell '1 contribution on March 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2392] gridcell '3 contributions on April 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2393] gridcell '1 contribution on April 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2394] gridcell '1 contribution on April 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2395] gridcell 'No contributions on April 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2396] gridcell '1 contribution on May 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2397] gridcell '1 contribution on May 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2398] gridcell 'No contributions on May 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2399] gridcell 'No contributions on May 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2400] gridcell '2 contributions on May 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2401] gridcell '1 contribution on June 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2402] gridcell 'No contributions on June 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2403] gridcell 'No contributions on June 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2404] gridcell '1 contribution on June 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2405] gridcell '3 contributions on July 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2406] gridcell '1 contribution on July 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2407] gridcell '1 contribution on July 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2408] gridcell '1 contribution on July 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2409] gridcell 'No contributions on August 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2410] gridcell '1 contribution on August 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2411] gridcell 'No contributions on August 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2412] gridcell '1 contribution on August 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2413] gridcell '1 contribution on August 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2414] gridcell '1 contribution on September 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2415] gridcell '1 contribution on September 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2416] gridcell '1 contribution on September 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2417] gridcell '1 contribution on September 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2418] gridcell '1 contribution on October 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2419] gridcell '2 contributions on October 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2420] gridcell '8 contributions on October 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2421] gridcell '1 contribution on October 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2422] gridcell '2 contributions on October 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2423] gridcell '1 contribution on November 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2424] gridcell '3 contributions on November 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2425] gridcell '2 contributions on November 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2426] gridcell '3 contributions on November 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2427] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2428] gridcell 'Friday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2429] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Friday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2431] gridcell 'No contributions on December 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2432] gridcell '1 contribution on December 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2433] gridcell '2 contributions on December 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2434] gridcell '1 contribution on December 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2435] gridcell '1 contribution on December 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2436] gridcell 'No contributions on January 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2437] gridcell '1 contribution on January 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2438] gridcell '1 contribution on January 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2439] gridcell 'No contributions on January 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2440] gridcell '1 contribution on February 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2441] gridcell 'No contributions on February 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2442] gridcell '1 contribution on February 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2443] gridcell 'No contributions on February 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2444] gridcell 'No contributions on March 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2445] gridcell 'No contributions on March 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2446] gridcell 'No contributions on March 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2447] gridcell 'No contributions on March 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2448] gridcell '1 contribution on March 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2449] gridcell 'No contributions on April 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2450] gridcell '2 contributions on April 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2451] gridcell 'No contributions on April 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2452] gridcell 'No contributions on April 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2453] gridcell 'No contributions on May 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2454] gridcell '1 contribution on May 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2455] gridcell '1 contribution on May 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2456] gridcell 'No contributions on May 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2457] gridcell 'No contributions on May 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2458] gridcell 'No contributions on June 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2459] gridcell 'No contributions on June 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2460] gridcell 'No contributions on June 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2461] gridcell '1 contribution on June 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2462] gridcell '1 contribution on July 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2463] gridcell '2 contributions on July 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2464] gridcell 'No contributions on July 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2465] gridcell '1 contribution on July 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2466] gridcell 'No contributions on August 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2467] gridcell '2 contributions on August 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2468] gridcell '2 contributions on August 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2469] gridcell 'No contributions on August 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2470] gridcell '1 contribution on August 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2471] gridcell 'No contributions on September 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2472] gridcell '1 contribution on September 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2473] gridcell '3 contributions on September 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2474] gridcell '1 contribution on September 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2475] gridcell 'No contributions on October 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2476] gridcell '3 contributions on October 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2477] gridcell '5 contributions on October 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2478] gridcell '3 contributions on October 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2479] gridcell '1 contribution on November 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2480] gridcell '1 contribution on November 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2481] gridcell '3 contributions on November 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2482] gridcell '1 contribution on November 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2483] gridcell ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2484] row ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2485] gridcell 'Saturday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2486] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Saturday'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2488] gridcell '10 contributions on December 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-3'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2489] gridcell '13 contributions on December 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-4'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2490] gridcell 'No contributions on December 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2491] gridcell '1 contribution on December 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2492] gridcell '10 contributions on December 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-3'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2493] gridcell '3 contributions on January 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2494] gridcell '1 contribution on January 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2495] gridcell '1 contribution on January 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2496] gridcell '3 contributions on January 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2497] gridcell 'No contributions on February 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2498] gridcell '1 contribution on February 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2499] gridcell 'No contributions on February 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2500] gridcell '1 contribution on February 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2501] gridcell 'No contributions on March 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2502] gridcell 'No contributions on March 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2503] gridcell 'No contributions on March 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2504] gridcell 'No contributions on March 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2505] gridcell '2 contributions on March 30th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2506] gridcell '1 contribution on April 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2507] gridcell '5 contributions on April 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2508] gridcell '1 contribution on April 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2509] gridcell 'No contributions on April 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2510] gridcell 'No contributions on May 4th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2511] gridcell '1 contribution on May 11th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2512] gridcell '1 contribution on May 18th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2513] gridcell 'No contributions on May 25th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2514] gridcell '2 contributions on June 1st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2515] gridcell 'No contributions on June 8th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2516] gridcell 'No contributions on June 15th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2517] gridcell 'No contributions on June 22nd.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2518] gridcell 'No contributions on June 29th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2519] gridcell '1 contribution on July 6th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2520] gridcell 'No contributions on July 13th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2521] gridcell '1 contribution on July 20th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2522] gridcell 'No contributions on July 27th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2523] gridcell '1 contribution on August 3rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2524] gridcell 'No contributions on August 10th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2525] gridcell 'No contributions on August 17th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2526] gridcell 'No contributions on August 24th.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2527] gridcell 'No contributions on August 31st.', clickable, selected=False, describedby='contribution-graph-legend-level-0'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2528] gridcell '1 contribution on September 7th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2529] gridcell '1 contribution on September 14th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2530] gridcell '1 contribution on September 21st.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2531] gridcell '1 contribution on September 28th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2532] gridcell '1 contribution on October 5th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2533] gridcell '5 contributions on October 12th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2534] gridcell '5 contributions on October 19th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2535] gridcell '7 contributions on October 26th.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2536] gridcell '5 contributions on November 2nd.', clickable, selected=False, describedby='contribution-graph-legend-level-2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2537] gridcell '17 contributions on November 9th.', clickable, selected=False, describedby='contribution-graph-legend-level-4'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2538] gridcell '1 contribution on November 16th.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2539] gridcell '1 contribution on November 23rd.', clickable, selected=False, describedby='contribution-graph-legend-level-1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2540] gridcell ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2541] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2542] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2543] link 'Learn how we count contributions', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2544] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2545] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Less'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2546] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2547] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'No contributions.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2548] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2549] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Low contributions.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2550] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2551] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Medium-low contributions.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2552] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2553] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Medium-high contributions.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2554] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2555] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'High contributions.'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2556] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'More'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2557] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2558] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2559] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2560] navigation 'Organizations'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2561] link '@All-Hands-AI', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2562] image ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2563] link '@Globe-NLP-Lab', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2564] image ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2565] link '@TransformerLensOrg', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2566] image ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2567] Details '', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2568] button 'More', clickable, hasPopup='menu', expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2569] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2591] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2592] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2593] heading 'Activity overview'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2594] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2597] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Contributed to'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2598] link 'All-Hands-AI/OpenHands', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ','\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2599] link 'All-Hands-AI/openhands-aci', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ','\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2600] link 'ryanhoangt/locify', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2601] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'and 36 other repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2602] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2603] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2604] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2608] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Loading'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2609] SvgRoot \"A graph representing ryanhoangt's contributions from November 26, 2023 to November 28, 2024. The contributions are 77% commits, 15% pull requests, 4% code review, 4% issues.\"\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2611] group ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2612] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2613] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2614] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2615] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2616] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2617] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2618] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2619] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '4%'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2620] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Code review'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2621] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '4%'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2622] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Issues'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2623] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '15%'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2624] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Pull requests'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2625] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '77%'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2626] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Commits'\n\t\t\t\t\t\t\t\t\t\t\t\t\t[2627] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2629] heading 'Contribution activity'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2630] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2631] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2632] heading 'November 2024'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2633] generic 'November 2024'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2634] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '2024'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2635] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2636] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2639] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2640] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2641] button 'Created 24 commits in 3 repositories', clickable, expanded=True\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2642] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Created 24 commits in 3 repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2643] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2644] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2650] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2651] list ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2652] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2653] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2654] link 'All-Hands-AI/openhands-aci', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2655] link '16 commits', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2656] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2657] image '67% of commits in November were made to All-Hands-AI/openhands-aci'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2658] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2659] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2660] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2661] link 'All-Hands-AI/OpenHands', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2662] link '4 commits', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2663] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2664] image '17% of commits in November were made to All-Hands-AI/OpenHands'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2665] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2666] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2667] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2668] link 'ryanhoangt/p4cm4n', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2669] link '4 commits', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2670] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2671] image '17% of commits in November were made to ryanhoangt/p4cm4n'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2672] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2673] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2674] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2677] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2678] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2679] button 'Created 3 repositories', clickable, expanded=True\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2680] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Created 3 repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2681] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2682] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2688] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2689] list ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2690] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2691] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2692] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2695] link 'ryanhoangt/TapeAgents', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2696] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2697] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2698] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2699] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2700] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2701] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'This contribution was made on Nov 21'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2703] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2704] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2705] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2708] link 'ryanhoangt/multilspy', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2709] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2710] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2711] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2712] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Python'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2713] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2714] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'This contribution was made on Nov 8'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2716] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2717] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2718] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2721] link 'ryanhoangt/anthropic-quickstarts', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2722] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2723] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2724] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2725] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'TypeScript'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2726] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2727] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'This contribution was made on Nov 3'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2729] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2730] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2733] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2734] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2735] heading 'Created a pull request in All-Hands-AI/OpenHands that received 20 comments'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2736] link 'All-Hands-AI/OpenHands', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2737] link 'Nov 17', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2738] time ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Nov 17'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2739] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2742] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2743] heading '[Experiment] Add symbol navigation commands into the editor'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2744] link '[Experiment] Add symbol navigation commands into the editor', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2745] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2746] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2747] strong ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'End-user friendly description of the problem this fixes or functionality that this introduces'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Include this change in the Release Notes. If checke…'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2748] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2749] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2750] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '+311'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2751] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '−105'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2752] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2753] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2754] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2755] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2756] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2757] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'lines changed'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2758] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '•'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '20 comments'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2759] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2760] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2763] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2764] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2765] button 'Opened 17 other pull requests in 5 repositories', clickable, expanded=True\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2766] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Opened 17 other pull requests in 5 repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2767] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2768] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2774] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2775] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2776] button 'All-Hands-AI/openhands-aci 2 open 8 merged', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2777] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2778] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/openhands-aci'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2779] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2780] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'open'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2781] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '8'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'merged'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2782] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2786] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2896] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2897] button 'All-Hands-AI/OpenHands 4 merged', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2898] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2899] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/OpenHands'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2900] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2901] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '4'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'merged'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2902] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2906] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2951] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2952] button 'ryanhoangt/multilspy 1 open', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2953] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2954] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'ryanhoangt/multilspy'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2955] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2956] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'open'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2957] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2961] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2976] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2977] button 'anthropics/anthropic-quickstarts 1 closed', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2978] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2979] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'anthropics/anthropic-quickstarts'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2980] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2981] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'closed'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2982] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[2986] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3001] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3002] button 'danbraunai/simple_stories_train 1 open', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3003] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3004] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'danbraunai/simple_stories_train'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3005] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3006] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'open'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3007] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3011] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3026] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3027] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3030] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3031] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3032] button 'Reviewed 6 pull requests in 2 repositories', clickable, expanded=True\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3033] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Reviewed 6 pull requests in 2 repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3034] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3035] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3041] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3042] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3043] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3044] button 'All-Hands-AI/openhands-aci 3 pull requests', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3045] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3046] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/openhands-aci'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3047] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '3 pull requests'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3048] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3052] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3087] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3088] button 'All-Hands-AI/OpenHands 3 pull requests', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3089] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3090] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/OpenHands'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3091] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '3 pull requests'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3092] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3096] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3131] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3132] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3135] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3136] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3137] heading 'Created an issue in All-Hands-AI/OpenHands that received 1 comment'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3138] link 'All-Hands-AI/OpenHands', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3139] link 'Nov 7', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3140] time ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Nov 7'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3141] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3145] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3146] heading '[Bug]: Patch collection after eval was empty although the agent did make changes'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3147] link '[Bug]: Patch collection after eval was empty although the agent did make changes', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3148] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3149] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText \"Is there an existing issue for the same bug? I have checked the existing issues. Describe the bug and reproduction steps I'm running eval for\"\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3150] link '#4782', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3151] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3152] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3153] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3154] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3158] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3159] SvgRoot ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3160] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3161] graphics-symbol ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3162] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1 task done'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3163] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '•'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1 comment'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3164] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3165] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3169] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3170] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3171] button 'Opened 3 other issues in 2 repositories', clickable, expanded=True\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3172] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Opened 3 other issues in 2 repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3173] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3174] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3180] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3181] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3182] button 'ryanhoangt/locify 2 open', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3183] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3184] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'ryanhoangt/locify'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3185] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3186] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '2'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'open'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3187] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3191] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3218] Details ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3219] button 'All-Hands-AI/openhands-aci 1 closed', clickable, expanded=False\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3220] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3221] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'All-Hands-AI/openhands-aci'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3222] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3223] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '1'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'closed'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3224] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3228] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3244] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3245] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3248] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3249] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '31 contributions in private repositories'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3250] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Nov 5 – Nov 25'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3251] Section ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3252] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3256] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Loading'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3257] button 'Show more activity', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3258] paragraph ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText 'Seeing something unexpected? Take a look at the'\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3259] link 'GitHub profile guide', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tStaticText '.'\n\t\t\t\t\t\t\t\t\t\t\t\t[3260] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t[3261] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3263] generic\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3264] list ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3265] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3266] link 'Contribution activity in 2024', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3267] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3268] link 'Contribution activity in 2023', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3269] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3270] link 'Contribution activity in 2022', clickable\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3271] listitem ''\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t[3272] link 'Contribution activity in 2021', clickable\n\t\t\t[3273] contentinfo ''\n\t\t\t\t[3274] heading 'Footer'\n\t\t\t\t[3275] generic\n\t\t\t\t\t[3276] generic\n\t\t\t\t\t\t[3277] link 'Homepage', clickable\n\t\t\t\t\t\t[3280] generic\n\t\t\t\t\t\t\tStaticText '© 2024 GitHub,\\xa0Inc.'\n\t\t\t\t\t[3281] navigation 'Footer'\n\t\t\t\t\t\t[3282] heading 'Footer navigation'\n\t\t\t\t\t\t[3283] list 'Footer navigation'\n\t\t\t\t\t\t\t[3284] listitem ''\n\t\t\t\t\t\t\t\t[3285] link 'Terms', clickable\n\t\t\t\t\t\t\t[3286] listitem ''\n\t\t\t\t\t\t\t\t[3287] link 'Privacy', clickable\n\t\t\t\t\t\t\t[3288] listitem ''\n\t\t\t\t\t\t\t\t[3289] link 'Security', clickable\n\t\t\t\t\t\t\t[3290] listitem ''\n\t\t\t\t\t\t\t\t[3291] link 'Status', clickable\n\t\t\t\t\t\t\t[3292] listitem ''\n\t\t\t\t\t\t\t\t[3293] link 'Docs', clickable\n\t\t\t\t\t\t\t[3294] listitem ''\n\t\t\t\t\t\t\t\t[3295] link 'Contact', clickable\n\t\t\t\t\t\t\t[3296] listitem ''\n\t\t\t\t\t\t\t\t[3297] generic\n\t\t\t\t\t\t\t\t\t[3298] button 'Manage cookies', clickable\n\t\t\t\t\t\t\t[3299] listitem ''\n\t\t\t\t\t\t\t\t[3300] generic\n\t\t\t\t\t\t\t\t\t[3301] button 'Do not share my personal information', clickable\n\t\t\t[3302] generic\n\t\t[3314] generic, live='polite', atomic, relevant='additions text'\n\t\t[3315] generic, live='assertive', atomic, relevant='additions text'\n============== END accessibility tree ==============\nThe screenshot of the current page is shown below.\n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
            "role": "tool",
            "cache_control": {"type": "ephemeral"},
            "tool_call_id": "tooluse_UxfOQT6jRq-SvoQ9La_1sA",
            "name": "browser",
        },
    ]

    result = prompt_factory(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        custom_llm_provider="anthropic",
    )

    assert b64_data in json.dumps(result)


def test_anthropic_map_openai_params_tools_and_json_schema():
    import json

    args = {
        "non_default_params": {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "properties": {
                            "question": {"title": "Question", "type": "string"},
                            "answer": {"title": "Answer", "type": "string"},
                        },
                        "required": ["question", "answer"],
                        "title": "RFormat",
                        "type": "object",
                        "additionalProperties": False,
                    },
                    "name": "RFormat",
                    "strict": True,
                },
            },
            "tools": [
                {
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
            ],
            "tool_choice": "required",
        }
    }

    mapped_params = litellm.AnthropicConfig().map_openai_params(
        non_default_params=args["non_default_params"],
        optional_params={},
        model="claude-3-5-sonnet-20240620",
        drop_params=False,
    )

    assert "Question" in json.dumps(mapped_params)


from litellm.constants import RESPONSE_FORMAT_TOOL_NAME


@pytest.mark.parametrize(
    "json_mode, tool_calls, expect_null_response",
    [
        (
            True,
            [
                {
                    "id": "toolu_013JszbnYBVygTxh6EGHEHia",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "New York, NY"}',
                    },
                    "index": 0,
                }
            ],
            True,
        ),
        (
            True,
            [
                {
                    "id": "toolu_013JszbnYBVygTxh6EGHEHia",
                    "type": "function",
                    "function": {
                        "name": RESPONSE_FORMAT_TOOL_NAME,
                        "arguments": '{"location": "New York, NY"}',
                    },
                    "index": 0,
                }
            ],
            False,
        ),
        (
            False,
            [
                {
                    "id": "toolu_013JszbnYBVygTxh6EGHEHia",
                    "type": "function",
                    "function": {
                        "name": RESPONSE_FORMAT_TOOL_NAME,
                        "arguments": '{"location": "New York, NY"}',
                    },
                    "index": 0,
                }
            ],
            True,
        ),
    ],
)
def test_anthropic_json_mode_and_tool_call_response(
    json_mode, tool_calls, expect_null_response
):
    result = litellm.AnthropicConfig()._transform_response_for_json_mode(
        json_mode=json_mode,
        tool_calls=tool_calls,
    )

    assert (
        result is None if expect_null_response else result is not None
    ), f"Expected result to be {None if expect_null_response else 'not None'}, but got {result}"


@pytest.mark.parametrize(
    "stop_input,expected_output,drop_params",
    [
        ("stop", ["stop"], True),  # basic string
        (["stop1", "stop2"], ["stop1", "stop2"], True),  # list of strings
        (
            "   ",
            None,
            True,
        ),  # whitespace string should be dropped when drop_params is True
        (
            "   ",
            ["   "],
            False,
        ),  # whitespace string should be kept when drop_params is False
        (
            ["stop1", "  ", "stop2"],
            ["stop1", "stop2"],
            True,
        ),  # list with whitespace that should be filtered
        (
            ["stop1", "  ", "stop2"],
            ["stop1", "  ", "stop2"],
            False,
        ),  # list with whitespace that should be kept
        (None, None, True),  # None input
    ],
)
def test_map_stop_sequences(stop_input, expected_output, drop_params):
    """Test the _map_stop_sequences method of AnthropicConfig"""
    litellm.drop_params = drop_params
    config = AnthropicConfig()
    result = config._map_stop_sequences(stop_input)
    assert result == expected_output


@pytest.mark.asyncio
async def test_anthropic_structured_output():
    """
    Test the _transform_response_for_structured_output

    Relevant Issue: https://github.com/BerriAI/litellm/issues/8291
    """
    from litellm import acompletion

    args = {
        "model": "claude-3-5-sonnet-20240620",
        "seed": 3015206306868917280,
        "stop": None,
        "messages": [
            {
                "role": "system",
                "content": 'You are a hello world agent.\nAlways respond in the following valid JSON format: {\n  "response": "response",\n}\n',
            },
            {"role": "user", "content": "Respond with hello world"},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "drop_params": True,
    }

    response = await acompletion(**args)
    assert response is not None

    print(response)


def test_anthropic_citations_api():
    """
    Test the citations API
    """
    from litellm import completion

    try:
        resp = completion(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "The grass is green. The sky is blue.",
                            },
                            "title": "My Document",
                            "context": "This is a trustworthy document.",
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "text",
                            "text": "What color is the grass and sky?",
                        },
                    ],
                }
            ],
        )

    except litellm.InternalServerError:
        pytest.skip("Anthropic overloaded")

    citations = resp.choices[0].message.provider_specific_fields["citations"]

    assert citations is not None
    if citations:
        citation = citations[0][0]
        assert "supported_text" in citation
        assert "cited_text" in citation
        assert "document_index" in citation
        assert "document_title" in citation
        assert "start_char_index" in citation
        assert "end_char_index" in citation


def test_anthropic_citations_api_streaming():
    from litellm import completion

    resp = completion(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The grass is green. The sky is blue.",
                        },
                        "title": "My Document",
                        "context": "This is a trustworthy document.",
                        "citations": {"enabled": True},
                    },
                    {
                        "type": "text",
                        "text": "What color is the grass and sky?",
                    },
                ],
            }
        ],
        stream=True,
    )

    has_citations = False
    for chunk in resp:
        print(f"returned chunk: {chunk}")
        if provider_specific_fields := chunk.choices[0].delta.provider_specific_fields:
            if "citation" in provider_specific_fields:
                has_citations = True

    assert has_citations


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-7-sonnet-20250219",
        "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ],
)
def test_anthropic_thinking_output(model):
    from litellm import completion

    litellm._turn_on_debug()

    resp = completion(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    print(resp)
    assert resp.choices[0].message.reasoning_content is not None
    assert isinstance(resp.choices[0].message.reasoning_content, str)
    assert resp.choices[0].message.thinking_blocks is not None
    assert isinstance(resp.choices[0].message.thinking_blocks, list)
    assert len(resp.choices[0].message.thinking_blocks) > 0

    assert resp.choices[0].message.thinking_blocks[0]["type"] == "thinking"
    assert resp.choices[0].message.thinking_blocks[0]["signature"] is not None


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-7-sonnet-20250219",
        # "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # "bedrock/invoke/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ],
)
def test_anthropic_thinking_output_stream(model):
    litellm.set_verbose = True
    try:
        # litellm._turn_on_debug()
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
            timeout=10,
        )

        reasoning_content_exists = False
        signature_block_exists = False
        for chunk in resp:
            print(f"chunk 2: {chunk}")
            if (
                hasattr(chunk.choices[0].delta, "thinking_blocks")
                and chunk.choices[0].delta.thinking_blocks is not None
                and chunk.choices[0].delta.reasoning_content is not None
                and isinstance(chunk.choices[0].delta.thinking_blocks, list)
                and len(chunk.choices[0].delta.thinking_blocks) > 0
                and isinstance(chunk.choices[0].delta.reasoning_content, str)
            ):
                reasoning_content_exists = True
                print(chunk.choices[0].delta.thinking_blocks[0])
                if chunk.choices[0].delta.thinking_blocks[0].get("signature"):
                    signature_block_exists = True
                    assert (
                        chunk.choices[0].delta.thinking_blocks[0]["type"] == "thinking"
                    )
        assert reasoning_content_exists
        assert signature_block_exists
    except litellm.Timeout:
        pytest.skip("Model is timing out")


def test_anthropic_custom_headers():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    tools = [
        {
            "type": "computer_20241022",
            "function": {
                "name": "get_current_weather",
                "parameters": {
                    "display_height_px": 100,
                    "display_width_px": 100,
                    "display_number": 1,
                },
            },
        }
    ]

    with patch.object(client, "post") as mock_post:
        try:
            resp = completion(
                model="claude-3-5-sonnet-20240620",
                headers={"anthropic-beta": "structured-output-2024-03-01"},
                messages=[
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                client=client,
                tools=tools,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert "structured-output-2024-03-01" in headers["anthropic-beta"]


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-7-sonnet-20250219",
        # "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ],
)
def test_anthropic_thinking_in_assistant_message(model):
    litellm._turn_on_debug()
    params = {
        "model": model,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Alright, let's get started with resolving this issue about implementing `__len__` for the Pipeline class in scikit-learn.\n\n### Step 1: Explore the repository structure to understand it better\n\nLet's first explore the repository structure to get a better understanding of the codebase:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nThis will give us an overview of the top-level directories and files. From there, we can delve deeper into relevant parts of the code.\n\nLet's look at the main organization of the repository first, then we'll locate the Pipeline class, understand how it's currently implemented, and then add the `__len__` method to it.\n\nNext, let's look at the pipeline module, as the issue mentions `pipe[:len(pipe)]`, indicating that there is indexing support for Pipeline, but it lacks a `__len__` method.\n\nSince the issue mentions that Pipeline supports indexing but not `__len__`, I'll check the Pipeline class to understand its implementation.\n\nThe Pipeline class is likely to be in sklearn/pipeline.py or within a pipeline directory. Let's look for it:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nThis will help us locate the file containing the Pipeline class.\n\nOnce we've located the file, we'll examine it to understand:\n1. How indexing is currently implemented\n2. Where we need to add the `__len__` method\n3. How the length of a pipeline should be determined\n\nThen, we'll implement the `__len__` method, test our changes to ensure they fix the issue, and verify that our changes don't break existing functionality.\n\nLet's begin by exploring the repository.\n\n### Step 2: Reproduce the error\n\nAfter exploring the repository, I'll create a script to reproduce the error described in the issue. This script will use the code provided in the \"Steps/Code to Reproduce\" section of the issue description.\n\nThen I'll run the script to confirm the error occurs as described, which will help us understand what needs to be fixed.\n\n### Step 3: Fix the issue\n\nOnce I've understood the problem and located the relevant files, I'll modify the code to add the `__len__` method to the Pipeline class. This method should return the number of steps in the pipeline, which is consistent with what a user would expect when calling `len(pipe)`.\n\n### Step 4: Test the fix\n\nAfter implementing the fix, I'll rerun the reproduction script to verify that the error is resolved and `len(pipe)` works correctly.\n\n### Step 5: Consider edge cases\n\nI'll think about potential edge cases like:\n- Empty pipelines\n- Very long pipelines\n- Any other special cases where the `__len__` implementation might behave unexpectedly\n\nI'll add tests for these cases to ensure our implementation handles them correctly.\n\n### Step 6: Comprehensive testing\n\nI'll run the existing tests to ensure our changes don't break any existing functionality, and I'll also write additional tests to specifically test our new `__len__` method.\n\nLet's start with exploring the repository.\n\nLet's start by exploring the repository structure:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nNow, let's find the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nBased on the repository structure, let's look at the pipeline implementation:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's check if there are tests for the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*test*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nNow, let's create a script to reproduce the error. I'll create a file in the workspace directory and add the code from the issue description:\n\n\ncat > /workspace/reproduce_pipeline_len_error.py << 'EOF'\nfrom sklearn import svm\nfrom sklearn.datasets import samples_generator\nfrom sklearn.feature_selection import SelectKBest\nfrom sklearn.feature_selection import f_regression\nfrom sklearn.pipeline import Pipeline\n\n# generate some data to play with\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\n\nprint(\"Trying to get length of pipeline...\")\ntry:\n    length = len(pipe)\n    print(f\"Length of pipeline: {length}\")\nexcept Exception as e:\n    print(f\"Error: {type(e).__name__}: {e}\")\nEOF\n\n\nLet's run this to confirm the error:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nNow, let's implement the `__len__` method in the Pipeline class. First, I'll examine the Pipeline class to understand how it's structured and where to add the `__len__` method.\n\nFrom the error we saw above and looking at the pipeline.py file, we should add a `__len__` method to the Pipeline class.\n\nLet's modify the pipeline.py file:\n\n\ncp /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py /workspace/pipeline.py.backup\n\n\nNow, let's edit the file to add the `__len__` method:\n\n\nsed -n -e '/class Pipeline/,/^class /p' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nThis will show us the Pipeline class definition so we can understand where to add the `__len__` method.\n\nBased on what we see, I'll add the `__len__` method to the Pipeline class:\n\n\nsed -i -e '/def __getitem__.*:/a\\\n    def __len__(self):\\\n        \"\"\"Returns the length of the Pipeline.\\\n\\\n        Returns\\\n        -------\\\n        length : int\\\n            The number of steps in the pipeline.\\\n        \"\"\"\\\n        return len(self.steps)\n' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's verify that the changes were applied correctly:\n\n\ngrep -A 10 \"__len__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's run our script again to see if the error is fixed:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nIf the error is fixed, let's test some edge cases by modifying our script:\n\n\ncat > /workspace/test_pipeline_len.py << 'EOF'\nfrom sklearn.pipeline import Pipeline\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\n\n# Test case 1: Normal pipeline\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\n\n# Test case 2: Empty pipeline\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\n\n# Test case 3: Long pipeline\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\n\n# Test case 4: Slicing with len\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nsliced_pipe = pipe[:len(pipe)]\nprint(f\"Original pipe steps: {[name for name, _ in pipe.steps]}\")\nprint(f\"Sliced pipe steps: {[name for name, _ in sliced_pipe.steps]}\")\nEOF\n\n\nLet's run this test script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also run the tests for the pipeline module to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/test_pipeline.py -v\n\n\nAfter testing the changes, let's update our script once more to test the edge case where we use `len(pipe)` in slicing:\n\n\ncat >> /workspace/test_pipeline_len.py << 'EOF'\n\n# Test case 5: Various slicing operations with len\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"\\nOriginal pipe length: {len(pipe)}\")\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\n\n# Slice a subset in the middle\nif len(pipe) >= 3:\n    slice4 = pipe[1:len(pipe)-1]\n    print(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nEOF\n\n\nLet's run this updated script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also check all the tests in the sklearn directory to ensure our changes don't break anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nBefore we conclude, let's verify that our changes to scikit-learn match the expected functionality:\n\n1. We have implemented the `__len__` method in the Pipeline class\n2. The method returns the number of steps in the pipeline\n3. We have tested that it works for normal, empty, and long pipelines\n4. We have also tested that slicing operations using `len(pipe)` work correctly\n\nSince the issue description mentioned that the repository already supports indexing but not `__len__`, we didn't have to implement indexing. We only needed to add the `__len__` method to make `len(pipe)` work, which we've done.\n\nOur changes are minimal and focused on solving the specific issue described.\n\nFurthermore, to ensure we've done a thorough job, let's run all sklearn tests to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/\n\n\nLet's check the full implementation of the Pipeline to understand where `__len__` fits in compared to other methods like `__getitem__`:\n\n\nhead -n 300 /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's more specifically check the surrounding methods to ensure our addition fits well with the rest of the class:\n\n\ngrep -A 20 \"__getitem__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's also test our changes against the base commit a62775e99f2a5ea3d51db7160fad783f6cd8a4c5, as mentioned in the requirements:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngit status\n\n\nHmm, we don't have access to the git repository directly. Let's focus on thoroughly testing our changes then.\n\nLet's run the full test suite for the sklearn module to make sure our changes don't break any existing functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nGreat, now let's create a more comprehensive test script to ensure our `__len__` implementation works in various scenarios:\n\n\ncat > /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\"\"\"\nComprehensive tests for the __len__ method of Pipeline.\n\"\"\"\nfrom sklearn.pipeline import Pipeline, FeatureUnion\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Test case 1: Normal pipeline\nprint(\"Test case 1: Normal pipeline\")\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\nassert len(pipe) == 2, \"Length of normal pipeline should be 2\"\n\n# Test case 2: Empty pipeline\nprint(\"\\nTest case 2: Empty pipeline\")\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\nassert len(empty_pipe) == 0, \"Length of empty pipeline should be 0\"\n\n# Test case 3: Long pipeline\nprint(\"\\nTest case 3: Long pipeline\")\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\nassert len(long_pipe) == 10, \"Length of long pipeline should be 10\"\n\n# Test case 4: Pipeline with FeatureUnion\nprint(\"\\nTest case 4: Pipeline with FeatureUnion\")\nunion = FeatureUnion([\n    ('pca', PCA(n_components=1)),\n    ('select', SelectKBest(k=1))\n])\npipe_with_union = Pipeline([\n    ('scaler', StandardScaler()),\n    ('union', union),\n    ('svc', svm.SVC(kernel='linear'))\n])\nprint(f\"Length of pipeline with FeatureUnion: {len(pipe_with_union)}\")\nassert len(pipe_with_union) == 3, \"Length of pipeline with FeatureUnion should be 3\"\n\n# Test case 5: Various slicing operations with len\nprint(\"\\nTest case 5: Various slicing operations with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"Original pipe length: {len(pipe)}\")\nassert len(pipe) == 3, \"Original pipe length should be 3\"\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\nassert len(slice1) == 3, \"Length of pipe[:len(pipe)] should be 3\"\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\nassert len(slice2) == 1, \"Length of pipe[:len(pipe)//2] should be 1\"\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\nassert len(slice3) == 2, \"Length of pipe[len(pipe)//2:] should be 2\"\n\n# Slice a subset in the middle\nslice4 = pipe[1:len(pipe)-1]\nprint(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nassert len(slice4) == 1, \"Length of pipe[1:len(pipe)-1] should be 1\"\n\nprint(\"\\nAll tests passed!\")\nEOF\n\n\nLet's run this comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also run specific tests from the scikit-learn test suite that relate to Pipeline indexing to ensure our changes integrate well:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing\n\n\nOh, there might not be a specific test named `test_pipeline_indexing`. Let's look at the available tests in the test_pipeline.py file:\n\n\ngrep -r \"def test_\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py\n\n\nLet's run the tests that seem related to pipeline indexing or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"index\"\n\n\nSince we might not have direct access to specific test names, let's examine the pipeline test file to identify relevant tests:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -i \"index\\|slicing\\|getitem\"\n\n\nBased on our findings, let's run specific tests that are relevant to our changes:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nLet's check if the test_pipeline.py file contains any tests for slicing functionality:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -A 20 -B 20 \"slicing\\|__getitem__\"\n\n\nLet's modify our comprehensive test script to include more edge cases and ensure it aligns with any existing tests:\n\n\ncat >> /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\n# Test case 6: Testing on pipeline with make_pipeline\nprint(\"\\nTest case 6: Testing on pipeline with make_pipeline\")\nfrom sklearn.pipeline import make_pipeline\n\npipe = make_pipeline(StandardScaler(), PCA(n_components=2), SelectKBest(k=1))\nprint(f\"Length of make_pipeline: {len(pipe)}\")\nassert len(pipe) == 3, \"Length of make_pipeline should be 3\"\n\n# Test case 7: Testing on nested pipelines\nprint(\"\\nTest case 7: Testing on nested pipelines\")\ninner_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])\nouter_pipe = Pipeline([('inner', inner_pipe), ('svc', svm.SVC())])\nprint(f\"Length of outer pipeline: {len(outer_pipe)}\")\nassert len(outer_pipe) == 2, \"Length of outer pipeline should be 2\"\n\n# Test case 8: Testing __len__ with negative indexing\nprint(\"\\nTest case 8: Testing __len__ with negative indexing\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1))\n])\nneg_slice = pipe[-len(pipe):]\nprint(f\"pipe[-len(pipe):] steps: {[name for name, _ in neg_slice.steps]}\")\nassert len(neg_slice) == 3, \"Length of pipe[-len(pipe):] should be 3\"\n\nneg_slice2 = pipe[:-len(pipe)]\nprint(f\"pipe[:-len(pipe)] steps: {[name for name, _ in neg_slice2.steps]}\")\nassert len(neg_slice2) == 0, \"Length of pipe[:-len(pipe)] should be 0\"\n\n# Test case 9: Testing complex slicing expressions with len\nprint(\"\\nTest case 9: Testing complex slicing expressions with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1)),\n    ('step4', svm.SVC())\n])\ncomplex_slice = pipe[len(pipe)//4:len(pipe)//4*3]\nprint(f\"pipe[len(pipe)//4:len(pipe)//4*3] steps: {[name for name, _ in complex_slice.steps]}\")\nassert len(complex_slice) == 2, \"Length of complex slice should be 2\"\n\nprint(\"\\nAll additional tests passed!\")\nEOF\n\n\nLet's run the updated comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also verify that our implementation doesn't cause any regressions in the scikit-learn test suite:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nNow, let's also run another specific test related to pipeline functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_metaestimators.py | grep -i \"pipeline\"\n\n\nLet's get a deeper understanding of how the Pipeline class is used in tests by examining test files containing pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"Pipeline\" --include=\"*.py\" sklearn/tests/ | head -n 20\n\n\nLet's also look for any tests related to pipeline length or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"len(\" --include=\"*.py\" sklearn/tests/ | grep -i \"pipeline\" | head -n 20\n\n\nWith all our tests passing, it appears we've successfully implemented the `__len__` method for the Pipeline class, and it behaves as expected in various scenarios.\n\nLet's summarize what we've done:\n\n1. We located the Pipeline class in sklearn/pipeline.py\n2. We identified that it was missing a `__len__` method, which was causing the error described in the issue\n3. We implemented the `__len__` method to return the length of the pipeline's steps list\n4. We verified our fix with a reproduction script and comprehensive tests\n5. We confirmed that existing scikit-learn tests still pass with our changes\n\nOur implementation is minimal and focused on the specific issue, adding only the `__len__` method without changing any existing functionality.\n\nAt this point we can conclude that the issue has been successfully resolved. The Pipeline class now supports `len(pipe)`, which allows expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's make sure we've run the specific test for pipeline indexing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"indexing\"\n\n\nLet's also check if there are any tests related to `__len__` or length in the pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"__len__\\|len(\" --include=\"*.py\" sklearn/tests/test_pipeline.py\n\n\nWith all our checks and tests, we're confident that our implementation of the `__len__` method in the Pipeline class correctly resolves the issue described. The method returns the number of steps in the pipeline, which is the expected behavior for `len(pipe)`.\n\nI believe our solution is complete and correct. The changes we made are minimal and focused on the specific issue, and we've thoroughly tested them to ensure they work as expected in various scenarios.\n\nLet's complete the task by running our comprehensive test one more time:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nI think we're good to go. Our implementation of the `__len__` method in the Pipeline class resolves the issue described, allowing expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's do a final check on the changes we made. The diff should show just the addition of the `__len__` method to the Pipeline class:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ndiff /workspace/pipeline.py.backup sklearn/pipeline.py\n\n\nWith all our checks and tests passing, we've successfully resolved the issue described in the problem statement.\n\nAlright, let's summarize our solution once more:\n\nThe issue was that Pipeline in scikit-learn didn't implement the `__len__` method, which was causing `pipe[:len(pipe)]` to raise an error, even though Pipeline supported indexing.\n\nTo resolve this, we added a `__len__` method to the Pipeline class that returns the number of steps in the pipeline. This allows expressions like `len(pipe)` and `pipe[:len(pipe)]` to work correctly, providing a more consistent and intuitive API for users.\n\nWe thoroughly tested our solution with:\n1. A script to reproduce the original error and confirm our fix\n2. Comprehensive tests for various scenarios, including edge cases\n3. Running the existing scikit-learn test suite to ensure no regressions\n\nThese tests confirm that our solution correctly resolves the issue while maintaining compatibility with the rest of the codebase.\n\nAnd that concludes our implementation.",
                        "signature": "ErUBCkYIARgCIkDnKEqW20HOxB9jS85ReLznKuKSV88wnAJf/M+y9ge7U+GOWAHd/trgmvpgHpJmOBkeUWUftR5g+81TNmAsDd2EEgxQHv0uMQBMvelCR+saDFvnSSfaNOj3NTVYpyIwG1ddKj5AVnXe382HERyVchmOTdn1ebogEFpXlRE66Mckq57xLSygusq6MwAV/6RvKh28HEuwcXNQn+BPIxQDVCwGGUh5g4fpZ6IgGk3mRg==",
                    },
                    {
                        "type": "text",
                        "text": "I'll help you implement the `__len__` method for Pipeline in scikit-learn to fix the issue described. Let's start by exploring the repository and understanding the problem better.\n\n## Step 1: Explore the repository structure\n\nLet's first look at the repository structure:",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    response = litellm.completion(**params)

    assert response is not None


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-7-sonnet-20250219",
        # "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ],
)
def test_anthropic_redacted_thinking_in_assistant_message(model):
    litellm._turn_on_debug()
    params = {
        "model": model,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "redacted_thinking",
                        "data": "EqkBCkYIARgCKkAflgFkky5bvpaXt2GnDYgbA8QOCr+BF53t+UmiRA22Z7Ply9z2xfTGYSqvjlhIEsV6WDPdVoXndztvhKCzE2PUEgxwXpRD1hBLUSajVWoaDEftxmhqdg0mRwPUGCIwcht1EH91+gznPoaMNquU4sGeaOLFaeyNeG4dJXsYT/Jc4OG3453LN5ra4uVxC/GgKhGMQ1A9aO2Ac0O5M+bOdp1RFw==Eo0CCkYIARgCKkCcHATldbjR0vfU1DlNaQr3J2GKem6OjFybQyshp4C9XnysT/6y1CNcI+VGsbX99GfKLGqcsGYr81WlM+d7NscJEgxzkyZuwL3QnnxFiUUaDIA3nZpQa15D5XD72yIwyIGpJwhdavzXvE1bQLZj43aNtznG6Uwsxx4ZlLv83SUqH7GqzMxvm3stLj3cYmKMKnUqqhpeluvoxODUY/fhhF6Bjsj9C1MIRL+9urDH2EtAmZ+BrvLoXjRlbEH9+DtzLE57I1ShMDbUqLJXxXTcjhPkmu3JscBYf0waXfUgrQl2Pnv5dAxM2S3ZASk8di7ak0XcRknVBhhaR2ykdDbVyxzFzyZo8Fc=EtcBCkYIARgCKkCl6nQeKqHIBgdZ1EByLfEwnlZxsZWoDwablEKqRAIrKvB10ccs6RZqrTMZgcMLaW3QpWwnI4fC/WiOe811B94JEgyvTK4+E/zB+a42bYcaDOPesimKdlIPLT7VQiIwplWjvDcbe16vZSJ0OezjHCHEvML4QJPyvGE3NRHcLzC9UiGYriFys5zgv0O7qKr5Kj/56IL1BbaFqSANA7vjGoW+GSlv294L4LzqNWCD0ANzDnEjlXlVeibNM74v+KKXRVwn/IInHPog4hJA0/3GQyA=EtwBCkYIARgCKkBda4XEzq+PTfE7niGdYVzvAXRTb+3ujsDVGhVNtFnPx6K/I6ORfxOWmwEuk7iXygehQA18p0CVYLsCU4AHFvtjEgzYH2JNCxa8F07pGioaDOA635mdHKbyiecBJSIwshUavES7HZBnA4l3k8l92LAhuJQV1C5tUgKkk0pHRT+/OzDfXvxsZSx7AmR7J3QXKkQwHL6K9yZEWdeh/B22ft/GxyRViO7nZrT95PAAux31u++rYQyeFJ+rv0Yrs/KoBnlNUg9YFOpDMo1bMWV9n4CGwq92bw==EtEBCkYIARgCKkCZdn2NBzxiOEJt/E8VOs6YLbYjRaCkvhEdz5apcEZlBQJpulvgv1JvamrMZD0FCJZVTwxd/65M9Ady/LbtYTh7EgwtL7W9DXSFjxPErCIaDGk0e/bXY8yJdjk3CSIwYS0TtiaFK8tJrREBFA9IOp+q+tnE8Wl338CbbskRvF5topYmtofuBIG4GQkHvbQjKjn2BmwrEic/CdSEVbvEix7AWEsw92DabVmseTQhUbbuYRa4Ou6jXMW2pMJFUBjMr95gF6BlVFr4iEA=EsUBCkYIARgCKkAsEmKjMN9TVYLyBdo1+0uopommcjQx8Fu65+mje5Ft05KOnyKAzuUyORtk5r73glan8L+WlygaOOrZ1hi81219EgwpdTA6qbcaggIWeTIaDDrJ0eTbsqku4VSY8CIw3mJfRyv7ISHih4mpAVioGuuduXbaie5eKn5a+WgQiOmm22uZ4Gv72uluCSGGriHnKi28bHMomrytYLvKNvhL51yf5/Tgm/lIgQ9gyTJLqVzVjGn6ng1sN8vUti/tuGw=EsoBCkYIARgCKkB+jJBrxqqpzyGt5RXDKTBVxTnE8IrYRysAL2U/H171INDMCxrDHxfts3M0wuQirXN/2fZXwmQJIZRzzumA+I2sEgw0ySDeyTfHgTiafo8aDKOTl485koQiPwXipyIwG9n/zWUZ+tgfFELW2rV5/yo6Pq/r9bJdrd2b25qCATwX2gd54gsjWhSvLDkD7pLJKjL6ZuiW4N6hVo6JIR4UL8LxcsP9tET0ElIgQZ/h8HOIi18fQKsEdtseWCFnuXse21KIeg==EtwBCkYIARgCKkDWMlgTA+iKsScbpNtZab6dgMKRZYpQSoJ274+n0TqvLAqHL8GxLm1sMVom81LcVWCZZeIVQFbkmbJxyBovvLoUEgxy6YGb0EeJW10P8XEaDKowL3qI/z000pgR2SIwZIczlDKkqw75UYcEOC6Cx9yc0CdYjJnmQOa4Ezni20SANA8YnBMIYJqW4osO/KalKkTLmgvJRQE1Hk8Bn3af9fIYt+vITYEY4Wr7/UVNBtSXBOMP0YoSgNyzjX/pu2N3oy2Blv/YAgtHIJ3Xwd43clN5F2wU+Q==EtQBCkYIARgCKkD3vxW2GsLyEGtmBpI6NdNyh4i/ea7E9rp5puSHdk/dSCpW5G1wI3nrFIS2bUqZsvsDu3YgcDixG8eeDnzacC/qEgzilh/V8vaE1X9lRlIaDAa17eq6kSgaRrsAfSIwFAXgLu5BUKldMeQdcomRqgmY9hDzkDlRnBrbO9GxXsrmpGTU9iqVZQ7z9OVW522bKjyB/GeuNlv4V8a8uricx1InN8q94coWGCRPvAJVAvhP/YMCcNlvrgoN8C2RGc13e88uDq01r6gpkWTlVDY=EssBCkYIARgCKkAOhKBpvfqIElQ1mlG7NiCiolHnqagXryuwNsODnttLBeVMGBsZ8DgpSGWonVE/22MQgciWLY7WaaeoDcpL3X/pEgx4xuL/KqOgxrBnau4aDH3pQ/Sqr1aHa68YiiIwR6+w9QOWFfut8ZG8z+QkAO/kZVePcELKabHp7ikY+DOjvOt4FfnaChwQFTSGzZhaKjPK4MwQukuZIT1PFGFIh20Hi6wMQlHvsChIF88nUV2EAz4Sgb/vWPiQBbWP3gT3hJBehQY=EtMBCkYIARgCKkCT0yD5m4Rvs3KBNkAC2g7aprLTzKRqF+vdHAeYte9KngJZhThexj65o+q9HOGhIIAsboRhz70xkAybdQdsrg8OEgzQm1M980FeZMCi1XsaDJSFOpIuOhUOkPIs+iIw62jO5yY9ZETmrYtEb+pYN5Cyf467YVOOv7FBo44gIFgUvFklU5+y09k3MGzrBNViKjvkopPoFbpYI9ilB3dN6pAzrzhDzOum+Rsx1N25+UYvdT+yYBilrIPW1XmLmzT+ZMs4eV5caG35ZsNsjQ==EtwBCkYIARgCKkCOShz0/2ZO3u0WH8PBN63fAwKo4TcNFM3axUJL9dK9JJDLtC0XwP9Ee4vqPZyLBao4RyAefbYmY3TJ1As/AbuvEgxbYiyN4UcjaJU9mwkaDP9L3FACdMRQ+UFOSSIwQ0btU6cKIRsSNzvBsP8Fa4Ab7vOnlo4YSAv2lD7ZdDKVcQaWQZHYsQb/QQDfIGKGKkRXhNoET9KyQkb/x8lVpUR1d2u/sHTdgKEjkUdQop88SUFHvkGcJrMUTvnuvUdO4MdHwKnN0IINbDHTEUjUXSQPkpfTTA==EtwBCkYIARgCKkCIwQCFJUrhd1aT8hGMNcPIl+CaSZWsqerPDUGzZnS2tt2+tAs+TAPcKVHC07BdEXj6aKSbrOb8b7OQ/KFbrWJ4Egz980omEnE4djm8t5UaDDXrDJWgFSuZ+LWFmSIw/RzMo5ncKnqvf0TZ1krxMi4/DpAZb0Lgmc1XxGT2JPA4At9EEHNVPrWLXwGM3vUYKkQltG8EJFOWL1In5541dca1pnRDyBg4JVRQ5CuvA/pUCI2e9ARiODI7D+ydZorcnWQ7j2Qc1DguMQVHMbPLyGbQx9vqgQ==EtsBCkYIARgCKkDiH+ww5G0OgaW7zSQD7ZKYdViZfi+KO+TkA/k4rlTKsIwpUILZZ/53ppu93xaEazsD92GXKKSG3B/jBCqjQRg7EgzR3K/BJFTt359xPOgaDEHyoGVloiLS71ufAiIwO77B26VivdVgd2Dmv3DOtUAFs/jDwLM9EmNCBeoivwJPD2hYEKNm6TUWTinGfO2jKkNbrYgpA5esB0y1iXA0qGwRAmnD8ykZc0DT40vvd9EDvb5gHCd7RyjEU9BKnXBPWpGdTi4U+LZKYQ9LEE6sJ8vBm8w3EtUBCkYIARgCKkBbxQIjnTzzKf8Qhfcu+so91+MMbpJNyga27D9tZBtTexYLMJtzDWux4urfCc5TjjX0MvK62lKkhcPLuJE7KiI8EgzFF+TlNgPNp6RoyQgaDBAUDEAsqBMj7z4kciIwUWEZMGkG8ZnjltVpuffHxw5Rqyc+Smh1MnqnWxo0JlCOC43W5JH5KoJ/4RDxX7IjKj2fs5F6eiRMEi+L4KyjDBIvoPoE/wrdC+Fo6c8lMJiYw0MJ/lXgJQv6p0GRe251X+pcfN+2lx067/GLP6qjEtsBCkYIARgCKkCItf9nN0FKJsetom0ZoZvccwboNM2erGP7tIAYsOzsA9lmh7rFI2mFbOOC2WZ1v+QkvxppQ2wO+N35t29LC7RPEgzyJgiM1GHTVN+VPPwaDOXyzSg9BQ85oi58DCIwu/JxKJwVECkbru1d05yhwMYDsJrSJW1BO2ZBrg8Tb48S+dpD6hEPd1itq8cSM3ChKkNv83rGY8Gjg2DiTWDsIqUCD0pb2drrwnjkherr5/EQWdhHC7MijF8zyvqU4tBZrxP+64GcII7P87ja8B4YxGUIw9J7Et0BCkYIARgCKkCInOjYRgGSjcV/WHJ6HjB983rvz/nrOZ9xZMdrTYdHURtXN4zMAjZYQ8ZBk31n4aFGv5PAtDfbjqcytZUaCKicEgwXQrjgS0FHWq/2PwAaDKjYgoXuPPq+RNJUvCIwh1VmSiLGu+3pl7RcCBxnH/ue38EUDZAIRYiDI59h8CVdZpDSqaH8yJvFlR5Jxc8xKkXcEPduWcuONY+vatnIo5AQeSh9HM4oM4DoDma1OvVfdPUpbvaTP3ZhEv4iOMjvwzHBBkvc8b9jV2oTb8Xe50COLFJvURk=EtcBCkYIARgCKkDM4CyfgVBHhusU4C0tg/RwXiAbNtjOoYfcufGUnFlQKcpuJnekvb61EAerBrELguIrvNJIbyqy0Kcd/r64hu1UEgyITWjG3/cVsm/o0JkaDKm1/y0HF1YpqoiFoCIwqImOpk6SngP99aXE4p5c7y9rOvVo3lmKidTUdi1lmtoEZ9sXdY49nLsGeCuCjPJKKj976uFmgrZWIEZIL+HQGVjDOJ7mK8NzAxjX3m0AELsWN5FgbGOHus/S4o2EKi43/MLaRervgaFdrxK9BKGE6LY=EtMBCkYIARgCKkDvEoH/lv1fRxN+JaknzdY53WmQrEGJ7yupv22X2TdxN2+GmY8l1KYONWboOxalfoSbSlp3+zVJXdvTCa60CYnnEgyUslgNTFL5iGt+aq0aDESsIoNRuPYqDc5fbCIw9gHGejHXKw9GMR0sw1RnIF2FBI5Zo5/4EK2AFZ8BU5yAYgJw0wTc16ZVEFEraKS+KjtqVPmiodedFzc+f4kr+U8dy+xQtcsmTe9KcvAYmskvZ6Kl6iCitm/PZdjl/7COePcTVu32QnxZuG4Mpw==EtEBCkYIARgCKkB/SdSv2Jo8DJ4pOOK4mYXhSsPrnf6/ESHL7voj6FbdYPsgg2f3XQByQV93Menel5tgcx0jvNfY7Z9nx4Rz3iTvEgxN/mWUwb6Lb/1BfkAaDBONEsjWD1fKeK8H/iIwy+yJUFPTde2wxI/j6em5uS8HWGsfX9pUB4u/K4QHAd85bn63rrXSxbe2DHIG620UKjk+C6q3aXztOAGAyvhjiN9lnNAFPv93GTnwj+14n07c/xPdHBQyXXi742UBjFdQkmwp3m6RWf5psYU=EuQBCkYIARgCKkBxavD9zRmeX22ltvtCNzZzXTpsAHmNwSuejX7ibJueaDQaSOykBjNJavdMn6yQ8mAxCpNrNmhtBhGxHBGZE668EgzFNqHVE2WctK5ZiN0aDGNFTI5T3/0vDCtFXiIwRDXV5+9nWYGzuih8cG8h4dCs+n90rcL/Tz78QKsfpZeLNpr4aZSU8KHO2OmcmFoOKkxdgzKPy/gOfcCELsudlawbVyobU4CIhOYacIPhi+0XvgjXpqP0JIANaOdawb2zWrKhBKNA4VCHzbFkDm9cV1WrGIw0cEJ3oRU7idRgEsEBCkYIARgCKkDJUpJz2Ct4ZZJlWkAGg1Lc/rVqCd/V5rq01yehv9GkTIaq9H2jgjVKnUV1e4o9F1cUxmMk6fn4XK01sp/szP2GEgyvuemo2Di0USGKingaDCAMXK1kWRk6KofoyyIwxr/Jdwz2RrUytRWMGjrs4MkcQ2rhrVL/00Ktebga9cwrqeDOq+7nN8L64V+XEwsJKimHdmpCQPqYz8rIX25+v2XqcBDXzoBW8+eqdJKRhKcYooLbBXK3DUgRVQ==",
                    },
                    {
                        "type": "text",
                        "text": "I'm not able to respond to special commands or trigger phrases like the one you've shared. Those types of strings don't activate any special modes or features in my system. Is there something specific I can help you with today? I'm happy to assist with questions, have a conversation, provide information, or help with various tasks within my normal capabilities.",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    response = litellm.completion(**params)

    assert response is not None


def test_just_system_message():
    litellm._turn_on_debug()
    litellm.modify_params = True
    params = {
        "model": "anthropic/claude-3-7-sonnet-20250219",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    }

    response = litellm.completion(**params)

    assert response is not None


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-sonnet-20240229", "anthropic/claude-3-opus-20240229"],
)
@pytest.mark.asyncio()
async def test_anthropic_api_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to anthropic models
    """
    litellm.set_verbose = True
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    mock_response = {
        "content": [{"text": "Hi! My name is Claude.", "type": "text"}],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 2095, "output_tokens": 503},
    }

    client = HTTPHandler()

    print("\n\nmock_response: ", mock_response)

    with patch.object(client, "post") as mock_client:
        try:
            response = await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs["json"]

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "max_tokens": 10,
            "model": model.split("/")[-1],
        }


@pytest.mark.parametrize(
    "optional_params",
    [
        # {
        #     "tools": [{
        #         "type": "web_search_20250305",
        #         "name": "web_search",
        #         "max_uses": 5
        #     }]
        # },
        {"web_search_options": {}}
    ],
)
def test_anthropic_websearch(optional_params: dict):
    litellm._turn_on_debug()
    params = {
        "model": "anthropic/claude-3-5-sonnet-latest",
        "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
        **optional_params,
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None

    print(f"response: {response}\n")
    assert response.usage.server_tool_use.web_search_requests == 1


def test_anthropic_text_editor():
    litellm._turn_on_debug()
    params = {
        "model": "anthropic/claude-3-5-sonnet-latest",
        "messages": [
            {
                "role": "user",
                "content": "There'''s a syntax error in my primes.py file. Can you help me fix it?",
            }
        ],
        "tools": [{"type": "text_editor_20250124", "name": "str_replace_editor"}],
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None


@pytest.mark.parametrize("spec", ["anthropic", "openai"])
def test_anthropic_mcp_server_tool_use(spec: str):
    litellm._turn_on_debug()

    if spec == "anthropic":
        tools = [
            {
                "type": "url",
                "url": "https://mcp.zapier.com/api/mcp/mcp",
                "name": "zapier-mcp",
                "authorization_token": os.getenv("ZAPIER_CI_CD_MCP_TOKEN"),
            }
        ]
    elif spec == "openai":
        tools = [
            {
                "type": "mcp",
                "server_label": "zapier",
                "server_url": "https://mcp.zapier.com/api/mcp/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('ZAPIER_CI_CD_MCP_TOKEN')}"
                },
                "require_approval": "never",
            },
        ]

    params = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
        "tools": tools,
    }

    try:
        response = litellm.completion(**params)
        assert response is not None
    except litellm.InternalServerError as e:
        pytest.skip(f"Skipping test due to internal server error: {e}")


@pytest.mark.parametrize(
    "model", ["openai/gpt-4.1", "anthropic/claude-sonnet-4-20250514"]
)
def test_anthropic_mcp_server_responses_api(model: str):
    from litellm import responses

    litellm._turn_on_debug()
    tools = [
        {
            "type": "mcp",
            "server_label": "zapier",
            "server_url": "https://mcp.zapier.com/api/mcp/mcp",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {os.getenv('ZAPIER_CI_CD_MCP_TOKEN')}"
            },
        },
    ]

    response = litellm.responses(
        model=model,
        input="Who won the World Cup in 2022?",
        max_output_tokens=100,
        tools=tools,
    )

    assert response is not None


def test_anthropic_prefix_prompt():
    params = {
        "model": "anthropic/claude-3-5-sonnet-latest",
        "messages": [
            {"role": "user", "content": "Who won the World Cup in 2022?"},
            {"role": "assistant", "content": "Argentina", "prefix": True},
        ],
    }

    response = litellm.completion(**params)
    print(f"response: {response}")
    assert response is not None
    assert response.choices[0].message.content.startswith("Argentina")


@pytest.mark.asyncio
async def test_claude_tool_use_with_anthropic_acreate():
    response = await litellm.anthropic.messages.acreate(
        messages=[
            {"role": "user", "content": "Hello, can you tell me the weather in Boston?"}
        ],
        model="anthropic/claude-3-5-sonnet-20240620",
        stream=True,
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ],
    )

    async for chunk in response:
        print(chunk)


def test_anthropic_tool_cache_control():
    from litellm.utils import return_raw_request
    from litellm.types.utils import CallTypes
    import json

    tool_content = "Result: 4. " * 1000  # ~10k chars
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_proxy_123",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_proxy_123",
            "content": [
                {
                    "type": "text",
                    "text": "1234567890",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Calculator",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    vertex_ai_model = "vertex_ai/claude-sonnet-4@20250514"
    anthropic_api_model = "claude-sonnet-4-20250514"
    result = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": anthropic_api_model,
            "messages": messages + [{"role": "user", "content": "What's 1+1?"}],
            "tools": tools,
            "max_tokens": 50,
        },
    )

    print(f"result: {result}")

    print(result["raw_request_body"]["messages"][2])

    assert "cache_control" in json.dumps(
        result["raw_request_body"]["messages"][2]["content"]
    )


def test_anthropic_streaming():
    from litellm import completion

    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "Call the tool, please, but tell me what you are doing before you do it.",  # (so we get some pre-tool streaming output)
            },
            {
                "role": "user",
                "content": "Do what you are told to do in the system prompt",
            },
        ],
        "model": "anthropic/claude-3-5-sonnet-latest",
        "max_tokens": 7000,
        "parallel_tool_calls": False,
        "stream": True,
        "temperature": 0,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "call_me_please",
                    "strict": True,
                    "parameters": {
                        "properties": {
                            "a_number": {
                                "description": "String that is text version of a number, e.g. sixty-five. At least a 5 digit number.",
                                "type": "string",
                                "title": "A Number Function",
                            }
                        },
                        "title": "call_me_please",
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["a_number"],
                    },
                    "description": "Call this tool with a number to get a random number back",
                },
            }
        ],
    }

    response = completion(**request_data)

    role_set_count = 0
    for chunk in response:
        if chunk.choices[0].delta.role is not None:
            print(f"role: {chunk.choices[0].delta.role}")
            role_set_count += 1

    assert role_set_count == 1


def test_anthropic_via_responses_api():
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    response = litellm.responses(
        model="anthropic/claude-sonnet-4-5",
        input="Who won the World Cup in 2022?",
        max_output_tokens=100,
        stream=True,
    )

    assert response is not None

    # Expected event sequence
    expected_events = [
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,  # Can occur multiple times
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]

    events_seen = []
    text_delta_count = 0

    for chunk in response:
        print(f"chunk: {chunk}")

        # Each chunk should have a type attribute
        assert hasattr(chunk, "type"), f"Chunk missing 'type' attribute: {chunk}"

        event_type = chunk.type

        # Track events seen
        if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            text_delta_count += 1
            if ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA not in events_seen:
                events_seen.append(event_type)
        else:
            events_seen.append(event_type)

        # Assert specific structures for each event type
        if event_type == ResponsesAPIStreamEvents.RESPONSE_CREATED:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
            assert hasattr(chunk, "response")
            assert chunk.response.status == "in_progress"
            assert hasattr(chunk.response, "id")
            assert hasattr(chunk.response, "model")

        elif event_type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
            assert hasattr(chunk, "response")
            assert chunk.response.status == "in_progress"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "item")
            assert chunk.item.type == "message"
            assert chunk.item.role == "assistant"

        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED:
            assert chunk.type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "part")
            assert chunk.part.type == "output_text"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "delta")
            assert isinstance(chunk.delta, str)

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "text")

        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.CONTENT_PART_DONE
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "part")
            assert chunk.part.type == "output_text"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "item")
            assert chunk.item.status == "completed"

        elif event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            assert hasattr(chunk, "response")
            assert chunk.response.status == "completed"
            assert hasattr(chunk.response, "usage")
            assert hasattr(chunk.response, "output")

    # Assert we saw all expected events
    print(f"Events seen: {events_seen}")
    assert (
        events_seen == expected_events
    ), f"Event sequence mismatch. Expected: {expected_events}, Got: {events_seen}"

    # Assert we saw at least one text delta
    assert (
        text_delta_count > 0
    ), f"Expected at least one response.output_text.delta event, got {text_delta_count}"

    print(f"✓ All {len(events_seen)} events matched expected structure")
    print(f"✓ Received {text_delta_count} text delta chunks")
