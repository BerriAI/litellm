# What is this?
## Unit tests for Anthropic Adapter

import asyncio
import os
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import (
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












@pytest.mark.parametrize(
    "tool_type, tool_config, message_content",
    [
        (
            "computer_20250124",
            {
                "type": "computer_20250124",
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

    litellm.turn_on_debug()

    tools = [tool_config]
    model = "claude-sonnet-4-5-20250929"
    messages = [{"role": "user", "content": message_content}]

    try:
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
        )
        print(f"Tool type: {tool_type}")
        print(resp)
        assert resp is not None
    except litellm.InternalServerError:
        pass








from litellm import completion


class TestAnthropicCompletion(BaseLLMChatTest, BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "anthropic/claude-sonnet-4-5-20250929"}

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }

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

    @pytest.mark.parametrize("sync_mode", [True])
    @pytest.mark.asyncio
    async def test_pdf_handling(self, pdf_messages, sync_mode):
        await super().test_pdf_handling(pdf_messages, sync_mode)
    test_content_list_handling = None
    test_image_url = None
    test_image_url_string = None
    test_web_search = None
















from litellm.constants import RESPONSE_FORMAT_TOOL_NAME






def test_anthropic_citations_api():
    """
    Test the citations API
    """

    try:
        resp = completion(
            model="claude-sonnet-4-5-20250929",
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

    resp = completion(
        model="claude-sonnet-4-5-20250929",
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
        "anthropic/claude-sonnet-4-5-20250929",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_thinking_output(model):

    litellm.turn_on_debug()

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
        "anthropic/claude-sonnet-4-5-20250929",
        # "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_thinking_output_stream(model):
    litellm.set_verbose = True
    try:
        # litellm.turn_on_debug()
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
                model="claude-sonnet-4-5-20250929",
                headers={"anthropic-beta": "computer-use-2025-01-24"},
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
        assert "computer-use-2025-01-24" in headers["anthropic-beta"]




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
    litellm.turn_on_debug()
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in Tokyo right now?. Make sure to search the web for an answer",
            }
        ],
        **optional_params,
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None

    print(f"response: {response}\n")
    # When web search is requested and used, server_tool_use should be present
    assert response.usage.server_tool_use is not None
    assert response.usage.server_tool_use.web_search_requests >= 1


def test_anthropic_text_editor():
    litellm.turn_on_debug()
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": "There'''s a syntax error in my primes.py file. Can you help me fix it?",
            }
        ],
        "tools": [
            {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
        ],
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None


@pytest.mark.parametrize("spec", ["anthropic", "openai"])
@pytest.mark.skipif(
    os.getenv("ZAPIER_CI_CD_MCP_TOKEN") is None, reason="ZAPIER_CI_CD_MCP_TOKEN not set"
)
def test_anthropic_mcp_server_tool_use(spec: str):
    litellm.turn_on_debug()

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
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
        "tools": tools,
    }

    try:
        response = litellm.completion(**params)
        assert response is not None
    except litellm.InternalServerError as e:
        pytest.skip(f"Skipping test due to internal server error: {e}")


@pytest.mark.parametrize(
    "model", ["openai/gpt-4.1", "anthropic/claude-sonnet-4-5-20250929"]
)
@pytest.mark.skipif(
    os.getenv("ZAPIER_CI_CD_MCP_TOKEN") is None, reason="ZAPIER_CI_CD_MCP_TOKEN not set"
)
def test_anthropic_mcp_server_responses_api(model: str):
    from litellm import responses

    litellm.turn_on_debug()
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
        "model": "anthropic/claude-sonnet-4-5-20250929",
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
        model="anthropic/claude-sonnet-4-5-20250929",
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




def test_anthropic_streaming():

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
        "model": "anthropic/claude-sonnet-4-5-20250929",
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






def _make_transform_request(optional_params: dict, litellm_params: dict) -> dict:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    return AnthropicConfig().transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )














def test_anthropic_basic_completion_replay():
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert response is not None
    content = response.choices[0].message.content
    assert isinstance(content, str) and content.strip(), content
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.choices[0].finish_reason in {"stop", "length"}


def test_anthropic_streaming_completion_replay():
    stream = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    collected_text = ""
    finish_reason = None
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            collected_text += delta.content
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    assert chunk_count > 1, "expected multiple SSE chunks from streaming response"
    assert collected_text.strip(), collected_text
    assert finish_reason in {"stop", "length"}
