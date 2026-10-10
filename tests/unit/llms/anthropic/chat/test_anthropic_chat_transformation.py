import copy
import json
from queue import Queue
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
from httpx import Headers
import pytest
import respx

import litellm
from litellm.constants import (
    ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
    RESPONSE_FORMAT_TOOL_NAME,
)
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import encrypted_reasoning_signature
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.llms.anthropic.chat import ModelResponseIterator
from litellm.llms.anthropic.common_utils import process_anthropic_headers
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
    VertexAIAnthropicConfig,
)
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.types.utils import ChatCompletionToolCallChunk, ServerToolUse, Usage


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

    result = config._create_json_tool_call_for_response_format(json_schema=response_format_json_schema)

    assert result["input_schema"]["properties"] == {"agent_doing": {"title": "Agent Doing", "type": "string"}}
    print(result)


def test_anthropic_json_mode_non_streaming_mixed_internal_and_user_tools():
    """Non-streaming + response_format: internal json tool must not require len(tool_calls)==1."""
    config = AnthropicConfig()
    tool_calls = [
        {
            "id": "toolu_json",
            "type": "function",
            "function": {
                "name": RESPONSE_FORMAT_TOOL_NAME,
                "arguments": '{"values": {"answer": 42}}',
            },
            "index": 0,
        },
        {
            "id": "toolu_user",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NY"}',
            },
            "index": 1,
        },
    ]
    replacement, filtered, extra = config._resolve_json_mode_non_streaming(
        json_mode=True,
        tool_calls=tool_calls,
    )
    assert replacement is None
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "get_weather"
    assert extra == '{"answer": 42}'


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


def test_calculate_usage_prefers_served_speed_from_response_usage():
    """
    Anthropic reports the speed a request was actually served at in the response
    usage (a fast request on a model without fast mode comes back
    ``"speed": "standard"``), so the served value must beat the requested one or
    spend gets multiplied for fast service that never happened.
    """
    config = AnthropicConfig()

    served_standard = config.calculate_usage(
        usage_object={"input_tokens": 12, "output_tokens": 1, "speed": "standard"},
        reasoning_content=None,
        speed="fast",
    )
    assert served_standard.speed == "standard"

    no_response_speed = config.calculate_usage(
        usage_object={"input_tokens": 12, "output_tokens": 1},
        reasoning_content=None,
        speed="fast",
    )
    assert no_response_speed.speed == "fast"


@pytest.mark.parametrize(
    "input_update, expected_fresh", [({}, 1000), ({"input_tokens": 0}, 0), ({"input_tokens": 2000}, 2000)]
)
def test_streaming_iterator_persists_cumulative_usage_across_partial_chunks(input_update, expected_fresh):
    """
    Omitted input/cache/pricing fields retain their last cumulative values;
    explicit input updates, including zero, replace them.
    """
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    iterator = ModelResponseIterator(None, sync_stream=True, speed="fast")

    start_usage = iterator._handle_usage(
        {
            "input_tokens": 1000,
            "output_tokens": 1,
            "speed": "standard",
            "inference_geo": "us",
            "cache_creation_input_tokens": 3000,
            "cache_read_input_tokens": 2000,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 3000},
        }
    )
    delta_usage = iterator._handle_usage({"output_tokens": 5, **input_update})

    assert start_usage.speed == "standard"
    assert delta_usage.speed == "standard"
    assert delta_usage.inference_geo == "us"
    assert delta_usage.prompt_tokens == expected_fresh + 5000
    assert delta_usage.completion_tokens == 5
    details = delta_usage.prompt_tokens_details
    assert (details.text_tokens, details.cached_tokens, details.cache_creation_tokens) == (expected_fresh, 2000, 3000)
    assert details.cache_creation_token_details.ephemeral_1h_input_tokens == 3000
    assert start_usage.prompt_tokens_details.text_tokens == 1000


def test_calculate_usage_aggregates_cache_creation_split_across_iterations():
    """
    In the iterations path each iteration can carry the 5m/1h cache_creation
    breakdown. calculate_usage must aggregate it into cache_creation_token_details
    so 1h writes are priced at the 1h rate instead of silently falling back to 5m.

    Regression for LIT-4868.
    """
    from litellm.llms.anthropic.cost_calculation import cost_per_token

    config = AnthropicConfig()
    usage_object = {
        "input_tokens": 0,
        "output_tokens": 5,
        "iterations": [
            {
                "type": "message",
                "input_tokens": 0,
                "output_tokens": 3,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 10000},
            },
            {
                "type": "message",
                "input_tokens": 0,
                "output_tokens": 2,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 10000},
            },
        ],
    }

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)

    details = usage.prompt_tokens_details.cache_creation_token_details
    assert details is not None
    assert details.ephemeral_5m_input_tokens == 0
    assert details.ephemeral_1h_input_tokens == 20000
    assert usage.prompt_tokens_details.cache_creation_tokens == 20000

    info = litellm.get_model_info(model="claude-opus-4-8", custom_llm_provider="anthropic")
    rate_5m = info["cache_creation_input_token_cost"]
    rate_1h = info["cache_creation_input_token_cost_above_1hr"]
    assert rate_1h > rate_5m

    prompt_cost, _ = cost_per_token(model="claude-opus-4-8", usage=usage)
    assert prompt_cost == pytest.approx(20000 * rate_1h)
    assert prompt_cost != pytest.approx(20000 * rate_5m)


def test_calculate_usage_bills_undetailed_iteration_cache_writes_at_5m_rate():
    """
    When only some iterations carry the cache_creation breakdown, the writes
    without a breakdown must still be billed (at the default 5m rate) instead
    of silently priced at zero once details exist.

    Regression for the Cursor Bugbot finding on the LIT-4868 fix.
    """
    from litellm.llms.anthropic.cost_calculation import cost_per_token

    config = AnthropicConfig()
    usage_object = {
        "input_tokens": 0,
        "output_tokens": 5,
        "iterations": [
            {
                "type": "message",
                "input_tokens": 0,
                "output_tokens": 3,
                "cache_creation_input_tokens": 10000,
                "cache_read_input_tokens": 0,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 10000},
            },
            {
                "type": "message",
                "input_tokens": 0,
                "output_tokens": 2,
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 0,
            },
        ],
    }

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)

    details = usage.prompt_tokens_details.cache_creation_token_details
    assert details is not None
    assert details.ephemeral_5m_input_tokens == 7000
    assert details.ephemeral_1h_input_tokens == 10000
    assert usage.prompt_tokens_details.cache_creation_tokens == 17000

    info = litellm.get_model_info(model="claude-opus-4-8", custom_llm_provider="anthropic")
    rate_5m = info["cache_creation_input_token_cost"]
    rate_1h = info["cache_creation_input_token_cost_above_1hr"]

    prompt_cost, _ = cost_per_token(model="claude-opus-4-8", usage=usage)
    assert prompt_cost == pytest.approx(7000 * rate_5m + 10000 * rate_1h)
    assert prompt_cost != pytest.approx(10000 * rate_1h)


def test_calculate_usage_clamps_text_tokens_when_reasoning_estimate_exceeds_output():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={"input_tokens": 10, "output_tokens": 1},
        reasoning_content="This reasoning text intentionally tokenizes above one output token.",
    )

    assert usage.completion_tokens == 1
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == usage.completion_tokens
    assert usage.completion_tokens_details.text_tokens == 0


def test_calculate_usage_prefers_provider_reported_thinking_tokens():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 32,
            "output_tokens": 421,
            "output_tokens_details": {"thinking_tokens": 372},
        },
        reasoning_content="",
        completion_response={
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "10"},
            ]
        },
    )

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 372
    assert usage.completion_tokens_details.text_tokens == 49


def test_calculate_usage_provider_thinking_tokens_win_over_visible_reasoning_estimate():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 50,
            "output_tokens": 811,
            "output_tokens_details": {"thinking_tokens": 747},
        },
        reasoning_content="short visible reasoning that tokenizes to far fewer than 747 tokens",
    )

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 747
    assert usage.completion_tokens_details.text_tokens == 64


def test_calculate_usage_sums_provider_thinking_tokens_across_iterations():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 10,
            "output_tokens": 300,
            "iterations": [
                {"input_tokens": 5, "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 60}},
                {"input_tokens": 5, "output_tokens": 200, "output_tokens_details": {"thinking_tokens": 90}},
            ],
        },
        reasoning_content=None,
    )

    assert usage.completion_tokens == 300
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 150
    assert usage.completion_tokens_details.text_tokens == 150


def test_calculate_usage_falls_back_when_only_some_iterations_report_thinking_tokens():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 10,
            "output_tokens": 300,
            "output_tokens_details": {"thinking_tokens": 240},
            "iterations": [
                {"input_tokens": 5, "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 60}},
                {"input_tokens": 5, "output_tokens": 200},
            ],
        },
        reasoning_content=None,
    )

    assert usage.completion_tokens == 300
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 240
    assert usage.completion_tokens_details.text_tokens == 60


def test_calculate_usage_reports_unknown_split_when_only_some_iterations_report_thinking_tokens():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 10,
            "output_tokens": 300,
            "iterations": [
                {"input_tokens": 5, "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 60}},
                {"input_tokens": 5, "output_tokens": 200},
            ],
        },
        reasoning_content="",
        completion_response={"content": [{"type": "thinking", "thinking": "", "signature": "sig"}]},
    )

    assert usage.completion_tokens == 300
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens is None
    assert usage.completion_tokens_details.text_tokens is None


def test_calculate_usage_reports_unknown_split_when_thinking_ran_without_a_count():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={"input_tokens": 32, "output_tokens": 580},
        reasoning_content="",
        completion_response={
            "content": [
                {"type": "redacted_thinking", "data": "encrypted"},
                {"type": "text", "text": "10"},
            ]
        },
    )

    assert usage.completion_tokens == 580
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens is None
    assert usage.completion_tokens_details.text_tokens is None


def test_calculate_usage_without_thinking_reports_all_output_as_text():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={"input_tokens": 32, "output_tokens": 171},
        reasoning_content=None,
        completion_response={"content": [{"type": "text", "text": "10"}]},
    )

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 0
    assert usage.completion_tokens_details.text_tokens == 171


def test_calculate_usage_ignores_malformed_provider_thinking_tokens():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={
            "input_tokens": 32,
            "output_tokens": 100,
            "output_tokens_details": {"thinking_tokens": "not-a-number"},
        },
        reasoning_content=None,
    )

    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 0
    assert usage.completion_tokens_details.text_tokens == 100


def test_calculate_usage_handles_mocked_output_tokens_with_reasoning_content():
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={"input_tokens": 10, "output_tokens": MagicMock()},
        reasoning_content="mocked response reasoning",
    )

    assert usage.completion_tokens == 0
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 0
    assert usage.completion_tokens_details.text_tokens == 0


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

    result, _ = config.map_tool_helper(tool)
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


@pytest.mark.parametrize("search_context_size, expected_max_uses", [("low", 1), ("medium", 5), ("high", 10)])
def test_web_search_tool_transformation_with_search_context_size(search_context_size, expected_max_uses):
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
                "input": {"query": "average weight african elephant kg"},
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
                        "snippet": "Adult African elephants weigh between 4,000-6,000 kg...",
                    }
                ],
            },
            {
                "type": "text",
                "text": "Based on my search, African elephants weigh around 5,000 kg.",
            },
            {
                "type": "tool_use",
                "id": "toolu_01XYZ789",
                "name": "add_numbers",
                "input": {"a": 5000, "b": 100},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "server_tool_use": {"web_search_requests": 1},
        },
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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
                "input": {"query": "test query"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_provider_test",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com/test",
                        "title": "Test Result",
                        "snippet": "Test snippet content",
                    }
                ],
            },
            {"type": "text", "text": "Here is the result."},
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 25,
            "server_tool_use": {"web_search_requests": 1},
        },
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
                "input": {"query": "african elephant weight"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_search1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example1.com",
                        "title": "Result 1",
                        "snippet": "First result",
                    }
                ],
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_search2",
                "name": "web_search",
                "input": {"query": "asian elephant weight"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_search2",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example2.com",
                        "title": "Result 2",
                        "snippet": "Second result",
                    }
                ],
            },
            {"type": "text", "text": "Found information about both elephants."},
        ]
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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
    assert result.choices[0].message.content == "You are a helpful assistant. The grass is green."


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
        api_key="fake-anthropic-key",
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
            "schema": {"type": "object", "properties": {}},
        }
    }

    result_headers = config.update_headers_with_optional_anthropic_beta(headers, optional_params)

    # Both beta headers should be present
    beta_value = result_headers["anthropic-beta"]
    assert "context-1m-2025-08-07" in beta_value, f"User's context-1m beta header missing from: {beta_value}"
    assert "structured-outputs-2025-11-13" in beta_value, f"Structured output beta header missing from: {beta_value}"


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
            "schema": {"type": "object", "properties": {}},
        },
        "context_management": _sample_context_management_payload(),
        "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
    }

    result_headers = config.update_headers_with_optional_anthropic_beta(headers, optional_params)

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
                        "properties": {"agent_doing": {"title": "Agent Doing", "type": "string"}},
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
    assert "structured-outputs-2025-11-13" in response["raw_request_headers"]["anthropic-beta"]


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-opus-4-8",
        "claude-opus-4-6-20260205",
        "claude-opus-4-5-20251101",
        "claude-opus-4.5-20251101",
    ],
)
def test_opus_uses_native_structured_output(model_name):
    """
    Test that supported Opus models use native Anthropic structured outputs
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


def test_native_structured_output_uses_bundled_capability_when_remote_map_lags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "claude-opus-4-8"
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {model: {"supports_response_schema": True}},
    )
    litellm.get_model_info.cache_clear()

    try:
        optional_params = AnthropicConfig().map_openai_params(
            non_default_params={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    },
                }
            },
            optional_params={},
            model=model,
            drop_params=False,
        )
    finally:
        litellm.get_model_info.cache_clear()

    assert "output_format" in optional_params
    assert "tools" not in optional_params


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
    tools = [{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]
    assert config.is_tool_search_used(tools) is True

    # Test without tool search
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    assert config.is_tool_search_used(tools) is False


def test_tool_search_bm25_detection():
    """Test that tool search BM25 tools are properly detected"""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    config = AnthropicModelInfo()

    # Test with tool search BM25 tool
    tools = [{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}]
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

    tool = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}

    mapped_tool, mcp_server = config.map_tool_helper(tool)

    assert mapped_tool is not None
    assert mapped_tool["type"] == "tool_search_tool_regex_20251119"
    assert mapped_tool["name"] == "tool_search_tool_regex"
    assert mcp_server is None


def test_tool_search_bm25_mapping():
    """Test that tool search BM25 tools are properly mapped"""
    config = AnthropicConfig()

    tool = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}

    mapped_tool, mcp_server = config.map_tool_helper(tool)

    assert mapped_tool is not None
    assert mapped_tool["type"] == "tool_search_tool_bm25_20251119"
    assert mapped_tool["name"] == "tool_search_tool_bm25"
    assert mcp_server is None


def test_deferred_tools_separation():
    """Test that deferred and non-deferred tools are properly separated"""
    config = AnthropicConfig()

    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "type": "function",
            "function": {"name": "get_weather"},
            "defer_loading": True,
        },
        {
            "type": "function",
            "function": {"name": "search_files"},
            "defer_loading": False,
        },
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
                "input": {"query": "weather"},
            }
        ]
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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
        "server_tool_use": {"tool_search_requests": 2},
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
            "function": {"name": "get_weather", "description": "Get weather"},
        }
    ]

    content = [
        {"type": "text", "text": "I'll search for tools"},
        {"type": "tool_reference", "tool_name": "get_weather"},
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
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        "defer_loading": True,
    }

    mapped_tool, mcp_server = config.map_tool_helper(tool)

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
                "text": "I'll search for weather-related tools that can help you.",
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_015i6aVA2niwzv4RG4DtnxDJ",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "weather", "limit": 5},
                "caller": {"type": "direct"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_015i6aVA2niwzv4RG4DtnxDJ",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
                },
            },
            {"type": "text", "text": "Great! I found a weather tool."},
            {
                "type": "tool_use",
                "id": "toolu_01CrCNx4ntSaeeV9iArT4JfQ",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            },
        ],
        "usage": {
            "input_tokens": 1639,
            "output_tokens": 170,
            "server_tool_use": {"web_search_requests": 0},
        },
    }

    # Extract content
    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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
        completion_response=completion_response,
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
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
        "allowed_callers": ["code_execution_20250825"],
    }

    transformed_tool, _ = config.map_tool_helper(tool_with_allowed_callers)
    assert transformed_tool is not None
    assert "allowed_callers" in transformed_tool
    assert transformed_tool["allowed_callers"] == ["code_execution_20250825"]


def test_programmatic_tool_calling_beta_header():
    """Test that beta header is automatically added when programmatic tool calling is detected."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model_info = AnthropicModelInfo()

    # Test detection with allowed_callers
    tools = [
        {"type": "code_execution_20250825", "name": "code_execution"},
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute a SQL query",
                "parameters": {"type": "object", "properties": {}},
            },
            "allowed_callers": ["code_execution_20250825"],
        },
    ]

    is_programmatic = model_info.is_programmatic_tool_calling_used(tools)
    assert is_programmatic is True

    # Test header generation
    headers = model_info.get_anthropic_headers(api_key="test-key", programmatic_tool_calling_used=True)

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
            {"type": "text", "text": "I'll query the database."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "query_database",
                "input": {"sql": "SELECT * FROM users"},
                "caller": {
                    "type": "code_execution_20250825",
                    "tool_id": "srvtoolu_abc",
                },
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }

    (
        text,
        citations,
        thinking,
        reasoning,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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

    tool = {"type": "code_execution_20250825", "name": "code_execution"}

    transformed_tool, _ = config.map_tool_helper(tool)
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
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
            "allowed_callers": ["code_execution_20250825"],
        },
    }

    transformed_tool, _ = config.map_tool_helper(tool)
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
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
        "input_examples": [
            {"location": "San Francisco, CA", "unit": "fahrenheit"},
            {"location": "Tokyo, Japan", "unit": "celsius"},
        ],
    }

    transformed_tool, _ = config.map_tool_helper(tool_with_examples)
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
                "parameters": {"type": "object", "properties": {}},
            },
            "input_examples": [{"location": "San Francisco, CA"}],
        }
    ]

    is_examples_used = model_info.is_input_examples_used(tools)
    assert is_examples_used is True

    # Test header generation
    headers = model_info.get_anthropic_headers(api_key="test-key", input_examples_used=True)

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
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            "input_examples": [
                {"location": "Paris, France"},
                {"location": "London, UK"},
            ],
        },
    }

    transformed_tool, _ = config.map_tool_helper(tool)
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
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
        "input_examples": [{"sql": "SELECT * FROM users WHERE id = 1"}],
        "defer_loading": True,
        "allowed_callers": ["code_execution_20250825"],
    }

    transformed_tool, _ = config.map_tool_helper(tool)
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
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
        "input_examples": [],
    }

    transformed_tool, _ = config.map_tool_helper(tool)
    assert transformed_tool is not None
    # Empty list should not be added
    assert "input_examples" not in transformed_tool or len(transformed_tool.get("input_examples", [])) == 0


# ============ Effort Parameter Tests ============


def test_effort_output_config_preservation():
    """Test that output_config with effort is preserved in transformation."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Analyze this code"}]
    optional_params = {"output_config": {"effort": "medium"}}

    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "output_config" in result
    assert result["output_config"]["effort"] == "medium"


def test_output_config_format_preservation_and_beta_header():
    """Test that output_config.format is preserved and treated as structured output."""
    config = AnthropicConfig()
    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    }
    optional_params = {"output_config": {"format": output_format, "effort": "xhigh"}}

    result = config.transform_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Test"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    headers = config.update_headers_with_optional_anthropic_beta({}, optional_params)

    assert result["output_config"]["format"] == output_format
    assert result["output_config"]["effort"] == "xhigh"
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]


def test_effort_beta_header_injection():
    """Test that effort beta header is automatically added when output_config is detected."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model_info = AnthropicModelInfo()

    # Test with effort parameter
    optional_params = {"output_config": {"effort": "low"}}

    effort_used = model_info.is_effort_used(optional_params=optional_params, custom_llm_provider="anthropic")
    assert effort_used is True

    headers = model_info.get_anthropic_headers(api_key="test-key", effort_used=effort_used)

    assert "anthropic-beta" in headers
    assert "effort-2025-11-24" in headers["anthropic-beta"]


def test_effort_validation():
    """Test that only valid effort values are accepted."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Test"}]

    # Valid values should work (xhigh is Opus 4.7+ only, not 4.5)
    for effort in ["high", "medium", "low"]:
        optional_params = {"output_config": {"effort": effort}}
        result = config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        assert result["output_config"]["effort"] == effort

    optional_params = {"output_config": {"effort": "invalid"}}

    with pytest.raises(litellm.exceptions.BadRequestError, match="Invalid effort value"):
        config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )


def test_effort_with_claude_opus_45():
    """Test effort parameter works with Claude Opus 4.5 model."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Complex analysis task"}]
    optional_params = {"output_config": {"effort": "high"}}

    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
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
            headers={},
        )
        assert result["output_config"]["effort"] == effort


def test_max_effort_rejected_for_opus_45():
    """Test that effort='max' is rejected when using Claude Opus 4.5."""
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Test"}]

    optional_params = {"output_config": {"effort": "max"}}

    with pytest.raises(
        litellm.exceptions.BadRequestError,
        match="effort='max' is not supported by this model",
    ):
        config.transform_request(
            model="claude-opus-4-5-20251101",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
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
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    optional_params = {
        "output_config": {"effort": "low"},
        "tools": tools,
        "thinking": {"type": "enabled", "budget_tokens": 1000},
    }

    result = config.transform_request(
        model="claude-opus-4-5-20251101",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    # Verify all features are present
    assert "output_config" in result
    assert result["output_config"]["effort"] == "low"
    assert "tools" in result
    assert len(result["tools"]) > 0
    assert "thinking" in result


def test_anthropic_drop_params_strips_output_config_for_pre_4_5_models():
    """``drop_params=True`` strips unsupported ``output_config`` for pre-4.5 models."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Hello"}]

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = config.transform_request(
            model="claude-3-haiku-20240307",
            messages=messages,
            optional_params={"output_config": {"effort": "low"}},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert "output_config" not in result


def test_anthropic_drop_params_keeps_output_config_for_supporting_models():
    """``drop_params=True`` must not strip on models that support effort."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Hello"}]

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = config.transform_request(
            model="claude-opus-4-7",
            messages=messages,
            optional_params={"output_config": {"effort": "high"}},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert result.get("output_config") == {"effort": "high"}


def test_anthropic_drop_params_false_forwards_to_unsupported_model():
    """Default ``drop_params=False`` forwards ``output_config`` and lets the provider 400."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Hello"}]

    original = litellm.drop_params
    litellm.drop_params = False
    try:
        result = config.transform_request(
            model="claude-3-haiku-20240307",
            messages=messages,
            optional_params={"output_config": {"effort": "low"}},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert result.get("output_config") == {"effort": "low"}


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-5-20251101",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "anthropic.claude-mythos-preview",
        "bedrock/anthropic.claude-mythos-preview",
    ],
)
def test_anthropic_model_supports_effort_param_recognizes_supporting_models(model):
    assert AnthropicConfig.model_supports_effort_param(model, "anthropic") is True


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-sonnet-4-20250514",
    ],
)
def test_anthropic_model_supports_effort_param_rejects_non_supporting_models(model):
    assert AnthropicConfig.model_supports_effort_param(model, "anthropic") is False


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-4-6-20260205",
        "claude-opus-4-7-20260416",
    ],
)
def test_anthropic_model_supports_speed_param_recognizes_supporting_models(model):
    assert AnthropicConfig._model_supports_speed_param(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-fable-5",
        "claude-3-haiku-20240307",
        "vertex_ai/claude-opus-4-8",
        "azure_ai/claude-opus-4-8",
        "anthropic.claude-opus-4-8",
    ],
)
def test_anthropic_model_supports_speed_param_rejects_non_supporting_models(model):
    assert AnthropicConfig._model_supports_speed_param(model) is False


@pytest.mark.parametrize("custom_llm_provider", ["vertex_ai", "azure_ai", "bedrock"])
def test_anthropic_model_supports_speed_param_rejects_non_anthropic_providers(
    custom_llm_provider,
):
    """Fast mode is direct-Anthropic-only. Vertex/Azure/Bedrock strip their prefix
    before the shared transform runs, so the bare Opus id must still be rejected."""
    assert AnthropicConfig._model_supports_speed_param("claude-opus-4-8", custom_llm_provider) is False
    assert AnthropicConfig._model_supports_speed_param("claude-opus-4-8", "anthropic") is True


def test_vertex_anthropic_drops_speed_for_opus_with_drop_params(monkeypatch):
    """Regression: vertex_ai Opus must drop ``speed`` even though the prefix-stripped
    ``claude-opus-4-8`` maps to a fast-mode-capable direct-Anthropic entry."""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    monkeypatch.setattr(litellm, "drop_params", True)
    result = VertexAIAnthropicConfig().transform_request(
        model="claude-opus-4-8",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"speed": "fast", "max_tokens": 1024},
        litellm_params={},
        headers={},
    )

    assert "speed" not in result


def test_vertex_anthropic_raises_on_speed_without_drop_params(monkeypatch):
    """Regression: vertex_ai Opus raises rather than forwarding an unsupported
    ``speed`` when neither global nor per-request drop_params is set."""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        VertexAIAnthropicConfig().map_openai_params(
            non_default_params={"speed": "fast"},
            optional_params={},
            model="claude-opus-4-8",
            drop_params=False,
        )


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
    # System message must be removed from messages so it doesn't reach anthropic_messages_pt
    assert all(m["role"] != "system" for m in messages)


def test_translate_system_message_skips_empty_list_content():
    """
    Test that translate_system_message skips empty text blocks in list content.

    Fixes: Vertex AI Anthropic API error "messages: text content blocks must be non-empty"
    """
    config = AnthropicConfig()

    # Test list content with empty text block
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "Valid content"},
                {"type": "text", "text": ""},
            ],
        },
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
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Cached content",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
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
    with patch(
        "litellm.llms.anthropic.chat.transformation.get_max_tokens",
        return_value=8192,
    ):
        max_tokens = config.get_max_tokens_for_model("claude-3-5-sonnet-20241022")
        assert max_tokens == 8192


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

    def _mock_get_max_tokens(model):
        """Return expected max_output_tokens for each model."""
        model_map = {
            "claude-3-sonnet-20240229": 4096,
            "claude-3-5-sonnet-20241022": 8192,
            "claude-3-7-sonnet-20250219": 64000,
        }
        result = model_map.get(model)
        if result is None:
            raise Exception(f"Model {model} not found")
        return result

    with patch(
        "litellm.llms.anthropic.chat.transformation.get_max_tokens",
        side_effect=_mock_get_max_tokens,
    ):
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


def test_get_config_does_not_leak_module_constants():
    """``get_config`` must not leak the reasoning-effort lookup table onto the wire."""
    cfg = AnthropicConfig.get_config(model="claude-opus-4-7")
    for forbidden in (
        "REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT",
        "_REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT",
    ):
        assert forbidden not in cfg


@pytest.mark.parametrize(
    "model,level,expected",
    [
        ("claude-opus-4-7", "max", True),
        ("claude-opus-4-7", "xhigh", True),
        ("claude-opus-4-6", "max", True),
        ("claude-opus-4-6", "xhigh", False),
        ("claude-sonnet-4-6", "max", True),
        ("claude-sonnet-4-6", "xhigh", False),
        ("bedrock/invoke/us.anthropic.claude-opus-4-7", "max", True),
        ("bedrock/invoke/us.anthropic.claude-opus-4-7", "xhigh", True),
        ("bedrock/invoke/us.anthropic.claude-opus-4-6-v1", "max", True),
        ("bedrock/invoke/us.anthropic.claude-opus-4-6-v1", "xhigh", False),
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", "max", True),
        ("vertex_ai/claude-opus-4-7", "xhigh", True),
        ("azure_ai/claude-opus-4-7", "xhigh", True),
    ],
)
def test_supports_effort_level_handles_provider_prefixes(model, level, expected):
    """``_supports_effort_level`` resolves bedrock/vertex/azure-prefixed model ids."""
    assert AnthropicConfig._supports_effort_level(model, level, "anthropic") is expected


@pytest.mark.parametrize(
    "model,effort,expect_error",
    [
        ("claude-opus-4-6", "max", False),
        ("claude-sonnet-4-6", "max", False),
        ("claude-opus-4-7", "max", False),
        ("claude-opus-4-5-20251101", "max", True),
        ("claude-sonnet-4-5", "max", True),
        ("claude-opus-4-7", "xhigh", False),
        ("claude-opus-4-6", "xhigh", True),
        ("claude-sonnet-4-6", "xhigh", True),
        ("claude-opus-4-5-20251101", "high", False),
        ("claude-haiku-4-5", "low", False),
        ("claude-opus-4-5-20251101", None, False),
    ],
)
def test_validate_effort_for_model_centralises_per_model_gating(model, effort, expect_error):
    err = AnthropicConfig.validate_effort_for_model(model, effort, "anthropic")
    if expect_error:
        assert err is not None
        assert effort in err
        assert model in err
    else:
        assert err is None


def test_transform_request_injects_dummy_tool_without_tools_param():
    """
    Anthropic rejects messages that contain tool turns when ``tools`` is omitted.
    LiteLLM must inject a dummy tool without ``litellm.modify_params``.
    """
    config = AnthropicConfig()
    prev_modify_params = litellm.modify_params
    litellm.modify_params = False
    try:
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [
                    {
                        "id": "toolu_test_dummy",
                        "type": "function",
                        "function": {"name": "get_x", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_test_dummy",
                "content": "{}",
            },
        ]
        result = config.transform_request(
            model="claude-3-5-haiku-20241022",
            messages=messages,
            optional_params={"max_tokens": 256},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.modify_params = prev_modify_params

    assert "tools" in result
    names = [t.get("name") for t in result["tools"] if isinstance(t, dict) and t.get("name") is not None]
    assert "dummy_tool" in names


def test_transform_request_respects_user_max_tokens():
    """
    Test that transform_request respects user-provided max_tokens
    and doesn't override it with dynamic value.
    """
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]

    # User provides explicit max_tokens=1000, should not be overridden
    result = config.transform_request(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        optional_params={"max_tokens": 1000},
        litellm_params={},
        headers={},
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
    assert usage.completion_tokens_details.reasoning_tokens == 0
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

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=reasoning_content)

    # completion_tokens_details should be populated with both reasoning and text tokens
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens is not None
    assert usage.completion_tokens_details.reasoning_tokens > 0
    # text_tokens should be total minus reasoning
    expected_text_tokens = 500 - usage.completion_tokens_details.reasoning_tokens
    assert usage.completion_tokens_details.text_tokens == expected_text_tokens
    assert usage.completion_tokens == 500


# ============ Reasoning Effort Tests ============


def test_reasoning_effort_maps_to_adaptive_thinking_for_claude_4_6_models():
    """
    Test that reasoning_effort maps to adaptive thinking type for Claude 4.6 models,
    and also sets output_config with the effort level.
    """
    config = AnthropicConfig()

    effort_map = {
        "low": "low",
        "minimal": "low",
        "medium": "medium",
        "high": "high",
        "max": "max",
    }

    # Test with different reasoning_effort values - all should map to adaptive
    for model in ["claude-opus-4-6-20250514", "claude-sonnet-4-6-20260219"]:
        for effort in ["low", "medium", "high", "minimal", "max"]:
            non_default_params = {"reasoning_effort": effort}
            optional_params = {}

            result = config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=False,
            )

            # Should map to adaptive thinking type
            assert "thinking" in result
            assert result["thinking"]["type"] == "adaptive"
            # Should not have budget_tokens for adaptive type
            assert "budget_tokens" not in result["thinking"]
            # reasoning_effort should not be in the result (it's transformed to thinking)
            assert "reasoning_effort" not in result
            # Should set output_config with the mapped effort value
            assert "output_config" in result, f"output_config missing for {model} with effort={effort}"
            assert result["output_config"]["effort"] == effort_map[effort]


def test_raw_adaptive_thinking_translates_to_legacy_for_pre_46_model():
    """Clients like Claude Code send ``thinking={"type": "adaptive"}`` directly
    (not via ``reasoning_effort``) on every request, regardless of which model
    the request routes to. For a pre-4.6 model that doesn't understand
    adaptive thinking, this must be translated to the legacy
    ``thinking={type: enabled, budget_tokens}`` interface instead of being
    forwarded raw, which Anthropic would reject."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"thinking": {"type": "adaptive"}, "max_tokens": 8192},
        optional_params={},
        model="claude-haiku-4-5-20251001",
        drop_params=False,
    )

    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET


def test_raw_adaptive_thinking_budget_capped_below_max_tokens():
    """Anthropic requires ``max_tokens > thinking.budget_tokens``. When the
    default medium budget wouldn't fit, it must be capped below max_tokens
    rather than forwarded as an invalid combination."""
    config = AnthropicConfig()

    max_tokens = DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET - 100
    result = config.map_openai_params(
        non_default_params={"thinking": {"type": "adaptive"}, "max_tokens": max_tokens},
        optional_params={},
        model="claude-haiku-4-5-20251001",
        drop_params=False,
    )

    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == max_tokens - 1


def test_raw_adaptive_thinking_dropped_when_max_tokens_too_small():
    """When max_tokens can't fit even the minimum thinking budget, thinking
    must be dropped entirely so the request still succeeds, matching how the
    native /v1/messages passthrough already handles this."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "adaptive"},
            "max_tokens": ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
        },
        optional_params={},
        model="claude-haiku-4-5-20251001",
        drop_params=False,
    )

    assert "thinking" not in result


def test_raw_adaptive_thinking_untouched_for_46_plus_model():
    """Adaptive-thinking models understand ``thinking={"type": "adaptive"}``
    natively, so it must pass through unmodified."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"thinking": {"type": "adaptive"}, "max_tokens": 8192},
        optional_params={},
        model="claude-sonnet-4-6-20260219",
        drop_params=False,
    )

    assert result["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize(
    "model, expected",
    [
        # explicit cost-map entries, across provider routes / separators / date suffix
        ("claude-opus-4-8", True),
        ("anthropic.claude-opus-4-8", True),
        ("vertex_ai/claude-opus-4-6@default", True),
        ("us.anthropic.claude-sonnet-4-6", True),
        ("claude-opus-4-6-20260205", True),
        # unmapped future models -> anthropic-claude fallback rule
        ("claude-opus-4-9", True),
        ("claude-sonnet-5-0", True),
        # Claude 4.0 (dated): "4-20250514" must not be read as minor 4.20250514
        ("claude-opus-4-20250514", False),
        ("us.anthropic.claude-opus-4-20250514-v1:0", False),
        ("bedrock/invoke/us.anthropic.claude-opus-4-20250514", False),
        # sub-4.6 and legacy names
        ("claude-opus-4-5", False),
        ("claude-sonnet-4-5-20250929", False),
        ("claude-3-7-sonnet", False),
        ("claude-3-opus-20240229", False),
        ("gpt-4o", False),
    ],
)
def test_is_adaptive_thinking_model_is_sourced_from_cost_map(local_model_cost_map, model, expected):
    """Adaptive thinking resolves from the cost map first (an explicit
    supports_adaptive_thinking entry, or the anthropic-claude fallback rule for unmapped
    future Claudes), then from a date-safe opus/sonnet/haiku >= 4.6 name version as a
    fallback for ids the cost map cannot resolve. The dated Claude 4.0 names stay
    non-adaptive because the date suffix is not read as a minor version, while 4.8/4.9/5.x
    are covered without a code change."""
    assert AnthropicConfig.is_adaptive_thinking_model(model, "anthropic") is expected


def test_get_supported_params_includes_reasoning_for_sonnet_4_6_alias(
    local_model_cost_map,
):
    """Sonnet 4.6 aliases should expose thinking + reasoning_effort in supported params."""
    config = AnthropicConfig()

    params = config.get_supported_openai_params(model="claude-sonnet-4-6-20260219")

    assert "thinking" in params
    assert "reasoning_effort" in params


def test_get_supported_params_includes_reasoning_for_sonnet_4_6_dotted_alias(
    local_model_cost_map,
):
    """Dotted Sonnet 4.6 aliases should expose thinking + reasoning_effort in supported
    params. The anthropic-claude fallback rule accepts a dotted minor (4.6) as well as
    a dashed one, so an unmapped dotted alias still degrades to adaptive thinking."""
    config = AnthropicConfig()

    params = config.get_supported_openai_params(model="claude-sonnet-4.6")

    assert "thinking" in params
    assert "reasoning_effort" in params


def test_sonnet_4_6_reasoning_effort_to_transform_request_payload():
    """
    Sonnet 4.6 should convert reasoning_effort to adaptive thinking in final request payload.
    """
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Think through this carefully."}]

    mapped_optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="claude-sonnet-4-6-20260219",
        drop_params=False,
    )
    result = config.transform_request(
        model="claude-sonnet-4-6-20260219",
        messages=messages,
        optional_params=mapped_optional_params,
        litellm_params={},
        headers={},
    )

    assert "thinking" in result
    assert result["thinking"]["type"] == "adaptive"
    assert "budget_tokens" not in result["thinking"]


def test_reasoning_effort_maps_to_budget_thinking_for_non_opus_4_6():
    """
    Test that reasoning_effort maps to budget-based thinking config for non-Opus 4.6 models.

    For models other than Claude Opus 4.6, reasoning_effort should map to
    thinking config with budget_tokens based on the effort level.
    """
    config = AnthropicConfig()

    # ``minimal`` floors at ANTHROPIC_MIN_THINKING_BUDGET_TOKENS (1024).
    test_cases = [
        ("low", DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET),
        ("medium", DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET),
        ("high", DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET),
        ("minimal", 1024),
    ]

    for effort, expected_budget in test_cases:
        non_default_params = {"reasoning_effort": effort}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="claude-sonnet-4-5-20250929",
            drop_params=False,
        )

        # Should map to enabled thinking type with budget_tokens
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == expected_budget
        # reasoning_effort should not be in the result (it's transformed to thinking)
        assert "reasoning_effort" not in result


def test_reasoning_effort_sets_output_config_for_46_models():
    """
    Test that reasoning_effort generates output_config for Claude 4.6 models.

    For Claude 4.6 models, reasoning_effort should produce both adaptive
    thinking AND output_config with the mapped effort level.
    """
    config = AnthropicConfig()

    for model in ["claude-opus-4-6-20250514", "claude-sonnet-4-6-20260219"]:
        for effort in ["low", "medium", "high"]:
            result = config.map_openai_params(
                non_default_params={"reasoning_effort": effort},
                optional_params={},
                model=model,
                drop_params=False,
            )

            assert "output_config" in result, f"output_config missing for {model} with effort={effort}"
            assert result["output_config"]["effort"] == effort


def test_reasoning_effort_minimal_maps_to_low_output_config_for_46():
    """
    Test that reasoning_effort='minimal' maps to output_config effort='low'
    for 4.6 models, since 'minimal' has no Anthropic equivalent.
    """
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="claude-opus-4-6-20250514",
        drop_params=False,
    )

    assert result["output_config"]["effort"] == "low"


def test_reasoning_effort_does_not_set_output_config_for_older_models():
    """
    Test that reasoning_effort does NOT generate output_config for pre-4.6 models.
    """
    config = AnthropicConfig()

    for model in [
        "claude-sonnet-4-5-20250929",
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-5-20251101",
    ]:
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "high"},
            optional_params={},
            model=model,
            drop_params=False,
        )

        assert "output_config" not in result, f"output_config should not be set for {model}"


@pytest.mark.parametrize(
    "reasoning_effort_value",
    [
        # String shape — what callers send when using `reasoning_effort="low"` directly.
        "low",
        # Dict shape with `effort` only — what the Responses->Chat parser produces
        # when `reasoning={"effort": "low"}` is set without `summary`.
        {"effort": "low"},
        # Dict shape with `effort` AND `summary` — what the Responses->Chat parser
        # produces when callers send `Reasoning(effort="low", summary="concise")`.
        # PR #25359 added the dict-keeping branch for this case, but the Anthropic
        # transformation must coerce the dict back to a string before mapping.
        {"effort": "low", "summary": "concise"},
        {"effort": "low", "summary": "detailed"},
    ],
)
def test_reasoning_effort_accepts_dict_shape_for_adaptive_model(reasoning_effort_value):
    """
    Adaptive-thinking (Claude 4.6+) branch: dict-shape reasoning_effort must
    map to ``thinking.type='adaptive'`` + ``output_config.effort``.

    Regression test for the dict-shape ``reasoning_effort`` produced by the
    Responses->Chat parser when ``summary`` is set on the request's
    ``reasoning`` field. Before this fix, the Anthropic transformation guarded
    on ``isinstance(value, str)`` and silently dropped the param — disabling
    extended thinking entirely.
    """
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort_value},
        optional_params={},
        model="claude-sonnet-4-6-20260219",
        drop_params=False,
    )

    # thinking must be set (adaptive for 4.6+)
    assert "thinking" in result, f"thinking missing for reasoning_effort={reasoning_effort_value!r}"
    assert result["thinking"]["type"] == "adaptive"
    # output_config must carry the mapped effort
    assert "output_config" in result, f"output_config missing for reasoning_effort={reasoning_effort_value!r}"
    assert result["output_config"]["effort"] == "low"


@pytest.mark.parametrize(
    "reasoning_effort_value",
    [
        "low",
        {"effort": "low"},
        {"effort": "low", "summary": "concise"},
    ],
)
def test_reasoning_effort_accepts_dict_shape_for_non_adaptive_model(
    reasoning_effort_value,
):
    """
    Non-adaptive (pre-4.6) branch: dict-shape reasoning_effort must still map
    to ``thinking.type='enabled'`` + ``budget_tokens``. ``output_config`` must
    NOT be set on these models.
    """
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort_value},
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert "thinking" in result, f"thinking missing for reasoning_effort={reasoning_effort_value!r}"
    assert result["thinking"]["type"] == "enabled"
    assert "budget_tokens" in result["thinking"]
    assert result["thinking"]["budget_tokens"] > 0
    # Older models must not get adaptive-thinking output_config
    assert "output_config" not in result, (
        f"output_config should not be set for non-adaptive model (reasoning_effort={reasoning_effort_value!r})"
    )


@pytest.mark.parametrize(
    "model,budget_tokens,expected",
    [
        ("claude-opus-4-8", 4096, ({"type": "adaptive"}, {"effort": "high"})),
        ("claude-opus-4-7", 24000, ({"type": "adaptive"}, {"effort": "xhigh"})),
        ("claude-opus-4-6", 4096, ({"type": "enabled", "budget_tokens": 4096}, None)),
        ("claude-sonnet-4-5-20250929", 4096, ({"type": "enabled", "budget_tokens": 4096}, None)),
    ],
)
def test_legacy_thinking_translated_to_adaptive_on_adaptive_only_models(model, budget_tokens, expected):
    """Adaptive-only models reject thinking={type: enabled} with a 400, so the
    legacy shape must be upgraded to adaptive + output_config.effort on
    /chat/completions too, while models that accept it keep the caller's budget."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": budget_tokens}, "max_tokens": 64000},
        optional_params={},
        model=model,
        drop_params=False,
    )

    assert (result["thinking"], result.get("output_config")) == expected


@pytest.mark.parametrize(
    "bad_value",
    [
        {"summary": "concise"},  # missing effort
        {"effort": None},  # explicit None effort
        {"effort": 123},  # non-string effort
    ],
)
def test_reasoning_effort_unparseable_dict_is_dropped(bad_value):
    """
    A dict shape that doesn't carry a usable ``effort`` key (e.g. only
    ``summary`` is set, or the value is some other unexpected type) should be
    silently dropped — not crash, not partially apply.
    """
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": bad_value},
        optional_params={},
        model="claude-sonnet-4-6-20260219",
        drop_params=False,
    )
    assert "thinking" not in result, f"thinking should not be set for bad value {bad_value!r}"
    assert "output_config" not in result, f"output_config should not be set for bad value {bad_value!r}"


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-sonnet-4-6-20260219",
        "us.anthropic.claude-sonnet-4-6",
        "bedrock/converse/us.anthropic.claude-sonnet-4-6",
        "vertex_ai/claude-sonnet-4-6",
        "openrouter/anthropic/claude-sonnet-4.6",
    ],
)
def test_max_effort_accepted_for_sonnet_46_variants(model):
    """``effort='max'`` is supported on Claude 4.6 (Opus + Sonnet) and 4.7."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Test"}]

    result = config.transform_request(
        model=model,
        messages=messages,
        optional_params={"output_config": {"effort": "max"}},
        litellm_params={},
        headers={},
    )

    assert result["output_config"]["effort"] == "max"


def test_max_effort_accepted_for_opus_46():
    """Test that effort='max' works for Opus 4.6."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Test"}]

    result = config.transform_request(
        model="claude-opus-4-6-20250514",
        messages=messages,
        optional_params={"output_config": {"effort": "max"}},
        litellm_params={},
        headers={},
    )

    assert result["output_config"]["effort"] == "max"


def test_max_effort_accepted_for_opus_47():
    """Test that effort='max' works for Opus 4.7."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Test"}]

    result = config.transform_request(
        model="claude-opus-4-7",
        messages=messages,
        optional_params={"output_config": {"effort": "max"}},
        litellm_params={},
        headers={},
    )

    assert result["output_config"]["effort"] == "max"


def test_effort_beta_header_not_injected_for_46_models():
    """
    Test that is_effort_used returns False for Claude 4.6 models.

    Claude 4.6 models use output_config as a stable API feature —
    no beta header should be injected.
    """
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model_info = AnthropicModelInfo()

    for model in ["claude-opus-4-6-20250514", "claude-sonnet-4-6-20260219"]:
        # Even with output_config present, should return False for 4.6 models
        result = model_info.is_effort_used(
            optional_params={"output_config": {"effort": "high"}},
            model=model,
            custom_llm_provider="anthropic",
        )
        assert result is False, f"is_effort_used should return False for {model}"


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-5-20251101",
        "claude-opus-4-6-20250514",
        "claude-sonnet-4-6-20260219",
        "claude-opus-4-7",
    ],
)
def test_reasoning_effort_none_omits_thinking_and_output_config(model):
    """reasoning_effort="none" must omit thinking and output_config from the request."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "none"},
        optional_params={},
        model=model,
        drop_params=False,
    )

    assert "thinking" not in result
    assert "output_config" not in result


@pytest.mark.parametrize(
    "effort",
    ["disabled", "invalid", ""],
)
def test_reasoning_effort_garbage_raises_bad_request(effort):
    """Unmapped reasoning_effort raises BadRequestError (clean 400, not a 500)."""
    config = AnthropicConfig()

    with pytest.raises(litellm.exceptions.BadRequestError):
        config.map_openai_params(
            non_default_params={"reasoning_effort": effort},
            optional_params={},
            model="claude-sonnet-4-5-20250929",
            drop_params=False,
        )


@pytest.mark.parametrize(
    "effort,expected_budget",
    [
        ("xhigh", DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET),
        ("max", DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET),
    ],
)
def test_reasoning_effort_xhigh_max_maps_to_budget_on_budget_model(effort, expected_budget):
    """``xhigh`` / ``max`` extend the budget_tokens progression on budget-mode models."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": effort},
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] == expected_budget
    assert "output_config" not in result


def test_output_config_effort_empty_string_raises_bad_request():
    """``output_config={"effort": ""}`` is rejected with a 400."""
    config = AnthropicConfig()

    with pytest.raises(litellm.exceptions.BadRequestError, match="Invalid effort"):
        config.transform_request(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"output_config": {"effort": ""}, "max_tokens": 32},
            litellm_params={},
            headers={},
        )


def test_reasoning_effort_minimal_floors_at_anthropic_provider_minimum():
    """``minimal`` floors at the Anthropic provider minimum (1024)."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"reasoning_effort": "minimal"},
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert result["thinking"]["type"] == "enabled"
    assert result["thinking"]["budget_tokens"] >= 1024


def test_effort_beta_header_still_injected_for_older_models():
    """
    Test that is_effort_used still returns True for pre-4.6 models
    when output_config is present.
    """
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    model_info = AnthropicModelInfo()

    result = model_info.is_effort_used(
        optional_params={"output_config": {"effort": "low"}},
        model="claude-opus-4-5-20251101",
        custom_llm_provider="anthropic",
    )
    assert result is True


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
            {"type": "text", "text": "I'll calculate that for you."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC",
                "name": "bash_code_execution",
                "input": {"command": "python3 << 'EOF'\nprint(2 + 2)\nEOF\n"},
            },
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01ABC",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "4\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01DEF",
                "name": "text_editor_code_execution",
                "input": {
                    "command": "create",
                    "path": "test.txt",
                    "file_text": "Hello",
                },
            },
            {
                "type": "text_editor_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01DEF",
                "content": {
                    "type": "text_editor_code_execution_result",
                    "is_file_update": False,
                },
            },
            {"type": "text", "text": "Done!"},
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
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


def test_code_execution_tool_results_in_hidden_params():
    """
    Test that tool_results reaches _hidden_params so the Responses API adapter
    can surface them via provider_specific_fields.

    The Responses API adapter reads _hidden_params.get("provider_specific_fields")
    to set provider_specific_fields on the response. Without this, server-side
    code execution results (stdout/stderr) are lost when using responses.create().
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
            {"type": "text", "text": "Here's the result."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01ABC",
                "name": "bash_code_execution",
                "input": {"command": "echo hello"},
            },
            {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01ABC",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "hello\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
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

    # Verify tool_results is in _hidden_params for the Responses API adapter
    hidden = transformed_response._hidden_params
    assert "provider_specific_fields" in hidden
    assert "tool_results" in hidden["provider_specific_fields"]
    assert len(hidden["provider_specific_fields"]["tool_results"]) == 1
    assert hidden["provider_specific_fields"]["tool_results"][0]["content"]["stdout"] == "hello\n"


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
            {"type": "text", "text": "Found tools."},
            {"type": "tool_search_tool_result", "tool_references": ["tool1", "tool2"]},
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
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
            {"type": "text", "text": "Here are the results."},
            {
                "type": "web_search_tool_result",
                "search_query": "test query",
                "results": [{"title": "Result 1", "url": "https://example.com"}],
            },
        ],
        "stop_reason": "stop",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
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
                "content": "Summary of the conversation: The user requested help building a web scraper...",
            },
            {
                "type": "text",
                "text": "I don't have access to real-time data, so I can't provide the current weather in San Francisco.",
            },
        ],
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "usage": {"input_tokens": 86, "output_tokens": 100},
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

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
                "content": "Summary of the conversation: The user requested help building a web scraper...",
            },
            {"type": "text", "text": "Here is the response."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 25},
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
            {"type": "compaction", "content": "First summary..."},
            {"type": "text", "text": "Some text."},
            {"type": "compaction", "content": "Second summary..."},
        ]
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

    # Verify both compaction blocks are extracted
    assert compaction_blocks is not None
    assert len(compaction_blocks) == 2
    assert compaction_blocks[0]["content"] == "First summary..."
    assert compaction_blocks[1]["content"] == "Second summary..."


@pytest.mark.parametrize(
    "messages_api,gateway,native_endpoint",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ],
)
async def test_native_compaction_wire_roundtrip(
    messages_api: bool,
    gateway: bool,
    native_endpoint: bool,
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    monkeypatch.setattr(litellm.anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    monkeypatch.setattr(litellm, "use_chat_completions_url_for_anthropic_messages", False)
    block: Final = {"type": "compaction", "content": "Exact summary", "signature": "opaque-signature"}
    operation: Final = {"type": "summarize", "instructions": "Keep identifiers"}
    usage: Final = {
        "input_tokens": 0,
        "output_tokens": 0,
        "iterations": [{"type": "compaction", "input_tokens": 103, "output_tokens": 165}],
    }
    chat_wire: Final = gateway and not native_endpoint
    base: Final = "https://gateway.test/v1" if gateway else "https://api.anthropic.com/v1"
    route: Final = respx_mock.post(f"{base}/{'chat/completions' if chat_wire else 'messages'}")

    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert len(request.headers.get_list("anthropic-beta")) == 1
        assert {value.strip() for value in request.headers["anthropic-beta"].split(",")} == {
            "compact-2026-09-04",
            "interleaved-thinking-2025-05-14",
        }
        if "compaction" in payload:
            assert payload["compaction"] == operation
        else:
            assert payload["messages"][0] == {"role": "assistant", "content": [block]}
        body: Final = (
            {
                "id": "chatcmpl_compact",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-sonnet-5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "provider_specific_fields": {"compaction_blocks": [block]},
                        },
                    }
                ],
                "usage": {"prompt_tokens": 103, "completion_tokens": 165, "total_tokens": 268},
            }
            if chat_wire
            else {
                "id": "msg_compact",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [block],
                "stop_reason": "compaction",
                "usage": usage,
            }
        )
        return httpx.Response(200, json=body)

    route.mock(side_effect=respond)
    call: Final = litellm.anthropic.messages.acreate if messages_api else litellm.acompletion
    params: Final = dict(
        model=f"{'openai/' if gateway else ''}anthropic/claude-sonnet-5",
        api_key="test",
        max_tokens=512,
        api_base=base if gateway else "https://api.anthropic.com",
        extra_headers={"Anthropic-Beta": f"interleaved-thinking-2025-05-14{',compact-2026-09-04' if gateway else ''}"},
        model_info={"supported_endpoints": ["/v1/messages"]} if native_endpoint else {},
    )
    response: Final = await call(
        messages=[{"role": "user", "content": "Remember identifiers"}], compaction=operation, **params
    )
    message: Final = response if messages_api else response.choices[0].message.model_dump()
    blocks: Final = message["content"] if messages_api else message["provider_specific_fields"]["compaction_blocks"]
    assert blocks == [block]
    if messages_api:
        assert response["stop_reason"] == "compaction"
        if not chat_wire:
            assert response["usage"] == usage
    if not gateway:
        replay: Final = {"role": "assistant", "content": blocks} if messages_api else message
        await call(messages=[replay, {"role": "user", "content": "Continue"}], **params)
    assert route.call_count == (1 if gateway else 2)


def test_compaction_block_request_transformation():
    """
    Test that compaction blocks from provider_specific_fields are correctly
    transformed back to Anthropic format in requests.
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        anthropic_messages_pt,
    )

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I don't have access to real-time data."}],
            "provider_specific_fields": {
                "compaction_blocks": [
                    {
                        "type": "compaction",
                        "content": "Summary of the conversation: The user requested help building a web scraper...",
                    }
                ]
            },
        },
        {"role": "user", "content": "What about New York?"},
    ]

    result = anthropic_messages_pt(messages=messages, model="claude-opus-4-6", llm_provider="anthropic")

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
        "context_management": {"edits": [{"type": "compact_20260112"}]},
        "max_tokens": 100,
    }

    result = config.transform_request(
        model="claude-opus-4-6",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
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
            {"type": "compaction", "content": "Summary of previous conversation..."},
            {"type": "thinking", "thinking": "Let me think about this..."},
            {"type": "text", "text": "Based on my analysis..."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco"},
            },
        ]
    }

    (
        text,
        citations,
        thinking_blocks,
        reasoning_content,
        tool_calls,
        web_search_results,
        tool_results,
        compaction_blocks,
    ) = config.extract_response_content(completion_response)

    # Verify all content types are extracted
    assert compaction_blocks is not None
    assert len(compaction_blocks) == 1
    assert thinking_blocks is not None
    assert len(thinking_blocks) == 1
    assert "Based on my analysis" in text
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_map_openai_context_management_to_anthropic():
    """
    Test mapping OpenAI Responses API context_management format to Anthropic format.
    """
    config = AnthropicConfig()

    # Test OpenAI list format with compaction
    openai_format = [{"type": "compaction", "compact_threshold": 200000}]
    result = config.map_openai_context_management_to_anthropic(openai_format)

    assert result is not None
    assert "edits" in result
    assert len(result["edits"]) == 1
    assert result["edits"][0]["type"] == "compact_20260112"
    assert result["edits"][0]["trigger"]["type"] == "input_tokens"
    assert result["edits"][0]["trigger"]["value"] == 200000

    # Test OpenAI format with instructions
    openai_format_with_instructions = [
        {
            "type": "compaction",
            "compact_threshold": 150000,
            "instructions": "Focus on preserving code snippets",
        }
    ]
    result = config.map_openai_context_management_to_anthropic(openai_format_with_instructions)

    assert result is not None
    assert result["edits"][0]["trigger"]["value"] == 150000
    assert result["edits"][0]["instructions"] == "Focus on preserving code snippets"

    # Test Anthropic format (should pass through)
    anthropic_format = {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 150000},
            }
        ]
    }
    result = config.map_openai_context_management_to_anthropic(anthropic_format)

    assert result == anthropic_format


def test_map_openai_params_with_context_management():
    """
    Test that map_openai_params correctly transforms context_management from OpenAI to Anthropic format.
    """
    config = AnthropicConfig()

    # Test with OpenAI list format
    non_default_params = {"context_management": [{"type": "compaction", "compact_threshold": 200000}]}
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-opus-4-6",
        drop_params=False,
    )

    assert "context_management" in result
    assert "edits" in result["context_management"]
    assert result["context_management"]["edits"][0]["type"] == "compact_20260112"
    assert result["context_management"]["edits"][0]["trigger"]["value"] == 200000

    # Test with Anthropic dict format (should pass through)
    non_default_params_anthropic = {
        "context_management": {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 150000},
                    "instructions": "Focus on preserving code",
                }
            ]
        }
    }
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params_anthropic,
        optional_params=optional_params,
        model="claude-opus-4-6",
        drop_params=False,
    )

    assert "context_management" in result
    assert result["context_management"] == non_default_params_anthropic["context_management"]


def test_cache_control_in_supported_params():
    """
    Test that cache_control is listed as a supported OpenAI param for Anthropic.
    """
    config = AnthropicConfig()
    params = config.get_supported_openai_params(model="claude-sonnet-4-20250514")
    assert "cache_control" in params


def test_map_openai_params_with_cache_control():
    """
    Test that map_openai_params correctly passes through top-level cache_control
    for Anthropic's automatic prompt caching.
    """
    config = AnthropicConfig()

    non_default_params = {"cache_control": {"type": "ephemeral"}}
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-sonnet-4-20250514",
        drop_params=False,
    )

    assert "cache_control" in result
    assert result["cache_control"] == {"type": "ephemeral"}


def test_map_openai_params_cache_control_ignored_when_not_dict():
    """
    Test that cache_control is ignored when it is not a dict.
    """
    config = AnthropicConfig()

    non_default_params = {"cache_control": "ephemeral"}
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-sonnet-4-20250514",
        drop_params=False,
    )

    assert "cache_control" not in result


def test_transform_request_includes_cache_control():
    """
    Test that transform_request includes top-level cache_control in the request body.
    """
    config = AnthropicConfig()

    messages = [{"role": "user", "content": "Hello"}]
    optional_params = {
        "max_tokens": 100,
        "cache_control": {"type": "ephemeral"},
    }

    result = config.transform_request(
        model="claude-sonnet-4-20250514",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert "cache_control" in result
    assert result["cache_control"] == {"type": "ephemeral"}


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
        "content": [{"type": "text", "text": "Just a regular response."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
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
        headers=headers, optional_params=optional_params
    )

    assert "anthropic-beta" in result_headers
    assert "fast-mode-2026-02-01" in result_headers["anthropic-beta"]


def test_fast_mode_with_other_beta_headers():
    """
    Test that fast mode beta header is combined with other beta headers.
    """
    config = AnthropicConfig()

    headers = {}
    optional_params = {"speed": "fast", "output_format": {"type": "json_object"}}

    result_headers = config.update_headers_with_optional_anthropic_beta(
        headers=headers, optional_params=optional_params
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

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None, speed="fast")

    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert hasattr(usage, "speed")
    assert usage.speed == "fast"


def test_fast_mode_cost_calculation():
    """
    Test that fast mode applies the 'fast' multiplier from provider_specific_entry
    on top of the base model cost (1.1x for claude-opus-4-6).
    """

    from litellm.llms.anthropic.cost_calculation import cost_per_token
    from litellm.types.utils import Usage

    base_prompt = 0.005
    base_completion = 0.025

    with (
        patch("litellm.llms.anthropic.cost_calculation.generic_cost_per_token") as mock_cost,
        patch("litellm.get_model_info") as mock_info,
    ):
        mock_cost.return_value = (base_prompt, base_completion)
        mock_info.return_value = {"provider_specific_entry": {"fast": 1.1, "us": 1.1}}

        usage_fast = Usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            speed="fast",
        )

        prompt_cost, completion_cost = cost_per_token(
            model="claude-opus-4-6",
            usage=usage_fast,
        )

        # generic_cost_per_token called with the plain base model name
        mock_cost.assert_called_once()
        assert mock_cost.call_args[1]["model"] == "claude-opus-4-6"
        assert mock_cost.call_args[1]["custom_llm_provider"] == "anthropic"

        # 1.1x multiplier applied
        assert abs(prompt_cost - base_prompt * 1.1) < 1e-10
        assert abs(completion_cost - base_completion * 1.1) < 1e-10


def test_fast_mode_with_inference_geo():
    """
    Test that fast mode + inference_geo both apply their multipliers from
    provider_specific_entry (1.1 * 1.1 = 1.21x for claude-opus-4-6).
    """

    from litellm.llms.anthropic.cost_calculation import cost_per_token
    from litellm.types.utils import Usage

    base_prompt = 0.005
    base_completion = 0.025

    with (
        patch("litellm.llms.anthropic.cost_calculation.generic_cost_per_token") as mock_cost,
        patch("litellm.get_model_info") as mock_info,
    ):
        mock_cost.return_value = (base_prompt, base_completion)
        mock_info.return_value = {"provider_specific_entry": {"fast": 1.1, "us": 1.1}}

        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=1000,
            speed="fast",
            inference_geo="us",
        )

        prompt_cost, completion_cost = cost_per_token(
            model="claude-opus-4-6",
            usage=usage,
        )

        # generic_cost_per_token called with the plain base model name
        mock_cost.assert_called_once()
        assert mock_cost.call_args[1]["model"] == "claude-opus-4-6"
        assert mock_cost.call_args[1]["custom_llm_provider"] == "anthropic"

        # 1.1 (fast) * 1.1 (us) = 1.21x multiplier applied
        expected_multiplier = 1.1 * 1.1
        assert abs(prompt_cost - base_prompt * expected_multiplier) < 1e-10
        assert abs(completion_cost - base_completion * expected_multiplier) < 1e-10


def test_calculate_usage_captures_service_tier():
    """
    Anthropic returns the assigned service tier on the response usage object
    (e.g. ``"priority"``). It must be surfaced on the Usage object so it is
    visible in logs and used to select tier-specific pricing.
    """
    config = AnthropicConfig()

    usage_object = {
        "input_tokens": 410,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 585,
        "service_tier": "priority",
    }

    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)

    assert usage.service_tier == "priority"


def test_calculate_usage_service_tier_defaults_to_none():
    """A response without a service tier must not invent one."""
    config = AnthropicConfig()

    usage = config.calculate_usage(
        usage_object={"input_tokens": 10, "output_tokens": 5},
        reasoning_content=None,
    )

    assert usage.service_tier is None


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
        drop_params=False,
    )

    assert "speed" in result
    assert result["speed"] == "fast"


def test_anthropic_drop_params_strips_speed_for_unsupported_models():
    """``drop_params=True`` strips unsupported ``speed`` for non-Opus models."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Hello"}]

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = config.transform_request(
            model="claude-sonnet-4-6",
            messages=messages,
            optional_params={"speed": "fast", "max_tokens": 1024},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert "speed" not in result


def test_anthropic_drop_params_keeps_speed_for_supporting_models():
    """``drop_params=True`` must not strip ``speed`` on Opus fast-mode models."""
    config = AnthropicConfig()
    messages = [{"role": "user", "content": "Hello"}]

    original = litellm.drop_params
    litellm.drop_params = True
    try:
        result = config.transform_request(
            model="claude-opus-4-6",
            messages=messages,
            optional_params={"speed": "fast", "max_tokens": 1024},
            litellm_params={},
            headers={},
        )
    finally:
        litellm.drop_params = original

    assert result.get("speed") == "fast"


def test_speed_raises_clean_error_without_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        config.map_openai_params(
            non_default_params={"speed": "fast"},
            optional_params={},
            model="claude-sonnet-4-6",
            drop_params=False,
        )


def test_map_openai_params_max_tokens_normalized_to_int():
    """
    Test that map_openai_params normalizes max_tokens to an integer (e.g. 0.7 -> 1).
    """
    config = AnthropicConfig()

    non_default_params = {"max_tokens": 0.7}
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="claude-3-5-sonnet-20241022",
        drop_params=False,
    )

    assert "max_tokens" in result
    assert result["max_tokens"] == 1


# ========================================================================
# Tool schema normalization tests
# ========================================================================


def test_map_tool_helper_enforces_object_type_when_missing():
    """
    Anthropic requires input_schema.type to be "object". When an OpenAI tool
    has parameters without a 'type' field (common with MCP servers), LiteLLM
    should inject type:"object" before forwarding to Anthropic.

    Without this fix, Anthropic rejects with:
        tools.N.custom.input_schema.type: Input should be 'object'
    """
    config = AnthropicConfig()

    # Tool with parameters that has properties but no 'type' field
    tool = {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for code patterns",
            "parameters": {
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    }

    original_params = tool["function"]["parameters"].copy()
    result, _ = config.map_tool_helper(tool)
    assert result is not None
    assert result["input_schema"]["type"] == "object"
    assert "properties" in result["input_schema"]
    assert "query" in result["input_schema"]["properties"]
    # Original parameters dict must not be modified in place
    assert tool["function"]["parameters"] == original_params, (
        "parameters dict was mutated; _map_tool_helper should not modify caller data"
    )


def test_map_tool_helper_enforces_object_type_when_wrong_type():
    """
    If a tool schema has type:"string" or type:"array" at the root level,
    LiteLLM should normalize it to type:"object" for Anthropic compatibility.
    """
    config = AnthropicConfig()

    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo input",
            "parameters": {
                "type": "string",
                "description": "The input to echo",
            },
        },
    }

    original_params = tool["function"]["parameters"].copy()
    result, _ = config.map_tool_helper(tool)
    assert result is not None
    assert result["input_schema"]["type"] == "object"
    assert result["input_schema"].get("properties") == {}, (
        "properties should be injected as {} when schema has non-object type and no properties key"
    )
    # Original parameters dict must not be modified in place
    assert tool["function"]["parameters"] == original_params, (
        "parameters dict was mutated; _map_tool_helper should not modify caller data"
    )


def test_map_tool_helper_preserves_valid_object_schema():
    """
    When a tool schema already has type:"object", it should be preserved
    without modification.
    """
    config = AnthropicConfig()

    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }

    result, _ = config.map_tool_helper(tool)
    assert result is not None
    assert result["input_schema"]["type"] == "object"
    assert "city" in result["input_schema"]["properties"]
    assert result["input_schema"]["required"] == ["city"]


def test_map_tool_helper_empty_parameters_get_default():
    """
    When parameters is entirely missing, the existing default should still
    produce a valid {type:"object", properties:{}} schema.
    """
    config = AnthropicConfig()

    tool = {
        "type": "function",
        "function": {
            "name": "no_params_tool",
            "description": "Tool with no parameters",
        },
    }

    result, _ = config.map_tool_helper(tool)
    assert result is not None
    assert result["input_schema"]["type"] == "object"
    assert result["input_schema"].get("properties") == {}


def test_extract_response_content_thinking_block_null_thinking():
    """
    Test that thinking blocks are not dropped when the 'thinking' field is null
    or missing. Regression test for https://github.com/BerriAI/litellm/issues/24026
    """
    config = AnthropicConfig()

    # Case 1: thinking key is explicitly null
    completion_response_null = {
        "content": [
            {"type": "thinking", "thinking": None, "signature": "sig123"},
            {"type": "text", "text": "Hello"},
        ]
    }
    text, _, thinking_blocks, _, _, _, _, _ = config.extract_response_content(completion_response_null)
    assert thinking_blocks is not None, "thinking blocks should not be None when thinking=null"
    assert len(thinking_blocks) == 1
    assert "Hello" in text

    # Case 2: thinking key is absent entirely
    completion_response_missing = {
        "content": [
            {"type": "thinking", "signature": "sig456"},
            {"type": "text", "text": "World"},
        ]
    }
    text, _, thinking_blocks, _, _, _, _, _ = config.extract_response_content(completion_response_missing)
    assert thinking_blocks is not None, "thinking blocks should not be None when thinking key is absent"
    assert len(thinking_blocks) == 1
    assert "World" in text

    # Case 3: thinking key has actual content (should still work)
    completion_response_text = {
        "content": [
            {"type": "thinking", "thinking": "Let me think...", "signature": "sig789"},
            {"type": "text", "text": "Done"},
        ]
    }
    text, _, thinking_blocks, _, _, _, _, _ = config.extract_response_content(completion_response_text)
    assert thinking_blocks is not None
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0]["thinking"] == "Let me think..."
    assert "Done" in text


def test_advisor_tool_map_tool_helper():
    """advisor_20260301 tool type should not raise ValueError."""
    config = AnthropicConfig()
    tool = {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "claude-opus-4-6",
    }
    returned_tool, mcp_server = config.map_tool_helper(tool)  # type: ignore
    assert returned_tool is not None
    assert returned_tool["type"] == "advisor_20260301"
    assert returned_tool["model"] == "claude-opus-4-6"
    assert mcp_server is None


def test_advisor_tool_map_tool_helper_with_optional_fields():
    """advisor_20260301 tool with max_uses and caching should be mapped correctly."""
    config = AnthropicConfig()
    tool = {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "claude-opus-4-6",
        "max_uses": 3,
        "caching": {"type": "ephemeral", "ttl": "5m"},
    }
    returned_tool, _ = config.map_tool_helper(tool)  # type: ignore
    assert returned_tool is not None
    assert returned_tool["max_uses"] == 3
    assert returned_tool["caching"] == {"type": "ephemeral", "ttl": "5m"}


def test_advisor_tool_map_tool_helper_missing_model():
    """advisor_20260301 without model should raise ValueError."""
    config = AnthropicConfig()
    tool = {"type": "advisor_20260301", "name": "advisor"}
    with pytest.raises(ValueError, match="valid model"):
        config.map_tool_helper(tool)  # type: ignore


def test_advisor_beta_header_injected():
    """advisor-tool-2026-03-01 beta header is auto-injected when advisor tool is present."""
    config = AnthropicConfig()
    headers: dict = {}
    optional_params = {
        "tools": [
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",
            }
        ]
    }
    result = config.update_headers_with_optional_anthropic_beta(headers, optional_params)
    assert ANTHROPIC_BETA_HEADER_VALUES.ADVISOR_TOOL_2026_03_01.value in result.get("anthropic-beta", "")


def test_advisor_beta_header_not_injected_without_tool():
    """advisor-tool-2026-03-01 beta header is NOT added when advisor tool is absent."""
    config = AnthropicConfig()
    headers: dict = {}
    optional_params: dict = {"tools": []}
    result = config.update_headers_with_optional_anthropic_beta(headers, optional_params)
    assert "advisor-tool-2026-03-01" not in result.get("anthropic-beta", "")


def test_advisor_tool_result_preserved_in_response():
    """advisor_tool_result blocks are preserved in tool_results (not dropped)."""
    config = AnthropicConfig()
    completion_response = {
        "content": [
            {"type": "text", "text": "Consulting advisor."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_abc123",
                "name": "advisor",
                "input": {},
            },
            {
                "type": "advisor_tool_result",
                "tool_use_id": "srvtoolu_abc123",
                "content": {
                    "type": "advisor_result",
                    "text": "Use a channel-based pattern.",
                },
            },
            {"type": "text", "text": "Here is the implementation."},
        ]
    }
    text, _, _, _, tool_calls, _, tool_results, _ = config.extract_response_content(completion_response)
    assert "Consulting advisor." in text
    assert "Here is the implementation." in text
    # server_tool_use (advisor) should be a tool_call
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "advisor"
    assert tool_calls[0]["id"] == "srvtoolu_abc123"
    # advisor_tool_result should be in tool_results
    assert tool_results is not None
    assert len(tool_results) == 1
    assert tool_results[0]["type"] == "advisor_tool_result"
    assert tool_results[0]["tool_use_id"] == "srvtoolu_abc123"


def test_messages_path_advisor_beta_header_injected():
    """advisor-tool-2026-03-01 beta header is auto-injected in /messages path."""
    config = AnthropicMessagesConfig()
    headers: dict = {}
    optional_params = {
        "tools": [
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",
            }
        ]
    }
    result = config._update_headers_with_anthropic_beta(headers, optional_params)
    assert "advisor-tool-2026-03-01" in result.get("anthropic-beta", "")


def test_messages_path_advisor_beta_header_preserved_when_user_sends_it():
    """Existing anthropic-beta headers are preserved and advisor header is merged."""
    config = AnthropicMessagesConfig()
    headers: dict = {"anthropic-beta": "advisor-tool-2026-03-01"}
    optional_params: dict = {"tools": []}
    result = config._update_headers_with_anthropic_beta(headers, optional_params)
    assert "advisor-tool-2026-03-01" in result.get("anthropic-beta", "")


def test_strip_advisor_blocks_when_no_advisor_tool():
    """
    Auto-strip removes server_tool_use(advisor) + advisor_tool_result blocks when
    advisor tool is absent, preventing Anthropic 400 on follow-up turns.
    """
    from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages

    messages = [
        {"role": "user", "content": "Build a worker pool."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me consult the advisor."},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_abc123",
                    "name": "advisor",
                    "input": {},
                },
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": "srvtoolu_abc123",
                    "content": {"type": "advisor_result", "text": "Use channels."},
                },
                {"type": "text", "text": "Here is the implementation."},
            ],
        },
    ]
    result = strip_advisor_blocks_from_messages(messages)
    assistant_content = result[1]["content"]
    types = [b["type"] for b in assistant_content]
    assert "server_tool_use" not in types
    assert "advisor_tool_result" not in types
    assert "text" in types
    assert len(assistant_content) == 2


def test_strip_advisor_blocks_no_op_when_no_advisor_blocks():
    """strip_advisor_blocks_from_messages is a no-op when no advisor blocks exist."""
    from litellm.llms.anthropic.common_utils import strip_advisor_blocks_from_messages

    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hi there"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                },
            ],
        },
    ]
    original_content = [dict(b) for b in messages[1]["content"]]
    result = strip_advisor_blocks_from_messages(messages)
    assert result[1]["content"] == original_content


# ---------------------------------------------------------------------------
# Tool-name sanitization for Anthropic compatibility (^[a-zA-Z0-9_-]{1,128}$)
# Repro: Slack-bot agent sent an MCP tool named
# "github_openapi_mcp-actions/download-job-logs-for-workflow-run" which 400'd
# with `tools.N.custom.name: String should match pattern`.
# ---------------------------------------------------------------------------


def test_basic_sanitize_anthropic_tool_name_replaces_invalid_chars():
    from litellm.llms.anthropic.chat.transformation import (
        _basic_sanitize_anthropic_tool_name,
    )

    assert (
        _basic_sanitize_anthropic_tool_name("github_openapi_mcp-actions/download-job-logs-for-workflow-run")
        == "github_openapi_mcp-actions_download-job-logs-for-workflow-run"
    )
    # other punctuation
    assert _basic_sanitize_anthropic_tool_name("foo.bar:baz qux") == "foo_bar_baz_qux"
    # already valid -> unchanged
    assert _basic_sanitize_anthropic_tool_name("plain_tool-1") == "plain_tool-1"
    # empty
    assert _basic_sanitize_anthropic_tool_name("") == ""
    # 128-char cap
    long = "a/" * 200
    out = _basic_sanitize_anthropic_tool_name(long)
    assert len(out) <= 128


def test_build_anthropic_tool_name_maps_no_collisions():
    """Names that need rewriting go in the maps; valid names stay out."""
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(
        [
            "fine_name",
            "actions/download-job-logs-for-workflow-run",
            "pulls/list-files",
        ]
    )
    assert forward == {
        "actions/download-job-logs-for-workflow-run": ("actions_download-job-logs-for-workflow-run"),
        "pulls/list-files": "pulls_list-files",
    }
    assert reverse == {v: k for k, v in forward.items()}
    # untouched names absent
    assert "fine_name" not in forward
    assert "fine_name" not in reverse


def test_build_anthropic_tool_name_maps_disambiguates_collision_with_existing_valid():
    """If `foo/bar` would collapse to `foo_bar` but `foo_bar` already exists,
    the rewritten one must get a unique suffix and only THAT one shows up in
    the reverse map. The legitimately-named `foo_bar` round-trips identically."""
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(["foo_bar", "foo/bar"])
    # The original valid name keeps its slot.
    assert "foo_bar" not in forward  # untouched
    # The rewritten one gets a disambiguating suffix.
    assert forward["foo/bar"] == "foo_bar_2"
    # Reverse map only has the rewritten entry.
    assert reverse == {"foo_bar_2": "foo/bar"}
    # CRITICAL: a legit `foo_bar` returned by the model must NOT round-trip
    # to `foo/bar`.
    assert "foo_bar" not in reverse


def test_build_anthropic_tool_name_maps_disambiguates_two_rewrites_to_same_target():
    """Two different invalid names that collapse to the same candidate must
    both end up with unique sanitized forms."""
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(["foo/bar", "foo.bar"])
    # First wins the canonical slot, second gets a suffix.
    assert forward["foo/bar"] == "foo_bar"
    assert forward["foo.bar"] == "foo_bar_2"
    # Round-trip is unambiguous.
    assert reverse["foo_bar"] == "foo/bar"
    assert reverse["foo_bar_2"] == "foo.bar"


def test_build_anthropic_tool_name_maps_three_way_collision():
    """`foo/bar`, `foo.bar`, and an existing `foo_bar` must all coexist."""
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(["foo_bar", "foo/bar", "foo.bar"])
    assert "foo_bar" not in forward  # untouched
    assert forward["foo/bar"] == "foo_bar_2"
    assert forward["foo.bar"] == "foo_bar_3"
    # All three sanitized names are distinct.
    sent_names = {"foo_bar", forward["foo/bar"], forward["foo.bar"]}
    assert len(sent_names) == 3
    assert reverse == {"foo_bar_2": "foo/bar", "foo_bar_3": "foo.bar"}


def test_build_anthropic_tool_name_maps_reverse_order_collision():
    """REGRESSION: when the invalid name appears *before* the valid name that
    its sanitized form collides with, both must still end up with distinct
    names on the wire."""
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(["foo/bar", "foo_bar"])
    # The valid name keeps its slot untouched.
    assert "foo_bar" not in forward
    # The rewritten one gets a disambiguating suffix.
    assert forward["foo/bar"] == "foo_bar_2"
    assert reverse == {"foo_bar_2": "foo/bar"}
    assert "foo_bar" not in reverse


def test_build_anthropic_tool_name_maps_duplicate_originals():
    """REGRESSION: duplicate originals must not corrupt the forward map.

    Previously, the second occurrence of the same invalid name would
    rewrite ``forward[original]`` to a suffixed name (``foo_bar_2``),
    leaving ``foo_bar`` orphaned in ``used`` with no reverse mapping —
    so when ``_sanitize_tool_names_in_request`` applied the forward
    map, *both* tool entries got the suffixed name and Anthropic 400'd
    on duplicates.
    """
    from litellm.llms.anthropic.chat.transformation import (
        _build_anthropic_tool_name_maps,
    )

    forward, reverse = _build_anthropic_tool_name_maps(["foo/bar", "foo/bar"])
    # Same original sanitizes to the same target — no spurious suffix.
    assert forward == {"foo/bar": "foo_bar"}
    assert reverse == {"foo_bar": "foo/bar"}


def test_map_openai_params_does_not_pollute_optional_params_with_internal_keys():
    """REGRESSION: ``optional_params`` is what becomes the JSON body sent to
    Anthropic (``data = {**optional_params}``). It MUST NOT carry LiteLLM-
    internal coordination state like the per-request forward/reverse name
    maps, or Anthropic 400s with ``Extra inputs are not permitted``.
    Sanitization belongs in ``transform_request``, not here."""
    config = AnthropicConfig()
    optional_params: dict = {}
    config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "actions/download-job-logs-for-workflow-run",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        },
        optional_params=optional_params,
        model="claude-sonnet-4",
        drop_params=False,
    )
    # No internal keys may appear in optional_params for ANY input.
    for key in optional_params:
        assert not key.startswith("_anthropic_tool_name"), (
            f"optional_params leaked internal key {key!r}: {optional_params}"
        )
    # And no key starting with `_` either; optional_params should only
    # contain documented Anthropic Messages API parameters.
    for key in optional_params:
        assert not key.startswith("_"), f"optional_params leaked underscore-prefixed key {key!r}: {optional_params}"


def test_map_openai_params_no_maps_when_all_names_already_valid():
    """Sanity check: an all-valid tool list adds nothing weird either."""
    config = AnthropicConfig()
    optional_params: dict = {}
    config.map_openai_params(
        non_default_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "plain_tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        },
        optional_params=optional_params,
        model="claude-sonnet-4",
        drop_params=False,
    )
    for key in optional_params:
        assert not key.startswith("_anthropic_tool_name")


def test_rewrite_tool_names_in_messages_uses_forward_map():
    config = AnthropicConfig()
    forward_map = {"actions/download-job-logs-for-workflow-run": ("actions_download-job-logs-for-workflow-run")}
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "actions/download-job-logs-for-workflow-run",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    out = config._rewrite_tool_names_in_messages(messages, forward_map)

    # input list must not be mutated
    assert messages[1]["tool_calls"][0]["function"]["name"] == "actions/download-job-logs-for-workflow-run"
    # output rewritten according to forward map
    assert out[1]["tool_calls"][0]["function"]["name"] == "actions_download-job-logs-for-workflow-run"
    # non-tool-call messages pass through unchanged (same object)
    assert out[0] is messages[0]
    assert out[2] is messages[2]


def test_rewrite_tool_names_in_messages_leaves_unmapped_names_alone():
    """A tool_call name not in the forward map must NOT be rewritten,
    even if it happens to look like a sanitized form of some other tool."""
    config = AnthropicConfig()
    # `foo_bar` is NOT in the forward map (only `foo/bar` -> `foo_bar_2` is).
    # If we naively re-sanitized, `foo_bar` would stay `foo_bar`, but more
    # subtly, in a buggy implementation we might collide it with the codomain
    # of some other rewrite. Either way: it must round-trip identically.
    forward_map = {"foo/bar": "foo_bar_2"}
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "foo_bar", "arguments": "{}"},
                }
            ],
        },
    ]
    out = config._rewrite_tool_names_in_messages(messages, forward_map)
    assert out[0]["tool_calls"][0]["function"]["name"] == "foo_bar"
    # input list must not be mutated either way
    assert messages[0]["tool_calls"][0]["function"]["name"] == "foo_bar"


def test_rewrite_tool_names_in_messages_with_tool_calls_and_none_function_call():
    """When a message has tool_calls but function_call is explicitly None,
    the rewrite must still apply to tool_calls and leave function_call as
    None. Pins behavior at the boundary where ``new_msg = dict(msg)``
    copies the explicit-None key forward."""
    config = AnthropicConfig()
    forward_map = {"foo/bar": "foo_bar"}
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "foo/bar", "arguments": "{}"},
                }
            ],
            "function_call": None,
        },
    ]
    out = config._rewrite_tool_names_in_messages(messages, forward_map)
    assert out[0]["tool_calls"][0]["function"]["name"] == "foo_bar"
    assert out[0]["function_call"] is None
    # input list must not be mutated
    assert messages[0]["tool_calls"][0]["function"]["name"] == "foo/bar"


def test_sanitize_tool_names_in_request_does_not_mutate_caller_tool_dicts():
    """REGRESSION: a caller reusing the same tool list/dicts across requests
    must not see its inputs permanently rewritten. _sanitize_tool_names_in_request
    builds a new list with copy-on-change entries."""
    config = AnthropicConfig()
    original_name = "actions/download-job-logs-for-workflow-run"
    caller_tool = {
        "type": "custom",
        "name": original_name,
        "input_schema": {"type": "object", "properties": {}},
    }
    caller_tools = [caller_tool]
    optional_params: dict = {"tools": caller_tools}

    forward, reverse = config._sanitize_tool_names_in_request(optional_params=optional_params)

    assert forward.get(original_name)
    sanitized = forward[original_name]
    assert optional_params["tools"][0]["name"] == sanitized
    # caller's original dict + list must not be touched
    assert caller_tool["name"] == original_name
    assert caller_tools[0] is caller_tool


def test_transform_parsed_response_reverse_maps_tool_names():
    """End-to-end: rewritten tool name in Anthropic response -> original in OpenAI tool_calls."""
    import json as _json

    config = AnthropicConfig()
    raw_response = MagicMock()
    raw_response.headers = {}
    raw_response.status_code = 200

    completion_response = {
        "id": "msg_x",
        "model": "claude-sonnet-4",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "actions_download-job-logs-for-workflow-run",
                "input": {"job_id": 123},
            }
        ],
    }
    from litellm.types.utils import ModelResponse

    model_response = ModelResponse()

    out = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        tool_name_reverse_map={
            "actions_download-job-logs-for-workflow-run": "actions/download-job-logs-for-workflow-run",
        },
    )

    tcs = out.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].function.name == "actions/download-job-logs-for-workflow-run"
    assert _json.loads(tcs[0].function.arguments) == {"job_id": 123}


def test_transform_parsed_response_does_not_rewrite_unmapped_names():
    """CRITICAL: a tool legitimately named `foo_bar` must NOT be rewritten
    to `foo/bar` just because some other request had that pair. The reverse
    map is per-request -- only entries we actually created go in it."""
    config = AnthropicConfig()
    raw_response = MagicMock()
    raw_response.headers = {}
    raw_response.status_code = 200

    # Caller registered `foo_bar` (valid) and `foo/bar` (rewrites to foo_bar_2).
    # The reverse map only contains the rewrite.
    reverse_map = {"foo_bar_2": "foo/bar"}

    completion_response = {
        "id": "msg_x",
        "model": "claude-sonnet-4",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "foo_bar",  # the legit one, NOT in reverse map
                "input": {},
            }
        ],
    }
    from litellm.types.utils import ModelResponse

    model_response = ModelResponse()
    out = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
        tool_name_reverse_map=reverse_map,
    )
    # Must come back as-is, not rewritten to "foo/bar".
    assert out.choices[0].message.tool_calls[0].function.name == "foo_bar"


def test_transform_parsed_response_no_reverse_map_is_noop():
    """When no map is provided, tool name is passed through unchanged."""
    config = AnthropicConfig()
    raw_response = MagicMock()
    raw_response.headers = {}
    raw_response.status_code = 200

    completion_response = {
        "id": "msg_x",
        "model": "claude-sonnet-4",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "plain_tool",
                "input": {},
            }
        ],
    }
    from litellm.types.utils import ModelResponse

    model_response = ModelResponse()
    out = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
    )
    assert out.choices[0].message.tool_calls[0].function.name == "plain_tool"


def test_streaming_iterator_reverse_maps_tool_use_name():
    """Streaming `content_block_start` for tool_use should reverse-map the name."""
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        tool_name_reverse_map={
            "actions_download-job-logs-for-workflow-run": "actions/download-job-logs-for-workflow-run",
        },
    )

    chunk = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "actions_download-job-logs-for-workflow-run",
            "input": {},
        },
    }
    parsed = iterator.chunk_parser(chunk=chunk)
    tool_calls = parsed.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "actions/download-job-logs-for-workflow-run"


def test_streaming_iterator_passthrough_when_name_not_in_map():
    from litellm.llms.anthropic.chat.handler import ModelResponseIterator

    iterator = ModelResponseIterator(
        streaming_response=iter([]),
        sync_stream=True,
        tool_name_reverse_map=None,
    )
    chunk = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "plain_tool",
            "input": {},
        },
    }
    parsed = iterator.chunk_parser(chunk=chunk)
    tool_calls = parsed.choices[0].delta.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "plain_tool"


# ---------------------------------------------------------------------------
# transform_request: end-to-end sanitization regression coverage
# ---------------------------------------------------------------------------


def _build_optional_params_for_tools(tools):
    """Run a tools list through ``map_openai_params`` to get the same shape
    ``transform_request`` will see from the router. Keeping this helper local
    avoids duplicating the OpenAI->Anthropic param mapping in tests."""
    config = AnthropicConfig()
    optional_params: dict = {}
    config.map_openai_params(
        non_default_params={"tools": tools},
        optional_params=optional_params,
        model="claude-sonnet-4",
        drop_params=False,
    )
    return optional_params


def test_transform_request_does_not_leak_internal_keys_into_body():
    """REGRESSION for "_anthropic_tool_name_forward_map: Extra inputs are not
    permitted". The dict returned by ``transform_request`` is what becomes
    the JSON body POSTed to Anthropic. It must contain ONLY documented
    Anthropic Messages fields -- no LiteLLM coordination state."""
    config = AnthropicConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "github_openapi_mcp-actions/download-job-logs-for-workflow-run",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plain_tool",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    optional_params = _build_optional_params_for_tools(tools)
    litellm_params: dict = {}

    data = config.transform_request(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "go"}],
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )

    # Body must not contain any LiteLLM-internal keys.
    for key in data.keys():
        assert not key.startswith("_"), (
            f"transformed request body leaked underscore-prefixed key {key!r}; "
            f"Anthropic will reject this with 'Extra inputs are not permitted'. "
            f"body keys: {list(data.keys())}"
        )

    # Tool names in the body match Anthropic's pattern.
    import re as _re

    for tool in data.get("tools", []):
        name = tool.get("name")
        assert isinstance(name, str)
        assert _re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name), (
            f"sanitized tool name {name!r} still violates Anthropic regex"
        )

    # Sent name for the bad tool is the disambiguated form, valid name passes through.
    sent_names = {t["name"] for t in data["tools"]}
    assert "github_openapi_mcp-actions_download-job-logs-for-workflow-run" in sent_names
    assert "plain_tool" in sent_names

    # Reverse map landed on litellm_params (NOT optional_params, NOT body).
    rmap = litellm_params["_anthropic_tool_name_map"]
    assert (
        rmap["github_openapi_mcp-actions_download-job-logs-for-workflow-run"]
        == "github_openapi_mcp-actions/download-job-logs-for-workflow-run"
    )
    # The legitimately-named tool is not in the reverse map -- it round-trips
    # untouched on the response side.
    assert "plain_tool" not in rmap


def test_transform_request_no_reverse_map_when_all_names_valid():
    """If every name is already valid, ``litellm_params`` stays clean
    (no reverse map key) -- minimizes blast radius for the common case."""
    config = AnthropicConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "plain_tool",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    optional_params = _build_optional_params_for_tools(tools)
    litellm_params: dict = {}

    data = config.transform_request(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "go"}],
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )
    assert data["tools"][0]["name"] == "plain_tool"
    assert "_anthropic_tool_name_map" not in litellm_params


def test_transform_request_sanitizes_tool_choice_named_tool():
    """``tool_choice={"type": "function", "function": {"name": "<bad/name>"}}``
    must arrive at Anthropic as ``{"type": "tool", "name": "<sanitized>"}``,
    matching the sanitized name in the tools array."""
    config = AnthropicConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "actions/download-job-logs-for-workflow-run",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    optional_params = AnthropicConfig().map_openai_params(
        non_default_params={
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "actions/download-job-logs-for-workflow-run"},
            },
        },
        optional_params={},
        model="claude-sonnet-4",
        drop_params=False,
    )
    litellm_params: dict = {}
    data = config.transform_request(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "go"}],
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )
    assert data["tool_choice"]["type"] == "tool"
    assert data["tool_choice"]["name"] == "actions_download-job-logs-for-workflow-run"
    assert data["tools"][0]["name"] == "actions_download-job-logs-for-workflow-run"


def test_transform_request_rewrites_tool_names_in_history():
    """Historical assistant messages with ``tool_calls`` referencing the bad
    name must be rewritten to the sanitized form so Anthropic doesn't 400 on
    ``tool_use.name`` mismatching the (sanitized) tools array."""
    config = AnthropicConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "actions/download-job-logs-for-workflow-run",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    optional_params = _build_optional_params_for_tools(tools)
    messages = [
        {"role": "user", "content": "logs please"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_old",
                    "type": "function",
                    "function": {
                        "name": "actions/download-job-logs-for-workflow-run",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_old", "content": "..."},
        {"role": "user", "content": "again"},
    ]
    litellm_params: dict = {}
    data = config.transform_request(
        model="claude-sonnet-4",
        messages=messages,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )
    # Find the assistant tool_use block in the Anthropic-shaped messages.
    tool_use_names = []
    for msg in data["messages"]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_use_names.append(block.get("name"))
    assert tool_use_names, "expected at least one tool_use block in transformed messages"
    for name in tool_use_names:
        assert name == "actions_download-job-logs-for-workflow-run", (
            f"history tool_use.name {name!r} not rewritten -- Anthropic will "
            f"400 because it doesn't match the (sanitized) tools array"
        )


def test_sanitize_tool_names_in_request_skips_hosted_tools():
    """Hosted tools (web_search, computer_*, code_execution, ...) own
    Anthropic-reserved names. The sanitizer must not enumerate them as
    ``custom`` and must not rename them."""
    optional_params = {
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {
                "type": "custom",
                "name": "actions/download-job-logs-for-workflow-run",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }
    forward, reverse = AnthropicConfig._sanitize_tool_names_in_request(optional_params)
    # Only the custom tool was rewritten.
    assert forward == {"actions/download-job-logs-for-workflow-run": "actions_download-job-logs-for-workflow-run"}
    assert reverse == {"actions_download-job-logs-for-workflow-run": "actions/download-job-logs-for-workflow-run"}
    # Hosted tool's name unchanged.
    assert optional_params["tools"][0]["name"] == "web_search"
    # Custom tool's name updated in place.
    assert optional_params["tools"][1]["name"] == "actions_download-job-logs-for-workflow-run"


def test_sanitize_tool_names_in_request_no_tools_is_noop():
    """Empty / missing tools must not error or pollute return."""
    forward, reverse = AnthropicConfig._sanitize_tool_names_in_request({})
    assert forward == {}
    assert reverse == {}
    forward, reverse = AnthropicConfig._sanitize_tool_names_in_request({"tools": []})
    assert forward == {}
    assert reverse == {}


# -----------------------------------------------------------------------------
# Regression tests for legacy / OpenAPI $ref defs in tool input_schema.
#
# Anthropic only resolves `$defs` (JSON Schema 2020-12). Tools coming from MCP
# servers (legacy `definitions`) or OpenAPI-derived gateways like AWS
# AgentCore (`components.schemas`) used to silently lose their def blocks
# while keeping dangling `$ref`s, causing upstream 400s. See
# https://github.com/BerriAI/litellm/issues/26692.
# -----------------------------------------------------------------------------


def _assert_no_unresolved_refs(input_schema: dict) -> None:
    import json

    blob = json.dumps(input_schema)
    assert "$ref" not in blob, f"unresolved $ref in transformed input_schema: {blob}"


def test_map_tool_helper_inlines_components_schemas_refs():
    """OpenAPI `components.schemas` $refs (AgentCore-style) must be inlined."""
    config = AnthropicConfig()
    tool = {
        "type": "function",
        "function": {
            "name": "slides_presentations_create",
            "description": "Create a Google Slides presentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"$ref": "#/components/schemas/Presentation"},
                },
                "required": ["body"],
                "components": {
                    "schemas": {
                        "Presentation": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "presentationId": {"type": "string"},
                            },
                        }
                    }
                },
            },
        },
    }

    transformed, _ = config.map_tool_helper(tool)

    assert transformed is not None
    schema = transformed["input_schema"]
    _assert_no_unresolved_refs(schema)
    assert schema["properties"]["body"] == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "presentationId": {"type": "string"},
        },
    }
    # The OpenAPI components block is not part of Anthropic's allow-list and
    # must not be forwarded.
    assert "components" not in schema


def test_map_tool_helper_inlines_legacy_definitions_refs():
    """Legacy draft-04 `definitions` $refs (DevRev MCP-style) must be inlined."""
    config = AnthropicConfig()
    tool = {
        "type": "function",
        "function": {
            "name": "create_thing",
            "description": "Create a thing",
            "parameters": {
                "type": "object",
                "properties": {
                    "thing": {"$ref": "#/definitions/Thing"},
                },
                "definitions": {
                    "Thing": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    }
                },
            },
        },
    }

    transformed, _ = config.map_tool_helper(tool)

    assert transformed is not None
    schema = transformed["input_schema"]
    _assert_no_unresolved_refs(schema)
    assert schema["properties"]["thing"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    assert "definitions" not in schema


def test_map_tool_helper_preserves_native_dollar_defs():
    """`$defs` is JSON Schema 2020-12 native; Anthropic resolves it itself.

    Re-implementation must not pop or unpack `$defs`.
    """
    config = AnthropicConfig()
    tool = {
        "type": "function",
        "function": {
            "name": "native_defs_tool",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"type": "string"}},
            },
        },
    }

    transformed, _ = config.map_tool_helper(tool)

    assert transformed is not None
    schema = transformed["input_schema"]
    assert schema["$defs"] == {"A": {"type": "string"}}
    assert schema["properties"]["a"] == {"$ref": "#/$defs/A"}


def test_map_tool_helper_does_not_mutate_caller_dict():
    """Caller-supplied tool dict must not be mutated by the inlining step."""
    import copy

    config = AnthropicConfig()
    tool = {
        "type": "function",
        "function": {
            "name": "create_thing",
            "description": "Create a thing",
            "parameters": {
                "type": "object",
                "properties": {"thing": {"$ref": "#/definitions/Thing"}},
                "definitions": {
                    "Thing": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    }
                },
            },
        },
    }
    snapshot = copy.deepcopy(tool)

    config.map_tool_helper(tool)

    assert tool == snapshot, "caller's tool dict was mutated in place"


def test_map_tool_helper_collision_prefers_definitions_over_components_schemas():
    """If both `definitions.X` and `components.schemas.X` exist with the same
    name, prefer the `definitions` body. ``unpack_defs`` keys refs by last path
    segment so only one body can win; pick the JSON-Schema-native one.

    This locks in the residual limitation as a deliberate contract: a ref
    written as ``#/components/schemas/X`` will *also* resolve to the
    ``definitions`` body when both namespaces define ``X``. Cross-namespace
    disambiguation would require teaching ``unpack_defs`` to key by full ref
    path, which is out of scope here.
    """
    config = AnthropicConfig()
    tool = {
        "type": "function",
        "function": {
            "name": "collision_tool",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_definitions": {"$ref": "#/definitions/Thing"},
                    "from_components": {"$ref": "#/components/schemas/Thing"},
                },
                "definitions": {
                    "Thing": {"type": "string", "description": "from-definitions"},
                },
                "components": {
                    "schemas": {
                        "Thing": {"type": "integer", "description": "from-components"},
                    }
                },
            },
        },
    }

    transformed, _ = config.map_tool_helper(tool)

    assert transformed is not None
    expected = {"type": "string", "description": "from-definitions"}
    # Direct ref resolves to the `definitions` body (the documented winner).
    assert transformed["input_schema"]["properties"]["from_definitions"] == expected
    # Cross-namespace ref *also* resolves to the `definitions` body because
    # ``unpack_defs`` keys by last path segment -- documented limitation.
    assert transformed["input_schema"]["properties"]["from_components"] == expected


BILLING_HEADER_BLOCK = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=1.0.abc; cc_entrypoint=cli; cch=00000;",
}


def _system_with_billing_header(real_text: str) -> list:
    return [
        {
            "role": "system",
            "content": [BILLING_HEADER_BLOCK, {"type": "text", "text": real_text}],
        }
    ]


def test_translate_system_message_keeps_billing_header_for_first_party_anthropic():
    config = AnthropicConfig()
    assert config.should_strip_billing_metadata() is False

    result = config.translate_system_message(
        messages=_system_with_billing_header("You are Claude Code, Anthropic's official CLI for Claude.")
    )

    texts = [block["text"] for block in result]
    assert any(t.startswith("x-anthropic-billing-header:") for t in texts)
    assert "You are Claude Code, Anthropic's official CLI for Claude." in texts


def test_translate_system_message_strips_billing_header_for_bedrock():
    from litellm.llms.bedrock.claude_platform.transformation import (
        BedrockClaudePlatformConfig,
    )

    config = BedrockClaudePlatformConfig()
    assert config.should_strip_billing_metadata() is True

    result = config.translate_system_message(messages=_system_with_billing_header("real system prompt"))

    texts = [block["text"] for block in result]
    assert all(not t.startswith("x-anthropic-billing-header:") for t in texts)
    assert "real system prompt" in texts


def test_anthropic_messages_request_keeps_billing_header_for_first_party():
    from litellm.types.router import GenericLiteLLMParams

    config = AnthropicMessagesConfig()
    assert config.should_strip_billing_metadata() is False

    optional_params = {
        "max_tokens": 16,
        "system": [
            BILLING_HEADER_BLOCK,
            {"type": "text", "text": "real system prompt"},
        ],
    }
    result = config.transform_anthropic_messages_request(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    texts = [block["text"] for block in result["system"]]
    assert any(t.startswith("x-anthropic-billing-header:") for t in texts)


def test_anthropic_messages_request_strips_billing_header_for_minimax():
    from litellm.llms.minimax.messages.transformation import MinimaxMessagesConfig
    from litellm.types.router import GenericLiteLLMParams

    config = MinimaxMessagesConfig()
    assert config.should_strip_billing_metadata() is True

    optional_params = {
        "max_tokens": 16,
        "system": [
            BILLING_HEADER_BLOCK,
            {"type": "text", "text": "real system prompt"},
        ],
    }
    result = config.transform_anthropic_messages_request(
        model="MiniMax-M2",
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params=optional_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    texts = [block["text"] for block in result.get("system", [])]
    assert all(not t.startswith("x-anthropic-billing-header:") for t in texts)


def test_translate_system_message_strips_billing_header_for_bedrock_invoke():
    from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
        AmazonAnthropicClaudeConfig,
    )

    config = AmazonAnthropicClaudeConfig()
    assert config.should_strip_billing_metadata() is True

    result = config.translate_system_message(messages=_system_with_billing_header("real system prompt"))

    texts = [block["text"] for block in result]
    assert all(not t.startswith("x-anthropic-billing-header:") for t in texts)
    assert "real system prompt" in texts


@pytest.mark.parametrize(
    "module_path, class_name, expected_strip",
    [
        ("litellm.llms.anthropic.chat.transformation", "AnthropicConfig", False),
        (
            "litellm.llms.anthropic.pass_through.messages.transformation",
            "AnthropicMessagesConfig",
            False,
        ),
        (
            "litellm.llms.bedrock.claude_platform.transformation",
            "BedrockClaudePlatformConfig",
            True,
        ),
        (
            "litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation",
            "AmazonAnthropicClaudeConfig",
            True,
        ),
        (
            "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation",
            "VertexAIAnthropicConfig",
            True,
        ),
        (
            "litellm.llms.azure_ai.anthropic.transformation",
            "AzureAnthropicConfig",
            True,
        ),
        ("litellm.llms.minimax.messages.transformation", "MinimaxMessagesConfig", True),
        (
            "litellm.llms.azure_ai.anthropic.messages_transformation",
            "AzureAnthropicMessagesConfig",
            True,
        ),
        (
            "litellm.llms.deepseek.messages.transformation",
            "DeepSeekAnthropicMessagesConfig",
            True,
        ),
        (
            "litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation",
            "VertexAIPartnerModelsAnthropicMessagesConfig",
            True,
        ),
    ],
)
def test_should_strip_billing_metadata_by_provider(module_path, class_name, expected_strip):
    import importlib

    config_cls = getattr(importlib.import_module(module_path), class_name)
    assert config_cls().should_strip_billing_metadata() is expected_strip


def test_namespace_tool_flat_nested_tools_are_extracted():
    """Codex sends nested tools in flat format {type, name, description, parameters} with no 'function' wrapper.
    These must be normalized and mapped without raising KeyError: 'function'."""
    config = AnthropicConfig()
    tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "tools": [
                {
                    "type": "function",
                    "name": "close_agent",
                    "description": "Close an agent.",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                        "additionalProperties": False,
                    },
                },
            ],
        }
    ]
    anthropic_tools, _ = config.map_tools(tools)
    assert len(anthropic_tools) == 1
    assert anthropic_tools[0]["name"] == "close_agent"


def test_namespace_tool_nested_tools_are_extracted():
    """Codex sends type='namespace' wrapping nested tools in Anthropic format.
    The namespace container must be dropped and its nested tools extracted individually.
    """
    config = AnthropicConfig()
    tools = [
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Tools for spawning and managing sub-agents.",
            "tools": [
                {
                    "name": "close_agent",
                    "type": "custom",
                    "description": "Close an agent.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    },
                },
                {
                    "name": "resume_agent",
                    "type": "custom",
                    "description": "Resume a closed agent.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
            ],
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Run a command.",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
    ]
    anthropic_tools, mcp_servers = config.map_tools(tools)
    names = [t["name"] for t in anthropic_tools]
    assert "close_agent" in names
    assert "resume_agent" in names
    assert "exec_command" in names
    assert "multi_agent_v1" not in names
    assert len(anthropic_tools) == 3
    assert mcp_servers == []


def test_client_metadata_stripped_from_anthropic_request():
    """client_metadata passed by codex must not reach the Anthropic (or Vertex Anthropic) payload."""
    config = AnthropicConfig()
    result = config.transform_request(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"max_tokens": 10, "client_metadata": {"originator": "codex"}},
        litellm_params={},
        headers={},
    )
    assert "client_metadata" not in result


@pytest.mark.parametrize(
    "model",
    ["claude-fable-5", "claude-opus-4-7", "claude-opus-4-8-20260120"],
)
def test_sampling_params_dropped_for_models_that_removed_them(model):
    """Fable 5 / Opus 4.7 / 4.8 reject temperature != 1 and any top_p with a
    400; with drop_params set they must be dropped, not forwarded (#30064)."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"temperature": 0.5, "top_p": 0.9},
        optional_params={},
        model=model,
        drop_params=True,
    )

    assert "temperature" not in result
    assert "top_p" not in result


@pytest.mark.parametrize("params", [{"temperature": 0.5}, {"top_p": 0.9}, {"top_p": 1}])
def test_sampling_params_raise_clean_error_without_drop_params(params, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="claude-fable-5",
            drop_params=False,
        )


def test_temperature_1_forwarded_on_models_that_removed_sampling_params():
    """temperature=1 (the API default) is still accepted and must pass through."""
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"temperature": 1},
        optional_params={},
        model="claude-fable-5",
        drop_params=False,
    )

    assert result["temperature"] == 1


@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6"])
def test_sampling_params_forwarded_on_models_that_accept_them(model):
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"temperature": 0.5, "top_p": 0.9},
        optional_params={},
        model=model,
        drop_params=True,
    )

    assert result["temperature"] == 0.5
    assert result["top_p"] == 0.9


def test_sampling_param_gating_driven_by_model_map_flag(monkeypatch):
    """The drop/raise decision must come from ``supports_sampling_params`` in
    the model map, not just name matching: a flagged entry gates a model whose
    name says nothing, and an explicit ``true`` overrides the name fallback."""
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_sampling_params": False})
    monkeypatch.setitem(litellm.model_cost, "claude-fable-5-test", {"supports_sampling_params": True})
    config = AnthropicConfig()

    flagged_off = config.map_openai_params(
        non_default_params={"top_p": 0.9},
        optional_params={},
        model="claude-zeta-9",
        drop_params=True,
    )
    assert "top_p" not in flagged_off

    flagged_on = config.map_openai_params(
        non_default_params={"top_p": 0.9},
        optional_params={},
        model="claude-fable-5-test",
        drop_params=True,
    )
    assert flagged_on["top_p"] == 0.9


def test_top_k_dropped_at_transform_for_models_that_removed_it():
    """``top_k`` is a provider-specific kwarg that bypasses
    ``map_openai_params``, so it must be stripped at the transform_request
    boundary shared by the direct, invoke, Vertex, and Azure paths (#30064)."""
    config = AnthropicConfig()

    result = config.transform_request(
        model="claude-fable-5",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"max_tokens": 10, "top_k": 40},
        litellm_params={"drop_params": True},
        headers={},
    )

    assert "top_k" not in result


def test_top_k_raises_at_transform_without_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="drop_params"):
        config.transform_request(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={"max_tokens": 10, "top_k": 40},
            litellm_params={},
            headers={},
        )


def test_top_k_forwarded_at_transform_on_models_that_accept_it():
    config = AnthropicConfig()

    result = config.transform_request(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={"max_tokens": 10, "top_k": 40},
        litellm_params={"drop_params": True},
        headers={},
    )

    assert result["top_k"] == 40


def test_is_anthropic_usage_object_distinguishes_chat_usage():
    """Chat-shaped Usage mirrors cache_read_input_tokens alongside prompt_tokens that already
    include the cache tokens, so treating it as Anthropic usage would re-add them and
    double-count the prompt. Only the Anthropic shape, where input_tokens excludes cache
    tokens, may take the Anthropic mapping."""
    assert AnthropicConfig.is_anthropic_usage_object(
        {"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 4014}
    )
    assert AnthropicConfig.is_anthropic_usage_object(
        {"input_tokens": 3, "output_tokens": 5, "cache_creation_input_tokens": 10}
    )
    assert not AnthropicConfig.is_anthropic_usage_object(
        Usage(
            prompt_tokens=4017,
            completion_tokens=5,
            total_tokens=4022,
            cache_read_input_tokens=4014,
        ).model_dump()
    )
    assert not AnthropicConfig.is_anthropic_usage_object({"input_tokens": 3, "output_tokens": 5})


def test_is_anthropic_usage_object_rejects_responses_api_usage():
    """completion_cost checks the Anthropic shape before the Responses API shape, so a
    Responses API usage payload, whose cache reads live in nested input_tokens_details,
    must never match; matching would route it past the converter that reads the nested
    field and its cache reads would be billed at the full input rate."""
    assert not AnthropicConfig.is_anthropic_usage_object(
        {
            "input_tokens": 4017,
            "output_tokens": 5,
            "total_tokens": 4022,
            "input_tokens_details": {"cached_tokens": 4014},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    )


@pytest.mark.parametrize(
    "model, expected_dropped",
    [
        # always-on-thinking models reject thinking.type=disabled with a 400
        ("claude-fable-5", True),
        ("claude-fable-5-1", True),
        ("claude-mythos-5", True),
        # unmapped future family member -> claude-always-on-thinking fallback rule
        ("claude-fable-6-1", True),
        # adaptive-capable models that ACCEPT disabled must keep it verbatim
        ("claude-opus-5", False),
        ("claude-sonnet-5", False),
        ("claude-opus-4-8", False),
        # legacy models keep it verbatim
        ("claude-sonnet-4-5-20250929", False),
    ],
)
def test_disabled_thinking_omitted_only_for_always_on_models(local_model_cost_map, model, expected_dropped):
    """``thinking={"type": "disabled"}`` is omitted for always-on-thinking models
    (Fable/Mythos, which 400 on it: the API remedy is to omit the param) and is
    forwarded verbatim for every model that accepts it."""
    config = AnthropicConfig()

    request = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 64, "thinking": {"type": "disabled"}},
        litellm_params={},
        headers={},
    )

    if expected_dropped:
        assert "thinking" not in request
    else:
        assert request["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "required"}, {"type": "function", "function": {"name": "get_weather"}}],
)
def test_forced_tool_choice_raises_clean_error_on_fable_5_1_without_drop_params(
    local_model_cost_map, tool_choice, monkeypatch
):
    """Fable 5.1 400s on tool_choice type any/tool (thinking is always on and a
    forced call would skip it); without drop_params the caller gets a clean
    client-side 400 that explains the workaround, not a provider error."""
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="forced tool use"):
        config.map_openai_params(
            non_default_params={"tool_choice": tool_choice},
            optional_params={},
            model="claude-fable-5-1",
            drop_params=False,
        )


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "required"}, {"type": "function", "function": {"name": "get_weather"}}],
)
def test_forced_tool_choice_downgraded_to_auto_on_fable_5_1_with_drop_params(local_model_cost_map, tool_choice):
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"tool_choice": tool_choice},
        optional_params={},
        model="claude-fable-5-1",
        drop_params=True,
    )

    assert result["tool_choice"] == {"type": "auto"}


def test_forced_tool_choice_downgrade_keeps_parallel_tool_calls_flag(local_model_cost_map):
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"tool_choice": "required", "parallel_tool_calls": False},
        optional_params={},
        model="claude-fable-5-1",
        drop_params=True,
    )

    assert result["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


@pytest.mark.parametrize("tool_choice, expected_type", [("auto", "auto"), ("none", "none")])
def test_unforced_tool_choice_forwarded_on_fable_5_1(local_model_cost_map, tool_choice, expected_type, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"tool_choice": tool_choice},
        optional_params={},
        model="claude-fable-5-1",
        drop_params=False,
    )

    assert result["tool_choice"]["type"] == expected_type


@pytest.mark.parametrize("model", ["claude-fable-5", "claude-opus-5", "claude-sonnet-5"])
def test_forced_tool_choice_forwarded_on_models_that_support_it(local_model_cost_map, model, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"tool_choice": "required"},
        optional_params={},
        model=model,
        drop_params=True,
    )

    assert result["tool_choice"] == {"type": "any"}


def test_forced_tool_choice_gating_driven_by_model_map_flag(local_model_cost_map, monkeypatch):
    """The gate must read ``supports_forced_tool_use`` from the model map, not
    the model name: a flagged entry gates a model whose name says nothing."""
    monkeypatch.setitem(litellm.model_cost, "claude-zeta-9", {"supports_forced_tool_use": False})
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={"tool_choice": "required"},
        optional_params={},
        model="claude-zeta-9",
        drop_params=True,
    )

    assert result["tool_choice"] == {"type": "auto"}


def test_anthropic_drop_params_keeps_format_only_output_config(monkeypatch):
    """``drop_params=True`` must not consume ``output_config.format``: the drop
    gate is an effort gate and ``format`` is a structured-output field."""
    monkeypatch.setattr(litellm, "drop_params", True)
    config = AnthropicConfig()
    schema_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"z": {"type": "integer"}}},
    }

    result = config.transform_request(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"output_config": {"format": schema_format}},
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"format": schema_format}


def test_anthropic_drop_params_reduces_mixed_output_config_to_format(monkeypatch):
    """``drop_params=True`` drops the effort key on unsupported models but keeps
    ``format`` so structured outputs still reach the provider."""
    monkeypatch.setattr(litellm, "drop_params", True)
    config = AnthropicConfig()
    schema_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"z": {"type": "integer"}}},
    }

    result = config.transform_request(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"output_config": {"effort": "low", "format": schema_format}},
        litellm_params={},
        headers={},
    )

    assert result.get("output_config") == {"format": schema_format}


def test_response_format_tool_path_skips_forced_tool_choice_when_unsupported(local_model_cost_map, monkeypatch):
    """Backstop: on the tool-based structured-output path, a model flagged
    ``supports_forced_tool_use: false`` must not get the forced response-format
    tool_choice the provider would 400 on."""
    monkeypatch.setitem(
        litellm.model_cost,
        "claude-test-no-forced-tools",
        {"litellm_provider": "anthropic", "mode": "chat", "supports_forced_tool_use": False},
    )
    config = AnthropicConfig()

    result = config.map_openai_params(
        non_default_params={
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
                },
            }
        },
        optional_params={},
        model="claude-test-no-forced-tools",
        drop_params=False,
    )

    assert "tools" in result
    assert "tool_choice" not in result


def _eager_chat_function(**extra: object) -> dict[str, object]:
    return {
        "name": "write_file",
        "description": "Write a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        **extra,
    }


def _eager_chat_tool(**extra: object) -> dict[str, object]:
    return {"type": "function", "function": _eager_chat_function(), **extra}


@pytest.mark.parametrize("flag", [True, False])
def test_eager_input_streaming_passed_through_from_tool_top_level(flag):
    mapped_tool, _ = AnthropicConfig().map_tool_helper(_eager_chat_tool(eager_input_streaming=flag))

    assert mapped_tool == {
        "name": "write_file",
        "description": "Write a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        "type": "custom",
        "eager_input_streaming": flag,
    }


def test_eager_input_streaming_passed_through_from_function():
    mapped_tool, _ = AnthropicConfig().map_tool_helper(
        {"type": "function", "function": _eager_chat_function(eager_input_streaming=True)}
    )

    assert mapped_tool["eager_input_streaming"] is True
    assert "eager_input_streaming" not in mapped_tool["input_schema"]


def test_eager_input_streaming_absent_stays_absent():
    mapped_tool, _ = AnthropicConfig().map_tool_helper(_eager_chat_tool())

    assert "eager_input_streaming" not in mapped_tool


def test_eager_input_streaming_rejects_non_boolean():
    with pytest.raises(litellm.BadRequestError, match="eager_input_streaming must be a boolean"):
        AnthropicConfig().map_tool_helper(_eager_chat_tool(eager_input_streaming="true"))


def test_eager_input_streaming_not_set_on_computer_use_tool():
    computer_tool = {
        "type": "computer_20250124",
        "function": {"name": "computer", "parameters": {"display_width_px": 1024, "display_height_px": 768}},
        "eager_input_streaming": True,
    }

    mapped_tool, _ = AnthropicConfig().map_tool_helper(computer_tool)

    assert mapped_tool["type"] == "computer_20250124"
    assert "eager_input_streaming" not in mapped_tool


def test_eager_input_streaming_reaches_anthropic_request_tools():
    result = AnthropicConfig().map_openai_params(
        non_default_params={"tools": [_eager_chat_tool(eager_input_streaming=True)], "stream": True},
        optional_params={},
        model="claude-sonnet-5",
        drop_params=False,
    )

    assert result["tools"][0]["eager_input_streaming"] is True
    assert result["tools"][0]["name"] == "write_file"


# ---------------------------------------------------------------------------
# Mid-conversation ``role: "system"`` on the chat completions path.
#
# Hoisting a later system message into the top-level ``system`` block rewrites
# the cached prefix and re-bills the whole conversation at cache-write pricing
# on every reminder (#36559). The chat path must keep the prefix stable: leading
# system messages still become the ``system`` param, later ones stay in place as
# ``role: "system"`` on models flagged ``supports_mid_conversation_system`` and
# become a user turn on models that reject the role inside ``messages``.
# ---------------------------------------------------------------------------

UNFLAGGED_CLAUDE = "claude-opus-4-7"
FLAGGED_CLAUDE = "claude-opus-4-8"
CONVERTED_SYSTEM_NOTE = (
    "Operator note (not from the user): the following was originally a mid-conversation system-role reminder."
)
REMINDER_TEXT = "<system-reminder>Answer with exactly one word.</system-reminder>"
CACHED_SYSTEM_BLOCK = {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}


def _chat_request(config: AnthropicConfig, model: str, messages: list[dict]) -> dict:
    return config.transform_request(
        model=model,
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )


def _reminder_conversation() -> list[dict]:
    """The shape Claude Code sends mid-session: cached system prompt, turns, a
    reminder right after a user turn, an assistant turn, a fresh user turn."""
    return [
        {"role": "system", "content": [dict(CACHED_SYSTEM_BLOCK)]},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "system", "content": REMINDER_TEXT},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]


def _texts(message: dict) -> list[str]:
    return [block["text"] for block in message["content"] if block.get("type") == "text"]


def test_chat_unflagged_model_converts_mid_conversation_system_to_user_turn(local_model_cost_map):
    result = _chat_request(AnthropicConfig(), UNFLAGGED_CLAUDE, _reminder_conversation())

    assert result["system"] == [CACHED_SYSTEM_BLOCK]
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert _texts(result["messages"][2]) == ["Second question", CONVERTED_SYSTEM_NOTE, REMINDER_TEXT]


def test_chat_flagged_model_keeps_mid_conversation_system_in_messages(local_model_cost_map):
    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, _reminder_conversation())

    assert result["system"] == [CACHED_SYSTEM_BLOCK]
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user", "system", "assistant", "user"]
    assert result["messages"][3] == {"role": "system", "content": [{"type": "text", "text": REMINDER_TEXT}]}


def test_chat_flagged_model_keeps_cache_control_on_mid_conversation_system(local_model_cost_map):
    messages = _reminder_conversation()
    messages[4] = {
        "role": "system",
        "content": [{"type": "text", "text": REMINDER_TEXT, "cache_control": {"type": "ephemeral"}}],
    }

    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, messages)

    assert result["messages"][3]["content"] == [
        {"type": "text", "text": REMINDER_TEXT, "cache_control": {"type": "ephemeral"}}
    ]


def test_chat_flagged_model_moves_system_after_the_user_turn_it_precedes(local_model_cost_map):
    """Anthropic only accepts role=system directly after a user turn; an
    OpenAI-shaped client that puts the reminder before its next question gets a
    placement-valid request without the reminder leaving ``messages``."""
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": REMINDER_TEXT},
        {"role": "user", "content": "Second question"},
    ]

    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, messages)

    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user", "system"]
    assert _texts(result["messages"][2]) == ["Second question"]
    assert _texts(result["messages"][3]) == [REMINDER_TEXT]


def test_chat_flagged_model_converts_system_with_no_following_user_turn(local_model_cost_map):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "system", "content": REMINDER_TEXT},
    ]

    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, messages)

    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]
    assert _texts(result["messages"][2]) == [CONVERTED_SYSTEM_NOTE, REMINDER_TEXT]


@pytest.mark.parametrize(
    "empty_content",
    [[], None, [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]],
    ids=["empty-list", "none", "unsupported-part-only"],
)
def test_chat_flagged_model_converts_a_system_behind_a_user_turn_that_sends_nothing(
    local_model_cost_map, empty_content
):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": empty_content},
        {"role": "system", "content": REMINDER_TEXT},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]

    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, messages)

    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]
    assert _texts(result["messages"][0]) == [CONVERTED_SYSTEM_NOTE, REMINDER_TEXT]


USER_PART_BY_TYPE = {
    "text": {"type": "text", "text": "hello"},
    "image_url": {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    "document": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hello"}},
    "file": {"type": "file", "file": {"file_data": "data:text/plain;base64,aGVsbG8=", "filename": "hello.txt"}},
    "input_audio": {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
    "video_url": {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
}


@pytest.mark.parametrize("part_type", sorted(USER_PART_BY_TYPE))
def test_chat_flagged_model_anchors_a_system_on_a_user_turn_exactly_when_that_turn_reaches_the_wire(
    local_model_cost_map, part_type
):
    part_only_turn = {"role": "user", "content": [USER_PART_BY_TYPE[part_type]]}
    tail = [{"role": "assistant", "content": "First answer"}, {"role": "user", "content": "Second question"}]

    without_reminder = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, [part_only_turn, *tail])
    with_reminder = _chat_request(
        AnthropicConfig(), FLAGGED_CLAUDE, [part_only_turn, {"role": "system", "content": REMINDER_TEXT}, *tail]
    )

    turn_reaches_wire = [m["role"] for m in without_reminder["messages"]] == ["user", "assistant", "user"]
    expected_roles = ["user", "system", "assistant", "user"] if turn_reaches_wire else ["user", "assistant", "user"]
    assert [m["role"] for m in with_reminder["messages"]] == expected_roles


ASSISTANT_TURN_BY_SHAPE = {
    "text": {"role": "assistant", "content": "First answer"},
    "tool-calls": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    },
    "signed-thinking-part": {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
    },
    "empty-string": {"role": "assistant", "content": ""},
    "whitespace-string": {"role": "assistant", "content": "   "},
    "empty-text-part": {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    "none": {"role": "assistant", "content": None},
    "empty-list": {"role": "assistant", "content": []},
    "unsigned-thinking-part": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}]},
    "signed-thinking-block": {
        "role": "assistant",
        "content": None,
        "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
    },
    "redacted-thinking-block": {
        "role": "assistant",
        "content": None,
        "thinking_blocks": [{"type": "redacted_thinking", "data": "x"}],
    },
    "encrypted-thinking-part": {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hm", "signature": encrypted_reasoning_signature("abc")}],
    },
    "encrypted-redacted-thinking-block": {
        "role": "assistant",
        "content": None,
        "thinking_blocks": [{"type": "redacted_thinking", "data": encrypted_reasoning_signature("abc")}],
    },
    "unsigned-inline-part-hides-signed-block": {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hm"}],
        "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
    },
    "inline-redacted-part-hides-redacted-block": {
        "role": "assistant",
        "content": [{"type": "redacted_thinking", "data": "x"}],
        "thinking_blocks": [{"type": "redacted_thinking", "data": "x"}],
    },
    "text-part-beside-signed-block": {
        "role": "assistant",
        "content": [{"type": "text", "text": "First answer"}],
        "thinking_blocks": [{"type": "thinking", "thinking": "hm", "signature": "s"}],
    },
}


@pytest.mark.parametrize("shape", sorted(ASSISTANT_TURN_BY_SHAPE))
def test_chat_flagged_model_keeps_a_system_exactly_when_the_assistant_turn_after_it_reaches_the_wire(
    local_model_cost_map, shape
):
    first_turn = {"role": "user", "content": "First question"}
    tail = [ASSISTANT_TURN_BY_SHAPE[shape], {"role": "user", "content": "Second question"}]

    without_reminder = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, [first_turn, *tail])
    with_reminder = _chat_request(
        AnthropicConfig(), FLAGGED_CLAUDE, [first_turn, {"role": "system", "content": REMINDER_TEXT}, *tail]
    )

    turn_reaches_wire = [m["role"] for m in without_reminder["messages"]] == ["user", "assistant", "user"]
    expected_roles = ["user", "system", "assistant", "user"] if turn_reaches_wire else ["user", "user"]
    assert [m["role"] for m in with_reminder["messages"]] == expected_roles
    if not turn_reaches_wire:
        assert _texts(with_reminder["messages"][0]) == ["First question", CONVERTED_SYSTEM_NOTE, REMINDER_TEXT]


def test_chat_flagged_model_merges_adjacent_system_messages(local_model_cost_map):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "First question"},
        {"role": "system", "content": "Reminder one."},
        {"role": "system", "content": "Reminder two."},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]

    result = _chat_request(AnthropicConfig(), FLAGGED_CLAUDE, messages)

    assert [m["role"] for m in result["messages"]] == ["user", "system", "assistant", "user"]
    assert _texts(result["messages"][1]) == ["Reminder one.", "Reminder two."]


def test_chat_unflagged_model_keeps_tool_result_first_when_system_precedes_tool_message(local_model_cost_map):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "system", "content": REMINDER_TEXT},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "user", "content": "Thanks"},
    ]

    result = _chat_request(AnthropicConfig(), UNFLAGGED_CLAUDE, messages)

    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]
    blocks = result["messages"][2]["content"]
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "call_1"
    assert _texts(result["messages"][2]) == [CONVERTED_SYSTEM_NOTE, REMINDER_TEXT, "Thanks"]


def test_chat_transform_request_does_not_mutate_caller_messages(local_model_cost_map):
    messages = _reminder_conversation()
    snapshot = copy.deepcopy(messages)

    _chat_request(AnthropicConfig(), UNFLAGGED_CLAUDE, messages)

    assert messages == snapshot


_CHAT_CONFIGS = [
    pytest.param(AnthropicConfig, UNFLAGGED_CLAUDE, id="anthropic-unflagged"),
    pytest.param(AnthropicConfig, FLAGGED_CLAUDE, id="anthropic-flagged"),
    pytest.param(VertexAIAnthropicConfig, UNFLAGGED_CLAUDE, id="vertex_ai-unflagged"),
    pytest.param(VertexAIAnthropicConfig, FLAGGED_CLAUDE, id="vertex_ai-flagged"),
    pytest.param(AzureAnthropicConfig, UNFLAGGED_CLAUDE, id="azure_ai-unflagged"),
    pytest.param(AzureAnthropicConfig, FLAGGED_CLAUDE, id="azure_ai-flagged"),
    pytest.param(AmazonAnthropicClaudeConfig, "invoke/us.anthropic.claude-opus-4-7", id="bedrock_invoke-unflagged"),
    pytest.param(AmazonAnthropicClaudeConfig, "invoke/us.anthropic.claude-opus-4-8", id="bedrock_invoke-flagged"),
]


@pytest.mark.parametrize("config_cls, model", _CHAT_CONFIGS)
def test_chat_mid_conversation_system_keeps_earlier_turns_a_prefix_of_the_next_request(
    local_model_cost_map, config_cls, model
):
    """The provider-side prompt cache is a prefix match over ``system`` +
    ``messages``. Whatever the policy for the reminder, turn N's request must
    stay a prefix of turn N+1's request or the whole conversation is re-billed.

    Anthropic combines consecutive same-role messages into one turn, so the
    cache-relevant sequence is ``(role, content block)`` pairs, not the message
    list: a reminder that joins the preceding user turn still extends the prefix.
    """
    conversation = _reminder_conversation()

    earlier = _chat_request(config_cls(), model, copy.deepcopy(conversation[:4]))
    later = _chat_request(config_cls(), model, copy.deepcopy(conversation))

    assert later["system"] == earlier["system"]
    earlier_blocks = _role_block_pairs(earlier["messages"])
    later_blocks = _role_block_pairs(later["messages"])
    assert later_blocks[: len(earlier_blocks)] == earlier_blocks
    assert len(later_blocks) > len(earlier_blocks)


def _role_block_pairs(messages: list[dict]) -> list[tuple[str, object]]:
    return [
        (message["role"], block)
        for message in messages
        for block in (message["content"] if isinstance(message["content"], list) else [message["content"]])
    ]


def _thinking_reply(text: str) -> dict:
    return {
        "role": "assistant",
        "content": text,
        "thinking_blocks": [{"type": "thinking", "thinking": "Working it out.", "signature": f"sig-{text}"}],
    }


def _preserved_thinking_turns(reminder_after_user: bool) -> tuple[list[dict], list[dict], list[dict]]:
    turn_n = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "First question"}]
    reminder = {"role": "system", "content": REMINDER_TEXT}
    second_question = {"role": "user", "content": "Second question"}
    second_turn = [second_question, reminder] if reminder_after_user else [reminder, second_question]
    turn_n_plus_one = [*turn_n, _thinking_reply("First answer"), *second_turn]
    turn_n_plus_two = [
        *turn_n_plus_one,
        _thinking_reply("Second answer"),
        {"role": "user", "content": "Third question"},
    ]
    return turn_n, turn_n_plus_one, turn_n_plus_two


def _replayed_prefix(request: dict, message_count: int) -> str:
    replayed = {
        "system": request.get("system"),
        "tools": request.get("tools"),
        "messages": request["messages"][:message_count],
    }
    return json.dumps(replayed, sort_keys=True)


def _assert_prefix_stable(requests: list[dict]) -> None:
    for earlier, later in zip(requests, requests[1:]):
        count = len(earlier["messages"])
        assert _replayed_prefix(later, count) == _replayed_prefix(earlier, count)


@pytest.mark.parametrize("reminder_after_user", [True, False])
def test_chat_flagged_model_replays_a_byte_identical_prefix_around_a_mid_conversation_reminder(
    local_model_cost_map, reminder_after_user
):
    """Preserved thinking binds each signed block to the request prefix it was created
    under (``system``, ``tools`` and the earlier messages), so turn N's transformed
    request must be a byte-identical prefix of turn N+1's or the block is dropped."""
    requests = [
        AnthropicConfig().transform_request(
            model="claude-fable-5-1", messages=copy.deepcopy(turn), optional_params={}, litellm_params={}, headers={}
        )
        for turn in _preserved_thinking_turns(reminder_after_user)
    ]

    _assert_prefix_stable(requests)
    assert [m["role"] for m in requests[1]["messages"]] == ["user", "assistant", "user", "system"]
    assert [m["role"] for m in requests[2]["messages"]] == ["user", "assistant", "user", "system", "assistant", "user"]


def test_chat_dummy_tool_result_for_an_orphaned_tool_call_replays_a_byte_identical_prefix(
    local_model_cost_map, monkeypatch
):
    monkeypatch.setattr(litellm, "modify_params", True)
    tools = [
        {"name": "lookup", "description": "Look something up", "input_schema": {"type": "object", "properties": {}}}
    ]
    orphaned_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    }
    turn_n = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "First question"},
        orphaned_call,
    ]
    turn_n_plus_one = [*turn_n, _thinking_reply("First answer"), {"role": "user", "content": "Second question"}]
    requests = [
        AnthropicConfig().transform_request(
            model="claude-fable-5-1",
            messages=copy.deepcopy(turn),
            optional_params={"tools": copy.deepcopy(tools)},
            litellm_params={},
            headers={},
        )
        for turn in (turn_n, turn_n_plus_one)
    ]

    _assert_prefix_stable(requests)
    assert [m["role"] for m in requests[0]["messages"]] == ["user", "assistant", "user"]
    assert requests[0]["messages"][2]["content"][0]["type"] == "tool_result"


def test_transform_parsed_response_preserves_existing_hidden_params():
    from litellm.types.utils import ModelResponse

    config = AnthropicConfig()
    raw_response = MagicMock()
    raw_response.headers = {"request-id": "req_vertex"}
    raw_response.status_code = 200
    completion_response = {
        "id": "msg_vertex",
        "model": "claude-sonnet-4-5@20250929",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "text", "text": "Hello"}],
    }
    model_response = ModelResponse()
    model_response._hidden_params = {
        "custom_llm_provider": "vertex_ai",
        "region_name": "us-east5",
    }

    result = config.transform_parsed_response(
        completion_response=completion_response,
        raw_response=raw_response,
        model_response=model_response,
    )

    assert result._hidden_params["custom_llm_provider"] == "vertex_ai"
    assert result._hidden_params["region_name"] == "us-east5"
    assert result.choices[0].message.content == "Hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_off_message_logging", [True, False])
async def test_anthropic_completion_respects_message_redaction_in_datadog_v1(
    monkeypatch: pytest.MonkeyPatch, turn_off_message_logging: bool
) -> None:
    monkeypatch.setattr(litellm, "datadog_use_v1", True)
    private_output: Final = "private-claude-output-for-redaction-regression"
    logged_requests: Final[Queue[httpx.Request]] = Queue()

    def provider_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_redaction",
                "type": "message",
                "role": "assistant",
                "model": "claude-fable-5-1",
                "content": [{"type": "text", "text": private_output}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    def datadog_response(request: httpx.Request) -> httpx.Response:
        logged_requests.put(request)
        return httpx.Response(202)

    logger: Final = DataDogLogger(dd_api_key="test-key", dd_site="datadoghq.com", allow_env_credentials=False)
    provider_client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(provider_response)))
    datadog_client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(datadog_response)))
    logger.sync_client = datadog_client
    try:
        response: Final = litellm.completion(
            model="anthropic/claude-fable-5-1",
            messages=[{"role": "user", "content": "Hello"}],
            api_key="test-key",
            client=provider_client,
            success_callback=[logger],
            turn_off_message_logging=turn_off_message_logging,
        )
        request: Final = logged_requests.get(timeout=10)
    finally:
        provider_client.close()
        datadog_client.close()

    assert response.choices[0].message.content == private_output
    assert (private_output in request.content.decode()) is (not turn_off_message_logging)
    payload: Final = json.loads(json.loads(request.content)["message"])
    assert payload["usage"]["prompt_tokens"] == 10
    assert payload["usage"]["completion_tokens"] == 5


anthropic_chunk_list: Final = [
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


def _make_transform_request(optional_params: dict, litellm_params: dict) -> dict:

    return AnthropicConfig().transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers={},
    )


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
    input_headers = Headers({"unrelated-header-1": "value1", "unrelated-header-2": "value2"})

    expected_output = {
        "llm_provider-unrelated-header-1": "value1",
        "llm_provider-unrelated-header-2": "value2",
    }

    result = process_anthropic_headers(input_headers)
    assert result == expected_output, "Unexpected output for non-matching headers"


@pytest.mark.parametrize(
    "computer_tool_used, prompt_caching_set, expected_beta_header",
    [
        (True, False, True),
        (False, True, False),
        (True, True, True),
        (False, False, False),
    ],
)
def test_anthropic_beta_header(computer_tool_used, prompt_caching_set, expected_beta_header):
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

    tool, _ = AnthropicConfig().map_tool_helper(tool=tool)

    assert tool["cache_control"] == {"type": "ephemeral"}


def test_create_json_tool_call_for_response_format():
    """
    tests using response_format=json with anthropic

    A tool call to anthropic is made when response_format=json is used.

    """

    config = AnthropicConfig()

    tool = config._create_json_tool_call_for_response_format()
    assert tool["name"] == "json_tool_call"
    _input_schema = tool.get("input_schema")
    assert _input_schema is not None
    assert _input_schema.get("type") == "object"
    assert _input_schema.get("additionalProperties") is True
    assert _input_schema.get("properties") == {}

    custom_schema = {"name": {"type": "string"}, "age": {"type": "integer"}}
    tool = config._create_json_tool_call_for_response_format(json_schema=custom_schema)
    assert tool["name"] == "json_tool_call"
    _input_schema = tool.get("input_schema")
    assert _input_schema is not None
    assert _input_schema.get("type") == "object"
    assert _input_schema.get("name") == custom_schema["name"]
    assert _input_schema.get("age") == custom_schema["age"]
    assert "additionalProperties" not in _input_schema


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

    message = AnthropicConfig.convert_tool_response_to_message(tool_calls=tool_calls)

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

    message = AnthropicConfig.convert_tool_response_to_message(tool_calls=tool_calls)

    assert message is not None
    assert message.content == '{"name": "John", "age": 30}'


def test_convert_tool_response_to_message_invalid_json():
    """Test converting a tool response with invalid JSON"""
    tool_calls = [
        ChatCompletionToolCallChunk(
            id="test_id",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(name="json_tool_call", arguments="invalid json"),
            index=0,
        )
    ]

    message = AnthropicConfig.convert_tool_response_to_message(tool_calls=tool_calls)

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

    message = AnthropicConfig.convert_tool_response_to_message(tool_calls=tool_calls)

    assert message is None


def test_anthropic_tool_with_image():
    import json

    from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory

    b64_data = "iVBORw0KGgoAAAANSUhEu6U3//C9t/fKv5wDgpP1r5796XwC4zyH1D565bHGDqbY85AMb0nIQe+u3J390Xbtb9XgXxcK0/aqRXpdYcwgARbCN03FJk"
    image_url = f"data:image/png;base64,{b64_data}"
    messages = [
        {
            "content": [
                {"type": "text", "text": "go to github ryanhoangt by browser"},
                {
                    "type": "text",
                    "text": '<extra_info>\nThe following information has been included based on a keyword match for "github". It may or may not be relevant to the user\'s request.\n\nYou have access to an environment variable, `GITHUB_TOKEN`, which allows you to interact with\nthe GitHub API.\n\nYou can use `curl` with the `GITHUB_TOKEN` to interact with GitHub\'s API.\nALWAYS use the GitHub API for operations instead of a web browser.\n\nHere are some instructions for pushing, but ONLY do this if the user asks you to:\n* NEVER push directly to the `main` or `master` branch\n* Git config (username and email) is pre-set. Do not modify.\n* You may already be on a branch called `openhands-workspace`. Create a new branch with a better name before pushing.\n* Use the GitHub API to create a pull request, if you haven\'t already\n* Use the main branch as the base branch, unless the user requests otherwise\n* After opening or updating a pull request, send the user a short message with a link to the pull request.\n* Do all of the above in as few steps as possible. E.g. you could open a PR with one step by running the following bash commands:\n```bash\ngit remote -v && git branch \x23 to find the current org, repo and branch\ngit checkout -b create-widget && git add . && git commit -m "Create widget" && git push -u origin create-widget\ncurl -X POST "https://api.github.com/repos/$ORG_NAME/$REPO_NAME/pulls" \\\n    -H "Authorization: Bearer $GITHUB_TOKEN" \\\n    -d \'{"title":"Create widget","head":"create-widget","base":"openhands-workspace"}\'\n```\n</extra_info>',
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
        model="claude-sonnet-4-5-20250929",
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
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert "Question" in json.dumps(mapped_params)


def test_anthropic_map_openai_params_tools_with_defs():
    args = {
        "non_default_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "create_user",
                        "description": "Create a user from provided profile data.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "user": {"$ref": "#/$defs/User"},
                            },
                            "required": ["user"],
                            "$defs": {
                                "User": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"},
                                    },
                                    "required": ["name", "email"],
                                }
                            },
                        },
                    },
                }
            ]
        }
    }

    mapped_params = litellm.AnthropicConfig().map_openai_params(
        non_default_params=args["non_default_params"],
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    tool = mapped_params["tools"][0]
    assert tool["input_schema"]["properties"]["user"]["$ref"] == "#/$defs/User"
    assert tool["input_schema"]["$defs"]["User"]["properties"]["name"]["type"] == "string"


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
def test_anthropic_json_mode_and_tool_call_response(json_mode, tool_calls, expect_null_response):
    result, _, _ = litellm.AnthropicConfig()._resolve_json_mode_non_streaming(
        json_mode=json_mode,
        tool_calls=tool_calls,
    )

    assert result is None if expect_null_response else result is not None, (
        f"Expected result to be {None if expect_null_response else 'not None'}, but got {result}"
    )


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
    result = config.map_stop_sequences(stop_input)
    assert result == expected_output


@pytest.mark.usefixtures("fake_provider_credentials")
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


def test_anthropic_tool_cache_control():
    from litellm.utils import return_raw_request
    from litellm.types.utils import CallTypes
    import json

    tool_content = "Result: 4. " * 1000
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

    vertex_ai_model = "vertex_ai/claude-sonnet-4-5@20250929"
    anthropic_api_model = "claude-sonnet-4-5-20250929"
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

    assert "cache_control" in json.dumps(result["raw_request_body"]["messages"][2]["content"])


def test_anthropic_strict_parameter_passthrough():
    """Test that the strict parameter in tool parameters is passed through to Anthropic input_schema"""
    args = {
        "non_default_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                            "strict": True,
                        },
                    },
                }
            ],
        }
    }

    mapped_params = litellm.AnthropicConfig().map_openai_params(
        non_default_params=args["non_default_params"],
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert "tools" in mapped_params
    assert len(mapped_params["tools"]) == 1
    tool = mapped_params["tools"][0]
    assert "input_schema" in tool
    assert tool["input_schema"]["strict"] is True


def test_anthropic_strict_not_present():
    """Test that the strict parameter in tool parameters is passed through to Anthropic input_schema"""
    args = {
        "non_default_params": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        }
    }

    mapped_params = litellm.AnthropicConfig().map_openai_params(
        non_default_params=args["non_default_params"],
        optional_params={},
        model="claude-sonnet-4-5-20250929",
        drop_params=False,
    )

    assert "tools" in mapped_params
    assert len(mapped_params["tools"]) == 1
    tool = mapped_params["tools"][0]
    assert "input_schema" in tool
    assert "strict" not in tool["input_schema"]


def test_metadata_only_user_id_passes_through():
    """metadata with only user_id is forwarded as-is."""
    data = _make_transform_request(
        optional_params={"metadata": {"user_id": "abc123"}},
        litellm_params={},
    )
    assert data.get("metadata") == {"user_id": "abc123"}


def test_metadata_extra_keys_are_stripped():
    """Extra keys in metadata are removed; only user_id is sent."""
    data = _make_transform_request(
        optional_params={"metadata": {"user_id": "abc123", "extra_key": "val"}},
        litellm_params={},
    )
    assert data.get("metadata") == {"user_id": "abc123"}


def test_metadata_without_user_id_is_dropped():
    """metadata with no user_id is removed entirely."""
    data = _make_transform_request(
        optional_params={"metadata": {"only_other_key": "val"}},
        litellm_params={},
    )
    assert "metadata" not in data


def test_metadata_user_id_from_litellm_params_strips_extras():
    """user_id from litellm_params metadata is extracted; extra keys are not forwarded."""
    data = _make_transform_request(
        optional_params={},
        litellm_params={"metadata": {"user_id": "abc123", "trace_id": "xyz"}},
    )
    assert data.get("metadata") == {"user_id": "abc123"}


def test_metadata_filter_applies_to_vertex_anthropic():
    """VertexAIAnthropicConfig inherits the metadata filter."""
    from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import (
        VertexAIAnthropicConfig,
    )

    data = VertexAIAnthropicConfig().transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"metadata": {"user_id": "u1", "extra": "drop_me"}},
        litellm_params={},
        headers={},
    )
    assert data.get("metadata") == {"user_id": "u1"}


def test_metadata_filter_applies_to_azure_anthropic():
    """AzureAnthropicConfig inherits the metadata filter."""
    from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig

    data = AzureAnthropicConfig().transform_request(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"metadata": {"user_id": "u2", "extra": "drop_me"}},
        litellm_params={},
        headers={},
    )
    assert data.get("metadata") == {"user_id": "u2"}


@pytest.mark.asyncio()
async def test_anthropic_api_prompt_caching_with_content_str():
    system_message = [
        {
            "role": "system",
            "content": "Here is the full text of a complex legal agreement",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    translated_system_message = litellm.AnthropicConfig().translate_system_message(
        messages=system_message
    )

    assert translated_system_message == [
        # System Message
        {
            "type": "text",
            "text": "Here is the full text of a complex legal agreement",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_messages = [
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": "What are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": "What are the key terms and conditions in this agreement?",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(
        messages=user_messages,
        model="claude-3-5-sonnet-20240620",
        llm_provider="anthropic",
    )

    expected_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
                }
            ],
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    assert len(translated_messages) == len(expected_messages)
    for idx, i in enumerate(translated_messages):
        assert (
            i == expected_messages[idx]
        ), "Error on idx={}. Got={}, Expected={}".format(idx, i, expected_messages[idx])


@pytest.fixture
def anthropic_messages():
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 500,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


@pytest.mark.usefixtures("fake_provider_credentials")
def test_is_prompt_caching_enabled(anthropic_messages):
    assert litellm.utils.is_prompt_caching_valid_prompt(
        messages=anthropic_messages,
        tools=None,
        custom_llm_provider="anthropic",
        model="anthropic/claude-sonnet-4-5-20250929",
    )


def test_calculate_usage_sums_compaction_and_message_iterations():
    usage: Final = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 100,
            "output_tokens": 50,
            "iterations": [
                {"iteration": 1, "type": "compaction", "input_tokens": 1000, "output_tokens": 500},
                {"iteration": 2, "type": "message", "input_tokens": 100, "output_tokens": 50},
            ],
        },
        reasoning_content=None,
    )
    assert usage.prompt_tokens == 1100
    assert usage.completion_tokens == 550
    assert usage.total_tokens == 1650
    assert usage.prompt_tokens_details.text_tokens == 1100
    assert usage.iterations is not None
    assert len(usage.iterations) == 2
    assert usage.iterations[0]["type"] == "compaction"


def test_calculate_usage_sums_cache_tokens_across_compaction_iterations():
    usage: Final = AnthropicConfig().calculate_usage(
        usage_object={
            "input_tokens": 100,
            "output_tokens": 50,
            "iterations": [
                {
                    "type": "compaction",
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 50,
                    "cache_read_input_tokens": 17000,
                },
                {
                    "type": "message",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                },
            ],
        },
        reasoning_content=None,
    )
    assert usage.prompt_tokens == 17680
    assert usage.completion_tokens == 250
    assert usage.prompt_tokens_details.cache_creation_tokens == 60
    assert usage.prompt_tokens_details.cached_tokens == 17020


PROMPT_CACHING_MODEL: Final = "claude-sonnet-5-5"
EPHEMERAL: Final = {"type": "ephemeral"}
WEATHER_PARAMETERS: Final = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["location"],
}


async def _send_prompt_caching_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, **params: object
) -> httpx.Request:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": PROMPT_CACHING_MODEL,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        )
    )
    await litellm.acompletion(
        api_key="mock_api_key",
        model=f"anthropic/{PROMPT_CACHING_MODEL}",
        extra_headers={"anthropic-version": "2023-06-01"},
        **params,
    )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["x-api-key"] == "mock_api_key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in request.headers
    return request


async def test_litellm_anthropic_prompt_caching_tools(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    request: Final = await _send_prompt_caching_request(
        respx_mock,
        monkeypatch,
        messages=[{"role": "user", "content": "What's the weather like in Boston today?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": WEATHER_PARAMETERS,
                    "cache_control": EPHEMERAL,
                },
            }
        ],
    )
    assert json.loads(request.content) == {
        "model": PROMPT_CACHING_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What's the weather like in Boston today?"}]}
        ],
        "tools": [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "input_schema": WEATHER_PARAMETERS,
                "type": "custom",
                "cache_control": EPHEMERAL,
            }
        ],
        "max_tokens": litellm.get_max_tokens(PROMPT_CACHING_MODEL),
    }


async def test_litellm_anthropic_prompt_caching_system(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_blocks: Final = [
        {"type": "text", "text": "You are an AI assistant tasked with analyzing legal documents."},
        {"type": "text", "text": "Here is the full text of a complex legal agreement", "cache_control": EPHEMERAL},
    ]
    request: Final = await _send_prompt_caching_request(
        respx_mock,
        monkeypatch,
        messages=[
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": "what are the key terms and conditions in this agreement?"},
        ],
    )
    assert json.loads(request.content) == {
        "model": PROMPT_CACHING_MODEL,
        "system": system_blocks,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "what are the key terms and conditions in this agreement?"}],
            }
        ],
        "max_tokens": litellm.get_max_tokens(PROMPT_CACHING_MODEL),
    }
