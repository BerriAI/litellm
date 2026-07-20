import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import litellm
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
    _sanitize_empty_content,
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


def test_transform_choices():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking.",
                                "signature": "ErcBCkgIAhABGAIiQMadog2CAJc8YJdce2Cmqvk0MFB+gGt4OyaH4c3l9p9v+0TKhYcNGliFkxddhCVkYR8zz8oaO1f3cHaEmYXN5SISDGAaomDR7CaTrhZxURoMbOR7AfFuHcIdVXFSIjC9ZamSyhzMg3maOtq2QHLXr6Z7tv0dut2S0Icdqk4g7MOFTSnCc0jA7lvnJyjI0wMqHR05PoVXEDSQjAV6NcUFkzFzp34z0xVMaK/VatCT",
                            }
                        ],
                    },
                    {"type": "text", "text": "# 5 Question and Answer Pairs"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "# 5 Question and Answer Pairs"
    assert choices[0].message.reasoning_content == "i'm thinking."
    assert choices[0].message.thinking_blocks is not None
    assert choices[0].message.tool_calls is None


def test_transform_choices_without_signature():
    """
    Test that the transformation works correctly when the signature field is missing
    from the summary, which occurs with new Databricks Foundation Models like
    databricks-gpt-oss-20b and databricks-gpt-oss-120b.
    """
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking without signature.",
                                # Note: no signature field here
                            }
                        ],
                    },
                    {"type": "text", "text": "Response without signature"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    # This should not raise a KeyError for missing signature
    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "Response without signature"
    assert choices[0].message.reasoning_content == "i'm thinking without signature."
    assert choices[0].message.thinking_blocks is not None
    assert len(choices[0].message.thinking_blocks) == 1

    # Verify the thinking block was created successfully without signature
    thinking_block = choices[0].message.thinking_blocks[0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == "i'm thinking without signature."


def test_convert_anthropic_tool_to_databricks_tool_with_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "description": "test description",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"]["description"] == "test description"


def test_convert_anthropic_tool_to_databricks_tool_without_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}},
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"].get("description") is None


def test_transform_choices_with_citations():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Blue",
                        "citations": [
                            {
                                "type": "char_location",
                                "cited_text": "The sky is blue.",
                                "document_index": 0,
                                "document_title": "My Document",
                                "start_char_index": 0,
                                "end_char_index": 50,
                            }
                        ],
                    }
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.provider_specific_fields == {
        "citations": [
            [
                {
                    "type": "char_location",
                    "cited_text": "The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "start_char_index": 0,
                    "end_char_index": 50,
                    "supported_text": "Blue",
                }
            ]
        ]
    }


def test_chunk_parser_with_citation():
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "delta": {
                    "content": [
                        {
                            "type": "text",
                            "text": "",
                            "citations": [
                                {
                                    "type": "char_location",
                                    "cited_text": "The sky is blue.",
                                    "document_index": 0,
                                    "document_title": "My Document",
                                    "start_char_index": 0,
                                    "end_char_index": 50,
                                }
                            ],
                        }
                    ],
                },
                "index": 0,
                "finish_reason": None,
            }
        ],
    }

    parsed = iterator.chunk_parser(chunk)
    assert parsed.choices[0].delta.provider_specific_fields == {
        "citation": {
            "type": "char_location",
            "cited_text": "The sky is blue.",
            "document_index": 0,
            "document_title": "My Document",
            "start_char_index": 0,
            "end_char_index": 50,
        }
    }


def test_sanitize_empty_content_pops_none():
    message = {"role": "user", "content": None}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_pops_empty_string():
    message = {"role": "user", "content": ""}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_pops_single_empty_text_block():
    message = {"role": "user", "content": [{"type": "text", "text": ""}]}
    _sanitize_empty_content(message)
    assert "content" not in message


def test_sanitize_empty_content_filters_empty_blocks_keeps_non_empty():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "  "},
        ],
    }
    _sanitize_empty_content(message)
    assert message["content"] == [{"type": "text", "text": "Hello"}]


def test_transform_messages_sanitizes_empty_content():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": "Hi"},
    ]
    result = config._transform_messages(messages=messages, model="databricks-claude", is_async=False)
    assert "content" not in result[0]
    assert result[1]["content"] == "Hi"


def _parallel_tool_calls():
    return [
        {
            "id": "call_A",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
        },
        {
            "id": "call_B",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        },
    ]


def _assert_every_tool_message_follows_tool_calls(messages):
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            previous = messages[index - 1] if index > 0 else {}
            assert previous.get("role") == "assistant" and previous.get("tool_calls"), (
                f"tool message at index {index} is not preceded by an assistant message with tool_calls: {messages}"
            )


def _declared_tool_call_ids(messages):
    return sorted(
        call["id"]
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
        for call in message["tool_calls"]
    )


def test_transform_request_splits_parallel_tool_calls_for_gpt():
    """Regression for LIT-3984: Databricks 400s with 'messages with role tool must
    be a response to a preceeding message with tool_calls' because parallel tool
    calls send consecutive tool messages. Each result must be re-paired with an
    assistant tool_calls message holding only its matching call."""
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "weather in SF and NYC?"},
        {"role": "assistant", "content": "checking", "tool_calls": _parallel_tool_calls()},
        {"role": "tool", "tool_call_id": "call_A", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_B", "content": "rainy"},
    ]

    result = config.transform_request(
        model="gpt-5.4-mini",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    _assert_every_tool_message_follows_tool_calls(result)
    assert _declared_tool_call_ids(result) == ["call_A", "call_B"]
    assistant_tool_call_messages = [m for m in result if m.get("role") == "assistant" and m.get("tool_calls")]
    assert all(len(m["tool_calls"]) == 1 for m in assistant_tool_call_messages), (
        "each split assistant message must declare exactly one tool call"
    )
    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_A", "call_B"]
    for tool_message, assistant_message in zip(tool_messages, assistant_tool_call_messages):
        assert assistant_message["tool_calls"][0]["id"] == tool_message["tool_call_id"]


def test_transform_request_pairs_out_of_order_parallel_results():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "checking", "tool_calls": _parallel_tool_calls()},
        {"role": "tool", "tool_call_id": "call_B", "content": "rainy"},
        {"role": "tool", "tool_call_id": "call_A", "content": "sunny"},
    ]

    result = config.transform_request(
        model="gpt-5.4-mini",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    _assert_every_tool_message_follows_tool_calls(result)
    for index, message in enumerate(result):
        if message.get("role") == "tool":
            assert result[index - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


def test_transform_request_leaves_single_tool_call_untouched():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_A",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": "sunny"},
    ]

    result = config.transform_request(
        model="gpt-5.4-mini",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    assert len(result) == 3
    _assert_every_tool_message_follows_tool_calls(result)
    assert _declared_tool_call_ids(result) == ["call_A"]


def test_transform_request_does_not_drop_tool_calls_on_incomplete_results():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "checking", "tool_calls": _parallel_tool_calls()},
        {"role": "tool", "tool_call_id": "call_A", "content": "sunny"},
        {"role": "user", "content": "thanks"},
    ]

    result = config.transform_request(
        model="gpt-5.4-mini",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    assert _declared_tool_call_ids(result) == ["call_A", "call_B"]


def test_transform_request_keeps_parallel_tool_calls_for_claude():
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "checking", "tool_calls": _parallel_tool_calls()},
        {"role": "tool", "tool_call_id": "call_A", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_B", "content": "rainy"},
    ]

    result = config.transform_request(
        model="databricks-claude-3-7-sonnet",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    assert len([m for m in result if m.get("role") == "assistant"]) == 1


def test_databricks_config_probes_capabilities_under_databricks_namespace():
    """Inherited AnthropicConfig capability probes read ``self.custom_llm_provider``;
    without this override they probed the ``anthropic`` cost-map namespace and
    ignored the exact ``databricks/databricks-claude-*`` entries."""
    assert DatabricksConfig().custom_llm_provider == "databricks"


# ---------------------------------------------------------------------------
# reasoning_effort translation
#
# Databricks foundation-model endpoints take reasoning controls via different
# payload shapes depending on the underlying model family:
#
#   Claude:        Anthropic-style `thinking={"type":"enabled","budget_tokens":N}`
#   Gemini 2.5:    Same Anthropic-style `thinking` payload as Claude
#                  (per docs.databricks.com/.../query-reason-models)
#   Gemini 3+:     Native OpenAI-style top-level `reasoning_effort`
#   GPT-5/GPT-OSS: Native OpenAI-style top-level `reasoning_effort`
#
# LiteLLM should translate `reasoning_effort` into the right shape for the
# first two families and pass it through unchanged for the latter two.
# ---------------------------------------------------------------------------


def _map_reasoning_effort(model: str, reasoning_effort, **extra_non_default):
    """Run map_openai_params with reasoning_effort + optional extras."""
    non_default = {"reasoning_effort": reasoning_effort}
    non_default.update(extra_non_default)
    return DatabricksConfig().map_openai_params(
        non_default_params=non_default,
        optional_params={},
        model=model,
        drop_params=False,
    )


def test_claude_translates_reasoning_effort_to_thinking():
    """Regression: Claude path must still translate to Anthropic-style thinking."""
    params = _map_reasoning_effort("databricks-claude-3-7-sonnet", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_low_translates_to_thinking_budget():
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_medium_translates_to_thinking_budget():
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "medium")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_high_translates_to_thinking_budget():
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "high")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_pro_translates_to_thinking_budget():
    """Cover the gemini-2-5-pro endpoint too, not just flash."""
    params = _map_reasoning_effort("databricks-gemini-2-5-pro", "high")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_with_dot_notation_translates():
    """A user passing the upstream Google-style `gemini-2.5-...` form should
    still trigger the Anthropic-thinking translation, not pass through."""
    params = _map_reasoning_effort("databricks-gemini-2.5-flash", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_0_does_not_match():
    """Guard against over-matching: `gemini-2-0` (hypothetical or future) is
    NOT a Gemini 2.5 endpoint and must not get the thinking translation."""
    params = _map_reasoning_effort("databricks-gemini-2-0-flash", "low")
    assert "thinking" not in params
    assert params.get("reasoning_effort") == "low"


def test_gemini_2_5_none_drops_thinking_and_reasoning_effort():
    """`reasoning_effort='none'` mirrors the Claude behavior: no thinking emitted."""
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "none")
    assert "thinking" not in params
    assert "reasoning_effort" not in params


def test_gemini_3_passes_reasoning_effort_through():
    """Databricks-Gemini-3+ accepts reasoning_effort natively — do not translate."""
    params = _map_reasoning_effort("databricks-gemini-3-1-pro", "low")
    assert params.get("reasoning_effort") == "low"
    assert "thinking" not in params


def test_gpt_5_passes_reasoning_effort_through():
    """Databricks-GPT-5 family accepts reasoning_effort natively."""
    params = _map_reasoning_effort("databricks-gpt-5-1", "low")
    assert params.get("reasoning_effort") == "low"
    assert "thinking" not in params


def test_gpt_oss_passes_reasoning_effort_through():
    """Databricks-GPT-OSS accepts reasoning_effort natively."""
    params = _map_reasoning_effort("databricks-gpt-oss-120b", "high")
    assert params.get("reasoning_effort") == "high"
    assert "thinking" not in params
