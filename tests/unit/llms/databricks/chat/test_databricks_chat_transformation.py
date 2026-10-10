import copy
import json
from collections.abc import Iterator
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
    _sanitize_empty_content,
)
import asyncio
from unittest.mock import Mock
from litellm._version import version
from litellm.utils import CustomStreamWrapper
from typing import Any, Dict
from typing import List

DATABRICKS_API_BASE: Final = "https://my.workspace.cloud.databricks.com/serving-endpoints"
DATABRICKS_API_KEY: Final = "dapimykey"
DATABRICKS_CHAT_COMPLETIONS_URL: Final = f"{DATABRICKS_API_BASE}/chat/completions"
DATABRICKS_EMBEDDINGS_URL: Final = f"{DATABRICKS_API_BASE}/embeddings"
JSON_SCHEMA_RESPONSE_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "P",
        "strict": True,
        "schema": {
            "$defs": {
                "Person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/Person"}},
            "required": ["p"],
        },
    },
}


@pytest.fixture()
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.fixture
def _databricks_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _databricks_chat_response(model: str, usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_3f78f09a-489c-4b8d-a587-f162c7497891",
        "object": "chat.completion",
        "created": 1726285449,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _databricks_embedding_response() -> dict[str, object]:
    return {
        "object": "list",
        "model": "bge-large-en-v1.5",
        "data": [
            {
                "index": 0,
                "object": "embedding",
                "embedding": [
                    0.06768798828125,
                    -0.01291656494140625,
                    -0.0501708984375,
                    0.0245361328125,
                    -0.030364990234375,
                ],
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
            "completion_tokens": 0,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }


def _databricks_anthropic_cache_response(
    cache_read_input_tokens: int,
    cache_creation_input_tokens: int,
) -> dict[str, object]:
    usage: Final = {
        "completion_tokens": 117,
        "prompt_tokens": 1549,
        "total_tokens": 1666,
        "completion_tokens_details": None,
        "prompt_tokens_details": {
            "cached_tokens": 0,
            "cache_creation_tokens": cache_creation_input_tokens,
        },
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
    }
    return _databricks_chat_response("claude-3-7-sonnet", usage)


def _databricks_streaming_chat_chunks() -> tuple[str, ...]:
    return (
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [{"delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 230, "completion_tokens": 1, "total_tokens": 231},
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [{"delta": {"content": " world"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 230, "completion_tokens": 1, "total_tokens": 231},
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 230, "completion_tokens": 1, "total_tokens": 231},
            }
        ),
    )


def _assert_databricks_request(request: httpx.Request, expected_url: str, api_key: str) -> None:
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {api_key}"
    assert str(request.url) == expected_url


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
    result = config.transform_messages(messages=messages, model="databricks-claude", is_async=False)
    assert "content" not in result[0]
    assert result[1]["content"] == "Hi"


def test_transform_request_preserves_unity_model_service_name():
    config = DatabricksConfig()
    result = config.transform_request(
        model="system.ai.kimi-k3",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert result["model"] == "system.ai.kimi-k3"


def test_transform_request_strips_thinking_blocks_and_reasoning_content():
    """Regression for LIT-6762: replaying an assistant turn that litellm decorated with
    `thinking_blocks` / `reasoning_content` made Databricks 400 with
    'messages.N.thinking_blocks: Extra inputs are not permitted'."""
    config = DatabricksConfig()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Hello! How can I help?",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "greet briefly", "signature": "sig_abc", "cache_control": {}}
            ],
            "reasoning_content": "greet briefly",
            "provider_specific_fields": {"foo": "bar"},
        },
        {"role": "user", "content": "thanks"},
    ]

    result = config.transform_request(
        model="databricks-claude-opus-5",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    assert result[1] == {"role": "assistant", "content": "Hello! How can I help?"}
    assert not any(
        key in message
        for message in result
        for key in ("thinking_blocks", "reasoning_content", "provider_specific_fields")
    )
    assert "thinking_blocks" in messages[1]


def test_transform_request_drops_thinking_only_assistant_turn_but_keeps_tool_call_turn():
    """A replayed thinking-only assistant turn has nothing left once `thinking_blocks` are stripped, so it must be
    dropped instead of being sent as a bare {"role": "assistant"}. A thinking + tool_use turn keeps its tool_calls."""
    config = DatabricksConfig()
    tool_call = {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "hmm", "signature": "sig_1"}],
            "reasoning_content": "hmm",
        },
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [{"type": "thinking", "thinking": "call f", "signature": "sig_2"}],
            "reasoning_content": "call f",
            "tool_calls": [tool_call],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    result = config.transform_request(
        model="databricks-claude-opus-5",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )["messages"]

    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]


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


@pytest.mark.parametrize(
    "model, expected_thinking, expected_output_config",
    [
        ("databricks-claude-opus-4-8", {"type": "adaptive"}, {"effort": "high"}),
        ("databricks-claude-opus-4-6", {"type": "enabled", "budget_tokens": 4096}, None),
    ],
    ids=["adaptive_only_upgrades_to_adaptive", "legacy_capable_forwards_verbatim"],
)
def test_map_openai_params_upgrades_legacy_thinking_on_adaptive_only_claude(
    model, expected_thinking, expected_output_config
):
    mapped = DatabricksConfig().map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 4096}},
        optional_params={},
        model=model,
        drop_params=False,
    )
    assert mapped["thinking"] == expected_thinking
    assert mapped.get("output_config") == expected_output_config


def _map_reasoning_effort(model: str, reasoning_effort: str):
    return DatabricksConfig().map_openai_params(
        non_default_params={"reasoning_effort": reasoning_effort},
        optional_params={},
        model=model,
        drop_params=False,
    )


def test_claude_translates_reasoning_effort_to_thinking(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-claude-3-7-sonnet", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_adaptive_claude_translates_reasoning_effort_to_output_config(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-claude-opus-4-7", "high")
    assert params.get("thinking") == {"type": "adaptive", "display": "summarized"}
    assert params.get("output_config") == {"effort": "high"}
    assert "reasoning_effort" not in params


def test_unmapped_claude_endpoint_still_translates(_use_local_model_cost_map):
    params = _map_reasoning_effort("my-claude-serving-endpoint", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_low_translates_to_thinking_budget(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_medium_translates_to_thinking_budget(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "medium")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_high_translates_to_thinking_budget(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "high")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_pro_translates_to_thinking_budget(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-5-pro", "high")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_5_with_dot_notation_translates(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2.5-flash", "low")
    assert params.get("thinking") == {
        "type": "enabled",
        "budget_tokens": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    }
    assert "reasoning_effort" not in params


def test_gemini_2_0_does_not_match(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-0-flash", "low")
    assert "thinking" not in params
    assert params.get("reasoning_effort") == "low"


def test_gemini_2_5_none_drops_thinking_and_reasoning_effort(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-2-5-flash", "none")
    assert "thinking" not in params
    assert "reasoning_effort" not in params


def test_gemini_3_passes_reasoning_effort_through(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gemini-3-1-pro", "low")
    assert params.get("reasoning_effort") == "low"
    assert "thinking" not in params


def test_gpt_5_passes_reasoning_effort_through(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gpt-5-1", "low")
    assert params.get("reasoning_effort") == "low"
    assert "thinking" not in params


def test_gpt_oss_passes_reasoning_effort_through(_use_local_model_cost_map):
    params = _map_reasoning_effort("databricks-gpt-oss-120b", "high")
    assert params.get("reasoning_effort") == "high"
    assert "thinking" not in params


def _streaming_chunk(usage=None, choices=None):
    base = {
        "id": "chatcmpl-test",
        "created": 1234567890,
        "model": "databricks-claude-sonnet-5",
        "choices": [{"delta": {"content": "hi"}}] if choices is None else choices,
    }
    return base if usage is None else {**base, "usage": usage}


@pytest.mark.parametrize(
    "cache_read, cache_creation, expected_cached, expected_written",
    [
        (12002, 0, 12002, 0),
        (0, 12002, 0, 12002),
    ],
    ids=["warm_cache_read", "cold_cache_write"],
)
def test_chunk_parser_surfaces_prompt_cache_usage(cache_read, cache_creation, expected_cached, expected_written):
    iterator = DatabricksChatResponseIterator(streaming_response=None, sync_stream=True)

    result = iterator.chunk_parser(
        _streaming_chunk(
            usage={
                "prompt_tokens": 12011,
                "completion_tokens": 8,
                "total_tokens": 12019,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        )
    )

    assert result.usage is not None
    assert result.usage.prompt_tokens == 12011
    assert result.usage.completion_tokens == 8
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == expected_cached
    assert result.usage._cache_creation_input_tokens == expected_written


def test_chunk_parser_surfaces_usage_only_final_chunk():
    """stream_options={"include_usage": True} emits a trailing chunk whose choices
    list is empty; usage must still reach the caller."""
    iterator = DatabricksChatResponseIterator(streaming_response=None, sync_stream=True)

    result = iterator.chunk_parser(
        _streaming_chunk(
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "cache_read_input_tokens": 90,
            },
            choices=[],
        )
    )

    assert result.choices == []
    assert result.usage is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 90


def test_chunk_parser_without_usage_still_parses_content():
    iterator = DatabricksChatResponseIterator(streaming_response=None, sync_stream=True)

    result = iterator.chunk_parser(_streaming_chunk())

    assert result.id == "chatcmpl-test"
    assert result.model == "databricks-claude-sonnet-5"
    assert result.choices[0]["delta"]["content"] == "hi"


@pytest.mark.parametrize("reasoning_key", ["reasoning_content", "reasoning"])
def test_transform_choices_surfaces_top_level_reasoning_content(reasoning_key: str) -> None:
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": "391",
                reasoning_key: "We need answer just number. 17*23=391.",
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.content == "391"
    assert choices[0].message.reasoning_content == "We need answer just number. 17*23=391."
    assert getattr(choices[0].message, "thinking_blocks", None) is None


def test_transform_choices_parses_think_tags_in_string_content():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {"role": "assistant", "content": "<think>17 times 23</think>391"},
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.content == "391"
    assert choices[0].message.reasoning_content == "17 times 23"


def test_transform_choices_prefers_reasoning_blocks_over_top_level_field():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "from block"}]},
                    {"type": "text", "text": "391"},
                ],
                "reasoning_content": "from field",
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.reasoning_content == "from block"
    assert choices[0].message.content == "391"


@pytest.mark.parametrize("reasoning_key", ["reasoning_content", "reasoning"])
def test_chunk_parser_surfaces_top_level_reasoning_delta(reasoning_key: str) -> None:
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lit-qa-deepseek-v4-flash",
        "choices": [
            {
                "delta": {"role": "assistant", "content": None, reasoning_key: "We need answer"},
                "index": 0,
                "finish_reason": None,
            }
        ],
    }

    parsed = iterator.chunk_parser(chunk)

    assert parsed.choices[0].delta.reasoning_content == "We need answer"
    assert parsed.choices[0].delta.content is None


def test_completion_merges_leading_system_and_developer_messages_for_chat_template_models(
    respx_mock: respx.MockRouter,
):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    response: Final = litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "developer", "content": "Skills: none."},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse.\n\nSkills: none."},
        {"role": "user", "content": "Hello"},
    ]
    assert response.choices[0].message.content == "Answer"


def test_completion_merges_system_messages_when_one_has_empty_content(respx_mock: respx.MockRouter):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]


def test_chunk_parser_relays_the_served_service_tier():
    iterator = DatabricksChatResponseIterator(streaming_response=None, sync_stream=True)

    with_tier: Final = iterator.chunk_parser({**_streaming_chunk(), "service_tier": "priority"})
    assert with_tier.model_dump()["service_tier"] == "priority"

    without_tier: Final = iterator.chunk_parser(_streaming_chunk())
    assert getattr(without_tier, "service_tier", None) is None


def test_completions_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        },
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


def test_completions_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        },
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response = asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages=messages,
                client=async_handler,
                temperature=0.5,
                extraparam="testpassingextraparam",
            )
        )

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


def test_completions_streaming_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()

    messages = [{"role": "user", "content": "How are you?"}]
    mock_response = mock_http_handler_chat_streaming_response()

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response_stream: CustomStreamWrapper = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
            stream=True,
        )
        response = list(response_stream)
        assert "dbrx-instruct-071224" in str(response)
        assert "chatcmpl" in str(response)
        assert len(response) == 4

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == True

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


def test_completions_streaming_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()

    messages = [{"role": "user", "content": "How are you?"}]
    mock_response = mock_http_handler_chat_async_streaming_response()

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response_stream: CustomStreamWrapper = asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages=messages,
                client=async_handler,
                temperature=0.5,
                extraparam="testpassingextraparam",
                stream=True,
            )
        )

        # Use async list gathering for the response
        async def gather_responses():
            return [item async for item in response_stream]

        response = asyncio.run(gather_responses())
        assert "dbrx-instruct-071224" in str(response)
        assert "chatcmpl" in str(response)
        assert len(response) == 4

        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == True

        actual_data = json.loads(
            mock_post.call_args.kwargs["data"]
        )  # Deserialize the actual data
        expected_data = {
            "model": "dbrx-instruct-071224",
            "messages": messages,
            "temperature": 0.5,
            "stream": True,
            "extraparam": "testpassingextraparam",
        }
        assert actual_data == expected_data, f"Unexpected JSON data: {actual_data}"


def test_embeddings_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    inputs = ["Hello", "World"]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.embedding(
            model="databricks/bge-large-en-v1.5",
            input=inputs,
            client=sync_handler,
            extraparam="testpassingextraparam",
        )
        assert response.to_dict() == mock_embedding_response()

        mock_post.assert_called_once_with(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"litellm/{version}",
            },
            data=json.dumps(
                {
                    "model": "bge-large-en-v1.5",
                    "input": inputs,
                    "extraparam": "testpassingextraparam",
                }
            ),
        )


def test_embeddings_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    inputs = ["Hello", "World"]

    with patch.object(
        AsyncHTTPHandler, "post", return_value=mock_response
    ) as mock_post:
        response = asyncio.run(
            litellm.aembedding(
                model="databricks/bge-large-en-v1.5",
                input=inputs,
                client=async_handler,
                extraparam="testpassingextraparam",
            )
        )
        assert response.to_dict() == mock_embedding_response()

        mock_post.assert_called_once_with(
            f"{base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"litellm/{version}",
            },
            data=json.dumps(
                {
                    "model": "bge-large-en-v1.5",
                    "input": inputs,
                    "extraparam": "testpassingextraparam",
                }
            ),
        )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_databricks_embeddings(sync_mode, monkeypatch):
    """
    Test Databricks embeddings with instruction parameter in both sync and async modes using mocked HTTP responses.
    """
    import openai

    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_embedding_response()

    inputs = ["good morning from litellm"]
    instruction = "Represent this sentence for searching relevant passages:"

    litellm.set_verbose = True
    litellm.drop_params = True

    if sync_mode:
        sync_handler = HTTPHandler()
        with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
            response = litellm.embedding(
                model="databricks/databricks-bge-large-en",
                input=inputs,
                instruction=instruction,
                client=sync_handler,
            )

            openai.types.CreateEmbeddingResponse.model_validate(
                response.model_dump(), strict=True
            )

            mock_post.assert_called_once_with(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"litellm/{version}",
                },
                data=json.dumps(
                    {
                        "model": "databricks-bge-large-en",
                        "input": inputs,
                        "instruction": instruction,
                    }
                ),
            )
    else:
        async_handler = AsyncHTTPHandler()
        with patch.object(
            AsyncHTTPHandler, "post", return_value=mock_response
        ) as mock_post:
            response = await litellm.aembedding(
                model="databricks/databricks-bge-large-en",
                input=inputs,
                instruction=instruction,
                client=async_handler,
            )

            openai.types.CreateEmbeddingResponse.model_validate(
                response.model_dump(), strict=True
            )

            mock_post.assert_called_once_with(
                f"{base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": f"litellm/{version}",
                },
                data=json.dumps(
                    {
                        "model": "databricks-bge-large-en",
                        "input": inputs,
                        "instruction": instruction,
                    }
                ),
            )


def test_completion_with_prompt_caching_anthropic_model_repeat(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = (
        mock_chat_response_anthropic_prompt_caching_repeat()
    )

    mock_text = "example text" * 512
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that explains the content of the given text.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": mock_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert "claude-3-7-sonnet" in response["model"]
        assert response["usage"]["cache_read_input_tokens"] == 1545
        assert response["usage"]["cache_creation_input_tokens"] == 0
        assert response["usage"]["prompt_tokens"] == 1549
        assert response["usage"]["completion_tokens"] == 117
        assert response["usage"]["total_tokens"] == 1666


def test_completion_with_prompt_caching_nonanthropic_model(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_nonanthropic_prompt_caching()

    mock_text = "example text" * 512
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that explains the content of the given text.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": mock_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-gpt-oss-20b",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert "gpt-oss-20b" in response["model"]
        assert ("cache_read_input_tokens" not in response["usage"]) or response[
            "usage"
        ]["cache_read_input_tokens"] in [0, None]
        assert ("cache_creation_input_tokens" not in response["usage"]) or response[
            "usage"
        ]["cache_creation_input_tokens"] in [0, None]
        assert response["usage"]["prompt_tokens"] == 1638
        assert response["usage"]["completion_tokens"] == 500
        assert response["usage"]["total_tokens"] == 2138


@pytest.mark.parametrize(
    "model",
    ["databricks/databricks-claude-3-7-sonnet"],
)
def test_databricks_anthropic_function_call_with_no_schema(model, monkeypatch):
    """
    Test function calling with tools that have no parameters schema using mocked HTTP responses.
    Relevant Issue: https://github.com/BerriAI/litellm/issues/6012
    """
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    mock_response_data = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1699896916,
        "model": "databricks-claude-3-7-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
        },
    }

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data

    sync_handler = HTTPHandler()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in New York",
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What is the current temperature in New York?"}
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response):
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            client=sync_handler,
        )

        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert (
            response.choices[0].message.tool_calls[0].function.name
            == "get_current_weather"
        )


def test_databricks_anthropic_user_string_content_cache_injection(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_anthropic_prompt_caching()

    mock_text = "example text" * 512
    messages = [
        {"role": "system", "content": "You are an expert summarizer."},
        {"role": "user", "content": mock_text},
    ]
    cache_control_injection_points = [{"location": "message", "role": "user"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            cache_control_injection_points=cache_control_injection_points,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert "claude-3-7-sonnet" in response["model"]
        assert response["usage"]["cache_read_input_tokens"] == 0
        assert response["usage"]["cache_creation_input_tokens"] == 1545
        assert response["usage"]["prompt_tokens"] == 1549
        assert response["usage"]["completion_tokens"] == 117
        assert response["usage"]["total_tokens"] == 1666


def test_databricks_anthropic_system_string_content_cache_injection(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response_anthropic_prompt_caching()

    mock_text = "example text" * 512
    messages = [
        {"role": "system", "content": mock_text},
        {"role": "user", "content": "You are an expert summarizer."},
    ]
    cache_control_injection_points = [{"location": "message", "role": "system"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            cache_control_injection_points=cache_control_injection_points,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert "claude-3-7-sonnet" in response["model"]
        assert response["usage"]["cache_read_input_tokens"] == 0
        assert response["usage"]["cache_creation_input_tokens"] == 1545
        assert response["usage"]["prompt_tokens"] == 1549
        assert response["usage"]["completion_tokens"] == 117
        assert response["usage"]["total_tokens"] == 1666


def test_databricks_anthropic_system_string_content_cache_injection_not_enough_tokens(
    monkeypatch,
):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = (
        mock_chat_response_anthropic_prompt_caching_not_enough_tokens()
    )

    mock_text = "example text" * 512
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that explains the content of the given text.",
        },
        {"role": "user", "content": mock_text},
    ]
    cache_control_injection_points = [{"location": "message", "role": "system"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/databricks-claude-3-7-sonnet",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
            cache_control_injection_points=cache_control_injection_points,
            extraparam="testpassingextraparam",
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Content-Type"] == "application/json"
        )
        assert (
            mock_post.call_args.kwargs["headers"]["Authorization"]
            == f"Bearer {api_key}"
        )
        assert mock_post.call_args.kwargs["url"] == f"{base_url}/chat/completions"
        assert mock_post.call_args.kwargs["stream"] == False

        # TODO: add test for entire expected output schema in the future
        # Check the response object returned from litellm.completion()
        assert "claude-3-7-sonnet" in response["model"]
        assert response["usage"]["cache_read_input_tokens"] == 0
        assert response["usage"]["cache_creation_input_tokens"] == 0
        assert response["usage"]["prompt_tokens"] == 1549
        assert response["usage"]["completion_tokens"] == 117
        assert response["usage"]["total_tokens"] == 1666


def mock_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl_3f78f09a-489c-4b8d-a587-f162c7497891",
        "object": "chat.completion",
        "created": 1726285449,
        "model": "dbrx-instruct-071224",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant. I'm doing well. How can I help?",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 230,
            "completion_tokens": 38,
            "completion_tokens_details": None,
            "total_tokens": 268,
            "prompt_tokens_details": None,
        },
        "system_fingerprint": None,
    }


def mock_chat_response_anthropic_prompt_caching() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761118943,
        "model": "claude-3-7-sonnet",  # Mock model name for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I notice that you've provided a repetitive text that simply repeats \"example text\" many times rather than actual content to summarize. \n\nTo provide you with a meaningful summary, I would need:\n- Actual substantive text with real information, arguments, or narrative\n- Content that has key points, themes, or conclusions to extract\n- Material with varying ideas or concepts to synthesize\n\nCould you please share the actual text you'd like me to summarize? I'm ready to help once you provide content with real information to work with.",
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "completion_tokens": 117,
            "prompt_tokens": 1549,
            "total_tokens": 1666,
            "completion_tokens_details": None,
            "prompt_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "text_tokens": None,
                "image_tokens": None,
                "cache_creation_tokens": 1545,
            },
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 1545,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def mock_chat_response_anthropic_prompt_caching_not_enough_tokens() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761118943,
        "model": "claude-3-7-sonnet",  # Mock model name for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I notice that you've provided a repetitive text that simply repeats \"example text\" many times rather than actual content to summarize. \n\nTo provide you with a meaningful summary, I would need:\n- Actual substantive text with real information, arguments, or narrative\n- Content that has key points, themes, or conclusions to extract\n- Material with varying ideas or concepts to synthesize\n\nCould you please share the actual text you'd like me to summarize? I'm ready to help once you provide content with real information to work with.",
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "completion_tokens": 117,
            "prompt_tokens": 1549,
            "total_tokens": 1666,
            "completion_tokens_details": None,
            "prompt_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "text_tokens": None,
                "image_tokens": None,
                "cache_creation_tokens": 0,
            },
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def mock_chat_response_anthropic_prompt_caching_repeat() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761118943,
        "model": "claude-3-7-sonnet",  # Mock model name for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I notice that you've provided a repetitive text that simply repeats \"example text\" many times rather than actual content to summarize. \n\nTo provide you with a meaningful summary, I would need:\n- Actual substantive text with real information, arguments, or narrative\n- Content that has key points, themes, or conclusions to extract\n- Material with varying ideas or concepts to synthesize\n\nCould you please share the actual text you'd like me to summarize? I'm ready to help once you provide content with real information to work with.",
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "completion_tokens": 117,
            "prompt_tokens": 1549,
            "total_tokens": 1666,
            "completion_tokens_details": None,
            "prompt_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
                "text_tokens": None,
                "image_tokens": None,
                "cache_creation_tokens": 1545,
            },
            "cache_read_input_tokens": 1545,
            "cache_creation_input_tokens": 0,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def mock_chat_response_nonanthropic_prompt_caching() -> Dict[str, Any]:
    return {
        "id": "msg_01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "object": "chat.completion",
        "created": 1761119150,
        "model": "gpt-oss-20b",  # Mock model nama for testing
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "The user just posted a block of text repeated: \"example textexample\" many times. It is unclear what they want. The instruction says: \"You are a helpful assistant that explains the content of the given text.\" So I need to explain the content.\n\nThe content is basically a repeated phrase 'example textexample' many times, possibly a demonstration of repeated words or filler text. Perhaps they test that the assistant enumerates or condenses. Should I explain that it is a repeated phrase used maybe as placeholder text? It looks like a placeholder or filler. Could say that it's essentially nonsense.\n\nExplain that the text consists of the word \"example\" concatenated with \"text\" repeated many times. It's not meaningful content. Might indicate filler text for page layout.\n\nAlternatively, explain why repeated 'example textexample' (without whitespace in some places?) is repeated. This could be a test. The user probably expects a response like: \"It says 'example textexample' several times.\" So I should summarize: The text is a repeated phrase used as filler.\n\nGiven the instruction, let's explain the content. Mention that it's repetitive placeholder, no meaningful content, just repeated phrase. Also note that \"example text\" repeated words. No specific meaning beyond being placeholder.\n\nSo respond: This is basically a placeholder used in design documents: the phrase \"example text\" repeated to fill a space, no distinct meaning beyond placeholder usage. 'text' might be part of the 'example text' phrase or 'textexample' it's concatenated. These might serve to fill text boxes, test fonts, etc.\n\nAlso mention the pattern: Could be used for testing text rendering, typographic layouts, measuring dimensions.\n\nAnswer accordingly.",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": 'The passage you pasted is essentially a block of **placeholder text**.  \nIt repeats the phrase "example textexample" (or "example text" in some places) over and over again.  There isn\'t any hidden message, concept, or argument buried in it – the purpose is purely to fill space, imitate real content, or test something like typography, layout, or rendering.\n\nIn design and copy‑editing, such repeated strings are often used to:\n\n* **Fill a page or template** so the designer can see how multiple lines of content will look.\n* **Test the appearance of fonts, line‑height, paragraph spacing, and other typographic settings.**\n* **Serve as a stand',
                        },
                    ],
                    "refusal": None,
                    "function_call": None,
                    "tool_calls": None,
                    "annotations": None,
                    "audio": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1638,
            "completion_tokens": 500,
            "total_tokens": 2138,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def mock_http_handler_chat_streaming_response() -> MagicMock:
    mock_stream_chunks = mock_chat_streaming_response_chunks()

    def mock_iter_lines():
        for chunk in mock_stream_chunks:
            for line in chunk.splitlines():
                yield line

    mock_response = MagicMock()
    mock_response.iter_lines.side_effect = mock_iter_lines
    mock_response.status_code = 200

    return mock_response


def mock_http_handler_chat_async_streaming_response() -> MagicMock:
    mock_stream_chunks = mock_chat_streaming_response_chunks()

    async def mock_iter_lines():
        for chunk in mock_stream_chunks:
            for line in chunk.splitlines():
                yield line

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = mock_iter_lines()
    mock_response.status_code = 200

    return mock_response


def mock_embedding_response() -> Dict[str, Any]:
    return {
        "object": "list",
        "model": "bge-large-en-v1.5",
        "data": [
            {
                "index": 0,
                "object": "embedding",
                "embedding": [
                    0.06768798828125,
                    -0.01291656494140625,
                    -0.0501708984375,
                    0.0245361328125,
                    -0.030364990234375,
                ],
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
            "completion_tokens": 0,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }


def mock_chat_streaming_response_chunks() -> List[str]:
    return [
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
        json.dumps(
            {
                "id": "chatcmpl_8a7075d1-956e-4960-b3a6-892cd4649ff3",
                "object": "chat.completion.chunk",
                "created": 1726469651,
                "model": "dbrx-instruct-071224",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "!"},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 230,
                    "completion_tokens": 1,
                    "total_tokens": 231,
                },
            }
        ),
    ]


@pytest.mark.parametrize(
    "model",
    (
        "databricks-meta-llama-3-3-70b-instruct",
        "databricks-qwen35-122b-a10b",
        "databricks-gpt-oss-120b",
    ),
)
def test_databricks_non_claude_json_schema_preserves_refs(model: str) -> None:
    optional_params: Final = litellm.utils.get_optional_params(
        model=model,
        custom_llm_provider="databricks",
        response_format=copy.deepcopy(JSON_SCHEMA_RESPONSE_FORMAT),
    )

    assert optional_params["response_format"] == JSON_SCHEMA_RESPONSE_FORMAT


def test_databricks_claude_json_schema_preserves_refs_in_tool_parameters() -> None:
    optional_params: Final = litellm.utils.get_optional_params(
        model="databricks-claude-haiku-4-5",
        custom_llm_provider="databricks",
        response_format=copy.deepcopy(JSON_SCHEMA_RESPONSE_FORMAT),
    )

    assert "response_format" not in optional_params
    assert optional_params["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "json_tool_call",
                "parameters": JSON_SCHEMA_RESPONSE_FORMAT["json_schema"]["schema"],
            },
        }
    ]


def test_databricks_pydantic_json_schema_preserves_nested_refs() -> None:
    class Person(BaseModel):
        model_config = ConfigDict(frozen=True)

        name: str

    class Order(BaseModel):
        model_config = ConfigDict(frozen=True)

        person: Person

    optional_params: Final = litellm.utils.get_optional_params(
        model="databricks-meta-llama-3-3-70b-instruct",
        custom_llm_provider="databricks",
        response_format=Order,
    )
    schema: Final = optional_params["response_format"]["json_schema"]["schema"]

    assert schema["properties"]["person"] == {"$ref": "#/$defs/Person"}
    assert schema["$defs"]["Person"] == {
        "additionalProperties": False,
        "properties": {"name": {"title": "Name", "type": "string"}},
        "required": ["name"],
        "title": "Person",
        "type": "object",
    }
