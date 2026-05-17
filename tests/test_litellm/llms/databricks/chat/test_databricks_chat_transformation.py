import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
    _sanitize_empty_content,
)


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
    result = config._transform_messages(
        messages=messages, model="databricks-claude", is_async=False
    )
    assert "content" not in result[0]
    assert result[1]["content"] == "Hi"


# --- Opus 4.7 ``temperature`` / ``top_p`` deprecation (issue #26444) ---
#
# Databricks model-serving routes ``databricks-claude-*`` requests to the
# Anthropic Messages API, which returns 400
# (``Model us.anthropic.claude-opus-4-7 does not support the temperature
# parameter``) when ``temperature`` or ``top_p`` is sent to Claude Opus 4.7.
# ``DatabricksConfig.get_supported_openai_params`` filters those params via
# the locally-defined ``_param_explicitly_unsupported`` helper, which reads
# ``supports_temperature: false`` / ``supports_top_p: false`` off the model
# registry and falls back to ``AnthropicConfig._is_claude_4_7_model`` for
# unreleased dated snapshots. Companion to the Anthropic / Bedrock fix in
# PR #28113.


@pytest.mark.parametrize(
    "model",
    [
        "databricks-claude-opus-4-7",
        "databricks-claude-opus-4-7-20260416",
        "databricks/databricks-claude-opus-4-7",
    ],
)
def test_opus_4_7_drops_temperature_and_top_p_from_supported_params(
    monkeypatch, model
):
    """Opus 4.7 on Databricks must not advertise ``temperature`` / ``top_p`` so
    ``drop_params=True`` strips them before the request leaves litellm."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import importlib

    import litellm as _litellm

    importlib.reload(_litellm)

    config = DatabricksConfig()
    params = config.get_supported_openai_params(model=model)

    assert "temperature" not in params, (
        f"temperature should be filtered from Databricks Opus 4.7 supported params; got {params!r}"
    )
    assert "top_p" not in params, (
        f"top_p should be filtered from Databricks Opus 4.7 supported params; got {params!r}"
    )


def test_opus_4_7_unknown_dated_variant_falls_back_to_family_check(monkeypatch):
    """Dated Databricks Opus 4.7 snapshots not yet in the model registry are
    still covered by the ``_is_claude_4_7_model`` family fallback."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    config = DatabricksConfig()

    params = config.get_supported_openai_params(
        model="databricks-claude-opus-4-7-20991231"
    )

    assert "temperature" not in params
    assert "top_p" not in params


@pytest.mark.parametrize(
    "model",
    [
        "databricks-claude-sonnet-4-5",
        "databricks-claude-haiku-4-5",
        "databricks-claude-opus-4-5",
        "databricks-claude-opus-4-1",
        "databricks-claude-3-7-sonnet",
    ],
)
def test_non_opus_4_7_databricks_models_still_support_temperature_and_top_p(
    monkeypatch, model
):
    """Regression guard: only Opus 4.7 deprecated temperature on Anthropic's
    Messages API. Every other Claude served via Databricks (Sonnet 4.5,
    Haiku 4.5, Opus 4.5/4.1, 3.7-sonnet) must keep advertising
    ``temperature`` and ``top_p`` so existing call sites keep working."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    config = DatabricksConfig()
    params = config.get_supported_openai_params(model=model)

    assert "temperature" in params, (
        f"temperature should remain supported for {model}; got {params!r}"
    )
    assert "top_p" in params, (
        f"top_p should remain supported for {model}; got {params!r}"
    )


def test_opus_4_7_map_openai_params_drops_temperature_and_top_p(monkeypatch):
    """``map_openai_params`` must not leak ``temperature`` / ``top_p`` into the
    Databricks request body for Opus 4.7, even when callers pass them
    explicitly without ``drop_params=True`` (defense in depth)."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    config = DatabricksConfig()

    mapped = config.map_openai_params(
        non_default_params={"temperature": 0.3, "top_p": 0.9, "max_tokens": 16},
        optional_params={},
        model="databricks-claude-opus-4-7",
        drop_params=False,
    )

    assert "temperature" not in mapped, (
        f"temperature leaked through Databricks Opus 4.7 mapping: {mapped!r}"
    )
    assert "top_p" not in mapped, (
        f"top_p leaked through Databricks Opus 4.7 mapping: {mapped!r}"
    )
    # ``max_tokens`` is unrelated to the deprecation and must still be forwarded.
    assert mapped.get("max_tokens") == 16


def test_opus_4_5_map_openai_params_preserves_temperature_and_top_p(monkeypatch):
    """Regression guard on the mapping site: Databricks Opus 4.5 still receives
    both sampling params verbatim."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    config = DatabricksConfig()

    mapped = config.map_openai_params(
        non_default_params={"temperature": 0.3, "top_p": 0.9},
        optional_params={},
        model="databricks-claude-opus-4-5",
        drop_params=False,
    )

    assert mapped["temperature"] == 0.3
    assert mapped["top_p"] == 0.9


def test_opus_4_7_drop_params_true_strips_temperature_end_to_end(monkeypatch):
    """End-to-end-shape check that mirrors @dgomez04's reproduction in issue
    #26444: ``get_supported_openai_params`` says ``temperature`` is NOT
    supported for ``databricks-claude-opus-4-7``, which is the exact contract
    ``litellm.utils.drop_params=True`` keys off to strip the param before the
    Databricks call. Without this PR the same lookup returned True for
    ``temperature``, so ``drop_params`` was a no-op and the request hit a 400."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    params = litellm.get_supported_openai_params(
        model="databricks-claude-opus-4-7", custom_llm_provider="databricks"
    )
    assert "temperature" not in params, (
        "regression: top-level litellm.get_supported_openai_params still "
        "claims temperature is supported for databricks-claude-opus-4-7; "
        "drop_params=True will be a no-op and Anthropic will return 400."
    )
    assert "top_p" not in params
