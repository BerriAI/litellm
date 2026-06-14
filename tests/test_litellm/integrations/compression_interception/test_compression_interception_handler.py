"""
Unit tests for Compression Interception Handler.
"""

from unittest.mock import MagicMock

import pytest

from litellm.integrations.compression_interception.handler import (
    CompressionInterceptionLogger,
)
from litellm.types.utils import CallTypes


def test_initialize_from_proxy_config():
    """Test initialization from proxy config with litellm_settings."""
    litellm_settings = {
        "compression_interception_params": {
            "enabled": True,
            "compression_trigger": 1234,
            "compression_target": 789,
        }
    }

    logger = CompressionInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params={},
    )

    assert logger.enabled is True
    assert logger.compression_trigger == 1234
    assert logger.compression_target == 789


@pytest.mark.asyncio
async def test_pre_call_hook_compresses_messages_and_injects_tool(monkeypatch):
    """Test pre-call hook compresses and stores per-call cache."""
    logger = CompressionInterceptionLogger()
    compressed_result = {
        "messages": [{"role": "user", "content": "stubbed"}],
        "original_tokens": 12000,
        "compressed_tokens": 5000,
        "compression_ratio": 0.58,
        "cache": {"auth.py": "full file content"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "litellm_content_retrieve",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    },
                },
            }
        ],
    }

    def _fake_compress(**kwargs):
        return compressed_result

    # The handler does ``from litellm.compression import compress`` at module
    # scope, so we must patch the binding on the handler module — patching
    # ``litellm.compress`` has no effect on the already-bound reference.
    monkeypatch.setattr(
        "litellm.integrations.compression_interception.handler.compress",
        _fake_compress,
    )

    kwargs = {
        "model": "bedrock/us.anthropic.claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "very large context"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "existing_tool", "parameters": {"type": "object"}},
            }
        ],
    }

    result = await logger.async_pre_call_deployment_hook(
        kwargs=kwargs, call_type=CallTypes.anthropic_messages
    )

    assert result is not None
    assert result["messages"] == compressed_result["messages"]
    tool_names = [t.get("function", {}).get("name") for t in result["tools"]]
    assert "existing_tool" in tool_names
    assert "litellm_content_retrieve" in tool_names
    assert result["litellm_call_id"] in logger._compression_cache_by_call_id


@pytest.mark.asyncio
async def test_pre_call_hook_below_trigger_does_not_inject_empty_tools(monkeypatch):
    """
    When compression is a no-op (below trigger / invalid tool sequence), the
    hook must NOT replace ``messages`` or inject an empty ``tools: []`` onto
    a request that originally had no tools — Anthropic Messages rejects
    ``tools: []``.
    """
    logger = CompressionInterceptionLogger()
    original_messages = [{"role": "user", "content": "short prompt"}]

    def _fake_compress_noop(**kwargs):
        return {
            "messages": original_messages,
            "original_tokens": 42,
            "compressed_tokens": 42,
            "compression_ratio": 0.0,
            "cache": {},
            "tools": [],
            "compression_skipped_reason": "below_trigger",
        }

    monkeypatch.setattr(
        "litellm.integrations.compression_interception.handler.compress",
        _fake_compress_noop,
    )

    kwargs = {
        "model": "bedrock/us.anthropic.claude-sonnet-4-5",
        "messages": original_messages,
    }

    result = await logger.async_pre_call_deployment_hook(
        kwargs=kwargs, call_type=CallTypes.anthropic_messages
    )

    assert result is not None
    # Original request had no ``tools`` — skipped compression must leave it that way.
    assert "tools" not in result
    # Cache must not be populated for a no-op.
    assert result.get("litellm_call_id") not in logger._compression_cache_by_call_id


@pytest.mark.asyncio
async def test_should_run_agentic_loop_detects_retrieval_tool_use():
    """Test should-run hook returns tool calls for retrieval tool_use blocks."""
    logger = CompressionInterceptionLogger()
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "litellm_content_retrieve",
                "input": {"key": "auth.py"},
            }
        ]
    }

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "litellm_content_retrieve",
                    "parameters": {"type": "object"},
                },
            }
        ],
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={},
    )

    assert should_run is True
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["input"]["key"] == "auth.py"


@pytest.mark.asyncio
async def test_build_agentic_loop_plan_returns_request_patch():
    """Callback should return typed patch with tool_result content."""
    logger = CompressionInterceptionLogger()
    call_id = "call_123"
    logger._compression_cache_by_call_id[call_id] = (
        {"auth.py": "full auth file"},
        9999999999.0,
    )

    logging_obj = MagicMock()
    logging_obj.litellm_call_id = call_id
    logging_obj.model_call_details = {
        "agentic_loop_params": {"model": "bedrock/invoke/claude-3-5-sonnet"}
    }

    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "toolu_abc",
                    "type": "tool_use",
                    "name": "litellm_content_retrieve",
                    "input": {"key": "auth.py"},
                }
            ]
        },
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "read auth.py"}],
        response=None,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "tools": [{"name": "litellm_content_retrieve"}],
        },
        logging_obj=logging_obj,
        stream=False,
        kwargs={
            "temperature": 0.1,
            "_compression_interception_internal": True,
            "litellm_logging_obj": object(),
        },
    )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None
    assert plan.request_patch.model == "bedrock/invoke/claude-3-5-sonnet"
    assert plan.request_patch.max_tokens == 1024
    assert plan.request_patch.messages is not None
    assert len(plan.request_patch.messages) == 3
    tool_result_content = plan.request_patch.messages[-1]["content"][0]["content"]
    assert tool_result_content == "full auth file"
    assert "_compression_interception_internal" not in plan.request_patch.kwargs
    assert "litellm_logging_obj" not in plan.request_patch.kwargs
    assert plan.request_patch.kwargs["temperature"] == 0.1
    assert "max_tokens" not in plan.request_patch.optional_params


@pytest.mark.asyncio
async def test_should_run_agentic_loop_with_custom_type_tools():
    """Test that async_should_run_agentic_loop returns True when tools contain
    litellm_content_retrieve as a custom-typed tool (e.g. Claude Code tool list)
    and the model response includes a matching tool_use block."""
    logger = CompressionInterceptionLogger()

    # Exact tools payload produced by Claude Code – litellm_content_retrieve is
    # the final entry and uses type="custom" (not type="function").
    tools = [
        {
            "name": "Agent",
            "description": "Launch a new agent to handle complex, multi-step tasks.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["description", "prompt"],
                "additionalProperties": False,
            },
        },
        {
            "name": "AskUserQuestion",
            "description": "Use this tool when you need to ask the user questions.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "questions": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
        },
        {
            "name": "Bash",
            "description": "Executes a given bash command and returns its output.",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
        {
            "name": "litellm_content_retrieve",
            "description": "Retrieve the full content of a file or message that was compressed to save tokens.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The identifier of the content to retrieve",
                        "enum": [
                            "message_0",
                            "HA_UPTIME_ROUTER_SPEC.md",
                            "message_159",
                            "message_160",
                        ],
                    }
                },
                "required": ["key"],
            },
            "type": "custom",
        },
    ]

    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "litellm_content_retrieve",
                "input": {"key": "message_0"},
            }
        ]
    }

    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="claude-3-5-sonnet",
        messages=[],
        tools=tools,
        stream=False,
        custom_llm_provider="anthropic",
        kwargs={},
    )

    assert should_run is True
    assert tools_dict["tool_type"] == "compression_retrieval"
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["input"]["key"] == "message_0"


@pytest.mark.asyncio
async def test_build_agentic_loop_plan_missing_key_fallback():
    """Missing cache keys should produce deterministic fallback content."""
    logger = CompressionInterceptionLogger()

    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "missing_call"
    logging_obj.model_call_details = {"agentic_loop_params": {}}

    plan = await logger.async_build_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "toolu_missing",
                    "type": "tool_use",
                    "name": "litellm_content_retrieve",
                    "input": {"key": "not_found.py"},
                }
            ]
        },
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "read file"}],
        response=None,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={},
        logging_obj=logging_obj,
        stream=False,
        kwargs={},
    )

    assert plan.request_patch is not None
    assert (
        plan.request_patch.messages[-1]["content"][0]["content"]
        == "[compressed content key 'not_found.py' not found]"
    )
