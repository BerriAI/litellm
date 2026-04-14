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

    monkeypatch.setattr("litellm.compress", _fake_compress)

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
