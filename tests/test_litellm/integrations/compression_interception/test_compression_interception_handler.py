from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.compression_interception.handler import (
    CompressionInterceptionLogger,
)


def test_initialize_from_proxy_config():
    litellm_settings = {
        "compression_interception_params": {
            "enabled_providers": ["openai"],
            "compression_trigger": 12000,
            "compression_target": 8000,
        }
    }
    logger = CompressionInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings, callback_specific_params={}
    )
    assert logger.enabled_providers == ["openai"]
    assert logger.compression_trigger == 12000
    assert logger.compression_target == 8000


@pytest.mark.asyncio
async def test_async_pre_request_hook_applies_compression_and_merges_tools():
    logger = CompressionInterceptionLogger(enabled_providers=["bedrock"])
    kwargs = {
        "messages": [{"role": "user", "content": "big context"}],
        "tools": [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}}
        ],
        "custom_llm_provider": "bedrock",
        "litellm_params": {"request_id": "abc"},
        "stream": True,
    }
    mock_result = {
        "messages": [{"role": "user", "content": "compressed context"}],
        "original_tokens": 2000,
        "compressed_tokens": 900,
        "compression_ratio": 0.55,
        "cache": {"auth.py": "full content"},
        "tools": [
            {
                "type": "function",
                "function": {"name": "litellm_content_retrieve", "parameters": {}},
            }
        ],
    }
    with patch(
        "litellm.integrations.compression_interception.handler.litellm.compress",
        return_value=mock_result,
    ):
        result = await logger.async_pre_request_hook(
            model="bedrock/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "big context"}],
            kwargs=kwargs,
        )

    assert result is not None
    assert result["messages"][0]["content"] == "compressed context"
    tool_names = [t.get("function", {}).get("name") for t in result["tools"]]
    assert "calculator" in tool_names
    assert "litellm_content_retrieve" in tool_names
    assert result["_compression_interception_cache"]["auth.py"] == "full content"
    assert (
        result["litellm_params"]["_compression_interception_cache"]["auth.py"]
        == "full content"
    )
    assert result["stream"] is False
    assert result["_websearch_interception_converted_stream"] is True


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop_detects_anthropic_tool_use():
    logger = CompressionInterceptionLogger(enabled_providers=["bedrock"])
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
    should_run, payload = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {"name": "litellm_content_retrieve", "parameters": {}},
            }
        ],
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={"_compression_interception_cache": {"auth.py": "full content"}},
    )
    assert should_run is True
    assert payload["tool_calls"][0]["id"] == "toolu_123"


@pytest.mark.asyncio
async def test_async_run_agentic_loop_executes_anthropic_followup():
    logger = CompressionInterceptionLogger(enabled_providers=["bedrock"])
    with patch(
        "litellm.integrations.compression_interception.handler.anthropic_messages.acreate",
        new=AsyncMock(return_value={"final": "answer"}),
    ) as mock_acreate:
        result = await logger.async_run_agentic_loop(
            tools={
                "tool_calls": [
                    {
                        "id": "toolu_123",
                        "name": "litellm_content_retrieve",
                        "input": {"key": "auth.py"},
                    }
                ],
                "cache": {"auth.py": "full file content"},
            },
            model="bedrock/claude",
            messages=[{"role": "user", "content": "fix auth"}],
            response={},
            anthropic_messages_provider_config=None,
            anthropic_messages_optional_request_params={"max_tokens": 512},
            logging_obj=None,
            stream=False,
            kwargs={},
        )

    assert result == {"final": "answer"}
    called_messages = mock_acreate.await_args.kwargs["messages"]
    assert called_messages[-1]["role"] == "user"
    tool_result = called_messages[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"] == "full file content"
