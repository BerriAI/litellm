import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.types.utils import Choices, Message, ModelResponse


def _fake_azure_client():
    """A Mock that passes the handler's isinstance(AsyncAzureOpenAI) check while
    still allowing arbitrary attribute access (e.g. .api_key)."""
    client = MagicMock()
    client.__class__ = AsyncAzureOpenAI
    return client


def _azure_tool_call_response_dict():
    return {
        "id": "azure-resp",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "litellm_web_search",
                                "arguments": '{"query": "weather in SF"}',
                            },
                        }
                    ],
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_azure_acompletion_runs_websearch_agentic_loop(monkeypatch):
    """Regression: web search interception must run on the Azure chat-completion
    handler. Azure uses its own handler that previously returned the raw response
    without ever invoking the agentic loop, so interception silently never ran.
    """
    final_response = ModelResponse(
        id="final-resp",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="It is sunny in SF."),
            )
        ],
        model="azure/gpt-4o",
        object="chat.completion",
        created=0,
    )

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=["azure"])
    websearch_logger._execute_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("SF weather: sunny, 65F", None)
    )
    monkeypatch.setattr(litellm, "callbacks", [websearch_logger])

    follow_up = AsyncMock(return_value=final_response)
    monkeypatch.setattr(litellm, "acompletion", follow_up)

    logging_obj = MagicMock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}

    azure_response = MagicMock()
    azure_response.model_dump.return_value = _azure_tool_call_response_dict()

    fake_client = MagicMock()
    fake_client.__class__ = AsyncAzureOpenAI

    handler = AzureChatCompletion()
    web_search_tool = {
        "type": "function",
        "function": {
            "name": "litellm_web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "weather in SF?"}],
        "tools": [web_search_tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": "litellm_web_search"},
        },
    }

    with (
        patch.object(
            handler,
            "get_azure_openai_client",
            return_value=_fake_azure_client(),
        ),
        patch.object(
            handler,
            "make_azure_openai_chat_completion_request",
            new=AsyncMock(return_value=({}, azure_response)),
        ),
    ):
        result = await handler.acompletion(
            api_key="sk-test",
            api_version="2024-02-01",
            model="gpt-4o",
            api_base="https://my-azure.openai.azure.com",
            data=data,
            timeout=60.0,
            dynamic_params=False,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            max_retries=2,
            litellm_params={},
        )

    follow_up.assert_awaited_once()
    _, follow_up_kwargs = follow_up.call_args
    assert "tool_choice" not in follow_up_kwargs
    assert result is final_response
    assert result.choices[0].message.content == "It is sunny in SF."


@pytest.mark.asyncio
async def test_azure_acompletion_skips_loop_for_disabled_provider(monkeypatch):
    """When azure is not enabled, the handler returns the original response and
    makes no follow-up call.
    """
    websearch_logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    monkeypatch.setattr(litellm, "callbacks", [websearch_logger])

    follow_up = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", follow_up)

    logging_obj = MagicMock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}

    azure_response = MagicMock()
    azure_response.model_dump.return_value = _azure_tool_call_response_dict()

    handler = AzureChatCompletion()
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "weather in SF?"}],
        "tools": [{"type": "function", "function": {"name": "litellm_web_search"}}],
    }

    with (
        patch.object(
            handler,
            "get_azure_openai_client",
            return_value=_fake_azure_client(),
        ),
        patch.object(
            handler,
            "make_azure_openai_chat_completion_request",
            new=AsyncMock(return_value=({}, azure_response)),
        ),
    ):
        result = await handler.acompletion(
            api_key="sk-test",
            api_version="2024-02-01",
            model="gpt-4o",
            api_base="https://my-azure.openai.azure.com",
            data=data,
            timeout=60.0,
            dynamic_params=False,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            max_retries=2,
            litellm_params={},
        )

    follow_up.assert_not_awaited()
    assert result.choices[0].message.tool_calls
    assert result.choices[0].message.tool_calls[0].function.name == "litellm_web_search"
