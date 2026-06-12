"""
Tests for the Bedrock Converse handler (converse_handler.py).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
)


def _web_search_model_response() -> ModelResponse:
    """ModelResponse where the model asked to call the litellm_web_search tool."""
    return ModelResponse(
        id="initial-resp",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "weather in SF"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="claude-sonnet-4",
        object="chat.completion",
        created=0,
    )


def _patch_converse_request(monkeypatch, initial_response: ModelResponse) -> None:
    """Stub out everything except the agentic-hook wiring under test."""
    monkeypatch.setattr(
        litellm.AmazonConverseConfig,
        "_async_transform_request",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        litellm.AmazonConverseConfig,
        "_transform_response",
        MagicMock(return_value=initial_response),
    )


async def _call_async_completion(handler: BedrockConverseLLM):
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

    handler.get_request_headers = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(headers={})
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=mock_resp)

    logging_obj = MagicMock()
    logging_obj.dynamic_success_callbacks = []
    logging_obj.model_call_details = {}
    logging_obj.pre_call = MagicMock()

    return await handler.async_completion(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        messages=[{"role": "user", "content": "weather in SF?"}],
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/model/x/converse",
        model_response=ModelResponse(),
        timeout=60.0,
        encoding=MagicMock(),
        logging_obj=logging_obj,
        stream=False,
        optional_params={"tools": [web_search_tool]},
        litellm_params={},
        credentials=MagicMock(),
        client=client,
    )


@pytest.mark.asyncio
async def test_bedrock_converse_runs_websearch_agentic_loop(monkeypatch):
    """Regression: web search interception must run on the Bedrock converse handler.
    Before the fix, async_completion returned the raw tool_calls response and the
    agentic loop never executed.
    """
    from litellm.integrations.websearch_interception.handler import (
        WebSearchInterceptionLogger,
    )

    initial_response = _web_search_model_response()
    final_response = ModelResponse(
        id="final-resp",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(role="assistant", content="It is sunny in SF."),
            )
        ],
        model="claude-sonnet-4",
        object="chat.completion",
        created=0,
    )

    _patch_converse_request(monkeypatch, initial_response)

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    websearch_logger._execute_search = AsyncMock(  # type: ignore[method-assign]
        return_value=("SF weather: sunny, 65F", None)
    )
    monkeypatch.setattr(litellm, "callbacks", [websearch_logger])

    follow_up = AsyncMock(return_value=final_response)
    monkeypatch.setattr(litellm, "acompletion", follow_up)

    handler = BedrockConverseLLM()
    result = await _call_async_completion(handler)

    follow_up.assert_awaited_once()
    assert result is final_response
    assert result.choices[0].message.content == "It is sunny in SF."
    assert not result.choices[0].message.tool_calls


@pytest.mark.asyncio
async def test_bedrock_converse_skips_loop_for_disabled_provider(monkeypatch):
    """When bedrock is not in enabled_providers, the converse handler must return
    the original response untouched (no follow-up call).
    """
    from litellm.integrations.websearch_interception.handler import (
        WebSearchInterceptionLogger,
    )

    initial_response = _web_search_model_response()
    _patch_converse_request(monkeypatch, initial_response)

    websearch_logger = WebSearchInterceptionLogger(enabled_providers=["openai"])
    monkeypatch.setattr(litellm, "callbacks", [websearch_logger])

    follow_up = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", follow_up)

    handler = BedrockConverseLLM()
    result = await _call_async_completion(handler)

    follow_up.assert_not_awaited()
    assert result is initial_response
