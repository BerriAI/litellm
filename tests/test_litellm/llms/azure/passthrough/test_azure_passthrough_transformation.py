import json
import os
import sys
from unittest.mock import MagicMock

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.azure.passthrough.transformation import AzurePassthroughConfig
from litellm.types.utils import ModelResponse


def _azure_chat_completion_body():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4.1-mini-2025-04-14",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


def _make_httpx_response(body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request(
            "POST",
            "https://example.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions",
        ),
    )


def test_azure_passthrough_logging_non_streaming_response_chat_completions():
    """
    Returns a populated ModelResponse (with usage + content) for a chat/completions
    endpoint. This is what _success_handler_helper_fn needs to build
    standard_logging_object — without it, Datadog/cost-tracking/router-success all
    raise on every Azure passthrough request.
    """
    config = AzurePassthroughConfig()
    logging_obj = MagicMock()

    result = config.logging_non_streaming_response(
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        httpx_response=_make_httpx_response(_azure_chat_completion_body()),
        request_data={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        logging_obj=logging_obj,
        endpoint="openai/deployments/gpt-4.1-mini/chat/completions",
    )

    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello! How can I assist you today?"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 18


def test_azure_passthrough_logging_non_streaming_response_unknown_endpoint_returns_none():
    """
    Endpoints other than chat/completions (responses, messages, images) fall
    through to None — matches base-class behavior and Bedrock's "unknown
    endpoint" handling. Not a regression; just scoping.
    """
    config = AzurePassthroughConfig()
    logging_obj = MagicMock()

    result = config.logging_non_streaming_response(
        model="gpt-4.1-mini",
        custom_llm_provider="azure",
        httpx_response=_make_httpx_response(_azure_chat_completion_body()),
        request_data={},
        logging_obj=logging_obj,
        endpoint="openai/responses",
    )

    assert result is None
