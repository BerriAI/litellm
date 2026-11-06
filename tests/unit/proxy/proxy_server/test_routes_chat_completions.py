"""Behavior pins for ``proxy_server.py`` chat-completions routes.

Pins (PR2):
    - POST /v1/chat/completions
    - POST /chat/completions
    - POST /engines/{model:path}/chat/completions
    - POST /openai/deployments/{model:path}/chat/completions
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import common_request_processing, proxy_server

from .conftest import normalize  # type: ignore[import-not-found]

HAPPY_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello from mock"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
def patched_chat(monkeypatch):
    """Stub chat-completions pipeline at ProxyBaseLLMRequestProcessing."""
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))

    async def _fake_process(self, *args, **kwargs):
        return dict(HAPPY_RESPONSE)

    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        _fake_process,
    )
    yield


@pytest.fixture
def patched_chat_error(monkeypatch):
    """Variant that makes the pipeline raise -> 400 via _handle_llm_api_exception."""
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", MagicMock(post_call_failure_hook=AsyncMock()))

    from litellm.proxy._types import ProxyException

    async def _raise(self, *args, **kwargs):
        raise ValueError("boom")

    async def _handler(self, *, e, user_api_key_dict, proxy_logging_obj):
        return ProxyException(message="boom", type="bad_request_error", param="model", code=400)

    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        _raise,
    )
    monkeypatch.setattr(
        common_request_processing.ProxyBaseLLMRequestProcessing,
        "handle_llm_api_exception",
        _handler,
    )
    yield


_CHAT_PATHS = [
    "/v1/chat/completions",
    "/chat/completions",
    "/engines/gpt-4/chat/completions",
    "/openai/deployments/gpt-4/chat/completions",
]


@pytest.mark.parametrize("path", _CHAT_PATHS)
def test_chat_completion_happy_path(client, auth_as, patched_chat, path):
    """Pins all four ``POST .../chat/completions`` aliases (happy path).

    Covers ``POST /v1/chat/completions``, ``POST /chat/completions``,
    ``POST /engines/{model:path}/chat/completions``, and
    ``POST /openai/deployments/{model:path}/chat/completions``.
    """
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert normalize(response.json()) == {
        "id": "<VOLATILE>",
        "object": "chat.completion",
        "created": "<VOLATILE>",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello from mock"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize("path", _CHAT_PATHS)
def test_chat_completion_pipeline_error(client, auth_as, patched_chat_error, path):
    """Pins all four ``POST .../chat/completions`` aliases (error: 400).

    Covers ``POST /v1/chat/completions``, ``POST /chat/completions``,
    ``POST /engines/{model:path}/chat/completions``, and
    ``POST /openai/deployments/{model:path}/chat/completions``.
    """
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 400
    assert "error" in response.json() or response.text != ""


@pytest.mark.parametrize("path", _CHAT_PATHS)
@pytest.mark.parametrize("enabled", [False, True])
def test_workflow_invocations_require_explicit_opt_in(client, auth_as, monkeypatch, path, enabled):
    from litellm.proxy.deps import database
    from litellm.proxy.management_endpoints import workflow_execution_endpoints

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True} if enabled else {})
    invocation = AsyncMock(return_value=dict(HAPPY_RESPONSE))
    connection = AsyncMock(return_value=None)
    monkeypatch.setattr(workflow_execution_endpoints, "invoke_workflow", invocation)
    monkeypatch.setattr(database, "get_database", connection)
    with auth_as():
        response = client.post(
            path,
            json={"model": "workflow/email-processing", "messages": [{"role": "user", "content": "Hello"}]},
        )
    if enabled:
        assert response.status_code == 200
        invocation.assert_awaited_once()
        connection.assert_awaited_once()
    else:
        assert response.status_code == 404
        invocation.assert_not_awaited()
        connection.assert_not_awaited()
