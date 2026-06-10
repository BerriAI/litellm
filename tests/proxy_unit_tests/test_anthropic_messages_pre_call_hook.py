import asyncio
import os
from unittest import mock

import litellm
import pytest
from fastapi.testclient import TestClient

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import app, initialize

EXAMPLE_ANTHROPIC_MESSAGES_RESULT = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from LiteLLM"}],
    "model": "gpt-3.5-turbo",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 5},
}


def mock_patch_anthropic_messages():
    return mock.patch(
        "litellm.proxy.proxy_server.llm_router.anthropic_messages",
        return_value=EXAMPLE_ANTHROPIC_MESSAGES_RESULT,
    )


@pytest.fixture(scope="function")
def fake_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_api_key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://fake-openai-api-base")
    monkeypatch.setenv("AZURE_AI_API_BASE", "http://fake-azure-api-base")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake_azure_api_key")
    monkeypatch.setenv("AZURE_SWEDEN_API_BASE", "http://fake-azure-sweden-api-base")
    monkeypatch.setenv("REDIS_HOST", "localhost")


@pytest.fixture(scope="function")
def client_no_auth(fake_env_vars):
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    cleanup_router_config_variables()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(test_dir, "test_configs", "test_config_no_auth.yaml")
    asyncio.run(initialize(config=config_path, debug=True))
    return TestClient(app)


@mock_patch_anthropic_messages()
def test_anthropic_messages_runs_proxy_async_pre_call_hook(
    mock_anthropic_messages, client_no_auth, monkeypatch
):
    hook_calls = []

    class AnthropicMessagesPreCallHook(CustomLogger):
        async def async_pre_call_hook(
            self, user_api_key_dict, cache, data, call_type, **kwargs
        ):
            hook_calls.append(call_type)
            data["metadata"] = {**(data.get("metadata") or {}), "source": "unit-test"}
            return data

    monkeypatch.setattr(litellm, "callbacks", [AnthropicMessagesPreCallHook()])

    response = client_no_auth.post(
        "/v1/messages",
        json={
            "model": "test_openai_models",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "Hello from LiteLLM"
    assert hook_calls == ["anthropic_messages"]
    mock_anthropic_messages.assert_called_once()
    metadata = mock_anthropic_messages.call_args.kwargs.get("metadata")
    assert metadata == {"source": "unit-test"}


@pytest.mark.asyncio
async def test_experimental_anthropic_messages_runs_proxy_async_pre_call_hook(
    monkeypatch,
):
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        anthropic_messages,
    )

    hook_calls = []

    class AnthropicMessagesPreCallHook(CustomLogger):
        async def async_pre_call_hook(
            self, user_api_key_dict, cache, data, call_type, **kwargs
        ):
            hook_calls.append((call_type, data["model"]))
            data["metadata"] = {
                **(data.get("metadata") or {}),
                "source": "experimental-unit-test",
            }
            data["temperature"] = 0.2
            return data

    monkeypatch.setattr(litellm, "callbacks", [AnthropicMessagesPreCallHook()])
    monkeypatch.setattr(
        litellm, "use_chat_completions_url_for_anthropic_messages", True
    )

    with mock.patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.handler.anthropic_messages_handler",
        return_value=EXAMPLE_ANTHROPIC_MESSAGES_RESULT,
    ) as mock_handler:
        response = await anthropic_messages(
            model="openai/gpt-4o-mini",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"existing": "keep"},
            custom_llm_provider="openai",
        )

    assert response == EXAMPLE_ANTHROPIC_MESSAGES_RESULT
    assert hook_calls == [("anthropic_messages", "openai/gpt-4o-mini")]
    mock_handler.assert_called_once()
    assert mock_handler.call_args.kwargs["metadata"] == {
        "existing": "keep",
        "source": "experimental-unit-test",
    }
    assert mock_handler.call_args.kwargs["temperature"] == 0.2
