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
    mock_anthropic_messages, client_no_auth
):
    hook_calls = []

    class AnthropicMessagesPreCallHook(CustomLogger):
        async def async_pre_call_hook(
            self, user_api_key_dict, cache, data, call_type, **kwargs
        ):
            hook_calls.append(call_type)
            data["metadata"] = {**(data.get("metadata") or {}), "source": "unit-test"}
            return data

    original_callbacks = litellm.callbacks
    litellm.callbacks = [AnthropicMessagesPreCallHook()]

    try:
        response = client_no_auth.post(
            "/v1/messages",
            json={
                "model": "gpt-3.5-turbo",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "Hello from LiteLLM"
        assert hook_calls == ["anthropic_messages"]
        mock_anthropic_messages.assert_called_once()
        assert (
            mock_anthropic_messages.call_args.kwargs["metadata"]["source"]
            == "unit-test"
        )
    finally:
        litellm.callbacks = original_callbacks
