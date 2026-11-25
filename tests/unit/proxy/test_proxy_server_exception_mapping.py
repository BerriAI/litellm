import asyncio
import json
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
import respx
from fastapi.testclient import TestClient

CONFIG_DIR: Final = Path(__file__).resolve().parent / "test_configs"
OPENAI_CHAT_COMPLETIONS: Final = "https://api.openai.com/v1/chat/completions"
AZURE_CHAT_COMPLETIONS: Final = (
    "https://azure-unit.example.com/openai/deployments/gpt-4.1-mini/chat/completions"
)
AZURE_EMBEDDINGS: Final = (
    "https://azure-unit.example.com/openai/deployments/text-embedding-ada-002/embeddings"
)

AZURE_401_BODY: Final = {
    "error": {"code": "401", "message": "Access denied due to invalid subscription key."}
}
CONTEXT_LENGTH_EXCEEDED_BODY: Final = {
    "error": {
        "message": "This model's maximum context length is 4096 tokens. However, your messages resulted in 10007 tokens. Please reduce the length of the messages.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, httpx_transport: None) -> TestClient:
    from litellm.proxy.proxy_server import (
        app,
        cleanup_router_config_variables,
        initialize,
    )

    monkeypatch.setenv("LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY", "true")
    cleanup_router_config_variables()
    asyncio.run(initialize(config=str(CONFIG_DIR / "test_bad_config.yaml")))
    return TestClient(app)


def _openai_exception(response: httpx.Response) -> openai.APIStatusError:
    return openai.OpenAI(api_key="anything")._make_status_error_from_response(
        response=response
    )


def test_chat_completion_exception(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.post(OPENAI_CHAT_COMPLETIONS).mock(
        return_value=httpx.Response(
            401,
            json={
                "error": {
                    "message": "Incorrect API key provided: bad-key. You can find your API key at https://platform.openai.com/account/api-keys.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    )

    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )

    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    assert "litellm.AuthenticationError: AuthenticationError" in json_response["error"]["message"]
    assert type(json_response["error"]["code"]) is str
    assert isinstance(_openai_exception(response), openai.AuthenticationError)


def test_chat_completion_exception_azure(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post(AZURE_CHAT_COMPLETIONS).mock(
        return_value=httpx.Response(401, json=AZURE_401_BODY)
    )

    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "azure-gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )

    assert response.status_code == 401, response.text
    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    assert "litellm.AuthenticationError" in json_response["error"]["message"]
    assert upstream.call_count == 1
    outbound: Final = upstream.calls[0].request
    assert json.loads(outbound.content)["messages"] == [
        {"role": "user", "content": "hi"}
    ]
    assert outbound.headers["api-key"] == "bad-key"
    assert isinstance(_openai_exception(response), openai.AuthenticationError)


def test_embedding_auth_exception_azure(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post(AZURE_EMBEDDINGS).mock(
        return_value=httpx.Response(401, json=AZURE_401_BODY)
    )

    response: Final = client.post(
        "/embeddings",
        json={"model": "azure-embedding", "input": ["hi"]},
    )

    assert response.status_code == 401, response.text
    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    assert upstream.call_count == 1
    outbound: Final = upstream.calls[0].request
    assert json.loads(outbound.content)["input"] == ["hi"]
    assert outbound.headers["api-key"] == "bad-key"
    assert isinstance(_openai_exception(response), openai.AuthenticationError)


def test_exception_openai_bad_model(client: TestClient) -> None:
    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "azure/GPT-12",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )

    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    assert isinstance(_openai_exception(response), openai.BadRequestError)


def test_chat_completion_exception_any_model(client: TestClient) -> None:
    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "Lite-GPT-12",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        },
    )

    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    openai_exception: Final = _openai_exception(response)
    assert isinstance(openai_exception, openai.BadRequestError)
    assert "/chat/completions: Invalid model name passed in model=Lite-GPT-12" in str(
        openai_exception.message
    )


def test_embedding_exception_any_model(client: TestClient) -> None:
    response: Final = client.post(
        "/embeddings",
        json={"model": "Lite-GPT-12", "input": ["hi"]},
    )

    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    openai_exception: Final = _openai_exception(response)
    assert isinstance(openai_exception, openai.BadRequestError)
    assert "/embeddings: Invalid model name passed in model=Lite-GPT-12" in str(
        openai_exception.message
    )


def test_chat_completion_exception_azure_context_window(
    client: TestClient, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.post(AZURE_CHAT_COMPLETIONS).mock(
        return_value=httpx.Response(400, json=CONTEXT_LENGTH_EXCEEDED_BODY)
    )

    response: Final = client.post(
        "/chat/completions",
        json={
            "model": "working-azure-gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi" * 10000}],
            "max_tokens": 10,
        },
    )

    assert response.status_code == 400, response.text
    json_response: Final = response.json()
    assert json_response.keys() == {"error"}
    assert upstream.call_count == 1
    error: Final = json_response["error"]
    assert "ContextWindowExceededError" in error["message"]
    assert "maximum context length is 4096 tokens" in error["message"]
    assert type(error["code"]) is str
    assert isinstance(_openai_exception(response), openai.BadRequestError)
