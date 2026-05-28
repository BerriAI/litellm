"""
Tests for `/v1/responses/input_tokens` endpoint.

Regression for LIT-2828: prior to this fix, POSTing to
`/v1/responses/input_tokens` returned HTTP 405 "Method Not Allowed" because
only `/v1/responses/{response_id}` was registered (as GET/DELETE), and the
path-parameter route does not match POST.
"""

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app

_AUTH = {"Authorization": "Bearer sk-1234"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _post(client: TestClient, path: str, body: Dict[str, Any]):
    return client.post(path, json=body, headers=_AUTH)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/responses/input_tokens",
        "/responses/input_tokens",
        "/openai/v1/responses/input_tokens",
    ],
)
def test_responses_input_tokens_string_input_returns_200(client, path):
    """All three route aliases accept a string `input` and return a count."""
    r = _post(
        client,
        path,
        {"model": "openai/gpt-4o", "input": "Hello there, how are you?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "input_tokens" in body
    assert isinstance(body["input_tokens"], int)
    assert body["input_tokens"] > 0


def test_responses_input_tokens_input_items_list_with_instructions(client):
    """Input-items array form (`input=[{role,content}]`) plus instructions
    flows through the Responses-API -> Chat Completion message translation."""
    r = _post(
        client,
        "/v1/responses/input_tokens",
        {
            "model": "openai/gpt-4o",
            "instructions": "You are a helpful assistant.",
            "input": [{"role": "user", "content": "What is the capital of France?"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_tokens"] > 0


def test_responses_input_tokens_with_tools_does_not_500(client):
    """Tools schema is forwarded to the token counter without breakage."""
    r = _post(
        client,
        "/v1/responses/input_tokens",
        {
            "model": "openai/gpt-4o",
            "input": "What is the weather?",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    assert "input_tokens" in r.json()


def test_responses_input_tokens_missing_model_returns_400(client):
    r = _post(client, "/v1/responses/input_tokens", {"input": "hi"})
    assert r.status_code == 400, r.text
    assert "model" in r.text


def test_responses_input_tokens_missing_input_returns_400(client):
    r = _post(client, "/v1/responses/input_tokens", {"model": "openai/gpt-4o"})
    assert r.status_code == 400, r.text
    # Updated wording: "input parameter is required (or instructions)".
    assert "input" in r.text
    assert "instructions" in r.text


def test_responses_input_tokens_route_registered_in_openai_routes_list():
    """Without explicit registration in `LiteLLMRoutes.openai_routes`,
    virtual-key allow-list checks would still 401 even after the route is
    served by FastAPI. Regression-guard that the route is whitelisted."""
    from litellm.proxy._types import LiteLLMRoutes

    routes = list(LiteLLMRoutes.openai_routes.value)
    assert "/v1/responses/input_tokens" in routes
    assert "/responses/input_tokens" in routes
    assert "/openai/v1/responses/input_tokens" in routes


# ---------------------------------------------------------------------------
# Explicit-mock fixture: the endpoint imports `token_counter` lazily from
# `litellm.proxy.proxy_server`. Patch the symbol on that module so the
# mocked function is what the endpoint resolves at request time. With this
# fixture active, the tests never call the real `litellm.token_counter()`
# (no local-tokenizer or provider-endpoint side effects), so behaviour is
# deterministic regardless of router state.
# ---------------------------------------------------------------------------


from unittest.mock import patch, AsyncMock

from litellm.types.utils import TokenCountResponse


@pytest.fixture
def mock_token_counter():
    async def _fake_token_counter(request, call_endpoint=False):
        # Token count proportional to a simple string-length of all message
        # contents (deterministic, so assertions can be specific).
        n = 0
        for m in request.messages or []:
            content = m.get("content")
            if isinstance(content, str):
                n += max(1, len(content) // 4)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        n += max(1, len(part["text"]) // 4)
        return TokenCountResponse(
            total_tokens=n,
            request_model=request.model,
            model_used=request.model,
            tokenizer_type="mocked",
        )

    with patch(
        "litellm.proxy.proxy_server.token_counter",
        new=AsyncMock(side_effect=_fake_token_counter),
    ) as m:
        yield m


def test_responses_input_tokens_mocked_endpoint_returns_expected_count(
    client, mock_token_counter
):
    """With the internal token_counter mocked, the endpoint must surface the
    mock’s `total_tokens` verbatim in the `input_tokens` field. Locks the
    contract independently of any live tokenizer."""
    r = _post(
        client,
        "/v1/responses/input_tokens",
        {"model": "openai/gpt-4o", "input": "abcd" * 10},  # 40 chars -> 10 tokens
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"input_tokens": 10}
    assert mock_token_counter.await_count == 1
