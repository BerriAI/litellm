"""Gateway coverage for Parallel AI's Responses integration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

PARALLEL_RESPONSES_URL: Final = "https://api.parallel.ai/v1/responses"


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy_server.app, raise_server_exceptions=False)


@pytest.fixture
def auth_as() -> Iterator[None]:
    async def _authorized_request() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="hashed-sk-test",
            user_id="parallel-test-user",
        )

    previous: Final = proxy_server.app.dependency_overrides.get(user_api_key_auth)
    proxy_server.app.dependency_overrides[user_api_key_auth] = _authorized_request
    try:
        yield
    finally:
        if previous is None:
            proxy_server.app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            proxy_server.app.dependency_overrides[user_api_key_auth] = previous


def _parallel_response_body() -> dict[str, object]:
    return {
        "id": "resp_parallel_gateway",
        "object": "response",
        "created_at": 1700000000,
        "model": "parallel",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_parallel_gateway",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Parallel grounded answer",
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _parallel_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "parallel-gateway",
                "litellm_params": {
                    "model": "parallel_ai/parallel",
                    "api_key": "parallel-responses-key",
                    "use_in_pass_through": True,
                },
            }
        ],
        num_retries=0,
    )


def _mock_async_post(
    monkeypatch,
    *,
    url: str,
    response_body: dict[str, object],
) -> AsyncMock:
    response = httpx.Response(
        status_code=200,
        json=response_body,
        request=httpx.Request("POST", url),
    )
    mock_post = AsyncMock(return_value=response)
    monkeypatch.setattr(AsyncHTTPHandler, "post", mock_post)
    return mock_post


def test_parallel_responses_gateway_route(client, auth_as, monkeypatch):
    """The primary LLM route reaches Parallel's native Responses endpoint."""
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", _parallel_router())
    mock_post = _mock_async_post(
        monkeypatch,
        url=PARALLEL_RESPONSES_URL,
        response_body=_parallel_response_body(),
    )

    response = client.post(
        "/v1/responses",
        json={"model": "parallel-gateway", "input": "Research Parallel"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["output"][0]["content"][0]["text"] == "Parallel grounded answer"

    request_kwargs = mock_post.await_args.kwargs
    assert request_kwargs["url"] == PARALLEL_RESPONSES_URL
    assert request_kwargs["headers"]["Authorization"] == "Bearer parallel-responses-key"
    assert request_kwargs["json"] == {
        "model": "parallel",
        "input": "Research Parallel",
    }
