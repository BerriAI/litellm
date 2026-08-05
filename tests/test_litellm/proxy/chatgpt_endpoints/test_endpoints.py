import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.chatgpt_endpoints.endpoints import (
    ChatGPTSearchLitellmParams,
    resolve_chatgpt_search_target,
    router,
    target_from_litellm_params,
)


class StubRouter:
    def __init__(self, litellm_params: dict) -> None:
        self.litellm_params = litellm_params
        self.calls: list[tuple[str, dict]] = []

    async def async_get_available_deployment(self, model: str, request_kwargs: dict) -> dict:
        self.calls.append((model, request_kwargs))
        return {"litellm_params": self.litellm_params}


def test_alpha_search_routes_are_classified_as_openai_data_routes() -> None:
    assert "/alpha/search" in LiteLLMRoutes.openai_routes.value
    assert "/v1/alpha/search" in LiteLLMRoutes.openai_routes.value


@pytest.mark.asyncio
async def test_resolver_selects_configured_chatgpt_deployment_with_auth_context() -> None:
    auth = UserAPIKeyAuth(
        api_key="sk-test",
        team_id="team-123",
        allowed_model_region="eu",
    )
    llm_router = StubRouter(
        {
            "model": "chatgpt/gpt-5.6-sol",
            "api_base": "https://chatgpt.test/backend-api/codex",
            "timeout": 45,
        }
    )

    target = await resolve_chatgpt_search_target(
        requested_model="sol",
        llm_router=llm_router,
        user_api_key_dict=auth,
        user_model=None,
        user_api_base=None,
        user_request_timeout=60,
    )

    assert target.model == "gpt-5.6-sol"
    assert target.api_base == "https://chatgpt.test/backend-api/codex"
    assert target.timeout == 45
    assert llm_router.calls[0][0] == "sol"
    request_kwargs = llm_router.calls[0][1]
    assert request_kwargs["metadata"]["user_api_key_auth"] is auth
    assert request_kwargs["metadata"]["user_api_key_team_id"] == "team-123"
    assert request_kwargs["allowed_model_region"] == "eu"


def test_target_rejects_non_chatgpt_provider() -> None:
    with pytest.raises(HTTPException) as exc_info:
        target_from_litellm_params(
            requested_model="sol",
            litellm_params=ChatGPTSearchLitellmParams(model="openai/gpt-5.6-sol"),
            default_api_base=None,
            default_timeout=None,
        )
    assert exc_info.value.status_code == 400
    assert "requires a ChatGPT subscription model" in str(exc_info.value.detail)


@pytest.mark.parametrize("path", ["/alpha/search", "/v1/alpha/search"])
def test_endpoint_preserves_future_payload_and_upstream_response(path: str) -> None:
    auth = UserAPIKeyAuth(api_key="sk-test")
    llm_router = StubRouter(
        {
            "model": "chatgpt/gpt-5.6-sol",
            "api_base": "https://chatgpt.test/backend-api/codex",
        }
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    upstream_response = httpx.Response(
        status_code=200,
        headers={
            "content-type": "application/json",
            "set-cookie": "private=upstream",
            "x-codex-primary-used-percent": "4",
            "x-request-id": "request-123",
        },
        json={
            "encrypted_output": "ciphertext",
            "output": "search output",
            "results": [{"type": "text_result", "future": {"preserved": True}}],
        },
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", llm_router),
        patch("litellm.proxy.proxy_server.user_model", None),
        patch("litellm.proxy.proxy_server.user_api_base", None),
        patch("litellm.proxy.proxy_server.user_request_timeout", None),
        patch(
            "litellm.proxy.chatgpt_endpoints.endpoints.ChatGPTSearchHandler.search",
            new=AsyncMock(return_value=upstream_response),
        ) as search,
    ):
        response = TestClient(app).post(
            path,
            headers={
                "Authorization": "Bearer sk-proxy",
                "originator": "codex_vscode",
                "x-codex-turn-metadata": '{"turn_id":"turn-123"}',
            },
            json={
                "id": "session-123",
                "model": "sol",
                "commands": {"search_query": [{"q": "LiteLLM"}]},
                "future_request_field": {"preserved": True},
            },
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["future"] == {"preserved": True}
    assert response.headers["x-codex-primary-used-percent"] == "4"
    assert response.headers["x-request-id"] == "request-123"
    assert "set-cookie" not in response.headers
    call = search.await_args.kwargs
    assert json.loads(call["payload"]) == {
        "id": "session-123",
        "model": "gpt-5.6-sol",
        "commands": {"search_query": [{"q": "LiteLLM"}]},
        "future_request_field": {"preserved": True},
    }
    assert call["session_id"] == "session-123"
    assert call["api_base"] == "https://chatgpt.test/backend-api/codex"
    assert call["extra_headers"] == {
        "originator": "codex_vscode",
        "x-codex-turn-metadata": '{"turn_id":"turn-123"}',
    }


@pytest.mark.parametrize("payload", [[], {}, {"model": 7}, {"model": ""}])
def test_endpoint_rejects_invalid_payload_before_routing(payload: object) -> None:
    auth = UserAPIKeyAuth(api_key="sk-test")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth

    with patch(
        "litellm.proxy.chatgpt_endpoints.endpoints.ChatGPTSearchHandler.search",
        new=AsyncMock(),
    ) as search:
        response = TestClient(app).post("/v1/alpha/search", json=payload)

    assert response.status_code == 400
    search.assert_not_awaited()
