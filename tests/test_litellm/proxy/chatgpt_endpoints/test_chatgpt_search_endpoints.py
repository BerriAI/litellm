from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLMRoutes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.chatgpt_endpoints.endpoints import router


def test_alpha_search_routes_are_classified_as_openai_data_routes() -> None:
    assert "/alpha/search" in LiteLLMRoutes.openai_routes.value
    assert "/v1/alpha/search" in LiteLLMRoutes.openai_routes.value


@pytest.mark.parametrize("path", ["/alpha/search", "/v1/alpha/search"])
def test_endpoint_uses_shared_passthrough_lifecycle(path: str) -> None:
    auth = UserAPIKeyAuth(api_key="sk-test")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    captured: dict = {}
    upstream_request = httpx.Request("POST", "https://chatgpt.test/backend-api/codex/alpha/search")
    upstream_response = httpx.Response(
        status_code=200,
        headers={
            "content-type": "application/json",
            "set-cookie": "private=upstream",
            "x-codex-primary-used-percent": "4",
            "x-litellm-call-id": "call-123",
            "x-request-id": "request-123",
        },
        json={"results": [{"future": {"preserved": True}}]},
        request=upstream_request,
    )

    async def route_request(**kwargs):
        captured.update(kwargs)

        async def send_request() -> httpx.Response:
            return upstream_response

        return send_request()

    with patch("litellm.proxy.common_request_processing.route_request", new=route_request):
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
    UUID(response.headers["x-litellm-call-id"])
    assert response.headers["x-request-id"] == "request-123"
    assert "set-cookie" not in response.headers
    assert captured["route_type"] == "allm_passthrough_route"
    assert captured["user_api_key_dict"] is auth
    assert captured["data"]["model"] == "sol"
    assert captured["data"]["method"] == "POST"
    assert captured["data"]["endpoint"] == "alpha/search"
    assert captured["data"]["required_custom_llm_provider"] == "chatgpt"
    assert captured["data"]["json"] == {
        "id": "session-123",
        "model": "sol",
        "commands": {"search_query": [{"q": "LiteLLM"}]},
        "future_request_field": {"preserved": True},
    }


@pytest.mark.parametrize("payload", [[], {}, {"model": 7}, {"model": ""}, {"model": "   "}])
def test_endpoint_rejects_invalid_payload_before_processing(payload: object) -> None:
    auth = UserAPIKeyAuth(api_key="sk-test")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth

    with patch("litellm.proxy.common_request_processing.route_request") as route_request:
        response = TestClient(app).post("/v1/alpha/search", json=payload)

    assert response.status_code == 400
    route_request.assert_not_called()


def test_endpoint_runs_shared_failure_lifecycle() -> None:
    auth = UserAPIKeyAuth(api_key="sk-test")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    failure_hook = AsyncMock(return_value=None)
    response_headers_hook = AsyncMock(return_value={"set-cookie": "private=upstream", "x-failure-hook": "called"})

    async def route_request(**kwargs):
        raise RuntimeError("search failed")

    with (
        patch("litellm.proxy.common_request_processing.route_request", new=route_request),
        patch.object(proxy_server.proxy_logging_obj, "post_call_failure_hook", new=failure_hook),
        patch.object(
            proxy_server.proxy_logging_obj,
            "post_call_response_headers_hook",
            new=response_headers_hook,
        ),
        pytest.raises(ProxyException, match="search failed") as exc_info,
    ):
        TestClient(app).post("/v1/alpha/search", json={"model": "sol"})

    failure_hook.assert_awaited_once()
    response_headers_hook.assert_awaited_once()
    assert exc_info.value.code == "500"
    assert "set-cookie" not in exc_info.value.headers
    assert exc_info.value.headers["x-failure-hook"] == "called"
