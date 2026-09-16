"""
Test for google_endpoints/endpoints.py
"""

import pytest
import sys, os
from dotenv import load_dotenv


from litellm.proxy.google_endpoints.endpoints import google_count_tokens
from litellm.types.llms.vertex_ai import TokenCountDetailsResponse
from starlette.requests import Request

load_dotenv()


@pytest.mark.asyncio
async def test_proxy_gemini_to_openai_like_model_token_counting():
    """
    Test the token counting endpoint for proxing gemini to openai-like models.
    """
    response: TokenCountDetailsResponse = await google_count_tokens(
        request=Request(
            scope={
                "type": "http",
                "parsed_body": (
                    ["contents"],
                    {"contents": [{"parts": [{"text": "Hello, how are you?"}]}]},
                ),
            }
        ),
        model_name="volcengine/foo",
    )

    assert response.get("totalTokens") > 0


SESSION_ID = "sesn_011CZkZAtmR3yMPDzynEDxu7"
ANTHROPIC_SESSION = f"https://api.anthropic.com/v1/sessions/{SESSION_ID}"
GEMINI_INTERACTION = f"https://generativelanguage.googleapis.com/v1beta/interactions/{SESSION_ID}"


class _NoopProxyLogging:
    async def pre_call_hook(self, user_api_key_dict, data, call_type):
        return data

    async def during_call_hook(self, *args, **kwargs):
        return None

    async def post_call_success_hook(self, data, user_api_key_dict, response):
        return response

    async def post_call_response_headers_hook(self, *args, **kwargs):
        return {}

    async def post_call_failure_hook(self, *args, **kwargs):
        return kwargs.get("original_exception")

    async def update_request_status(self, *args, **kwargs):
        return None

    async def _arelease_max_parallel_requests_on_disconnect(self, *args, **kwargs):
        return None


@pytest.fixture
def interactions_client(monkeypatch: pytest.MonkeyPatch):
    import litellm
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm import Router
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.google_endpoints import endpoints as google_endpoints
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setattr(proxy_server, "llm_router", Router(model_list=[]))
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", _NoopProxyLogging())
    monkeypatch.setattr(proxy_server, "general_settings", {})
    app = FastAPI()
    app.include_router(google_endpoints.router)

    async def _admin():
        return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN)

    app.dependency_overrides[user_api_key_auth] = _admin
    return TestClient(app)


def test_get_interaction_reaches_the_provider_named_in_the_query(interactions_client):
    import respx
    from httpx import Response

    with respx.mock:
        session = respx.get(ANTHROPIC_SESSION).mock(
            return_value=Response(
                200,
                json={
                    "id": SESSION_ID,
                    "status": "idle",
                    "agent": {"id": "agent_1", "model": {"id": "claude-haiku-4-5"}},
                },
            )
        )
        respx.get(f"{ANTHROPIC_SESSION}/events").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {"id": "sevt_idle", "type": "session.status_idle", "stop_reason": {"type": "requires_action"}}
                    ],
                    "next_page": None,
                },
            )
        )

        response = interactions_client.get(f"/v1beta/interactions/{SESSION_ID}?custom_llm_provider=anthropic")

    assert response.status_code == 200
    assert response.json()["status"] == "requires_action"
    assert response.json()["model"] == "claude-haiku-4-5"
    assert session.calls.last.request.headers["anthropic-beta"] == "managed-agents-2026-04-01"


def test_get_interaction_still_defaults_to_gemini(interactions_client):
    import respx
    from httpx import Response

    with respx.mock:
        gemini = respx.get(GEMINI_INTERACTION).mock(
            return_value=Response(200, json={"id": SESSION_ID, "status": "completed", "steps": []})
        )
        response = interactions_client.get(f"/v1beta/interactions/{SESSION_ID}")

    assert response.status_code == 200
    assert gemini.calls.last.request.headers["x-goog-api-key"] == "AIza-test"


def test_cancel_interaction_reaches_the_provider_named_in_the_query(interactions_client):
    import respx
    from httpx import Response

    with respx.mock:
        events = respx.post(f"{ANTHROPIC_SESSION}/events").mock(return_value=Response(200, json={"data": []}))
        response = interactions_client.post(f"/v1beta/interactions/{SESSION_ID}/cancel?custom_llm_provider=anthropic")

    assert response.status_code == 200
    assert response.json() == {"id": SESSION_ID, "status": "in_progress"}
    assert events.calls.last.request.content == b'{"events":[{"type":"user.interrupt"}]}'


def test_delete_interaction_reaches_the_provider_named_in_the_query(interactions_client):
    import respx
    from httpx import Response

    with respx.mock:
        delete = respx.delete(ANTHROPIC_SESSION).mock(
            return_value=Response(200, json={"id": SESSION_ID, "type": "session_deleted"})
        )
        response = interactions_client.delete(f"/v1beta/interactions/{SESSION_ID}?custom_llm_provider=anthropic")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert delete.called
