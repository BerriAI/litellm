"""The managed-agent proxy endpoints default to Gemini but let ``litellm_params_template``
name another provider, including on the GET / DELETE routes that carry it as a query parameter."""

import json
from urllib.parse import quote

import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

import litellm
from litellm import Router
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.google_endpoints import agents_endpoints

ANTHROPIC_AGENT = "https://api.anthropic.com/v1/agents/agent_123"
GEMINI_AGENT = "https://generativelanguage.googleapis.com/v1beta/agents/agent_123"


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
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    monkeypatch.setattr(
        agents_endpoints,
        "_proxy_server_imports",
        lambda: {
            "general_settings": {},
            "llm_router": Router(model_list=[]),
            "proxy_config": None,
            "proxy_logging_obj": _NoopProxyLogging(),
            "select_data_generator": None,
            "user_api_base": None,
            "user_max_tokens": None,
            "user_model": None,
            "user_request_timeout": None,
            "user_temperature": None,
            "version": "0.0.0",
        },
    )
    app = FastAPI()
    app.include_router(agents_endpoints.router)

    async def _admin():
        return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN)

    app.dependency_overrides[user_api_key_auth] = _admin
    return TestClient(app)


def _template(**fields) -> str:
    return quote(json.dumps(fields), safe="")


def _anthropic_agent(version: int = 3) -> dict:
    return {
        "type": "agent",
        "id": "agent_123",
        "name": "support-bot",
        "version": version,
        "model": {"id": "claude-haiku-4-5"},
    }


@respx.mock
def test_get_agent_reaches_anthropic_when_the_template_names_it(client: TestClient):
    upstream = respx.get(ANTHROPIC_AGENT).mock(return_value=Response(200, json=_anthropic_agent()))

    response = client.get(
        f"/v1beta/agents/agent_123?litellm_params_template={_template(custom_llm_provider='anthropic', api_key='sk-ant')}"
    )

    assert response.status_code == 200
    assert response.json()["id"] == "agent_123"
    assert response.json()["version"] == 3
    sent = upstream.calls.last.request
    assert sent.headers["x-api-key"] == "sk-ant"
    assert sent.headers["anthropic-beta"] == "managed-agents-2026-04-01"


@respx.mock
def test_get_agent_defaults_to_gemini(client: TestClient):
    upstream = respx.get(GEMINI_AGENT).mock(return_value=Response(200, json={"id": "agent_123", "name": "agent_123"}))

    response = client.get(f"/v1beta/agents/agent_123?litellm_params_template={_template(api_key='AIza-test')}")

    assert response.status_code == 200
    assert upstream.calls.last.request.headers["x-goog-api-key"] == "AIza-test"


@respx.mock
def test_list_versions_reaches_anthropic_when_the_template_names_it(client: TestClient):
    upstream = respx.get(f"{ANTHROPIC_AGENT}/versions").mock(
        return_value=Response(200, json={"data": [_anthropic_agent(1), _anthropic_agent(2)], "next_page": None})
    )

    response = client.get(
        f"/v1beta/agents/agent_123/versions?litellm_params_template={_template(custom_llm_provider='anthropic', api_key='sk-ant')}"
    )

    assert response.status_code == 200
    assert [v["version"] for v in response.json()["agent_versions"]] == [1, 2]
    assert upstream.calls.last.request.headers["anthropic-beta"] == "managed-agents-2026-04-01"


@respx.mock
def test_list_agents_reaches_anthropic_when_the_template_names_it(client: TestClient):
    upstream = respx.get("https://api.anthropic.com/v1/agents").mock(
        return_value=Response(200, json={"data": [_anthropic_agent()], "next_page": "page_2"})
    )

    response = client.get(
        f"/v1beta/agents?litellm_params_template={_template(custom_llm_provider='anthropic', api_key='sk-ant', page_size=1)}"
    )

    assert response.status_code == 200
    assert response.json()["next_page_token"] == "page_2"
    assert upstream.calls.last.request.url.params["limit"] == "1"


def test_delete_agent_is_refused_for_anthropic_with_the_archive_hint(client: TestClient):
    with pytest.raises(ProxyException) as excinfo:
        client.delete(
            f"/v1beta/agents/agent_123?litellm_params_template={_template(custom_llm_provider='anthropic', api_key='sk-ant')}"
        )

    assert excinfo.value.code == "400"
    assert f"POST {ANTHROPIC_AGENT}/archive" in excinfo.value.message
