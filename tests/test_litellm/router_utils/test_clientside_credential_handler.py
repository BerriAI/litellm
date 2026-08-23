"""OAuth client-credentials deployment config must not be forwarded to a
client-redirected api_base.

Regression for the security finding on PR #31026: when a deployment permits
clientside api_base override, the router must clear the admin's OAuth flag and
fields so the bearer token litellm mints with them is never sent to the caller's
endpoint.
"""

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.router_utils.clientside_credential_handler import (
    _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE,
    get_dynamic_litellm_params,
)

_OAUTH_FIELDS = (
    "oauth_client_credentials",
    "oauth_token_url",
    "oauth_client_id",
    "oauth_client_secret",
    "oauth_scope",
)


def _deployment_params() -> dict:
    return {
        "model": "openai/gpt-4o",
        "api_base": "https://gateway.internal/v1",
        "oauth_client_credentials": True,
        "oauth_token_url": "https://idp.internal/oauth/token",
        "oauth_client_id": "admin-id",
        "oauth_client_secret": "admin-secret",
        "oauth_scope": "admin-scope",
    }


def test_oauth_fields_in_base_override_clear_list():
    for field in _OAUTH_FIELDS:
        assert field in _ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE


def test_oauth_creds_cleared_when_client_overrides_api_base():
    params = get_dynamic_litellm_params(
        _deployment_params(), {"api_base": "https://client.example/v1"}
    )
    assert params["api_base"] == "https://client.example/v1"
    for field in _OAUTH_FIELDS:
        assert field not in params


def test_oauth_creds_preserved_without_base_override():
    params = get_dynamic_litellm_params(_deployment_params(), {"temperature": 0.5})
    for field in _OAUTH_FIELDS:
        assert field in params


OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "oauth-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-admin",
                    **_deployment_params(),
                },
            }
        ]
    )


@pytest.mark.asyncio
async def test_router_base_override_mints_nothing_for_client_upstream(monkeypatch, respx_mock: respx.MockRouter):
    # The in-flight request must use the cleared litellm_params: no token grant
    # fires and the client-redirected upstream sees only the client's own key,
    # never a bearer minted with the admin's client_secret.
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    token_route = respx_mock.post("https://idp.internal/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-admin", "expires_in": 3600})
    )
    client_route = respx_mock.post("https://client.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    await _router().acompletion(
        model="oauth-model",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://client.example/v1",
        api_key="sk-client",
    )

    assert not token_route.called
    assert client_route.calls.last.request.headers["authorization"] == "Bearer sk-client"


@pytest.mark.asyncio
async def test_router_without_override_mints_bearer_for_admin_upstream(monkeypatch, respx_mock: respx.MockRouter):
    from litellm.llms.openai_like.oauth_authenticator import _token_cache

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    _token_cache.flush_cache()
    respx_mock.post("https://idp.internal/oauth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "tok-admin", "expires_in": 3600})
    )
    gateway_route = respx_mock.post("https://gateway.internal/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    await _router().acompletion(
        model="oauth-model",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert gateway_route.calls.last.request.headers["authorization"] == "Bearer tok-admin"
