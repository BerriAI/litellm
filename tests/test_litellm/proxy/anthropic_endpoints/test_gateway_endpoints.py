"""
Tests for the Claude Code gateway protocol (anthropic_endpoints/gateway_endpoints.py).

Covers the OAuth device-flow surface (RFC 8414 discovery, RFC 8628 device
authorization + token), managed settings, OTLP ingestion, and the enable flag.
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.caching.dual_cache import DualCache
from litellm.proxy.anthropic_endpoints import gateway_endpoints
from litellm.proxy.management_endpoints.ui_sso import _get_cli_sso_flow_cache_key


@contextmanager
def _gateway_env(
    *,
    enabled: bool = True,
    managed_settings: Optional[dict[str, Any]] = None,
) -> Iterator[tuple[TestClient, DualCache]]:
    general_settings: dict[str, Any] = {"enable_claude_code_gateway": enabled}
    if managed_settings is not None:
        general_settings["claude_code_gateway_managed_settings"] = managed_settings
    cache = DualCache(default_in_memory_ttl=600)

    app = FastAPI()
    app.include_router(gateway_endpoints.router)

    async def _fake_auth() -> Any:
        return object()

    app.dependency_overrides[gateway_endpoints.user_api_key_auth] = _fake_auth

    with patch("litellm.proxy.proxy_server.general_settings", general_settings), patch(
        "litellm.proxy.proxy_server.cli_sso_session_cache", cache
    ):
        with TestClient(app) as client:
            yield client, cache


def _complete_flow(cache: DualCache, device_code: str) -> None:
    key = _get_cli_sso_flow_cache_key(device_code)
    flow = cache.get_cache(key=key)
    assert isinstance(flow, dict)
    flow["sso_complete"] = True
    flow["user_code_verified"] = True
    flow["session_data"] = {
        "user_id": "user-123",
        "user_role": "internal_user",
        "models": ["claude-sonnet-4-5"],
        "teams": ["team-a"],
    }
    cache.set_cache(key=key, value=flow, ttl=600)


def test_discovery_shape():
    with _gateway_env() as (client, _):
        resp = client.get("/claude_code_gateway/.well-known/oauth-authorization-server")
    assert resp.status_code == 200
    body = resp.json()
    assert body["device_authorization_endpoint"].endswith("/claude_code_gateway/oauth/device_authorization")
    assert body["token_endpoint"].endswith("/claude_code_gateway/oauth/token")
    assert body["grant_types_supported"] == [
        "urn:ietf:params:oauth:grant-type:device_code",
        "refresh_token",
    ]
    # authorization_endpoint is intentionally absent (device flow only).
    assert "authorization_endpoint" not in body
    # Both endpoints must be same-origin with the issuer.
    assert body["device_authorization_endpoint"].startswith(body["issuer"])
    assert body["token_endpoint"].startswith(body["issuer"])


def test_discovery_404_when_disabled():
    with _gateway_env(enabled=False) as (client, _):
        resp = client.get("/claude_code_gateway/.well-known/oauth-authorization-server")
    assert resp.status_code == 404


def test_device_authorization_returns_rfc8628_shape_and_persists_flow():
    with _gateway_env() as (client, cache):
        resp = client.post("/claude_code_gateway/oauth/device_authorization")
        assert resp.status_code == 200
        body = resp.json()
        device_code = body["device_code"]
        assert device_code.startswith("cli-")
        assert body["user_code"]
        assert body["expires_in"] == 600
        assert body["interval"] == 5
        # verification_uri_complete carries the user_code; the short uri does not.
        assert f"user_code={body['user_code']}" in body["verification_uri_complete"]
        assert "user_code=" not in body["verification_uri"]
        assert f"key={device_code}" in body["verification_uri"]
        # The device flow is stored under the device_code so the browser SSO leg can complete it.
        stored = cache.get_cache(key=_get_cli_sso_flow_cache_key(device_code))
        assert isinstance(stored, dict)
        assert stored["sso_complete"] is False


def test_token_authorization_pending_before_browser_completes():
    with _gateway_env() as (client, _):
        device_code = client.post("/claude_code_gateway/oauth/device_authorization").json()["device_code"]
        resp = client.post(
            "/claude_code_gateway/oauth/token",
            data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code},
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "authorization_pending"


def test_token_success_mints_bearer_and_is_single_use():
    with _gateway_env() as (client, cache):
        device_code = client.post("/claude_code_gateway/oauth/device_authorization").json()["device_code"]
        _complete_flow(cache, device_code)

        with patch(
            "litellm.proxy.auth.auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token",
            return_value="sk-litellm-session-token",
        ) as mint:
            resp = client.post(
                "/claude_code_gateway/oauth/token",
                data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["access_token"] == "sk-litellm-session-token"
            assert body["token_type"] == "Bearer"
            assert body["expires_in"] > 0

            called_user = mint.call_args.kwargs["user_info"]
            assert called_user.user_id == "user-123"
            assert mint.call_args.kwargs["team_id"] == "team-a"

            # Single-use: the flow is deleted, so a replay returns expired_token.
            replay = client.post(
                "/claude_code_gateway/oauth/token",
                data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code},
            )
    assert replay.status_code == 400
    assert replay.json()["error"] == "expired_token"


def test_token_unknown_device_code_is_expired_token():
    with _gateway_env() as (client, _):
        resp = client.post(
            "/claude_code_gateway/oauth/token",
            data={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": "cli-does-not-exist"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"] == "expired_token"


def test_refresh_grant_forces_relogin():
    with _gateway_env() as (client, _):
        resp = client.post(
            "/claude_code_gateway/oauth/token",
            data={"grant_type": "refresh_token", "refresh_token": "whatever"},
        )
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid_grant"


def test_unsupported_grant_type():
    with _gateway_env() as (client, _):
        resp = client.post("/claude_code_gateway/oauth/token", data={"grant_type": "password"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "unsupported_grant_type"


def test_managed_settings_404_when_unset():
    with _gateway_env() as (client, _):
        resp = client.get("/claude_code_gateway/managed/settings")
    assert resp.status_code == 404


def test_managed_settings_returns_json_with_etag_and_304():
    settings = {"permissions": {"defaultMode": "acceptEdits"}, "env": {"FOO": "bar"}}
    with _gateway_env(managed_settings=settings) as (client, _):
        resp = client.get("/claude_code_gateway/managed/settings")
        assert resp.status_code == 200
        assert resp.json() == settings
        etag = resp.headers["ETag"]
        assert etag

        not_modified = client.get("/claude_code_gateway/managed/settings", headers={"If-None-Match": etag})
    assert not_modified.status_code == 304
    assert not_modified.headers["ETag"] == etag


def test_managed_settings_404_when_gateway_disabled():
    with _gateway_env(enabled=False, managed_settings={"env": {}}) as (client, _):
        resp = client.get("/claude_code_gateway/managed/settings")
    assert resp.status_code == 404


@pytest.mark.parametrize("signal", ["metrics", "logs", "traces"])
def test_otlp_endpoints_accept_and_return_200(signal: str):
    with _gateway_env() as (client, _):
        resp = client.post(f"/claude_code_gateway/v1/{signal}", content=b"\x00\x01binary-otlp")
    assert resp.status_code == 200


@pytest.mark.parametrize("signal", ["metrics", "logs", "traces"])
def test_otlp_endpoints_404_when_disabled(signal: str):
    with _gateway_env(enabled=False) as (client, _):
        resp = client.post(f"/claude_code_gateway/v1/{signal}", content=b"payload")
    assert resp.status_code == 404


def test_messages_gated_by_enable_flag():
    with _gateway_env(enabled=False) as (client, _):
        resp = client.post("/claude_code_gateway/v1/messages", json={"model": "claude-sonnet-4-5", "messages": []})
    assert resp.status_code == 404
