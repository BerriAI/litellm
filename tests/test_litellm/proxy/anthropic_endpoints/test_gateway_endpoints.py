"""
Tests for the Claude Code gateway protocol (anthropic_endpoints/gateway_endpoints.py).

Covers the OAuth device-flow surface (RFC 8414 discovery, RFC 8628 device
authorization + token), managed settings, OTLP ingestion, and the enable flag.
"""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import ProxyException
from litellm.proxy.anthropic_endpoints import gateway_endpoints
from litellm.proxy.management_endpoints.ui_sso import (
    _get_cli_sso_flow_cache_key,
    _hash_cli_sso_secret,
    _set_cli_sso_flow,
)
from litellm.proxy.middleware.prometheus_auth_middleware import PrometheusAuthMiddleware

_DEVICE_CODE_GRANT: Final = "urn:ietf:params:oauth:grant-type:device_code"
_MASTER_KEY: Final = "sk-master-key"
_SHARED_LOGIN_ID: Final = "cli-shared-login-code"
_SHARED_POLL_SECRET: Final = "shared-poll-secret"
_SHARED_DEVICE_CODE: Final = f"{_SHARED_LOGIN_ID}.{_SHARED_POLL_SECRET}"
_MINT: Final = "litellm.proxy.auth.auth_checks.ExperimentalUIJWTToken.get_cli_jwt_auth_token"
_PROTOBUF_BODY: Final = b"\x0a\x05hello\x12\x03{{{"
_COMPLETED_SESSION: Final = MappingProxyType(
    {
        "user_id": "user-123",
        "user_role": "internal_user",
        "models": ["claude-sonnet-4-5"],
        "teams": ["team-a"],
        "team_details": [
            {
                "team_id": "team-a",
                "team_alias": "Team A",
                "team_models": ["claude-sonnet-4-5"],
                "team_model_aliases": None,
            }
        ],
    }
)


class _SharedRedisFake:
    def __init__(self) -> None:
        self.values: Mapping[str, object] = MappingProxyType({})
        self.counters: Mapping[str, float] = MappingProxyType({})

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.values = MappingProxyType({**self.values, key: value})

    def get_cache(self, key: str, **kwargs: object) -> object:
        return self.values.get(key)

    def delete_cache(self, key: str) -> None:
        self.values = MappingProxyType({name: value for name, value in self.values.items() if name != key})

    async def async_delete_cache(self, key: str) -> None:
        self.delete_cache(key)

    async def async_increment(self, key: str, value: float, **kwargs: object) -> float:
        incremented: Final = self.counters.get(key, 0) + value
        self.counters = MappingProxyType({**self.counters, key: incremented})
        return incremented


def _replica(redis: _SharedRedisFake) -> DualCache:
    return DualCache(redis_cache=redis, default_in_memory_ttl=600)  # pyright: ignore[reportArgumentType]  # duck-typed Redis double


def _real_auth_proxy_attrs() -> Mapping[str, object]:
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    return MappingProxyType(
        {
            "master_key": _MASTER_KEY,
            "prisma_client": None,
            "user_api_key_cache": DualCache(),
            "proxy_logging_obj": proxy_logging_obj,
            "llm_router": None,
            "llm_model_list": [],
            "user_custom_auth": None,
            "litellm_proxy_admin_name": "admin",
            "jwt_handler": None,
            "open_telemetry_logger": None,
            "model_max_budget_limiter": MagicMock(),
        }
    )


@contextmanager
def _gateway_env(
    *,
    enabled: bool = True,
    managed_settings: Mapping[str, object] | None = None,
    cache: DualCache | None = None,
    real_auth: bool = False,
    extra_settings: Mapping[str, object] = MappingProxyType({}),
) -> Iterator[tuple[TestClient, DualCache]]:
    general_settings: Final = {
        "enable_claude_code_gateway": enabled,
        **({} if managed_settings is None else {"claude_code_gateway_managed_settings": dict(managed_settings)}),
        **extra_settings,
    }
    session_cache: Final = cache or DualCache(default_in_memory_ttl=600)

    app: Final = FastAPI()
    app.add_middleware(PrometheusAuthMiddleware)
    app.include_router(gateway_endpoints.router)

    async def _fake_auth() -> object:
        return object()

    with ExitStack() as stack:
        stack.enter_context(
            patch(  # test-quality-ok: the gateway reads this proxy_server module global and has no injection seam
                "litellm.proxy.proxy_server.general_settings", general_settings
            )
        )
        stack.enter_context(
            patch(  # test-quality-ok: the CLI SSO flow cache is this proxy_server module global shared with ui_sso
                "litellm.proxy.proxy_server.cli_sso_session_cache", session_cache
            )
        )
        if real_auth:
            for name, value in _real_auth_proxy_attrs().items():
                stack.enter_context(patch(f"litellm.proxy.proxy_server.{name}", value))
        else:
            app.dependency_overrides[gateway_endpoints.user_api_key_auth] = _fake_auth
        with TestClient(app) as client:
            yield client, session_cache


def _start_device_flow(client: TestClient) -> str:
    return client.post("/claude_code_gateway/oauth/device_authorization").json()["device_code"]


def _request_token(client: TestClient, device_code: str) -> httpx.Response:
    return client.post(
        "/claude_code_gateway/oauth/token",
        data={"grant_type": _DEVICE_CODE_GRANT, "device_code": device_code},
    )


def _completed_flow(session_data: Mapping[str, object] = _COMPLETED_SESSION) -> dict[str, object]:
    return {
        "poll_secret_hash": _hash_cli_sso_secret(_SHARED_POLL_SECRET),
        "user_code_hash": "unused",
        "sso_complete": True,
        "user_code_verified": True,
        "session_data": dict(session_data),
    }


def _login_id(device_code: str) -> str:
    return device_code.partition(".")[0]


def _complete_flow(
    cache: DualCache, device_code: str, session_data: Mapping[str, object] = _COMPLETED_SESSION
) -> None:
    key: Final = _get_cli_sso_flow_cache_key(_login_id(device_code))
    flow: Final = cache.get_cache(key=key)
    assert isinstance(flow, dict)
    completed: Final = {**flow, **_completed_flow(session_data), "poll_secret_hash": flow["poll_secret_hash"]}
    cache.set_cache(key=key, value=completed, ttl=600)


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
        login_id, separator, poll_secret = device_code.partition(".")
        assert login_id.startswith("cli-")
        assert separator == "."
        assert len(poll_secret) >= 32
        assert body["user_code"]
        assert body["expires_in"] == 600
        assert body["interval"] == 5
        assert "verification_uri_complete" not in body
        assert body["verification_uri"].endswith(f"/sso/key/generate?source=litellm-cli&key={login_id}")
        assert poll_secret not in body["verification_uri"]
        stored = cache.get_cache(key=_get_cli_sso_flow_cache_key(login_id))
        assert isinstance(stored, dict)
        assert stored["sso_complete"] is False
        assert stored["poll_secret_hash"] == _hash_cli_sso_secret(poll_secret)
        assert cache.get_cache(key=_get_cli_sso_flow_cache_key(device_code)) is None


@pytest.mark.parametrize("opted_in", [True, False])
def test_verification_uri_complete_carries_the_user_code_only_when_the_operator_opts_in(opted_in: bool):
    with _gateway_env(extra_settings={"allow_cli_sso_verification_uri_complete": opted_in}) as (client, _):
        body = client.post("/claude_code_gateway/oauth/device_authorization").json()
    login_id = _login_id(body["device_code"])
    if not opted_in:
        assert "verification_uri_complete" not in body
        return
    assert body["verification_uri_complete"].endswith(
        f"/sso/key/generate?source=litellm-cli&key={login_id}&user_code={body['user_code']}"
    )
    assert "user_code=" not in body["verification_uri"]


def test_token_authorization_pending_before_browser_completes():
    with _gateway_env() as (client, _):
        resp = _request_token(client, _start_device_flow(client))
    assert resp.status_code == 400
    assert resp.json()["error"] == "authorization_pending"


@pytest.mark.parametrize("tamper", ["login_id_only", "wrong_secret"])
def test_token_refuses_the_browser_login_id_without_the_client_secret(tamper: str):
    with _gateway_env() as (client, cache):
        device_code = _start_device_flow(client)
        _complete_flow(cache, device_code)
        login_id = _login_id(device_code)
        presented = login_id if tamper == "login_id_only" else f"{login_id}.not-the-secret"
        with patch(_MINT, return_value="sk-session") as mint:
            resp = _request_token(client, presented)
            assert resp.status_code == 400
            assert resp.json()["error"] == "expired_token"
            mint.assert_not_called()
            with_secret = _request_token(client, device_code)
    assert with_secret.status_code == 200


def test_token_success_mints_bearer_and_is_single_use():
    with _gateway_env() as (client, cache):
        device_code = _start_device_flow(client)
        _complete_flow(cache, device_code)

        with patch(_MINT, return_value="sk-litellm-session-token") as mint:
            resp = _request_token(client, device_code)
            assert resp.status_code == 200
            body = resp.json()
            assert body["access_token"] == "sk-litellm-session-token"
            assert body["token_type"] == "Bearer"
            assert body["expires_in"] > 0

            called_user = mint.call_args.kwargs["user_info"]
            assert called_user.user_id == "user-123"
            assert mint.call_args.kwargs["team_id"] == "team-a"
            assert mint.call_args.kwargs["team_alias"] == "Team A"
            assert mint.call_args.kwargs["team_models"] == ("claude-sonnet-4-5",)

            # Single-use: the flow is deleted, so a replay returns expired_token.
            replay = _request_token(client, device_code)
    assert replay.status_code == 400
    assert replay.json()["error"] == "expired_token"


def test_token_teamless_user_mints_without_a_team():
    with _gateway_env() as (client, cache):
        device_code = _start_device_flow(client)
        _complete_flow(cache, device_code, session_data={**_COMPLETED_SESSION, "teams": [], "team_details": []})
        with patch(_MINT, return_value="sk-litellm-session-token") as mint:
            resp = _request_token(client, device_code)
    assert resp.status_code == 200
    assert mint.call_args.kwargs["team_id"] is None
    assert mint.call_args.kwargs["team_models"] == ()


@pytest.mark.parametrize(
    "session_data",
    [
        {"user_role": "internal_user"},
        {**_COMPLETED_SESSION, "user_role": None},
        {**_COMPLETED_SESSION, "user_role": "not-a-role"},
    ],
    ids=["missing_user_id", "no_role", "unknown_role"],
)
def test_token_malformed_session_is_invalid_grant_and_does_not_consume_the_login(session_data: Mapping[str, object]):
    with _gateway_env() as (client, cache):
        device_code = _start_device_flow(client)
        _complete_flow(cache, device_code, session_data=session_data)
        with patch(_MINT) as mint:
            resp = _request_token(client, device_code)
            again = _request_token(client, device_code)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_grant"
    assert again.json()["error"] == "invalid_grant"
    mint.assert_not_called()


def test_token_unknown_team_grants_is_invalid_grant():
    with _gateway_env() as (client, cache):
        device_code = _start_device_flow(client)
        _complete_flow(cache, device_code, session_data={**_COMPLETED_SESSION, "team_details": []})
        with patch(_MINT) as mint:
            resp = _request_token(client, device_code)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_grant"
    mint.assert_not_called()


def test_token_mints_on_a_replica_that_did_not_start_the_login():
    redis: Final = _SharedRedisFake()
    _set_cli_sso_flow(login_id=_SHARED_LOGIN_ID, cache=_replica(redis), flow=_completed_flow())

    with _gateway_env(cache=_replica(redis)) as (client, _), patch(_MINT, return_value="sk-session") as mint:
        resp = _request_token(client, _SHARED_DEVICE_CODE)
    assert resp.status_code == 200
    assert resp.json()["access_token"] == "sk-session"
    assert mint.call_args.kwargs["team_id"] == "team-a"
    assert mint.call_args.kwargs["user_info"].user_role == "internal_user"


def test_token_refuses_a_device_code_another_replica_already_claimed():
    redis: Final = _SharedRedisFake()
    replica_a: Final = _replica(redis)
    _set_cli_sso_flow(login_id=_SHARED_LOGIN_ID, cache=replica_a, flow=_completed_flow())
    assert asyncio.run(gateway_endpoints._claim_device_code(_SHARED_LOGIN_ID, replica_a)) is True

    with _gateway_env(cache=_replica(redis)) as (client, _), patch(_MINT) as mint:
        resp = _request_token(client, _SHARED_DEVICE_CODE)
    assert resp.status_code == 400
    assert resp.json()["error"] == "expired_token"
    mint.assert_not_called()


def test_token_unknown_device_code_is_expired_token():
    with _gateway_env() as (client, _):
        resp = _request_token(client, "cli-does-not-exist")
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


def test_managed_settings_returns_client_envelope_and_304_on_cached_checksum():
    settings = {"permissions": {"defaultMode": "acceptEdits"}, "env": {"FOO": "bar"}}
    with _gateway_env(managed_settings=settings) as (client, _):
        resp = client.get("/claude_code_gateway/managed/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["settings"] == settings
        checksum = body["checksum"]
        assert checksum.startswith("sha256:")
        assert body["uuid"] == checksum
        assert resp.headers["ETag"] == f'"{checksum}"'

        not_modified = client.get(
            "/claude_code_gateway/managed/settings", headers={"If-None-Match": f'"{checksum}"'}
        )
        assert not_modified.status_code == 304
        assert not_modified.headers["ETag"] == f'"{checksum}"'

        stale = client.get("/claude_code_gateway/managed/settings", headers={"If-None-Match": '"sha256:stale"'})
    assert stale.status_code == 200
    assert stale.json()["checksum"] == checksum


def test_managed_settings_checksum_tracks_policy_content():
    with _gateway_env(managed_settings={"env": {"FOO": "bar"}}) as (client, _):
        first = client.get("/claude_code_gateway/managed/settings").json()["checksum"]
    with _gateway_env(managed_settings={"env": {"FOO": "baz"}}) as (client, _):
        second = client.get("/claude_code_gateway/managed/settings").json()["checksum"]
    assert first != second


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


@pytest.mark.parametrize("signal", ["metrics", "logs", "traces"])
def test_otlp_protobuf_body_is_accepted_through_real_auth(signal: str):
    with _gateway_env(real_auth=True) as (client, _):
        resp = client.post(
            f"/claude_code_gateway/v1/{signal}",
            content=_PROTOBUF_BODY,
            headers={"Authorization": f"Bearer {_MASTER_KEY}", "Content-Type": "application/x-protobuf"},
        )
    assert resp.status_code == 200


def test_otlp_without_a_bearer_is_rejected_by_real_auth():
    with _gateway_env(real_auth=True) as (client, _), pytest.raises(ProxyException) as exc_info:
        client.post(
            "/claude_code_gateway/v1/metrics",
            content=_PROTOBUF_BODY,
            headers={"Content-Type": "application/x-protobuf"},
        )
    assert exc_info.value.code == "401"


def test_messages_gated_by_enable_flag():
    with _gateway_env(enabled=False) as (client, _):
        resp = client.post("/claude_code_gateway/v1/messages", json={"model": "claude-sonnet-4-5", "messages": []})
    assert resp.status_code == 404
