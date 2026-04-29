"""
Regression tests for the OAuth2-proxy header-forgery fix
(GHSA-5c3m-qffq-4r9m).

The hook reads HTTP request headers per ``oauth2_config_mappings`` and
constructs a ``UserAPIKeyAuth`` from them. Two separate failure modes
the fix closes:

1. The path was not gated on ``premium_user`` (the sibling
   ``enable_oauth2_auth`` and ``enable_jwt_auth`` paths are). Open-source
   deployments could enable the feature without realising it requires
   a hardened deployment topology.
2. Any ``UserAPIKeyAuth`` field could be mapped from a header — including
   ``user_role``, which Pydantic coerces from the string ``"proxy_admin"``
   into ``LitellmUserRoles.PROXY_ADMIN``. An attacker who reaches the
   proxy directly (or via a misconfigured reverse proxy) sets the mapped
   header and gains full admin privileges.
"""

import os
import sys
from unittest.mock import patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.oauth2_proxy_hook import (
    PRIVILEGED_OAUTH2_PROXY_FIELDS,
    handle_oauth2_proxy_request,
)


def _request_with_headers(headers: dict) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    request = Request(scope=scope)
    request._headers = Headers(headers)
    return request


@pytest.fixture
def premium_proxy_settings(monkeypatch):
    """
    Patch the proxy_server module attributes the hook reads so each test
    starts from "premium=True, mapping={user_id: x-user-id}".
    """
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True, raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"oauth2_config_mappings": {"user_id": "x-user-id"}},
        raising=False,
    )


@pytest.mark.asyncio
async def test_returns_auth_for_simple_user_id_mapping(premium_proxy_settings):
    request = _request_with_headers({"x-user-id": "alice"})

    auth = await handle_oauth2_proxy_request(request)

    assert auth.user_id == "alice"
    assert auth.user_role is None


@pytest.mark.asyncio
async def test_rejects_when_not_premium(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", False, raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"oauth2_config_mappings": {"user_id": "x-user-id"}},
        raising=False,
    )
    request = _request_with_headers({"x-user-id": "alice"})

    with pytest.raises(ValueError, match="enterprise"):
        await handle_oauth2_proxy_request(request)


@pytest.mark.parametrize(
    "privileged_field",
    sorted(PRIVILEGED_OAUTH2_PROXY_FIELDS),
)
@pytest.mark.asyncio
async def test_refuses_to_map_privileged_fields(monkeypatch, privileged_field):
    """
    The exact privesc shape from GHSA-5c3m-qffq-4r9m: an admin maps
    ``user_role`` (or any other privileged field) to a header and a
    caller forges ``X-User-Role: proxy_admin``. The hook must reject
    this configuration outright at request time.
    """
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True, raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {"oauth2_config_mappings": {privileged_field: f"x-{privileged_field}"}},
        raising=False,
    )
    request = _request_with_headers({f"x-{privileged_field}": "proxy_admin"})

    with pytest.raises(ValueError) as exc:
        await handle_oauth2_proxy_request(request)
    assert privileged_field in str(exc.value)


@pytest.mark.asyncio
async def test_user_role_header_forgery_attack_is_blocked(monkeypatch):
    """
    End-to-end shape from the GHSA: with ``user_role`` mapped, a forged
    ``X-User-Role: proxy_admin`` header would have produced a
    ``UserAPIKeyAuth`` with PROXY_ADMIN role. Now the request raises
    before any auth object is constructed.
    """
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True, raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "oauth2_config_mappings": {
                "user_id": "x-user-id",
                "user_role": "x-user-role",
            }
        },
        raising=False,
    )
    request = _request_with_headers(
        {
            "x-user-id": "attacker",
            "x-user-role": LitellmUserRoles.PROXY_ADMIN.value,
        }
    )

    with pytest.raises(ValueError, match="user_role"):
        await handle_oauth2_proxy_request(request)


@pytest.mark.asyncio
async def test_safe_fields_still_pass_through(monkeypatch):
    """
    Sanity check that non-privileged fields (the documented use case
    for OAuth2 proxy auth — asserting identity from a trusted upstream)
    still work after the fix.
    """
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True, raising=False)
    monkeypatch.setattr(
        proxy_server,
        "general_settings",
        {
            "oauth2_config_mappings": {
                "user_id": "x-user-id",
                "user_email": "x-user-email",
                "team_id": "x-team-id",
                "models": "x-models",
            }
        },
        raising=False,
    )
    request = _request_with_headers(
        {
            "x-user-id": "alice",
            "x-user-email": "alice@example.com",
            "x-team-id": "team-corp",
            "x-models": "gpt-4, gpt-3.5-turbo",
        }
    )

    auth = await handle_oauth2_proxy_request(request)

    assert auth.user_id == "alice"
    assert auth.user_email == "alice@example.com"
    assert auth.team_id == "team-corp"
    assert auth.models == ["gpt-4", "gpt-3.5-turbo"]
