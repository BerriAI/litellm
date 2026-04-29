"""
Regression tests for the OAuth2-proxy header-forgery fix
(GHSA-5c3m-qffq-4r9m).

The hook reads HTTP request headers per ``oauth2_config_mappings`` and
constructs a ``UserAPIKeyAuth`` from them. Without the
identity-only allowlist any field could be mapped — including
``user_role``, which Pydantic coerces from the string
``"proxy_admin"`` into ``LitellmUserRoles.PROXY_ADMIN``. An attacker
who reaches the proxy directly (or via a misconfigured reverse
proxy) sets the mapped header and gains full admin privileges.
"""

import os
import sys

import pytest
from fastapi import Request
from starlette.datastructures import Headers

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.oauth2_proxy_hook import (
    ALLOWED_OAUTH2_PROXY_FIELDS,
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
def configure_proxy(monkeypatch):
    """
    Yields a callable that sets ``oauth2_config_mappings`` on the
    proxy_server module for the duration of one test. Default mapping
    is a single ``user_id -> x-user-id`` (identity-only).
    """
    import litellm.proxy.proxy_server as proxy_server

    def _configure(*, mappings=None):
        if mappings is None:
            mappings = {"user_id": "x-user-id"}
        monkeypatch.setattr(
            proxy_server,
            "general_settings",
            {"oauth2_config_mappings": mappings},
            raising=False,
        )

    return _configure


@pytest.mark.asyncio
async def test_returns_auth_for_simple_user_id_mapping(configure_proxy):
    configure_proxy()
    request = _request_with_headers({"x-user-id": "alice"})

    auth = await handle_oauth2_proxy_request(request)

    assert auth.user_id == "alice"
    assert auth.user_role is None


@pytest.mark.parametrize(
    "privileged_field",
    [
        # The GHSA-5c3m-qffq-4r9m primary privesc field.
        "user_role",
        # Key-level enforcement bypass shapes.
        "api_key",
        "token",
        "permissions",
        "allowed_routes",
        "max_budget",
        "spend",
        "tpm_limit",
        "rpm_limit",
        "model_max_budget",
        "metadata",
        # User-level enforcement bypass — flagged by Greptile as a denylist gap.
        "user_max_budget",
        "user_tpm_limit",
        "user_rpm_limit",
        "user_spend",
        # Team / org / end-user / region — same class, all denied by the
        # identity-only allowlist.
        "team_max_budget",
        "team_spend",
        "team_member_tpm_limit",
        "organization_max_budget",
        "organization_tpm_limit",
        "end_user_max_budget",
        "allowed_model_region",
        # Anything not on ALLOWED_OAUTH2_PROXY_FIELDS is blocked, even
        # fabricated field names admins might try.
        "definitely_not_a_real_field",
    ],
)
@pytest.mark.asyncio
async def test_refuses_to_map_non_identity_fields(configure_proxy, privileged_field):
    # GHSA-5c3m-qffq-4r9m attack shape: admin maps a privileged field
    # to a header and a caller forges the value. The allowlist rejects
    # any non-identity mapping at request time, regardless of whether
    # the field ever appeared on a denylist — which is the whole reason
    # we use an allowlist instead.
    configure_proxy(mappings={privileged_field: f"x-{privileged_field}"})
    request = _request_with_headers({f"x-{privileged_field}": "proxy_admin"})

    with pytest.raises(ValueError) as exc:
        await handle_oauth2_proxy_request(request)
    assert privileged_field in str(exc.value)


@pytest.mark.parametrize("identity_field", sorted(ALLOWED_OAUTH2_PROXY_FIELDS))
def test_allowlist_is_identity_only(identity_field):
    # Lock in the allowlist's intent: only identity-assertion fields are
    # safe to populate from a header. If anyone proposes adding budget /
    # spend / role / permission to ``ALLOWED_OAUTH2_PROXY_FIELDS``, this
    # assertion forces them to update the test deliberately.
    assert identity_field in {
        "user_id",
        "user_email",
        "team_id",
        "team_alias",
        "org_id",
        "models",
    }


@pytest.mark.asyncio
async def test_user_role_header_forgery_attack_is_blocked(configure_proxy):
    # End-to-end form of the privesc: with ``user_role`` mapped, the
    # forged ``X-User-Role: proxy_admin`` header would have produced
    # a ``UserAPIKeyAuth(user_role=PROXY_ADMIN)``. Now rejected before
    # any auth object is constructed.
    configure_proxy(
        mappings={"user_id": "x-user-id", "user_role": "x-user-role"},
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
async def test_safe_fields_still_pass_through(configure_proxy):
    # The documented use case for OAuth2 proxy auth: identity assertion
    # from a trusted upstream. Must remain unaffected by the denylist.
    configure_proxy(
        mappings={
            "user_id": "x-user-id",
            "user_email": "x-user-email",
            "team_id": "x-team-id",
            "models": "x-models",
        },
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
