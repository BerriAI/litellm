"""
Regression tests for the OAuth2-proxy header-forgery fix
(GHSA-5c3m-qffq-4r9m).

The hook reads HTTP request headers per ``oauth2_config_mappings`` and
constructs a ``UserAPIKeyAuth`` from them. The fix has two parts:

1. Only requests from configured trusted proxy CIDR ranges may provide
   identity headers.
2. Only identity fields may be mapped from those headers. Without the
   identity-only allowlist any field could be mapped — including
   ``user_role``, which Pydantic coerces from the string
   ``"proxy_admin"`` into ``LitellmUserRoles.PROXY_ADMIN``.
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


def _request_with_headers(headers: dict, *, client_host: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "client": (client_host, 12345),
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    request = Request(scope=scope)
    request._headers = Headers(headers)
    return request


@pytest.fixture
def configure_proxy(monkeypatch):
    """
    Yields a callable that sets ``oauth2_config_mappings`` and
    ``trusted_proxy_ranges`` on the proxy_server module for the duration
    of one test. Defaults to a single identity mapping and localhost as
    a trusted proxy.
    """
    import litellm.proxy.proxy_server as proxy_server

    def _configure(*, mappings=None, trusted_proxy_ranges=("127.0.0.1/32",)):
        if mappings is None:
            mappings = {"user_id": "x-user-id"}
        settings = {
            "oauth2_config_mappings": mappings,
            "trusted_proxy_ranges": trusted_proxy_ranges,
        }
        monkeypatch.setattr(
            proxy_server,
            "general_settings",
            settings,
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


@pytest.mark.asyncio
async def test_rejects_identity_headers_without_trusted_proxy_ranges(configure_proxy):
    configure_proxy(trusted_proxy_ranges=None)
    request = _request_with_headers({"x-user-id": "alice"})

    with pytest.raises(ValueError, match="trusted_proxy_ranges"):
        await handle_oauth2_proxy_request(request)


@pytest.mark.asyncio
async def test_rejects_identity_headers_from_untrusted_direct_client(configure_proxy):
    configure_proxy(trusted_proxy_ranges=["10.0.0.0/24"])
    request = _request_with_headers({"x-user-id": "alice"}, client_host="203.0.113.10")

    with pytest.raises(ValueError, match="not trusted"):
        await handle_oauth2_proxy_request(request)


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
