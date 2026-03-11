"""
Route Access Contract Tests
============================

Parameterized matrix test that encodes which auth_type x role x route combinations
are allowed or denied. This acts as a regression safety net — any PR that changes
auth behavior will break specific rows and force the author to explicitly update
the contract.

Background:
- PR #22164 added common_checks() to custom auth, breaking custom routes.
- PR #22662 patched it with skip_route_check.
- PR (b44755db) replaced both with an opt-in flag.
All three PRs would have been caught (or made unnecessary) by this test.
"""

import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import _is_allowed_route, common_checks
from litellm.proxy.auth.route_checks import RouteChecks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request() -> MagicMock:
    req = MagicMock(spec=Request)
    req.query_params = {}
    return req


def _make_user(role: str) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id="test_user",
        user_email="test@example.com",
        user_role=role,
    )


def _make_token(
    role: Optional[str] = None,
    team_id: Optional[str] = None,
    allowed_routes: Optional[list] = None,
) -> UserAPIKeyAuth:
    kwargs = {"token": "sk-test", "user_id": "test_user"}
    if role:
        kwargs["user_role"] = role
    if team_id:
        kwargs["team_id"] = team_id
    if allowed_routes:
        kwargs["allowed_routes"] = allowed_routes
    return UserAPIKeyAuth(**kwargs)


# ---------------------------------------------------------------------------
# 1. Route check contract for key-based auth (non-admin roles)
#    Tests _is_allowed_route / RouteChecks.non_proxy_admin_allowed_routes_check
# ---------------------------------------------------------------------------

# (role, route, expected_outcome)
# "allowed" = no exception raised, "denied" = HTTPException / Exception raised
_KEY_BASED_ROUTE_MATRIX = [
    # --- LLM API routes: all non-admin roles can call them (except admin_view_only) ---
    (LitellmUserRoles.INTERNAL_USER.value, "/chat/completions", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/v1/chat/completions", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/v1/embeddings", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/v1/models", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/v1/messages", "allowed"),
    # --- Info routes: accessible to all non-admin roles ---
    (LitellmUserRoles.INTERNAL_USER.value, "/model/info", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/key/info", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/health", "allowed"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/model/info", "allowed"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/key/info", "allowed"),
    # --- Management write routes: denied for non-admin ---
    (LitellmUserRoles.INTERNAL_USER.value, "/user/new", "denied"),
    (LitellmUserRoles.INTERNAL_USER.value, "/user/delete", "denied"),
    (LitellmUserRoles.INTERNAL_USER.value, "/team/new", "denied"),
    (LitellmUserRoles.INTERNAL_USER.value, "/team/delete", "denied"),
    (LitellmUserRoles.INTERNAL_USER.value, "/config/update", "denied"),
    # internal_user CAN access key_management_routes and self_managed_routes
    (LitellmUserRoles.INTERNAL_USER.value, "/key/generate", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/model/new", "allowed"),
    # Admin view-only: write management routes denied
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/user/new", "denied"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/key/generate", "denied"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/team/delete", "denied"),
    # Admin view-only: read management routes allowed
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/user/list", "allowed"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/user/info", "allowed"),
    (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value, "/team/info", "allowed"),
    # --- Internal user: spend tracking routes allowed ---
    (LitellmUserRoles.INTERNAL_USER.value, "/global/spend/tags", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/global/spend/keys", "allowed"),
    # Internal user view-only: spend routes allowed, management denied
    (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value, "/spend/keys", "allowed"),
    (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value, "/global/spend/logs", "allowed"),
    (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value, "/key/generate", "denied"),
    (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value, "/user/new", "denied"),
    # --- Self-managed routes: accessible to non-admin ---
    (LitellmUserRoles.INTERNAL_USER.value, "/team/member_add", "allowed"),
    (LitellmUserRoles.INTERNAL_USER.value, "/team/member_delete", "allowed"),
]


@pytest.mark.parametrize(
    "role,route,expected",
    _KEY_BASED_ROUTE_MATRIX,
    ids=[f"{r}|{rt}|{exp}" for r, rt, exp in _KEY_BASED_ROUTE_MATRIX],
)
def test_key_based_route_access(role, route, expected):
    """Contract test: key-based auth route access for each role."""
    request = _make_request()
    user_obj = _make_user(role)
    valid_token = _make_token(role=role)

    if expected == "allowed":
        # Should NOT raise
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=role,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    else:
        with pytest.raises(Exception):
            RouteChecks.non_proxy_admin_allowed_routes_check(
                user_obj=user_obj,
                _user_role=role,
                route=route,
                request=request,
                valid_token=valid_token,
                request_data={},
            )


# ---------------------------------------------------------------------------
# 2. Proxy admin bypasses all route checks
# ---------------------------------------------------------------------------

_ADMIN_ROUTES = [
    "/chat/completions",
    "/key/generate",
    "/user/new",
    "/config/update",
    "/global/spend/reset",
    "/model/delete",
]


@pytest.mark.parametrize("route", _ADMIN_ROUTES)
def test_proxy_admin_can_access_all_routes(route):
    """Proxy admin should pass _is_allowed_route for any route."""
    request = _make_request()
    user_obj = _make_user(LitellmUserRoles.PROXY_ADMIN.value)
    valid_token = _make_token(role=LitellmUserRoles.PROXY_ADMIN.value)

    result = _is_allowed_route(
        route=route,
        token_type="api",
        request=request,
        request_data={},
        valid_token=valid_token,
        user_obj=user_obj,
    )
    assert result is True


# ---------------------------------------------------------------------------
# 3. Custom auth: common_checks opt-in contract
#    This is the exact regression that caused the 3-PR chain.
# ---------------------------------------------------------------------------

_CUSTOM_AUTH_ROUTES = [
    "/chat/completions",
    "/v1/chat/completions",
    "/ldap/ngs/ready",  # custom user-defined route (the one that broke)
    "/my-app/webhook",  # another hypothetical custom route
    "/key/info",
    "/health",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _CUSTOM_AUTH_ROUTES)
async def test_custom_auth_default_skips_common_checks(route):
    """
    Default custom auth: common_checks is NOT called, so ANY route passes.
    This is the backwards-compatible behavior (pre-#22164).
    """
    from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

    valid_token = UserAPIKeyAuth(token="sk-custom")
    mock_request = MagicMock()

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks",
        new_callable=AsyncMock,
    ) as mock_common, patch(
        "litellm.proxy.proxy_server.general_settings",
        {},
    ):
        mock_common.return_value = True
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=mock_request,
            request_data={},
            route=route,
            parent_otel_span=None,
        )
        mock_common.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", _CUSTOM_AUTH_ROUTES)
async def test_custom_auth_opt_in_runs_common_checks(route):
    """
    With custom_auth_run_common_checks=True, common_checks IS called.
    """
    from litellm.proxy.auth.user_api_key_auth import _run_post_custom_auth_checks

    valid_token = UserAPIKeyAuth(token="sk-custom")
    mock_request = MagicMock()

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks",
        new_callable=AsyncMock,
    ) as mock_common, patch(
        "litellm.proxy.proxy_server.general_settings",
        {"custom_auth_run_common_checks": True},
    ):
        mock_common.return_value = True
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=mock_request,
            request_data={},
            route=route,
            parent_otel_span=None,
        )
        mock_common.assert_called_once()


# ---------------------------------------------------------------------------
# 4. common_checks route enforcement (key-based path)
#    Verifies that _is_allowed_route is always called inside common_checks
#    (i.e. the skip_route_check parameter from #22662 stays removed).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_common_checks_always_runs_route_check():
    """
    Regression: common_checks must always run route authorization.
    The skip_route_check param (PR #22662) must NOT come back.
    """
    import inspect

    sig = inspect.signature(common_checks)
    param_names = list(sig.parameters.keys())

    # skip_route_check must NOT exist as a parameter
    assert "skip_route_check" not in param_names, (
        "skip_route_check parameter found in common_checks — "
        "this was removed in favor of the custom_auth_run_common_checks flag"
    )


# ---------------------------------------------------------------------------
# 5. Virtual key allowed_routes override
# ---------------------------------------------------------------------------

_VIRTUAL_KEY_CASES = [
    # (allowed_routes on key, route being accessed, expected)
    (["/chat/completions", "/key/info"], "/chat/completions", "allowed"),
    # allowed_routes on the key restricts to only those routes;
    # /config/update is not in any non-admin allowed set, so it's denied
    (["/chat/completions"], "/config/update", "denied"),
    (["llm_api_routes"], "/v1/chat/completions", "allowed"),
]


@pytest.mark.parametrize(
    "allowed_routes,route,expected",
    _VIRTUAL_KEY_CASES,
    ids=[f"{ar}|{rt}|{exp}" for ar, rt, exp in _VIRTUAL_KEY_CASES],
)
def test_virtual_key_allowed_routes(allowed_routes, route, expected):
    """Keys with explicit allowed_routes restrict access to those routes only."""
    request = _make_request()
    user_obj = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    valid_token = _make_token(
        role=LitellmUserRoles.INTERNAL_USER.value,
        allowed_routes=allowed_routes,
    )

    if expected == "allowed":
        RouteChecks.non_proxy_admin_allowed_routes_check(
            user_obj=user_obj,
            _user_role=LitellmUserRoles.INTERNAL_USER.value,
            route=route,
            request=request,
            valid_token=valid_token,
            request_data={},
        )
    else:
        with pytest.raises(Exception):
            RouteChecks.non_proxy_admin_allowed_routes_check(
                user_obj=user_obj,
                _user_role=LitellmUserRoles.INTERNAL_USER.value,
                route=route,
                request=request,
                valid_token=valid_token,
                request_data={},
            )
