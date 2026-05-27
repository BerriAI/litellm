"""Regression tests for LIT-3300.

Virtual keys created with key_type=management / key_type=llm_api / key_type=read_only
get `allowed_routes` set to LiteLLMRoutes enum-member NAMES like "management_routes",
"llm_api_routes", "info_routes" (by handle_key_type in key_management_endpoints.py).

`is_virtual_key_allowed_to_call_route` (first gate) already expands those names to
the concrete route list. The fallback `non_proxy_admin_allowed_routes_check` did not —
it compared the bucket-name string against the request path, which always failed,
so callers saw 401 on routes the bucket should cover.

These tests cover the fallback bucket-name expansion.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from fastapi import Request

from litellm.proxy._types import (
    LiteLLM_UserTable,
    LiteLLMRoutes,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks


def _make_request() -> Request:
    request = MagicMock(spec=Request)
    request.query_params = {}
    return request


def _make_token(allowed_routes):
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id=None,
        allowed_routes=allowed_routes,
        team_id=None,
        key_name=None,
        permissions={},
        models=[],
    )


def _call(allowed_routes, route, user_role=None):
    user_obj = None
    if user_role is not None:
        user_obj = LiteLLM_UserTable(
            user_id="u1",
            user_email=None,
            user_role=user_role,
        )
    RouteChecks.non_proxy_admin_allowed_routes_check(
        user_obj=user_obj,
        _user_role=user_role,
        route=route,
        request=_make_request(),
        valid_token=_make_token(allowed_routes),
        request_data={},
    )
    return "ALLOW"


@pytest.mark.parametrize(
    "route",
    [
        "/team/new",
        "/team/delete",
        "/team/block",
        "/team/unblock",
        "/key/generate",
        "/key/delete",
        "/key/regenerate",
    ],
)
def test_management_routes_bucket_allows_management_routes(route):
    """Bucket-name 'management_routes' should allow any route covered by management_routes."""
    assert _call(["management_routes"], route) == "ALLOW"


def test_info_routes_bucket_allows_key_info():
    assert _call(["info_routes"], "/key/info") == "ALLOW"


def test_info_routes_bucket_denies_management_route():
    """info_routes (a strict subset of management_routes) should NOT allow /team/new."""
    with pytest.raises(Exception) as exc_info:
        _call(["info_routes"], "/team/new")
    assert "Only proxy admin" in str(exc_info.value)


def test_management_routes_bucket_denies_unknown_route():
    """management_routes does not contain '/some/random/route' — must deny."""
    with pytest.raises(Exception) as exc_info:
        _call(["management_routes"], "/some/random/route")
    assert "Only proxy admin" in str(exc_info.value)


def test_unknown_bucket_string_falls_through_to_literal_match_and_denies():
    """A string that looks like a bucket name but is not a LiteLLMRoutes member
    should fall through to literal-path matching; since it doesn't match the
    request path either, the request must be denied."""
    with pytest.raises(Exception):
        _call(["not_a_real_bucket"], "/team/new")


def test_mixed_bucket_and_explicit_path_allows_bucket_route():
    """allowed_routes = ['management_routes', '/custom/route'] — /team/new (in bucket) allowed."""
    assert _call(["management_routes", "/custom/route"], "/team/new") == "ALLOW"


def test_mixed_bucket_and_explicit_path_allows_explicit_route():
    """allowed_routes = ['info_routes', '/custom/route'] — /custom/route (explicit) allowed."""
    assert _call(["info_routes", "/custom/route"], "/custom/route") == "ALLOW"


def test_mixed_bucket_and_wildcard_allows_wildcard_match():
    """allowed_routes = ['info_routes', '/admin/*'] — /admin/foo matched via wildcard."""
    assert _call(["info_routes", "/admin/*"], "/admin/foo") == "ALLOW"


def test_exact_match_still_works():
    assert _call(["/team/new"], "/team/new") == "ALLOW"


def test_wildcard_match_still_works():
    assert _call(["/team/*"], "/team/new") == "ALLOW"


def test_no_match_still_denies():
    with pytest.raises(Exception):
        _call(["/admin/x"], "/team/new")


def test_non_string_entries_are_skipped_safely():
    """allowed_routes could (in theory) contain a non-string entry; bucket
    expansion must require isinstance(str) and skip safely."""
    with pytest.raises(Exception):
        _call([123, None], "/team/new")


def test_bucket_name_then_explicit_path_allows_explicit():
    assert _call(["info_routes", "/team/new"], "/team/new") == "ALLOW"


def test_llm_api_routes_bucket_does_not_grant_arbitrary_passthrough():
    """A key with allowed_routes=['llm_api_routes'] must NOT be granted access to an
    arbitrary route via the fallback. Pass-through access is governed by
    check_passthrough_route_access (earlier gate), which consults explicit per-key
    / per-team allowed_passthrough_routes. The fallback must not have a blanket
    carve-out that bypasses that — Veria flagged exactly this in PR #28949 review.
    """
    arbitrary_route = "/azure-passthrough-only-explicitly-allowed/v1/whatever"
    assert arbitrary_route not in LiteLLMRoutes.llm_api_routes.value
    with pytest.raises(Exception) as exc_info:
        _call(["llm_api_routes"], arbitrary_route)
    assert "Only proxy admin" in str(exc_info.value)


@pytest.mark.parametrize(
    "bucket", ["management_routes", "info_routes", "llm_api_routes"]
)
def test_bucket_names_are_valid_litellm_routes_members(bucket):
    """If this assertion ever fails, handle_key_type wrote a bucket name that the
    expansion path doesn't recognize, and LIT-3300 would resurface."""
    assert bucket in LiteLLMRoutes._member_names_
