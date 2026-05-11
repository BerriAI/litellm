import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
import litellm.proxy.proxy_server
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLMRoutes,
    LiteLLM_JWTAuth,
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
    JWTRoutingOverride,
)
from litellm.proxy.auth.handle_jwt import JWTHandler
from litellm.proxy.auth.auth_checks import get_key_object, _cache_key_object
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import (
    _matches_routing_override,
    _reserve_budget_after_common_checks,
    _route_requires_auth_despite_public,
    _routing_selector_matches_claim,
    _run_centralized_common_checks,
    _run_post_custom_auth_checks,
    get_api_key,
    user_api_key_auth,
)


class _RoutingRequest:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}
        self.state = SimpleNamespace()


def test_get_api_key():
    bearer_token = "Bearer sk-12345678"
    api_key = "sk-12345678"
    passed_in_key = "Bearer sk-12345678"
    assert get_api_key(
        custom_litellm_key_header=None,
        api_key=bearer_token,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)


def test_route_requires_auth_despite_public_for_metrics(monkeypatch):
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", True)

    assert _route_requires_auth_despite_public("/metrics", {}) is True
    assert _route_requires_auth_despite_public("/metrics/", {}) is True

    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", False)

    assert _route_requires_auth_despite_public("/metrics", {}) is False


def test_public_ai_hub_routes_remain_public():
    for route in (
        "/public/model_hub",
        "/public/model_hub/info",
        "/public/agent_hub",
        "/public/mcp_hub",
        "/public/skill_hub",
    ):
        assert route in LiteLLMRoutes.public_routes.value
        assert _route_requires_auth_despite_public(route, {}) is False


@pytest.mark.asyncio
async def test_should_clear_stale_budget_reservation_when_budget_checks_skip():
    user_api_key_auth_obj = UserAPIKeyAuth(
        token="test_token",
        budget_reservation={
            "reserved_cost": 0.5,
            "entries": [{"counter_key": "spend:key:test_token"}],
        },
    )

    await _reserve_budget_after_common_checks(
        user_api_key_auth_obj=user_api_key_auth_obj,
        request_data={"model": "free-model"},
        route="/v1/chat/completions",
        llm_router=None,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=MagicMock(),
        proxy_logging_obj=MagicMock(),
        skip_budget_checks=True,
    )

    assert user_api_key_auth_obj.budget_reservation is None


@pytest.mark.asyncio
async def test_should_not_reuse_cached_key_object_for_request_state():
    key_cache = DualCache()
    cached_key = UserAPIKeyAuth(
        token="cached-token",
        request_route="/old-route",
        budget_reservation={
            "reserved_cost": 0.5,
            "entries": [{"counter_key": "spend:key:cached-token"}],
        },
    )

    await _cache_key_object(
        hashed_token="cached-token",
        user_api_key_obj=cached_key,
        user_api_key_cache=key_cache,
        proxy_logging_obj=None,
    )

    first_request_key = await get_key_object(
        hashed_token="cached-token",
        prisma_client=MagicMock(),
        user_api_key_cache=key_cache,
    )
    first_request_key.budget_reservation = {
        "reserved_cost": 0.9,
        "entries": [{"counter_key": "spend:key:cached-token"}],
    }
    first_request_key.request_route = "/chat/completions"

    second_request_key = await get_key_object(
        hashed_token="cached-token",
        prisma_client=MagicMock(),
        user_api_key_cache=key_cache,
    )

    assert first_request_key is not cached_key
    assert second_request_key is not first_request_key
    assert second_request_key.budget_reservation is None
    assert second_request_key.request_route is None


@pytest.mark.asyncio
async def test_custom_auth_does_not_enforce_key_model_access_by_default():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o"}

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            new_callable=AsyncMock,
        ) as mock_can_key,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data=request_data,
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_can_key.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_auth_honors_key_level_model_access_restriction_allowed_with_opt_in():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o-mini"}

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            new_callable=AsyncMock,
        ) as mock_can_key,
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data=request_data,
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_can_key.assert_awaited_once_with(
            model="gpt-4o-mini",
            llm_model_list=ANY,
            valid_token=valid_token,
            llm_router=ANY,
        )


@pytest.mark.asyncio
async def test_custom_auth_enforces_key_model_access_from_file_route_header_with_opt_in():
    valid_token = UserAPIKeyAuth(token="test_token", models=["allowed-model"])
    request = _RoutingRequest(headers={"x-litellm-model": "restricted-model"})

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            new_callable=AsyncMock,
        ) as mock_can_key,
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=request,
            request_data={},
            route="/v1/files",
            parent_otel_span=None,
        )
        mock_can_key.assert_awaited_once_with(
            model="restricted-model",
            llm_model_list=ANY,
            valid_token=valid_token,
            llm_router=ANY,
        )


@pytest.mark.asyncio
async def test_custom_auth_honors_key_level_model_access_restriction_denied_with_opt_in():
    valid_token = UserAPIKeyAuth(token="test_token", models=["gpt-4o-mini"])
    request_data = {"model": "gpt-4o"}

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            new_callable=AsyncMock,
        ) as mock_can_key,
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        mock_can_key.side_effect = ProxyException(
            message="Key not allowed to access model",
            type=ProxyErrorTypes.key_model_access_denied,
            param="model",
            code=401,
        )
        with pytest.raises(ProxyException) as exc:
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data=request_data,
                route="/v1/chat/completions",
                parent_otel_span=None,
            )

        assert exc.value.type == ProxyErrorTypes.key_model_access_denied


def _proxy_server_attrs_for_custom_auth(*, user_custom_auth):
    """
    Build the minimal set of proxy_server module attributes that
    _user_api_key_auth_builder reads when exercising a custom-auth return path.
    """
    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.delete_cache = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    return {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": user_custom_auth,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }


@pytest.mark.asyncio
async def test_user_custom_auth_skips_post_custom_auth_checks_by_default():
    """
    Regression test: after v1.82.6, _run_post_custom_auth_checks was unconditionally
    invoked on the user_custom_auth return path, which caused a ~44% RPS drop for
    custom-auth deployments due to per-request DB lookups on trusted tokens.
    The outer gate (litellm.enable_post_custom_auth_checks, default False) must
    short-circuit that call so the fast path returns the validated token unchanged.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    trusted_token = UserAPIKeyAuth(
        api_key="sk-custom-auth-trusted",
        user_id="custom-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    mock_user_custom_auth = AsyncMock(return_value=trusted_token)

    attrs = _proxy_server_attrs_for_custom_auth(user_custom_auth=mock_user_custom_auth)
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}
    original_flag = getattr(litellm, "enable_post_custom_auth_checks", False)

    try:
        for attr, val in attrs.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = False  # explicit: documents default

        with patch(
            "litellm.proxy.auth.user_api_key_auth._run_post_custom_auth_checks",
            new_callable=AsyncMock,
        ) as mock_post_checks:
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            result = await _user_api_key_auth_builder(
                request=request,
                api_key="Bearer sk-custom-auth-trusted",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

            mock_user_custom_auth.assert_awaited_once()
            mock_post_checks.assert_not_awaited()
            assert result.user_id == "custom-user-123"
    finally:
        for attr, val in originals.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = original_flag


@pytest.mark.asyncio
async def test_user_custom_auth_runs_post_custom_auth_checks_when_opt_in():
    """
    Opt-in half of the outer-gate regression test: when
    litellm.enable_post_custom_auth_checks=True, the user_custom_auth return path
    must invoke _run_post_custom_auth_checks so deployments that rely on the
    v1.82.6 DB-lookup behavior keep working after an explicit opt-in.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    trusted_token = UserAPIKeyAuth(
        api_key="sk-custom-auth-trusted",
        user_id="custom-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    mock_user_custom_auth = AsyncMock(return_value=trusted_token)

    attrs = _proxy_server_attrs_for_custom_auth(user_custom_auth=mock_user_custom_auth)
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}
    original_flag = getattr(litellm, "enable_post_custom_auth_checks", False)

    try:
        for attr, val in attrs.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = True

        with patch(
            "litellm.proxy.auth.user_api_key_auth._run_post_custom_auth_checks",
            new_callable=AsyncMock,
            return_value=trusted_token,
        ) as mock_post_checks:
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            await _user_api_key_auth_builder(
                request=request,
                api_key="Bearer sk-custom-auth-trusted",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

            mock_user_custom_auth.assert_awaited_once()
            mock_post_checks.assert_awaited_once()
    finally:
        for attr, val in originals.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = original_flag


@pytest.mark.asyncio
async def test_enterprise_custom_auth_skips_post_custom_auth_checks_by_default():
    """
    Mirror of test_user_custom_auth_skips_post_custom_auth_checks_by_default for the
    enterprise_custom_auth branch. Greptile explicitly asked for both branches to
    be covered in PR #24589 and the fix touches both return paths.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    trusted_token = UserAPIKeyAuth(
        api_key="sk-enterprise-custom-auth-trusted",
        user_id="enterprise-user-456",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    mock_enterprise_custom_auth = AsyncMock(return_value=trusted_token)

    attrs = _proxy_server_attrs_for_custom_auth(user_custom_auth=None)
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}
    original_flag = getattr(litellm, "enable_post_custom_auth_checks", False)

    try:
        for attr, val in attrs.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = False

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth",
                new=mock_enterprise_custom_auth,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._run_post_custom_auth_checks",
                new_callable=AsyncMock,
            ) as mock_post_checks,
        ):
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            result = await _user_api_key_auth_builder(
                request=request,
                api_key="Bearer sk-enterprise-custom-auth-trusted",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

            mock_enterprise_custom_auth.assert_awaited_once()
            mock_post_checks.assert_not_awaited()
            assert result.user_id == "enterprise-user-456"
    finally:
        for attr, val in originals.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = original_flag


@pytest.mark.asyncio
async def test_enterprise_custom_auth_runs_post_custom_auth_checks_when_opt_in():
    """
    Opt-in mirror for the enterprise_custom_auth branch: when the outer flag is
    set, _run_post_custom_auth_checks must still fire so users who depend on the
    v1.82.6 behavior have a working migration path.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm
    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    trusted_token = UserAPIKeyAuth(
        api_key="sk-enterprise-custom-auth-trusted",
        user_id="enterprise-user-456",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    mock_enterprise_custom_auth = AsyncMock(return_value=trusted_token)

    attrs = _proxy_server_attrs_for_custom_auth(user_custom_auth=None)
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}
    original_flag = getattr(litellm, "enable_post_custom_auth_checks", False)

    try:
        for attr, val in attrs.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = True

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth",
                new=mock_enterprise_custom_auth,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._run_post_custom_auth_checks",
                new_callable=AsyncMock,
                return_value=trusted_token,
            ) as mock_post_checks,
        ):
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            await _user_api_key_auth_builder(
                request=request,
                api_key="Bearer sk-enterprise-custom-auth-trusted",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

            mock_enterprise_custom_auth.assert_awaited_once()
            mock_post_checks.assert_awaited_once()
    finally:
        for attr, val in originals.items():
            setattr(_proxy_server_mod, attr, val)
        litellm.enable_post_custom_auth_checks = original_flag


def _assert_get_api_key_with_custom_litellm_key_header(
    custom_litellm_key_header, api_key, passed_in_key
):
    assert get_api_key(
        custom_litellm_key_header=custom_litellm_key_header,
        api_key=None,
        azure_api_key_header=None,
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        pass_through_endpoints=None,
        route="",
        request=MagicMock(),
    ) == (api_key, passed_in_key)


@pytest.mark.parametrize(
    "selector_value, claim_value, expected, split_space_delimited",
    [
        (None, "any-value", True, False),
        ("issuer.example.com", "issuer.example.com", True, False),
        ("issuer.example.com", "other-issuer.example.com", False, False),
        # iss (and other non-scope claims) must not match via space-split injection
        (
            "trusted.example.com",
            "trusted.example.com attacker.example.com",
            False,
            False,
        ),
        # Wildcard iss must not match space-containing claim strings (fnmatch * spans spaces)
        (
            "trusted.*",
            "trusted.example.com attacker.example.com",
            False,
            False,
        ),
        ("trusted.*", "trusted.example.com", True, False),
        (
            ["issuer-a.example.com", "issuer-b.example.com"],
            "issuer-b.example.com",
            True,
            False,
        ),
        ("*MID_LITELLM", "STREAM_MID_LITELLM", True, False),
        ("*MID_LITELLM", "REDIS_LITELLM", False, False),
        ("machine-??", "machine-01", True, False),
        ("machine-??", "machine-001", False, False),
        # Wildcard matching is case-sensitive (fnmatch.fnmatchcase)
        ("*litellm", "BATCH_LITELLM", False, False),
        ("*LITELLM", "BATCH_LITELLM", True, False),
        ("App:LiteLLM", "App:LiteLLM openid", True, True),
        ("App:*", "App:LiteLLM openid", True, True),
        (["openid", "App:LiteLLM"], "openid profile", True, True),
        (["service-*", "batch-*"], "batch-123", True, False),
        (["service-*", "batch-*"], "other-123", False, False),
        ("App:LiteLLM", ["openid", "App:LiteLLM"], True, False),
        ("App:LiteLLM", None, False, False),
    ],
)
def test_routing_selector_matches_claim_parametrized(
    selector_value, claim_value, expected, split_space_delimited
):
    assert (
        _routing_selector_matches_claim(
            selector_value=selector_value,
            claim_value=claim_value,
            split_space_delimited=split_space_delimited,
        )
        is expected
    )


@pytest.mark.parametrize(
    "override, token_claims, expected",
    [
        # Only iss selector is required and should match.
        (
            JWTRoutingOverride(iss="oauth-issuer.example.com", path="oauth2"),
            {"iss": "oauth-issuer.example.com"},
            True,
        ),
        # Scope selector narrows the match.
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                scope="App:LiteLLM",
                path="oauth2",
            ),
            {"iss": "oauth-issuer.example.com", "scope": "App:LiteLLM openid"},
            True,
        ),
        # client_id wildcard selector narrows the match.
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                client_id="*MID_LITELLM",
                path="oauth2",
            ),
            {"iss": "oauth-issuer.example.com", "client_id": "BATCH_MID_LITELLM"},
            True,
        ),
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                client_id="*MID_LITELLM",
                path="oauth2",
            ),
            {"iss": "oauth-issuer.example.com", "client_id": "BATCH_PORTAL"},
            False,
        ),
        # aud selector still works with list claims.
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                aud=["api://litellm", "api://fallback"],
                path="oauth2",
            ),
            {
                "iss": "oauth-issuer.example.com",
                "aud": ["api://other", "api://litellm"],
            },
            True,
        ),
        # All provided selectors are AND-ed.
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                scope="App:LiteLLM",
                client_id="*MID_LITELLM",
                path="oauth2",
            ),
            {
                "iss": "oauth-issuer.example.com",
                "scope": "App:LiteLLM openid",
                "client_id": "BATCH_MID_LITELLM",
            },
            True,
        ),
        (
            JWTRoutingOverride(
                iss="oauth-issuer.example.com",
                scope="App:LiteLLM",
                client_id="*MID_LITELLM",
                path="oauth2",
            ),
            {
                "iss": "oauth-issuer.example.com",
                "scope": "App:Other openid",
                "client_id": "BATCH_MID_LITELLM",
            },
            False,
        ),
    ],
)
def test_matches_routing_override_parametrized(override, token_claims, expected):
    assert (
        _matches_routing_override(token_claims=token_claims, override=override)
        is expected
    )


def test_get_api_key_with_custom_litellm_key_header_bearer_prefix():
    token = "sk-" + "1" * 8
    header = f"Bearer {token}"
    _assert_get_api_key_with_custom_litellm_key_header(
        custom_litellm_key_header=header, api_key=token, passed_in_key=header
    )


def test_get_api_key_with_custom_litellm_key_header_basic_prefix():
    token = "sk-" + "1" * 8
    header = f"Basic {token}"
    _assert_get_api_key_with_custom_litellm_key_header(
        custom_litellm_key_header=header, api_key=token, passed_in_key=header
    )


def test_get_api_key_with_custom_litellm_key_header_lowercase_bearer_prefix():
    token = "sk-" + "1" * 8
    header = f"bearer {token}"
    _assert_get_api_key_with_custom_litellm_key_header(
        custom_litellm_key_header=header, api_key=token, passed_in_key=header
    )


def test_get_api_key_with_custom_litellm_key_header_no_prefix():
    token = "sk-" + "1" * 8
    _assert_get_api_key_with_custom_litellm_key_header(
        custom_litellm_key_header=token, api_key=token, passed_in_key=token
    )


def test_get_api_key_with_custom_litellm_key_header_aws_sigv4():
    """AWS Signature V4 format (LangChain AWS SDK)."""
    token = "sk-" + "1" * 8
    header = (
        f"AWS4-HMAC-SHA256 Credential=Bearer {token}/20260210/us-east-1/bedrock/"
        "aws4_request, SignedHeaders=host, Signature=abc123"
    )
    _assert_get_api_key_with_custom_litellm_key_header(
        custom_litellm_key_header=header, api_key=token, passed_in_key=header
    )


def test_team_metadata_with_tags_flows_through_jwt_auth():
    """
    Test that team_metadata (specifically tags) flows through JWT authentication.

    This is a regression test for the issue where JWT auth was not populating
    team_metadata, causing team-level tags to be missing in litellm_pre_call_utils.py
    """
    from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth

    # Create a team object with metadata containing tags
    team_object = LiteLLM_TeamTable(
        team_id="test-team-id",
        team_alias="test-team-alias",
        metadata={"tags": ["production", "high-priority"], "department": "engineering"},
        tpm_limit=1000,
        rpm_limit=100,
        models=["gpt-4", "gpt-3.5-turbo"],
    )

    # Simulate constructing UserAPIKeyAuth like we do in JWT auth
    # This is the pattern from user_api_key_auth.py lines 552-587
    user_api_key_auth = UserAPIKeyAuth(
        api_key=None,
        team_id=team_object.team_id,
        team_tpm_limit=team_object.tpm_limit if team_object is not None else None,
        team_rpm_limit=team_object.rpm_limit if team_object is not None else None,
        team_models=team_object.models if team_object is not None else [],
        team_metadata=team_object.metadata if team_object is not None else None,
        user_role="internal_user",
        user_id="test-user",
    )

    # Verify team_metadata is set
    assert (
        user_api_key_auth.team_metadata is not None
    ), "team_metadata should be populated"
    assert user_api_key_auth.team_metadata == team_object.metadata, (
        f"team_metadata not correctly mapped. "
        f"Expected: {team_object.metadata}, Got: {user_api_key_auth.team_metadata}"
    )

    # Specifically verify tags are present
    assert "tags" in user_api_key_auth.team_metadata, "tags should be in team_metadata"
    assert user_api_key_auth.team_metadata["tags"] == ["production", "high-priority"], (
        f"tags not correctly mapped. "
        f"Expected: ['production', 'high-priority'], Got: {user_api_key_auth.team_metadata.get('tags')}"
    )


def test_route_checks_is_llm_api_route():
    """Test RouteChecks.is_llm_api_route() correctly identifies LLM API routes including passthrough endpoints"""

    # Test OpenAI routes
    openai_routes = [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/completions",
        "/completions",
        "/v1/embeddings",
        "/embeddings",
        "/v1/images/generations",
        "/images/generations",
        "/v1/audio/transcriptions",
        "/audio/transcriptions",
        "/v1/audio/speech",
        "/audio/speech",
        "/v1/moderations",
        "/moderations",
        "/v1/models",
        "/models",
        "/v1/rerank",
        "/rerank",
        "/v1/realtime",
        "/realtime",
    ]

    for route in openai_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test Anthropic routes
    anthropic_routes = [
        "/v1/messages",
        "/v1/messages/count_tokens",
    ]

    for route in anthropic_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test passthrough routes (this is the key improvement over the old route checking)
    passthrough_routes = [
        "/bedrock/v1/chat/completions",
        "/vertex-ai/v1/chat/completions",
        "/vertex_ai/v1/chat/completions",
        "/cohere/v1/chat/completions",
        "/gemini/v1/chat/completions",
        "/anthropic/v1/messages",
        "/langfuse/v1/chat/completions",
        "/azure/v1/chat/completions",
        "/openai/v1/chat/completions",
        "/assemblyai/v1/transcript",
        "/eu.assemblyai/v1/transcript",
        "/vllm/v1/chat/completions",
        "/mistral/v1/chat/completions",
    ]

    for route in passthrough_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test MCP routes
    mcp_routes = [
        "/mcp",
        "/mcp/",
        "/mcp/test",
    ]

    for route in mcp_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test LiteLLM native RAG routes
    rag_routes = [
        "/rag/ingest",
        "/v1/rag/ingest",
        "/rag/query",
        "/v1/rag/query",
    ]
    for route in rag_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test routes with placeholders
    placeholder_routes = [
        "/v1/threads/thread_49EIN5QF32s4mH20M7GFKdlZ",
        "/threads/thread_49EIN5QF32s4mH20M7GFKdlZ",
        "/v1/assistants/assistant_123",
        "/assistants/assistant_123",
        "/v1/files/file_123",
        "/files/file_123",
        "/v1/batches/batch_123",
        "/batches/batch_123",
    ]

    for route in placeholder_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test Azure OpenAI routes
    azure_routes = [
        "/openai/deployments/gpt-4/chat/completions",
        "/openai/deployments/gpt-3.5-turbo/completions",
        "/engines/gpt-4/chat/completions",
        "/engines/gpt-3.5-turbo/completions",
    ]

    for route in azure_routes:
        assert RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should be identified as LLM API route"

    # Test non-LLM routes (should return False)
    non_llm_routes = [
        "/health",
        "/metrics",
        "/key/list",
        "/team/list",
        "/user/list",
        "/config",
        "/routes",
        "/",
        "/admin/settings",
        "/logs",
        "/debug",
        "/test",
    ]

    for route in non_llm_routes:
        assert not RouteChecks.is_llm_api_route(
            route=route
        ), f"Route {route} should NOT be identified as LLM API route"

    # Test invalid inputs
    invalid_inputs = [
        None,
        123,
        [],
        {},
        "",
    ]

    for invalid_input in invalid_inputs:
        assert not RouteChecks.is_llm_api_route(
            route=invalid_input
        ), f"Invalid input {invalid_input} should return False"


@pytest.mark.asyncio
async def test_proxy_admin_expired_key_from_cache():
    """
    Test that PROXY_ADMIN keys retrieved from cache are checked for expiration
    before being returned. This prevents expired keys from bypassing expiration checks
    when retrieved from cache (which normally happens at lines 1014-1036).

    Regression test for issue where PROXY_ADMIN keys from cache skipped expiration check.
    """
    from datetime import datetime, timedelta, timezone

    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LitellmUserRoles,
        ProxyErrorTypes,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    from litellm.proxy.proxy_server import hash_token

    # Create an expired PROXY_ADMIN key
    api_key = "sk-test-proxy-admin-key"
    hashed_key = hash_token(api_key)
    expired_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Expired 1 hour ago

    expired_token = UserAPIKeyAuth(
        api_key=api_key,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        expires=expired_time,
        token=hashed_key,
    )

    # Mock cache to return the expired token
    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=expired_token)
    mock_cache.delete_cache = MagicMock()

    # Mock proxy_logging_obj
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    # Mock post_call_failure_hook as async function returning None (no transformation)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    # Mock prisma_client
    mock_prisma_client = MagicMock()

    # Mock get_key_object to return expired token from cache
    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_key_object",
            new_callable=AsyncMock,
        ) as mock_get_key_object,
        patch(
            "litellm.proxy.auth.user_api_key_auth._delete_cache_key_object",
            new_callable=AsyncMock,
        ) as mock_delete_cache,
    ):
        mock_get_key_object.return_value = expired_token

        # Set attributes on proxy_server module (these are imported inside _user_api_key_auth_builder)
        import litellm.proxy.proxy_server as _proxy_server_mod

        _attrs_to_set = {
            "prisma_client": mock_prisma_client,
            "user_api_key_cache": mock_cache,
            "proxy_logging_obj": mock_proxy_logging_obj,
            "master_key": "sk-master-key",
            "general_settings": {},
            "llm_model_list": [],
            "llm_router": None,
            "open_telemetry_logger": None,
            "model_max_budget_limiter": MagicMock(),
            "user_custom_auth": None,
            "jwt_handler": None,
            "litellm_proxy_admin_name": "admin",
        }
        _original_values = {
            attr: getattr(_proxy_server_mod, attr, None) for attr in _attrs_to_set
        }
        try:
            for attr, val in _attrs_to_set.items():
                setattr(_proxy_server_mod, attr, val)

            # Create a mock request
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")
            request_data = {}

            # Call the auth builder - should raise ProxyException for expired key
            # Note: api_key needs "Bearer " prefix for get_api_key() to process it correctly
            with pytest.raises(ProxyException) as exc_info:
                await _user_api_key_auth_builder(
                    request=request,
                    api_key=f"Bearer {api_key}",  # Add Bearer prefix
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data=request_data,
                )

            # Verify that ProxyException was raised with expired_key type
            assert hasattr(
                exc_info.value, "type"
            ), "Exception should have 'type' attribute"
            assert (
                exc_info.value.type == ProxyErrorTypes.expired_key
            ), f"Expected expired_key error type, got {exc_info.value.type}"
            assert "Expired Key" in str(
                exc_info.value.message
            ), f"Exception message should mention 'Expired Key', got: {exc_info.value.message}"

            # Verify that the param field does NOT leak the full API key (Issue #18731)
            # The param should be abbreviated like "sk-...XXXX" not the full plaintext key
            assert (
                exc_info.value.param is not None
            ), "Exception should have 'param' attribute"
            assert exc_info.value.param != api_key, (
                f"SECURITY: Full API key should NOT be in param field! "
                f"Got: {exc_info.value.param}, Expected abbreviated format like 'sk-...XXXX'"
            )
            assert exc_info.value.param.startswith(
                "sk-..."
            ), f"Param should be abbreviated to 'sk-...XXXX' format. Got: {exc_info.value.param}"

            # Verify that cache deletion was called
            mock_delete_cache.assert_called_once()
            call_args = mock_delete_cache.call_args
            assert (
                call_args[1]["hashed_token"] == hashed_key
            ), "Cache deletion should be called with the hashed key"
        finally:
            # Restore all module-level attributes so subsequent tests are not affected
            for attr, val in _original_values.items():
                setattr(_proxy_server_mod, attr, val)


@pytest.mark.asyncio
async def test_scim_deactivated_user_key_is_rejected():
    """A virtual key whose owning user has metadata.scim_active=False must be
    rejected by the auth flow (defense in depth on top of key-level blocking).
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    from litellm.proxy.proxy_server import hash_token

    api_key = "sk-scim-deactivated-user-key"
    hashed_key = hash_token(api_key)

    valid_token = UserAPIKeyAuth(
        api_key=api_key,
        token=hashed_key,
        user_id="scim-disabled-user",
    )
    deactivated_user = LiteLLM_UserTable(
        user_id="scim-disabled-user",
        metadata={"scim_active": False},
    )

    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.delete_cache = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    mock_prisma_client = MagicMock()

    import litellm.proxy.proxy_server as _proxy_server_mod

    _attrs_to_set = {
        "prisma_client": mock_prisma_client,
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    _original_values = {
        attr: getattr(_proxy_server_mod, attr, None) for attr in _attrs_to_set
    }
    try:
        for attr, val in _attrs_to_set.items():
            setattr(_proxy_server_mod, attr, val)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=valid_token,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                return_value=deactivated_user,
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await _user_api_key_auth_builder(
                    request=request,
                    api_key=f"Bearer {api_key}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={},
                )

        assert "deactivated via SCIM" in str(exc_info.value.message)
    finally:
        for attr, val in _original_values.items():
            setattr(_proxy_server_mod, attr, val)


@pytest.mark.asyncio
async def test_return_user_api_key_auth_obj_user_spend_and_budget():
    """
    Test that _return_user_api_key_auth_obj correctly sets user_spend and user_max_budget
    from user_obj attributes.
    """
    from datetime import datetime

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _return_user_api_key_auth_obj

    user_obj = type(
        "LiteLLM_UserTable",
        (),
        {
            "tpm_limit": 1000,
            "rpm_limit": 100,
            "user_email": "test@example.com",
            "spend": 250.0,
            "max_budget": 1000.0,
            "user_role": "internal_user",
        },
    )

    api_key = "sk-test-key"
    valid_token_dict = {
        "user_id": "test-user",
        "org_id": "test-org",
    }
    route = "/chat/completions"
    start_time = datetime.now()

    mock_service_logger = MagicMock()
    mock_service_logger.async_service_success_hook = AsyncMock()

    with patch(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_service_logger_obj",
        new=mock_service_logger,
    ):
        result = await _return_user_api_key_auth_obj(
            user_obj=user_obj,
            api_key=api_key,
            parent_otel_span=None,
            valid_token_dict=valid_token_dict,
            route=route,
            start_time=start_time,
            user_role=None,
        )

    assert isinstance(result, UserAPIKeyAuth)
    assert result.user_spend == 250.0
    assert result.user_max_budget == 1000.0
    assert result.user_tpm_limit == 1000
    assert result.user_rpm_limit == 100
    assert result.user_email == "test@example.com"


def test_proxy_admin_jwt_auth_includes_identity_fields():
    """
    Test that the proxy admin early-return path in JWT auth populates
    user_id, team_id, team_alias, team_metadata, org_id, and end_user_id.

    Regression test: previously the is_proxy_admin branch only set user_role
    and parent_otel_span, discarding all identity fields resolved from the JWT.
    This caused blank Team Name and Internal User in Request Logs UI.
    """
    from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth

    team_object = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="my-team",
        metadata={"tags": ["prod"], "env": "production"},
    )

    # Simulate the proxy admin early-return path (user_api_key_auth.py ~line 586)
    result = UserAPIKeyAuth(
        api_key=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="user-abc",
        team_id="team-123",
        team_alias=(team_object.team_alias if team_object is not None else None),
        team_metadata=team_object.metadata if team_object is not None else None,
        org_id="org-456",
        end_user_id="end-user-789",
        parent_otel_span=None,
    )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
    assert result.user_id == "user-abc"
    assert result.team_id == "team-123"
    assert result.team_alias == "my-team"
    assert result.team_metadata == {"tags": ["prod"], "env": "production"}
    assert result.org_id == "org-456"
    assert result.end_user_id == "end-user-789"
    assert result.api_key is None


def test_proxy_admin_jwt_auth_handles_no_team_object():
    """
    Test that the proxy admin early-return path works correctly when
    team_object is None (user has admin role but no team association).
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    team_object = None

    result = UserAPIKeyAuth(
        api_key=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        team_id=None,
        team_alias=(team_object.team_alias if team_object is not None else None),
        team_metadata=team_object.metadata if team_object is not None else None,
        org_id=None,
        end_user_id=None,
        parent_otel_span=None,
    )

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
    assert result.user_id == "admin-user"
    assert result.team_id is None
    assert result.team_alias is None
    assert result.team_metadata is None
    assert result.org_id is None
    assert result.end_user_id is None


class TestJWTOAuth2Coexistence:
    """
    Test that JWT and OAuth2 auth can coexist on the same instance.

    When both enable_jwt_auth and enable_oauth2_auth are True, the proxy should
    route tokens based on their format:
    - JWT tokens (3 dot-separated parts) -> JWT auth handler
    - Opaque tokens -> OAuth2 auth handler
    """

    def test_is_jwt_detects_jwt_tokens(self):
        """JWT tokens have 3 dot-separated parts."""
        assert JWTHandler.is_jwt("header.payload.signature") is True
        assert (
            JWTHandler.is_jwt("eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.sig123")
            is True
        )

    def test_is_jwt_rejects_opaque_tokens(self):
        """Opaque OAuth2 tokens do not have 3 dot-separated parts."""
        assert JWTHandler.is_jwt("some-opaque-oauth2-token") is False
        assert JWTHandler.is_jwt("sk-12345678") is False
        assert JWTHandler.is_jwt("Bearer token") is False
        assert JWTHandler.is_jwt("two.parts") is False

    def test_is_jwt_returns_false_for_none(self):
        """None token (missing Authorization header) should not be treated as JWT."""
        assert JWTHandler.is_jwt(None) is False

    @pytest.mark.asyncio
    async def test_both_enabled_opaque_token_uses_oauth2(self):
        """
        When both enable_jwt_auth and enable_oauth2_auth are True,
        an opaque token should be handled by OAuth2 auth (not JWT).
        """
        opaque_token = "some-opaque-m2m-oauth2-token"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }

        mock_oauth2_response = UserAPIKeyAuth(
            api_key=opaque_token,
            user_id="machine-client-1",
            team_id="m2m-team",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {opaque_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {opaque_token}",
            )

            # OAuth2 SHOULD be called for opaque tokens
            mock_oauth2.assert_called_once_with(token=opaque_token)
            # JWT auth should NOT be called
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-1"

    @pytest.mark.asyncio
    async def test_oauth2_path_requires_premium_user(self):
        """
        OAuth2 token validation should fail when enterprise premium is disabled.
        """
        opaque_token = "some-opaque-m2m-oauth2-token"
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {opaque_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", False),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(),
            )

            with pytest.raises(ProxyException) as exc_info:
                await user_api_key_auth(
                    request=mock_request,
                    api_key=f"Bearer {opaque_token}",
                )

            assert exc_info.value.type == ProxyErrorTypes.auth_error
            assert (
                "Oauth2 token validation is only available for premium users"
                in exc_info.value.message
            )
            mock_oauth2.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_enabled_jwt_token_skips_oauth2(self):
        """
        When both enable_jwt_auth and enable_oauth2_auth are True,
        a JWT-formatted token should skip OAuth2 and reach the JWT handler.
        """
        jwt_token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.signature"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }

        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": "jwt-team",
            "user_id": "jwt-human-user",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {"sub": "user1"},
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
                return_value=mock_jwt_result,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            # OAuth2 should NOT be called for JWT tokens
            mock_oauth2.assert_not_called()
            # JWT auth SHOULD be called
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-human-user"

    @pytest.mark.asyncio
    async def test_routing_override_routes_matching_jwt_to_oauth2(self):
        """
        When routing_overrides match JWT claims, route JWT-shaped token to OAuth2.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-override",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-override"

    @pytest.mark.asyncio
    async def test_routing_override_does_not_match_client_id_falls_back_to_jwt(self):
        """
        If override ISS matches but client_id does not, continue default JWT flow.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6IlVTRVJfUE9SVEFMIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": "jwt-team",
            "user_id": "jwt-user-no-override",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {"sub": "user1"},
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
                return_value=mock_jwt_result,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_not_called()
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-user-no-override"

    @pytest.mark.asyncio
    async def test_routing_override_matches_aud_claim_list_and_list_selectors(self):
        """
        Match routing override when selectors are lists and token aud claim is a list.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIiwiYXVkIjpbImFwaTovL2xpdGVsbG0iLCJhcGk6Ly9vdGhlciJdfQ."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-aud-list",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss=[
                                "machine-issuer.example.com",
                                "other-issuer.example.com",
                            ],
                            client_id=["MID_LITELLM", "MID_BACKUP"],
                            aud=["api://litellm", "api://fallback"],
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-aud-list"

    @pytest.mark.asyncio
    async def test_routing_override_matches_scope_claim(self):
        """
        Match routing override when scope selector is configured and scope claim matches.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJvYXV0aC1pc3N1ZXIuZXhhbXBsZS5jb20iLCJzY29wZSI6IkFwcDpMaXRlTExNIiwiY2xpZW50X2lkIjoiTUFDSElORV9NSURfTElURUxMTSJ9."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-scope-match",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="oauth-issuer.example.com",
                            scope="App:LiteLLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-scope-match"

    @pytest.mark.asyncio
    async def test_routing_override_scope_mismatch_falls_back_to_jwt(self):
        """
        If scope selector does not match, continue default JWT flow.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJvYXV0aC1pc3N1ZXIuZXhhbXBsZS5jb20iLCJzY29wZSI6IkFwcDpPdGhlciIsImNsaWVudF9pZCI6IlBPUlRBTF9NSURfTElURUxMTSJ9."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": "jwt-team",
            "user_id": "jwt-user-scope-mismatch",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {"sub": "user1"},
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
                return_value=mock_jwt_result,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="oauth-issuer.example.com",
                            scope="App:LiteLLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_not_called()
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-user-scope-mismatch"

    @pytest.mark.asyncio
    async def test_routing_override_matches_scope_and_client_wildcard_when_scope_claim_is_space_delimited(
        self,
    ):
        """
        Integration check: combined scope + wildcard selectors match on OAuth2 path
        when scope claim is a space-delimited string.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJvYXV0aC1pc3N1ZXIuZXhhbXBsZS5jb20iLCJzY29wZSI6IkFwcDpMaXRlTExNIG9wZW5pZCIsImNsaWVudF9pZCI6IkJBVENIX01JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-space-delimited-scope-match",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="oauth-issuer.example.com",
                            scope="App:LiteLLM",
                            client_id="*MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-space-delimited-scope-match"

    @pytest.mark.asyncio
    async def test_routing_override_routes_jwt_to_oauth2_when_oauth2_globally_disabled(
        self,
    ):
        """
        If enable_oauth2_auth is false, JWT tokens matching routing_overrides
        should still route to OAuth2 introspection.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-override-oauth2-off",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-override-oauth2-off"

    @pytest.mark.asyncio
    async def test_opaque_token_does_not_use_oauth2_when_oauth2_globally_disabled(
        self,
    ):
        """
        With enable_oauth2_auth=false, opaque tokens must not be sent to OAuth2.
        """
        opaque_token = "sk-ui-session-token"
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {opaque_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
        ):
            with pytest.raises(ProxyException) as exc_info:
                await user_api_key_auth(
                    request=mock_request,
                    api_key=f"Bearer {opaque_token}",
                )

            assert exc_info.value.type in (
                ProxyErrorTypes.auth_error,
                ProxyErrorTypes.no_db_connection,
            )
            mock_oauth2.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_override_on_info_route_uses_oauth2_when_oauth2_globally_disabled(
        self,
    ):
        """
        With enable_oauth2_auth=false, a JWT matching routing_overrides should
        still route to OAuth2 on info routes.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_token,
            user_id="machine-client-info-override-oauth2-off",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/team/list"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_called_once_with(token=jwt_token)
            mock_jwt_auth.assert_not_called()
            assert result.user_id == "machine-client-info-override-oauth2-off"

    @pytest.mark.asyncio
    async def test_routing_override_on_management_route_does_not_use_oauth2(self):
        """
        JWT routing_overrides should not force OAuth2 on management routes.
        """
        jwt_token = (
            "eyJhbGciOiJSUzI1NiJ9."
            "eyJpc3MiOiJtYWNoaW5lLWlzc3Vlci5leGFtcGxlLmNvbSIsImNsaWVudF9pZCI6Ik1JRF9MSVRFTExNIn0."
            "c2ln"
        )
        general_settings = {
            "enable_oauth2_auth": False,
            "enable_jwt_auth": True,
        }
        mock_jwt_result = {
            "is_proxy_admin": True,
            "team_object": None,
            "user_object": None,
            "end_user_object": None,
            "org_object": None,
            "token": jwt_token,
            "team_id": None,
            "user_id": "jwt-admin-user",
            "end_user_id": None,
            "org_id": None,
            "team_membership": None,
            "jwt_claims": {
                "iss": "machine-issuer.example.com",
                "client_id": "MID_LITELLM",
            },
        }

        mock_request = MagicMock()
        mock_request.url.path = "/key/generate"
        mock_request.headers = {"authorization": f"Bearer {jwt_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
            ) as mock_oauth2,
            patch(
                "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
                new_callable=AsyncMock,
                return_value=mock_jwt_result,
            ) as mock_jwt_auth,
        ):
            litellm.proxy.proxy_server.jwt_handler.update_environment(
                prisma_client=None,
                user_api_key_cache=DualCache(),
                litellm_jwtauth=LiteLLM_JWTAuth(
                    routing_overrides=[
                        JWTRoutingOverride(
                            iss="machine-issuer.example.com",
                            client_id="MID_LITELLM",
                            path="oauth2",
                        )
                    ]
                ),
            )

            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_token}",
            )

            mock_oauth2.assert_not_called()
            mock_jwt_auth.assert_called_once()
            assert result.user_id == "jwt-admin-user"

    @pytest.mark.asyncio
    async def test_only_oauth2_enabled_handles_all_tokens(self):
        """
        When only enable_oauth2_auth is True (no JWT), all LLM API tokens
        should go through OAuth2 - backward compatible behavior.
        """
        jwt_like_token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.signature"

        general_settings = {
            "enable_oauth2_auth": True,
            "enable_jwt_auth": False,
        }

        mock_oauth2_response = UserAPIKeyAuth(
            api_key=jwt_like_token,
            user_id="oauth2-user",
        )

        mock_request = MagicMock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {"authorization": f"Bearer {jwt_like_token}"}
        mock_request.query_params = {}

        with (
            patch("litellm.proxy.proxy_server.general_settings", general_settings),
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.master_key", "sk-master"),
            patch("litellm.proxy.proxy_server.prisma_client", None),
            patch(
                "litellm.proxy.auth.user_api_key_auth.Oauth2Handler.check_oauth2_token",
                new_callable=AsyncMock,
                return_value=mock_oauth2_response,
            ) as mock_oauth2,
        ):
            result = await user_api_key_auth(
                request=mock_request,
                api_key=f"Bearer {jwt_like_token}",
            )

            # OAuth2 should handle it since JWT auth is disabled
            mock_oauth2.assert_called_once_with(token=jwt_like_token)
            assert result.user_id == "oauth2-user"


@pytest.mark.asyncio
async def test_user_api_key_auth_builder_no_blocking_calls():
    """
    _user_api_key_auth_builder must never call any synchronous DualCache method
    (set_cache, get_cache, batch_get_cache, increment_cache, delete_cache) on
    the hot auth path — those methods call Redis synchronously and block the
    event loop. Only async_* variants are allowed.
    """
    from starlette.datastructures import URL
    from starlette.requests import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    _blocking_methods = [
        "set_cache",
        "get_cache",
        "batch_get_cache",
        "increment_cache",
        "delete_cache",
    ]

    api_key = "sk-test-no-blocking-cache"
    valid_token = UserAPIKeyAuth(
        api_key=api_key,
        token=api_key,
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team-abc",
    )

    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=valid_token)
    mock_cache.async_set_cache = AsyncMock(return_value=None)
    # Wire sync methods on the instance as plain MagicMocks (no side_effect) so
    # calls are recorded but not raised — the function's broad except Exception
    # would swallow a raised error. We assert not_called() after the run instead.
    for _m in _blocking_methods:
        setattr(mock_cache, _m, MagicMock())

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    import litellm.proxy.proxy_server as _proxy_server_mod

    _attrs = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    _originals = {k: getattr(_proxy_server_mod, k, None) for k in _attrs}

    try:
        for k, v in _attrs.items():
            setattr(_proxy_server_mod, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        import contextlib

        from litellm.caching.dual_cache import DualCache

        blocking_patches = [
            patch.object(
                DualCache,
                m,
                MagicMock(
                    side_effect=AssertionError(
                        f"Blocking DualCache.{m}() called on async hot path — use async_{m}() instead"
                    )
                ),
            )
            for m in _blocking_methods
        ]

        with contextlib.ExitStack() as stack:
            for p in blocking_patches:
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_key_object",
                    new_callable=AsyncMock,
                    return_value=valid_token,
                )
            )
            stack.enter_context(
                patch(
                    "litellm.proxy.auth.user_api_key_auth.get_team_object",
                    new_callable=AsyncMock,
                    return_value=None,
                )
            )
            await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {api_key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

        for _m in _blocking_methods:
            mock = getattr(mock_cache, _m)
            assert mock.call_count == 0, (
                f"Blocking DualCache.{_m}() was called {mock.call_count} time(s) "
                f"on the async hot path — use async_{_m}() instead"
            )

    finally:
        for k, v in _originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_team_metadata_refreshed_from_team_object_during_auth():
    """
    Regression test: when a cached API key has stale team_metadata (e.g. a
    guardrail was added to the team after the key was cached), the auth flow
    must update valid_token.team_metadata from the freshly fetched team object
    so that move_guardrails_to_metadata picks up the new guardrail.

    Before the fix: valid_token.team_metadata was never updated from _team_obj
    at the "Check 6" team-auth step in _user_api_key_auth_builder, so stale
    team_metadata persisted for the lifetime of the key cache entry.
    """
    from starlette.datastructures import URL
    from starlette.requests import Request

    from litellm.proxy._types import (
        LiteLLM_TeamTableCachedObj,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    api_key = "sk-test-team-metadata-refresh"

    # Simulate a cached key whose team_metadata was captured BEFORE the
    # guardrail was added — so it has no "guardrails" key.
    stale_team_metadata: dict = {"some_old_key": "some_old_value"}
    valid_token = UserAPIKeyAuth(
        api_key=api_key,
        token=api_key,
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team-guardrail-test",
        team_metadata=stale_team_metadata,
    )

    # The fresh team object returned by get_team_object has the new guardrail.
    fresh_team_obj = LiteLLM_TeamTableCachedObj(
        team_id="team-guardrail-test",
        metadata={"guardrails": ["test-guardrail-333"]},
    )

    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=valid_token)
    mock_cache.async_set_cache = AsyncMock(return_value=None)

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    import litellm.proxy.proxy_server as _proxy_server_mod

    _attrs = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    _originals = {k: getattr(_proxy_server_mod, k, None) for k in _attrs}

    try:
        for k, v in _attrs.items():
            setattr(_proxy_server_mod, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=valid_token,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_team_object",
                new_callable=AsyncMock,
                return_value=fresh_team_obj,
            ),
        ):
            result = await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {api_key}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

        assert result.team_metadata == {
            "guardrails": ["test-guardrail-333"]
        }, f"team_metadata was not updated from fresh team object. Got: {result.team_metadata}"

    finally:
        for k, v in _originals.items():
            setattr(_proxy_server_mod, k, v)


# ---------------------------------------------------------------------------

# _run_centralized_common_checks — centralized authz gate
# ---------------------------------------------------------------------------


def _proxy_attrs_for_centralized_checks(
    user_custom_auth=None, flag=False, master_key="sk-test-master"
):
    """Build the minimal proxy_server module attributes that
    _run_centralized_common_checks reads.

    ``master_key`` defaults to a non-None value because setting it to
    None short-circuits the gate (no-auth dev mode); tests that want to
    exercise that branch must pass ``master_key=None`` explicitly.
    """
    return {
        "prisma_client": None,
        "user_api_key_cache": DualCache(),
        "proxy_logging_obj": MagicMock(),
        "general_settings": ({"custom_auth_run_common_checks": True} if flag else {}),
        "llm_router": None,
        "user_custom_auth": user_custom_auth,
        "litellm_proxy_admin_name": "admin",
        "master_key": master_key,
    }


@pytest.mark.asyncio
async def test_centralized_common_checks_runs_for_standard_auth():
    """Regardless of which _user_api_key_auth_builder path returned, the
    wrapper must run common_checks. This is the structural fix: no
    early-return path can skip authorization."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"model": "gpt-4o"},
                route="/chat/completions",
            )
            mock_checks.assert_awaited_once()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_skipped_for_custom_auth_without_flag():
    """Existing RPS guarantee: custom-auth deployments without
    custom_auth_run_common_checks must not pay the centralized gate.
    Custom-auth paths don't use OAuth2/DB-fallback so this skip does
    not widen any bypass."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(
        user_custom_auth=AsyncMock(), flag=False
    )
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"model": "gpt-4o"},
                route="/chat/completions",
            )
            mock_checks.assert_not_awaited()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_runs_for_custom_auth_with_flag():
    """Custom-auth deployments that opt in via custom_auth_run_common_checks
    get the centralized gate."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=AsyncMock(), flag=True)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"model": "gpt-4o"},
                route="/chat/completions",
            )
            mock_checks.assert_awaited_once()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_runs_for_oauth2_fallback_token():
    """VERIA-18 regression: an OAuth2 token that would previously early-
    return without common_checks is now subject to it. If common_checks
    raises, the gate propagates the failure."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="oauth2-token", user_id="oauth-user")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
            side_effect=ProxyException(
                message="Key not allowed to access model",
                type=ProxyErrorTypes.key_model_access_denied,
                param="model",
                code=401,
            ),
        ):
            with pytest.raises(ProxyException) as exc:
                await _run_centralized_common_checks(
                    user_api_key_auth_obj=token,
                    request=request,
                    request_data={"model": "gpt-4"},
                    route="/chat/completions",
                )
            assert exc.value.type == ProxyErrorTypes.key_model_access_denied
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_tolerates_db_errors_when_fetching_context():
    """DB-outage scenario: the fallback UserAPIKeyAuth is issued when the
    DB is down, then the gate tries to fetch team/user/etc. Those fetches
    fail — the gate must swallow and still call common_checks with None
    objects so enforcement runs against whatever the token recorded."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy.auth.auth_exception_handler import (
        DB_UNAVAILABLE_FALLBACK_USER_ID,
    )

    token = UserAPIKeyAuth(
        api_key="fallback",
        user_id=DB_UNAVAILABLE_FALLBACK_USER_ID,
        team_id="team-x",
    )
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_team_object",
                new_callable=AsyncMock,
                side_effect=Exception("DB down"),
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"model": "gpt-4o"},
                route="/chat/completions",
            )
            mock_checks.assert_awaited_once()
            # team_object kwarg should have ended up as None after the
            # DB fetch was swallowed.
            assert mock_checks.call_args.kwargs["team_object"] is None
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_propagates_end_user_budget_error():
    """Regression: ``get_end_user_object`` raises ``litellm.BudgetExceededError``
    internally when an end user is over budget. ``_safe_fetch`` must
    re-raise it so the wrapper surfaces the budget violation, rather
    than swallowing it and letting ``common_checks`` see
    ``end_user_object=None`` and skip enforcement."""
    import litellm
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u", end_user_id="alice")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    request._body = json.dumps({"user": "alice", "model": "gpt-4o"}).encode()

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                side_effect=litellm.BudgetExceededError(
                    message="End-user budget exceeded",
                    current_cost=20.0,
                    max_budget=10.0,
                ),
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            with pytest.raises(litellm.BudgetExceededError):
                await _run_centralized_common_checks(
                    user_api_key_auth_obj=token,
                    request=request,
                    request_data={"user": "alice", "model": "gpt-4o"},
                    route="/chat/completions",
                )
            # common_checks must not be invoked if the budget violation
            # propagates from the context gathering — the wrapper should
            # fail before reaching it.
            mock_checks.assert_not_awaited()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_reserves_request_end_user_budget():
    """Regression: reservation runs before user_api_key_auth() copies the
    request end-user onto the token, so centralized checks must pass the
    locally extracted end_user_id/end_user_object into reservation."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u")
    request = Request(scope={"type": "http", "headers": []})
    request._url = URL(url="/chat/completions")
    request_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "user": "alice",
    }
    end_user_object = LiteLLM_EndUserTable(
        user_id="alice",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    counter_cache = DualCache()
    attrs["spend_counter_cache"] = counter_cache
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=end_user_object,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
                return_value=0.6,
            ),
        ):
            assert token.end_user_id is None

            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data=request_data,
                route="/chat/completions",
            )

    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)

    assert token.end_user_id is None
    assert token.budget_reservation is not None
    assert token.budget_reservation["entries"] == [
        {
            "counter_key": "spend:end_user:alice",
            "entity_type": "EndUser",
            "entity_id": "alice",
            "reserved_cost": 0.6,
            "applied_adjustment": 0.0,
        }
    ]
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:end_user:alice"
    ) == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_centralized_common_checks_short_circuits_when_master_key_unset():
    """master_key=None is no-auth dev mode — admin-only routes and
    common_checks must not run. Deployments in this mode have no proxy-
    level authentication, so applying authz would block every admin
    route for a test/dev setup that was previously wide-open."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.proxy._types import LitellmUserRoles

    token = UserAPIKeyAuth(
        api_key="sk-test", user_id="u", user_role=LitellmUserRoles.INTERNAL_USER
    )
    request = Request(scope={"type": "http"})
    request._url = URL(url="/get/config/callbacks")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None, master_key=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={},
                route="/get/config/callbacks",
            )
            mock_checks.assert_not_awaited()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_skips_public_routes():
    """Regression: public routes (e.g. /health/liveness) are exempted
    by the builder fast-path. The wrapper must not retroactively run
    common_checks on top — the synthetic INTERNAL_USER_VIEW_ONLY token
    has no user_id, so common_checks would reject the request as
    admin-only."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
    request = Request(scope={"type": "http"})
    request._url = URL(url="/health/liveness")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={},
                route="/health/liveness",
            )
            mock_checks.assert_not_awaited()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_skips_passthrough_endpoint_with_auth_false():
    """Regression: user-configured pass-through endpoints with
    ``auth: false`` are explicitly unauthenticated. The builder
    short-circuits and returns a fresh empty UserAPIKeyAuth(); running
    common_checks on that empty token would reject the request as
    admin-only. The "auth" flag on the endpoint config is the contract
    — when it's anything other than True, skip the gate."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth()
    request = Request(scope={"type": "http"})
    request._url = URL(url="/api/public/ingestion")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    attrs["general_settings"] = {
        "pass_through_endpoints": [
            {
                "path": "/api/public/ingestion",
                "target": "https://us.cloud.langfuse.com/api/public/ingestion",
                "auth": False,
            }
        ]
    }
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={},
                route="/api/public/ingestion",
            )
            mock_checks.assert_not_awaited()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_runs_for_passthrough_endpoint_with_auth_true():
    """Companion to the auth=False test: when a pass-through endpoint
    has ``auth: true``, the builder runs full authentication and the
    centralized gate must run too. Skipping based on path-match alone
    would re-open every ``auth: true`` pass-through endpoint."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u1")
    request = Request(scope={"type": "http"})
    request._url = URL(url="/api/public/ingestion")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    attrs["general_settings"] = {
        "pass_through_endpoints": [
            {
                "path": "/api/public/ingestion",
                "target": "https://us.cloud.langfuse.com/api/public/ingestion",
                "auth": True,
            }
        ]
    }
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_checks:
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={},
                route="/api/public/ingestion",
            )
            mock_checks.assert_awaited_once()
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_master_key_admin_overrides_db_user_role():
    """Regression: master_key tokens have user_id=litellm_proxy_admin_name
    (default 'default_user_id') and user_role=PROXY_ADMIN. If a row with
    that user_id exists in litellm_usertable with a non-admin user_role
    (created as a side effect of team membership), get_user_object
    returns it and the synthesized admin user_object is skipped — so
    common_checks demotes the master_key request to internal_user and
    blocks /team/update. The token is the source of truth for admin
    status; the DB row must not override it."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(
        api_key="sk-master",
        user_id="default_user_id",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    request = Request(scope={"type": "http"})
    request._url = URL(url="/team/update")

    # Simulate the DB row that team-creation created: same user_id, but
    # with user_role=internal_user (the default for new user rows).
    db_user = LiteLLM_UserTable(
        user_id="default_user_id",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        spend=1.5,
        max_budget=None,
    )

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                return_value=db_user,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"team_id": "t1", "max_budget": 10},
                route="/team/update",
            )
            mock_checks.assert_awaited_once()
            forwarded = mock_checks.call_args.kwargs["user_object"]
            assert forwarded is not None
            assert forwarded.user_role == LitellmUserRoles.PROXY_ADMIN
            assert forwarded.user_id == "default_user_id"
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_http_exception_without_team_id():
    """Regression: an HTTPException raised by any of the five parallel
    fetches (user/project/end_user/global_spend) must not trigger the
    _team_obj_from_token reconstruction when the token has no team_id —
    the helper asserts team_id is not None. This is the Greptile P1
    finding: the ``except HTTPException`` arm was team-fetch-biased."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import HTTPException, Request
    from starlette.datastructures import URL

    token = UserAPIKeyAuth(api_key="sk-test", user_id="u", team_id=None)
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        # Make the user fetch raise HTTPException. asyncio.gather with
        # return_exceptions=False propagates it.
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=404, detail="user-not-found"),
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            # Should NOT raise AssertionError from _team_obj_from_token;
            # should proceed with team_object=None.
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"model": "gpt-4o"},
                route="/chat/completions",
            )
            mock_checks.assert_awaited_once()
            assert mock_checks.call_args.kwargs["team_object"] is None
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_team_404_does_not_zero_other_contexts():
    """Per-fetch isolation: an HTTPException(404) from get_team_object
    (token references a deleted team) must reconstruct the team from the
    token but leave user_object / end_user_object / project_object intact.
    Pre-fix a bare ``except HTTPException`` over ``asyncio.gather`` zeroed
    every context, silently skipping user-budget, end-user-budget, and
    project enforcement whenever the token's team_id was stale."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import HTTPException, Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LiteLLM_EndUserTable,
        LiteLLM_ProjectTableCachedObj,
    )

    token = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        team_id="deleted-team",
        team_max_budget=5.0,
        team_models=["gpt-4o"],
        project_id="proj-1",
        end_user_id="alice",
    )
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    request._body = json.dumps({"user": "alice", "model": "gpt-4o"}).encode()

    fetched_user = LiteLLM_UserTable(
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        max_budget=10.0,
        spend=2.0,
    )
    fetched_end_user = LiteLLM_EndUserTable(user_id="alice", blocked=False, spend=1.0)
    fetched_project = LiteLLM_ProjectTableCachedObj(
        project_id="proj-1",
        project_alias="Proj 1",
        metadata={},
        created_by="admin",
        updated_by="admin",
    )

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_team_object",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=404, detail="team-not-found"),
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                return_value=fetched_user,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_project_object",
                new_callable=AsyncMock,
                return_value=fetched_project,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=fetched_end_user,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"user": "alice", "model": "gpt-4o"},
                route="/chat/completions",
            )

        mock_checks.assert_awaited_once()
        kwargs = mock_checks.call_args.kwargs
        # team reconstructed from the token
        assert kwargs["team_object"] is not None
        assert kwargs["team_object"].team_id == "deleted-team"
        assert kwargs["team_object"].max_budget == 5.0
        # other contexts must NOT be zeroed by the team fetch failure
        assert kwargs["user_object"] is fetched_user
        assert kwargs["end_user_object"] is fetched_end_user
        assert kwargs["project_object"] is fetched_project
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_centralized_common_checks_user_http_exception_isolates_to_user_only():
    """Per-fetch isolation, mirror of the team case: an HTTPException
    from get_user_object must zero only ``user_object``. The successfully
    fetched team / end_user / project / global_spend must reach
    common_checks intact so their enforcement still runs."""
    import litellm.proxy.proxy_server as _proxy_server_mod
    from fastapi import HTTPException, Request
    from starlette.datastructures import URL

    from litellm.proxy._types import (
        LiteLLM_EndUserTable,
        LiteLLM_ProjectTableCachedObj,
        LiteLLM_TeamTableCachedObj,
    )

    token = UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        team_id="t1",
        project_id="proj-1",
        end_user_id="alice",
    )
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    request._body = json.dumps({"user": "alice", "model": "gpt-4o"}).encode()

    fetched_team = LiteLLM_TeamTableCachedObj(
        team_id="t1", max_budget=20.0, models=["gpt-4o"]
    )
    fetched_end_user = LiteLLM_EndUserTable(user_id="alice", blocked=False, spend=1.0)
    fetched_project = LiteLLM_ProjectTableCachedObj(
        project_id="proj-1",
        project_alias="Proj 1",
        metadata={},
        created_by="admin",
        updated_by="admin",
    )

    attrs = _proxy_attrs_for_centralized_checks(user_custom_auth=None)
    originals = {a: getattr(_proxy_server_mod, a, None) for a in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)
        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_team_object",
                new_callable=AsyncMock,
                return_value=fetched_team,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=404, detail="user-not-found"),
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_project_object",
                new_callable=AsyncMock,
                return_value=fetched_project,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=fetched_end_user,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
            ) as mock_checks,
        ):
            await _run_centralized_common_checks(
                user_api_key_auth_obj=token,
                request=request,
                request_data={"user": "alice", "model": "gpt-4o"},
                route="/chat/completions",
            )

        mock_checks.assert_awaited_once()
        kwargs = mock_checks.call_args.kwargs
        assert kwargs["team_object"] is fetched_team
        assert kwargs["end_user_object"] is fetched_end_user
        assert kwargs["project_object"] is fetched_project
        # only the user_object is zeroed by its own fetch failing
        assert kwargs["user_object"] is None
    finally:
        for k, v in originals.items():
            setattr(_proxy_server_mod, k, v)


@pytest.mark.asyncio
async def test_master_key_auth_substitutes_alias_for_api_key():
    """
    When the master key authenticates a request, the resulting
    ``UserAPIKeyAuth.api_key`` must be the stable alias
    ``LITELLM_PROXY_MASTER_KEY_ALIAS`` — never the raw master key (which
    would propagate downstream and be hashed into spend logs, Prometheus
    ``/metrics`` labels, or audit trails) and never the master-key hash.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder
    from litellm.proxy.utils import hash_token

    import litellm.proxy.proxy_server as _proxy_server_mod

    attrs = _proxy_server_attrs_for_custom_auth(user_custom_auth=None)
    master_key = attrs["master_key"]
    _orig = {k: getattr(_proxy_server_mod, k, None) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(_proxy_server_mod, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/chat/completions")

        result = await _user_api_key_auth_builder(
            request=request,
            api_key=f"Bearer {master_key}",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )

        assert result.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS
        assert result.api_key != master_key
        assert result.api_key != hash_token(master_key)
    finally:
        for k, v in _orig.items():
            setattr(_proxy_server_mod, k, v)
