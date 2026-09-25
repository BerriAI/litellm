"""
Regression tests for the pass-through endpoint auth-default fix
(GHSA-7h34-mmrh-6g58).

Two failures the fix closes:

1. ``PassThroughGenericEndpoint.auth`` defaulted to ``False`` — an
   admin who added a pass-through to ``general_settings`` without
   explicitly setting ``auth: true`` shipped an unauthenticated
   forwarder.
2. Setting ``auth: true`` was rejected at startup unless the operator
   had a LiteLLM Enterprise license, leaving OSS deployments with no
   safe configuration.

The fix flips the default to ``True`` (safe-by-default) and removes
the enterprise gate so OSS operators can register an authenticated
pass-through. The runtime check in ``user_api_key_auth.py`` also now
defaults to ``True`` so a config dict (raw, not Pydantic) without an
``auth`` key still requires authentication.
"""

import logging
import uuid
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from litellm.caching.caching import DualCache
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import (
    ConfigFieldInfo,
    LiteLLMRoutes,
    PassThroughGenericEndpoint,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import (
    _run_centralized_common_checks,
    check_api_key_for_custom_headers_or_pass_through_endpoints,
)
from litellm.proxy.pass_through_endpoints.common_utils import _warn_once_per_process
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _get_pass_through_endpoints_from_db,
    _register_pass_through_endpoint,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassThroughAuthMode,
    pass_through_auth_mode,
)


def test_passthrough_auth_defaults_to_true():
    # Regression: an admin who configures a pass-through without setting
    # auth explicitly used to ship an unauthenticated forwarder. The
    # default is now safe.
    endpoint = PassThroughGenericEndpoint(
        path="/canary-forwarder",
        target="https://postman-echo.com/get",
    )
    assert endpoint.auth is True


def test_passthrough_auth_can_still_be_explicitly_disabled():
    # Operators who genuinely need an unauthenticated forwarder (e.g.
    # public webhook receiver) can opt in explicitly.
    endpoint = PassThroughGenericEndpoint(
        path="/public-webhook",
        target="https://example.com/webhook",
        auth=False,
    )
    assert endpoint.auth is False


@pytest.mark.asyncio
async def test_register_passthrough_with_auth_true_works_for_oss(monkeypatch):
    # Regression: setting ``auth: true`` used to raise at startup
    # unless ``premium_user`` was True, leaving OSS with no safe
    # configuration.
    app = MagicMock(spec=FastAPI)
    visited: set = set()

    endpoint = PassThroughGenericEndpoint(
        path="/forwarder",
        target="https://example.com",
        auth=True,
    )

    # Should not raise; OSS premium_user=False is allowed to use auth=True.
    await _register_pass_through_endpoint(
        endpoint=endpoint,
        app=app,
        premium_user=False,
        visited_endpoints=visited,
    )


@pytest.mark.asyncio
async def test_runtime_check_treats_missing_auth_key_as_authenticated():
    # The runtime dispatch in user_api_key_auth pulls
    # pass_through_endpoints from general_settings as raw dicts (the
    # Pydantic default never applies). A dict without an ``auth`` key
    # must default to "authenticated" — without this, the previous
    # behaviour (``endpoint.get("auth") is not True`` -> True -> empty
    # auth) ships an unauthenticated forwarder.
    request = MagicMock()
    request.headers = {}
    raw_endpoint_no_auth_key = {
        "path": "/forwarder",
        "target": "https://example.com",
        # ``auth`` deliberately omitted
    }

    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/forwarder",
        pass_through_endpoints=[raw_endpoint_no_auth_key],
        api_key="sk-1234",
    )

    # Result is the api_key string (auth is REQUIRED for this endpoint
    # — flow continues to normal key validation), NOT an empty
    # ``UserAPIKeyAuth()`` (which was the unauthenticated-forwarder
    # bug).
    assert result == "sk-1234"


@pytest.mark.asyncio
async def test_runtime_check_explicit_auth_false_still_skips_validation():
    # Operators who explicitly set ``auth: False`` get the legacy
    # behaviour — an empty UserAPIKeyAuth, no key required.
    from litellm.proxy._types import UserAPIKeyAuth

    request = MagicMock()
    request.headers = {}
    raw_endpoint_auth_false = {
        "path": "/public-webhook",
        "target": "https://example.com",
        "auth": False,
    }

    result = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/public-webhook",
        pass_through_endpoints=[raw_endpoint_auth_false],
        api_key="",
    )

    assert isinstance(result, UserAPIKeyAuth)


_OMITTED: Final = object()


def _config_entry(path: str, auth: object) -> dict[str, object]:
    base: Final[dict[str, object]] = {"path": path, "target": "https://example.com", "include_subpath": True}
    return base if auth is _OMITTED else {**base, "auth": auth}


@pytest.mark.parametrize(
    ("auth", "expected"),
    [
        (True, PassThroughAuthMode.GRANTED_KEYS),
        ("true", PassThroughAuthMode.GRANTED_KEYS),
        (" TRUE ", PassThroughAuthMode.GRANTED_KEYS),
        (False, PassThroughAuthMode.PUBLIC),
        ("false", PassThroughAuthMode.PUBLIC),
        ("False", PassThroughAuthMode.PUBLIC),
        (None, PassThroughAuthMode.ANY_KEY),
        ("yes", PassThroughAuthMode.ANY_KEY),
        (1, PassThroughAuthMode.ANY_KEY),
        (0, PassThroughAuthMode.ANY_KEY),
        ("", PassThroughAuthMode.ANY_KEY),
    ],
)
def test_pass_through_auth_mode_reads_every_config_spelling(auth: object, expected: PassThroughAuthMode) -> None:
    assert pass_through_auth_mode(auth) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "policy_checked", "grant_gated"),
    [
        (_OMITTED, True, False),
        (None, True, False),
        ("yes", True, False),
        (True, True, True),
        ("true", True, True),
        (False, False, False),
        ("false", False, False),
    ],
)
async def test_register_pass_through_endpoint_auth_tiers(auth: object, policy_checked: bool, grant_gated: bool) -> None:
    app = MagicMock(spec=FastAPI)
    path: Final = f"/lit8631-{uuid.uuid4().hex}"
    entry: Final = _config_entry(path, auth)
    await _register_pass_through_endpoint(endpoint=entry, app=app, premium_user=False, visited_endpoints=set())
    try:
        assert (path in LiteLLMRoutes.openai_routes.value) is policy_checked
        assert (f"{path}/*" in LiteLLMRoutes.openai_routes.value) is policy_checked
        assert RouteChecks.is_auth_enforced_pass_through_route(path) is grant_gated
        assert RouteChecks.is_auth_enforced_pass_through_route(f"{path}/sub") is grant_gated
        dependencies: Final = {
            bool(call.kwargs["dependencies"]) for call in app.add_api_route.call_args_list  # pyright: ignore[reportAny]  # MagicMock call record
        }
        assert dependencies == {grant_gated}
    finally:
        InitPassThroughEndpointHelpers.remove_endpoint_routes(cast(str, entry["id"]))
        assert path not in LiteLLMRoutes.openai_routes.value


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", [_OMITTED, "yes", "true", True])
async def test_runtime_check_requires_a_key_unless_auth_is_false(auth: object) -> None:
    request = MagicMock()
    request.headers = {}
    result: Final = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/forwarder",
        pass_through_endpoints=[_config_entry("/forwarder", auth)],
        api_key="sk-1234",
    )
    assert result == "sk-1234"


@pytest.mark.asyncio
async def test_runtime_check_string_false_is_public() -> None:
    request = MagicMock()
    request.headers = {}
    result: Final = await check_api_key_for_custom_headers_or_pass_through_endpoints(
        request=request,
        route="/public-webhook",
        pass_through_endpoints=[_config_entry("/public-webhook", "false")],
        api_key="",
    )
    assert isinstance(result, UserAPIKeyAuth)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "policy_enforced"),
    [(_OMITTED, True), ("yes", True), (True, True), (False, False), ("false", False)],
)
async def test_over_budget_team_is_blocked_on_every_authenticated_pass_through(
    auth: object, policy_enforced: bool
) -> None:
    import litellm.proxy.proxy_server as proxy_server_module
    from fastapi import Request
    from starlette.datastructures import URL

    route: Final = "/lit8631-policy"
    attrs: Final = {
        "prisma_client": None,
        "user_api_key_cache": DualCache(),
        "proxy_logging_obj": MagicMock(),
        "general_settings": {"pass_through_endpoints": [_config_entry(route, auth)]},
        "llm_router": None,
        "user_custom_auth": None,
        "litellm_proxy_admin_name": "admin",
        "master_key": "sk-test-master",
    }
    originals: Final = {name: getattr(proxy_server_module, name, None) for name in attrs}
    request = Request(scope={"type": "http"})
    request._url = URL(url=route)
    over_budget: Final = BudgetExceededError(current_cost=2.0, max_budget=1.0)
    try:
        for name, value in attrs.items():
            setattr(proxy_server_module, name, value)
        with patch("litellm.proxy.auth.user_api_key_auth.common_checks", new=AsyncMock(side_effect=over_budget)):
            run: Final = _run_centralized_common_checks(
                user_api_key_auth_obj=UserAPIKeyAuth(api_key="sk-test", user_id="u1"),
                request=request,
                request_data={},
                route=route,
            )
            if policy_enforced:
                with pytest.raises(BudgetExceededError):
                    await run
            else:
                assert await run is None
    finally:
        for name, value in originals.items():
            setattr(proxy_server_module, name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "expected_paths"), [("config", []), ("db", ["/from-db"]), ("unset", ["/from-db"])])
async def test_db_reader_never_mirrors_config_owned_entries(source: str, expected_paths: list[str]) -> None:
    field: Final = ConfigFieldInfo(
        field_name="pass_through_endpoints",
        field_value=[{"path": "/from-db", "target": "https://example.com"}],
        source=source,
    )
    with patch("litellm.proxy.proxy_server.get_config_general_settings", new=AsyncMock(return_value=field)):
        endpoints: Final = await _get_pass_through_endpoints_from_db()
    assert [endpoint.path for endpoint in endpoints] == expected_paths


def test_yaml_load_warns_once_per_process_per_entry_without_auth(caplog: pytest.LogCaptureFixture) -> None:
    import litellm.proxy.proxy_server as proxy_server_module

    _warn_once_per_process.cache_clear()
    original: Final = proxy_server_module.config_passthrough_endpoints
    config: Final = {
        "general_settings": {
            "pass_through_endpoints": [
                _config_entry("/no-auth-key", _OMITTED),
                _config_entry("/granted", True),
                _config_entry("/public", False),
                _config_entry("/also-no-auth-key", None),
            ]
        }
    }
    try:
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            proxy_server_module.ProxyConfig()._load_yaml_settings_stores(config)
            proxy_server_module.ProxyConfig()._load_yaml_settings_stores(config)
    finally:
        proxy_server_module.config_passthrough_endpoints = original
    warned: Final = [record.getMessage() for record in caplog.records if "sets no `auth`" in record.getMessage()]
    assert [message.split("'")[1] for message in warned] == ["/no-auth-key", "/also-no-auth-key"]
    assert all("`auth: false`" in message and "`auth: true`" in message for message in warned)
