import dataclasses
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Final, cast

import pytest
from fastapi import Request
from starlette.datastructures import URL

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
import litellm
from litellm.proxy.auth import core
from litellm.proxy.auth.auth_utils import is_authentication_failure
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.caching.dual_cache import DualCache
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustAuthStep, RustDenial, RustPlan

JWT: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"


def _accept(claims: dict[str, object]) -> bool:
    return True


@pytest.mark.parametrize(
    ("custom_validate", "expected"),
    [(None, RustPlan.Jwt), (_accept, RustPlan.JwtWithOperatorCode)],
    ids=["no_operator_code", "custom_validate_set"],
)
def test_a_jwt_is_planned_for_operator_code_only_when_custom_validate_is_set(
    custom_validate: Callable[[dict[str, object]], bool] | None, expected: RustPlan
) -> None:
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth().model_copy(update={"custom_validate": custom_validate}),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )
    request: Final = core.AuthRequest(
        route="/chat/completions",
        method="POST",
        token=JWT,
        model=None,
        team_header=None,
        is_llm_api_route=True,
        runs_inference_checks=True,
        auth_enforced_pass_through=False,
        forwarded_routes=(),
        is_info_route=False,
        now=0.0,
    )

    assert RustAuth.plan(request, settings) == expected


@pytest.mark.parametrize(
    ("enable_jwt_auth", "expected"),
    [(True, RustPlan.Jwt), ("true", RustPlan.Key), (1, RustPlan.Key), (object(), RustPlan.Key)],
    ids=["true", "string_true", "one", "object"],
)
def test_jwt_auth_is_on_only_for_the_boolean_true(enable_jwt_auth: object, expected: RustPlan) -> None:
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": enable_jwt_auth},
        LiteLLM_JWTAuth(),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )

    assert RustAuth.plan(core.request(JWT, "/chat/completions", "POST"), settings) == expected


@pytest.mark.asyncio
async def test_a_host_failure_the_core_cannot_skip_propagates_as_a_failed_authentication(
    monkeypatch: pytest.MonkeyPatch, auth_host: Callable[..., PythonAuthHost]
) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://idp.example/jwks")
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth(),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )

    class _Unreadable(DualCache):
        async def async_get_cache(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("cache down")

    failing: Final = dataclasses.replace(auth_host(), jwks_cache=_Unreadable())
    token: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"

    with pytest.raises(ConnectionError, match="cache down") as raised:
        await core.drive(RustAuth.verify_jwt(token, 0.0, settings), failing)
    assert is_authentication_failure(raised.value)


@pytest.mark.parametrize(
    ("public_routes", "route", "expected"),
    [
        (["/custom/route", 5, "/after/*"], "/after/x", RustPlan.Key),
        (["/custom/route", None], "/custom/route", RustPlan.Key),
        ({"/custom/route"}, "/custom/route", RustPlan.Public),
        (["/custom/route", 5], "/chat/completions", RustPlan.Key),
        (["/custom/route", 5], "/health/liveliness", RustPlan.Public),
        ("/custom/*", "/custom/route", RustPlan.Key),
        ("/custom/*", "/health/liveliness", RustPlan.Public),
        (5, "/custom/route", RustPlan.Key),
    ],
    ids=[
        "non_string_entry_voids_the_list",
        "null_entry_voids_the_list",
        "set_of_routes",
        "non_string_entry_grants_nothing",
        "built_in_route_beside_a_non_string_entry",
        "bare_string_is_not_a_route_list",
        "built_in_route_beside_a_bare_string",
        "number_is_not_a_route_list",
    ],
)
def test_a_malformed_public_routes_setting_never_fails_the_request(
    public_routes: object, route: str, expected: RustPlan
) -> None:
    settings: Final = RustAuthSettings(
        {"public_routes": public_routes}, None, core.HostFacts(master_key=None, premium=True, has_db=False)
    )

    assert RustAuth.plan(core.request(None, route, "POST"), settings) == expected


def _planned(route: str, token: str | None) -> RustPlan | RustDenial:
    return RustAuth.plan(core.request(token, route, "POST"), core.proxy_settings())


def _write_general_settings_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(proxy_server.general_settings, "enable_jwt_auth", True)


def _rebind_general_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_jwt_auth": True})


def _switch_to_a_premium_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", True)


@pytest.mark.parametrize(
    ("premium", "general_settings", "route", "token", "change", "expected"),
    [
        (True, {}, "/chat/completions", JWT, _write_general_settings_in_place, RustPlan.Jwt),
        (True, {}, "/chat/completions", JWT, _rebind_general_settings, RustPlan.Jwt),
        (
            False,
            {"public_routes": ["/custom/route"]},
            "/custom/route",
            None,
            _switch_to_a_premium_license,
            RustPlan.Public,
        ),
    ],
    ids=["general_settings_written_in_place", "general_settings_rebound", "license_switched"],
)
def test_a_config_change_while_serving_decides_the_next_request(
    monkeypatch: pytest.MonkeyPatch,
    premium: bool,
    general_settings: dict[str, object],
    route: str,
    token: str | None,
    change: Callable[[pytest.MonkeyPatch], None],
    expected: RustPlan,
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", dict(general_settings))
    monkeypatch.setattr(proxy_server, "premium_user", premium)
    monkeypatch.setattr(proxy_server, "master_key", None)
    before: Final = _planned(route, token)

    change(monkeypatch)

    assert before == RustPlan.Key
    assert _planned(route, token) == expected


async def _authenticate(api_key: str) -> UserAPIKeyAuth:
    request: Final = Request(scope={"type": "http", "headers": [], "method": "POST", "path": "/chat/completions"})
    request._url = URL(url="/chat/completions")
    return await user_api_key_auth(
        request=request,
        api_key=api_key,
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )


@pytest.mark.asyncio
async def test_a_master_key_loaded_while_serving_admits_the_next_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "master_key", "sk-before-reload")
    with pytest.raises(ProxyException):
        _ = await _authenticate("Bearer sk-after-reload")

    monkeypatch.setattr(proxy_server, "master_key", "sk-after-reload")

    assert (await _authenticate("Bearer sk-after-reload")).user_role == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.parametrize(
    ("database_url_configured", "code", "error_type"),
    [
        pytest.param(False, "401", ProxyErrorTypes.auth_error, id="no_database_configured"),
        pytest.param(True, "400", ProxyErrorTypes.no_db_connection, id="configured_database_not_connected"),
    ],
)
@pytest.mark.asyncio
async def test_a_wrong_key_without_a_database_is_an_outage_only_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch, database_url_configured: bool, code: str, error_type: ProxyErrorTypes
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "database_url_configured", database_url_configured)
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "master_key", "sk-the-master-key")

    assert (await _authenticate("Bearer sk-the-master-key")).user_role == LitellmUserRoles.PROXY_ADMIN
    with pytest.raises(ProxyException) as denied:
        _ = await _authenticate("Bearer sk-not-the-master-key")
    assert (denied.value.code, denied.value.type) == (code, error_type), denied.value.message


_STEPS: Final = frozenset(
    variant.__name__
    for variant in vars(RustAuthStep).values()
    if isinstance(variant, type) and issubclass(variant, RustAuthStep) and variant is not RustAuthStep.Done
)
_BEFORE_AUTHENTICATION: Final = frozenset(
    {
        "CachedKey",
        "DbKey",
        "EvictKey",
        "ValidatedEndUserId",
        "PrefetchIdentity",
        "EndUser",
        "FetchJson",
        "FetchUserinfo",
        "CachedDocument",
        "CacheDocument",
        "CustomValidate",
        "KeyMappingHash",
        "CachedKeyMapping",
        "CacheKeyMapping",
        "ForgetKeyMapping",
        "KeyByHash",
        "ReusableKeyHash",
        "CreateKeyMapping",
    }
)
_AFTER_AUTHENTICATION: Final = frozenset(
    {
        "UiKey",
        "KeyOwner",
        "CachedTeam",
        "KeyTeam",
        "ObjectPermission",
        "Project",
        "WithTeam",
        "WithTempBudget",
        "NormalizeModel",
        "PrefetchAuthObjects",
        "Team",
        "TeamByAlias",
        "GrantingTeam",
        "BindTeamMember",
        "LoadTeamMembership",
        "User",
        "ResolveUserIdentity",
        "Org",
        "OrgByAlias",
        "JwtEndUser",
        "SetUserRole",
        "PatchTeamMembership",
        "AddTeamMember",
        "ManagedAgent",
        "RegisteredAgent",
        "RecordAgentAuthentication",
        "KeySkipsBudgetChecks",
        "TeamMemberBudget",
        "KeyBudgets",
        "ProxyBudgetAlert",
        "FinishJwt",
        "FinishAutoRegisteredJwt",
        "TeamModelAccess",
        "DelegatedAgentTeam",
    }
)


def test_every_host_step_is_decided_as_before_or_after_authentication() -> None:
    assert _BEFORE_AUTHENTICATION | _AFTER_AUTHENTICATION == _STEPS
    assert frozenset(step.__name__ for step in core.PRE_AUTHENTICATION_STEPS) == _BEFORE_AUTHENTICATION


class _Rows:
    def __init__(self) -> None:
        self.written: tuple[dict[str, object], ...] = ()

    async def update_database(self, **kwargs: object) -> None:
        self.written = (*self.written, kwargs)


async def _raise_from_custom_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    raise ValueError("user not allowed")


async def _expired_custom_auth_token(request: Request, api_key: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="custom-caller", expires=datetime(2020, 1, 1, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    ("api_key", "headers", "general_settings", "custom_auth", "premium", "rows"),
    [
        pytest.param("", (), {}, None, False, 0, id="no_key"),
        pytest.param("Bearer sk-not-the-master-key", (), {}, None, False, 0, id="wrong_key_without_a_database"),
        pytest.param(
            "Bearer sk-the-master-key",
            (),
            {"allowed_routes": ["/embeddings"]},
            None,
            False,
            0,
            id="route_outside_the_proxy_allowed_routes",
        ),
        pytest.param("Bearer sk-anything", (), {}, _raise_from_custom_auth, False, 0, id="custom_auth_raises"),
        pytest.param("Bearer sk-anything", (), {}, _expired_custom_auth_token, False, 0, id="custom_auth_expired"),
        pytest.param(f"Bearer {JWT}", (), {"enable_jwt_auth": True}, None, False, 0, id="jwt_without_a_license"),
        pytest.param(
            "Bearer opaque-oauth2-token",
            (),
            {"enable_oauth2_auth": True},
            None,
            True,
            0,
            id="oauth2_introspection_not_configured",
        ),
        pytest.param(
            "Bearer sk-the-master-key",
            ((b"x-litellm-impersonate-user", b"u1"), (b"x-litellm-impersonate-encoding", b"rot13")),
            {},
            None,
            False,
            1,
            id="master_key_refused_after_authenticating",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_refusal_writes_a_spend_row_only_once_the_credential_authenticated(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    headers: tuple[tuple[bytes, bytes], ...],
    general_settings: dict[str, object],
    custom_auth: Callable[[Request, str], Awaitable[UserAPIKeyAuth]] | None,
    premium: bool,
    rows: int,
) -> None:
    written: Final = _Rows()
    monkeypatch.delenv("OAUTH_TOKEN_INFO_ENDPOINT", raising=False)
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger(spend_writer=lambda: cast(DBSpendUpdateWriter, written))])
    monkeypatch.setattr(litellm, "enable_post_custom_auth_checks", True, raising=False)
    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    monkeypatch.setattr(proxy_server, "premium_user", premium)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "database_url_configured", False)
    monkeypatch.setattr(proxy_server, "user_custom_auth", custom_auth)
    monkeypatch.setattr(proxy_server, "master_key", "sk-the-master-key")
    request: Final = Request(
        scope={"type": "http", "headers": list(headers), "method": "POST", "path": "/chat/completions"}
    )
    request._url = URL(url="/chat/completions")

    with pytest.raises(ProxyException):
        await user_api_key_auth(
            request=request,
            api_key=api_key,
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )
    assert len(written.written) == rows
