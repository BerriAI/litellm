import dataclasses
from collections.abc import Callable
from typing import Final

import pytest
from fastapi import Request
from starlette.datastructures import URL

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import core
from litellm.proxy.auth.host.python import PythonAuthHost
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustDenial, RustPlan

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
        auth_enforced_pass_through=False,
        forwarded_routes=(),
        is_info_route=False,
        now=0.0,
    )

    assert RustAuth.plan(request, settings) == expected


@pytest.mark.asyncio
async def test_a_host_failure_the_core_cannot_skip_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch, auth_host: Callable[..., PythonAuthHost]
) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", "https://idp.example/jwks")
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth(),
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )

    class _Unreadable(UserApiKeyCache):
        async def async_get_cache(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("cache down")

    failing: Final = dataclasses.replace(auth_host(), user_api_key_cache=_Unreadable())
    token: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"

    with pytest.raises(ConnectionError, match="cache down"):
        await core.drive(RustAuth.verify_jwt(token, 0.0, settings), failing)


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
