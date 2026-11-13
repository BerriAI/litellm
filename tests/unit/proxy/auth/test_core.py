import base64
import json
from collections.abc import Callable
from typing import Final

import pytest

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import JWTRoutingOverride, LiteLLM_JWTAuth
from litellm.proxy.auth import core
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustDenial, RustPlan

JWT: Final = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ1In0.c2ln"
ISSUER: Final = "oauth-issuer.example.com"


def _unsigned_jwt(claims: dict[str, object]) -> str:
    def segment(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'RS256'})}.{segment(claims)}.c2ln"


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
        core.HostFacts(premium=True),
    )

    assert RustAuth.plan(core.request(JWT, "/chat/completions"), settings) == expected


@pytest.mark.parametrize(
    ("iss", "expected"),
    [(ISSUER, True), ("other-issuer.example.com", False)],
    ids=["routed_issuer", "other_issuer"],
)
def test_a_routing_override_sends_its_jwts_to_introspection(iss: str, expected: bool) -> None:
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        LiteLLM_JWTAuth(routing_overrides=[JWTRoutingOverride(iss=ISSUER)]),
        core.HostFacts(premium=True),
    )
    request: Final = core.request(_unsigned_jwt({"iss": iss}), "/chat/completions")

    assert RustAuth.oauth2_introspection_applies(request, settings) is expected
    assert RustAuth.plan(request, settings) == (RustPlan.OAuth2Introspection if expected else RustPlan.Jwt)


@pytest.mark.parametrize(
    ("public_routes", "route", "expected"),
    [
        (["/custom/route", 5, "/after/*"], "/after/x", RustPlan.Public),
        (["/custom/route", None], "/custom/route", RustPlan.Public),
        ({"/custom/route"}, "/custom/route", RustPlan.Public),
        (["/custom/route", 5], "/chat/completions", RustPlan.Key),
        (["/custom/route", 5], "/health/liveliness", RustPlan.Public),
        ("/custom/*", "/custom/route", RustPlan.Key),
        ("/custom/*", "/health/liveliness", RustPlan.Public),
        (5, "/custom/route", RustPlan.Key),
    ],
    ids=[
        "route_after_a_non_string_entry",
        "route_before_a_null_entry",
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
    settings: Final = RustAuthSettings({"public_routes": public_routes}, None, core.HostFacts(premium=True))

    assert RustAuth.plan(core.request(None, route), settings) == expected


def _planned(route: str, token: str | None) -> RustPlan | RustDenial:
    return RustAuth.plan(core.request(token, route), core.proxy_settings())


def _write_general_settings_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    settings: Final = proxy_server.current_general_settings()
    assert isinstance(settings, dict)
    monkeypatch.setitem(settings, "enable_jwt_auth", True)


def _rebind_general_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_jwt_auth": True})


def _switch_to_a_premium_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "premium_user", True)


def _make_metrics_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", False)


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
        (True, {"public_routes": ["/metrics"]}, "/metrics", None, _make_metrics_public, RustPlan.Public),
    ],
    ids=["general_settings_written_in_place", "general_settings_rebound", "license_switched", "metrics_made_public"],
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
    monkeypatch.setattr(proxy_server.jwt_handler, "litellm_jwtauth", LiteLLM_JWTAuth(), raising=False)
    monkeypatch.setattr(litellm, "require_auth_for_metrics_endpoint", True)
    before: Final = _planned(route, token)

    change(monkeypatch)

    assert before == RustPlan.Key
    assert _planned(route, token) == expected
