from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest

from litellm.proxy.auth.core.identity import (
    AnonymousReason,
)
from litellm.proxy.auth.core.inputs import PlanFacts, PlanSettings, RequestFacts, RoutingOverride
from litellm.proxy.auth.core.python_core import PythonCore
from litellm.proxy.auth.core.validators import (
    AnonymousValidator,
    CustomValidator,
    JwtValidator,
    KeyValidator,
    OAuth2Mode,
    OAuth2Validator,
    PremiumFeature,
    PremiumRequired,
)

PLANNER: Final = PythonCore().plan()
JWT: Final = "header.payload.signature"
LLM_ROUTE: Final = "/chat/completions"
SETTINGS: Final = PlanSettings(
    custom_auth=False,
    premium_user=True,
    public_routes=frozenset({"/health/liveness", "/metrics"}),
    additional_public_routes=(),
    metrics_require_auth=False,
    oauth2_introspection=False,
    oauth2_proxy=False,
    jwt_enabled=False,
    oauth2_routing_overrides=(),
    unauthenticated_pass_through_routes=frozenset(),
)


def _facts(
    route: str = LLM_ROUTE,
    token: str | None = "sk-key",
    *,
    is_llm_api_route: bool = True,
    is_info_route: bool = False,
    claims: Mapping[str, object] | None = None,
) -> PlanFacts:
    return PlanFacts(
        request=RequestFacts(route=route, token=token, is_llm_api_route=is_llm_api_route),
        is_info_route=is_info_route,
        unverified_claims=claims,
    )


def _selectors(value: str | list[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(value) if isinstance(value, list) else (value,)


def _override(
    iss: str | list[str] | None = None,
    client_id: str | list[str] | None = None,
    scope: str | list[str] | None = None,
    aud: str | list[str] | None = None,
) -> RoutingOverride:
    return RoutingOverride(
        iss=_selectors(iss), client_id=_selectors(client_id), scope=_selectors(scope), aud=_selectors(aud)
    )


def _routes_to_oauth2(override: RoutingOverride, claims: Mapping[str, object]) -> bool:
    settings: Final = replace(SETTINGS, jwt_enabled=True, oauth2_routing_overrides=(override,))
    return PLANNER.oauth2_introspection_applies(_facts(token=JWT, claims=claims), settings)


def test_custom_auth_runs_before_every_other_path() -> None:
    settings: Final = replace(SETTINGS, custom_auth=True, oauth2_introspection=True)
    assert PLANNER.plan(_facts("/health/liveness"), settings) == CustomValidator()


def test_a_public_route_needs_no_credential_even_when_oauth2_is_on() -> None:
    settings: Final = replace(SETTINGS, oauth2_introspection=True, jwt_enabled=True)
    assert PLANNER.plan(_facts("/health/liveness", JWT), settings) == AnonymousValidator(AnonymousReason.PUBLIC_ROUTE)


@pytest.mark.parametrize("route", ["/metrics", "/metrics/"])
def test_metrics_requires_a_credential_when_configured(route: str) -> None:
    settings: Final = replace(SETTINGS, metrics_require_auth=True, public_routes=frozenset({route}))
    assert PLANNER.plan(_facts(route, is_llm_api_route=False), settings) == KeyValidator(pass_through_key=True)


@pytest.mark.parametrize(
    ("route", "premium_user", "expected_public"),
    [
        ("/api/users/123", True, True),
        ("/api/users/123", False, False),
        ("/spend/calculate", True, True),
        ("/spend/calculate/more", True, False),
        ("/other", True, False),
    ],
)
def test_configured_public_routes_match_exactly_or_by_prefix_wildcard_for_premium_only(
    route: str, premium_user: bool, expected_public: bool
) -> None:
    settings: Final = replace(
        SETTINGS, premium_user=premium_user, additional_public_routes=("/spend/calculate", "/api/*")
    )
    plan: Final = PLANNER.plan(_facts(route, is_llm_api_route=False), settings)
    assert (plan == AnonymousValidator(AnonymousReason.PUBLIC_ROUTE)) is expected_public


@pytest.mark.parametrize(
    ("route", "is_llm_api_route", "is_info_route", "expected"),
    [
        (LLM_ROUTE, True, False, True),
        ("/key/info", False, True, True),
        ("/lens", False, False, True),
        ("/lens/traces", False, False, True),
        ("/lensing", False, False, False),
        ("/key/generate", False, False, False),
    ],
)
def test_introspection_covers_llm_info_and_lens_routes_only(
    route: str, is_llm_api_route: bool, is_info_route: bool, expected: bool
) -> None:
    facts: Final = _facts(route, is_llm_api_route=is_llm_api_route, is_info_route=is_info_route)
    plan: Final = PLANNER.plan(facts, replace(SETTINGS, oauth2_introspection=True))
    assert (plan == OAuth2Validator(OAuth2Mode.INTROSPECTION)) is expected


@pytest.mark.parametrize(
    ("feature", "settings", "token"),
    [
        (PremiumFeature.OAUTH2, replace(SETTINGS, oauth2_introspection=True), "sk-key"),
        (PremiumFeature.JWT, replace(SETTINGS, jwt_enabled=True), JWT),
    ],
)
def test_oauth2_introspection_and_jwt_need_premium(feature: PremiumFeature, settings: PlanSettings, token: str) -> None:
    assert PLANNER.plan(_facts(token=token), replace(settings, premium_user=False)) == PremiumRequired(feature)


def test_introspection_wins_over_proxy_headers_and_proxy_headers_over_jwt() -> None:
    both: Final = replace(SETTINGS, oauth2_introspection=True, oauth2_proxy=True, jwt_enabled=True)
    assert PLANNER.plan(_facts(), both) == OAuth2Validator(OAuth2Mode.INTROSPECTION)
    assert PLANNER.plan(_facts(token=JWT), both) == OAuth2Validator(OAuth2Mode.PROXY)
    assert PLANNER.plan(_facts(token=JWT), replace(both, oauth2_proxy=False)) == JwtValidator()


def test_a_jwt_skips_global_introspection_unless_an_override_routes_it() -> None:
    settings: Final = replace(SETTINGS, oauth2_introspection=True, jwt_enabled=True)
    assert PLANNER.plan(_facts(token=JWT, claims={"iss": "idp"}), settings) == JwtValidator()
    routed: Final = replace(settings, oauth2_routing_overrides=(_override(iss="idp"),))
    assert PLANNER.plan(_facts(token=JWT, claims={"iss": "idp"}), routed) == OAuth2Validator(OAuth2Mode.INTROSPECTION)


def test_a_jwt_is_validated_before_an_unauthenticated_pass_through_route() -> None:
    settings: Final = replace(SETTINGS, jwt_enabled=True, unauthenticated_pass_through_routes=frozenset({"/webhook"}))
    assert PLANNER.plan(_facts("/webhook", JWT, is_llm_api_route=False), settings) == JwtValidator()
    assert PLANNER.plan(_facts("/webhook", is_llm_api_route=False), settings) == AnonymousValidator(
        AnonymousReason.PASS_THROUGH_AUTH_OFF
    )


@pytest.mark.parametrize(
    ("issued_key", "expected"),
    [
        (False, AnonymousValidator(AnonymousReason.PASS_THROUGH_AUTH_OFF)),
        (True, KeyValidator(pass_through_key=False)),
    ],
)
def test_a_key_issued_by_custom_auth_skips_pass_through_handling(
    issued_key: bool, expected: AnonymousValidator | KeyValidator
) -> None:
    settings: Final = replace(SETTINGS, custom_auth=True, unauthenticated_pass_through_routes=frozenset({"/webhook"}))
    facts: Final = _facts("/webhook", is_llm_api_route=False)
    assert PLANNER.after_custom_auth(facts, settings, issued_key=issued_key) == expected
    assert PLANNER.after_jwt(facts, settings, issued_key=issued_key) == expected


def test_after_custom_auth_still_plans_public_routes_and_oauth2() -> None:
    settings: Final = replace(SETTINGS, custom_auth=True, oauth2_introspection=True)
    assert PLANNER.after_custom_auth(_facts("/health/liveness"), settings, issued_key=True) == AnonymousValidator(
        AnonymousReason.PUBLIC_ROUTE
    )
    assert PLANNER.after_custom_auth(_facts(), settings, issued_key=True) == OAuth2Validator(OAuth2Mode.INTROSPECTION)


def test_a_plain_key_request_is_a_key_validation_with_pass_through_handling() -> None:
    assert PLANNER.plan(_facts(), SETTINGS) == KeyValidator(pass_through_key=True)


@pytest.mark.parametrize(
    ("selector", "claim", "expected", "scope"),
    [
        (None, "any-value", True, False),
        ("issuer.example.com", "issuer.example.com", True, False),
        ("issuer.example.com", "other-issuer.example.com", False, False),
        ("trusted.example.com", "trusted.example.com attacker.example.com", False, False),
        ("trusted.*", "trusted.example.com attacker.example.com", False, False),
        ("trusted.*", "trusted.example.com", True, False),
        (["issuer-a.example.com", "issuer-b.example.com"], "issuer-b.example.com", True, False),
        ("*MID_LITELLM", "STREAM_MID_LITELLM", True, False),
        ("*MID_LITELLM", "REDIS_LITELLM", False, False),
        ("machine-??", "machine-01", True, False),
        ("machine-??", "machine-001", False, False),
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
def test_routing_selectors_match_claims(
    selector: str | list[str] | None, claim: object, expected: bool, scope: bool
) -> None:
    override: Final = _override(scope=selector) if scope else _override(iss=selector)
    claims: Final = {"scope": claim} if scope else {"iss": claim}
    assert _routes_to_oauth2(override, claims) is expected


@pytest.mark.parametrize(
    ("override", "claims", "expected"),
    [
        (_override(iss="oauth-issuer.example.com"), {"iss": "oauth-issuer.example.com"}, True),
        (
            _override(iss="oauth-issuer.example.com", scope="App:LiteLLM"),
            {"iss": "oauth-issuer.example.com", "scope": "App:LiteLLM openid"},
            True,
        ),
        (
            _override(iss="oauth-issuer.example.com", client_id="*MID_LITELLM"),
            {"iss": "oauth-issuer.example.com", "client_id": "BATCH_MID_LITELLM"},
            True,
        ),
        (
            _override(iss="oauth-issuer.example.com", client_id="*MID_LITELLM"),
            {"iss": "oauth-issuer.example.com", "client_id": "BATCH_PORTAL"},
            False,
        ),
        (
            _override(iss="oauth-issuer.example.com", aud=["api://litellm", "api://fallback"]),
            {"iss": "oauth-issuer.example.com", "aud": ["api://other", "api://litellm"]},
            True,
        ),
        (
            _override(iss="oauth-issuer.example.com", scope="App:LiteLLM", client_id="*MID_LITELLM"),
            {"iss": "oauth-issuer.example.com", "scope": "App:LiteLLM openid", "client_id": "BATCH_MID_LITELLM"},
            True,
        ),
        (
            _override(iss="oauth-issuer.example.com", scope="App:LiteLLM", client_id="*MID_LITELLM"),
            {"iss": "oauth-issuer.example.com", "scope": "App:Other openid", "client_id": "BATCH_MID_LITELLM"},
            False,
        ),
    ],
)
def test_every_selector_of_an_override_must_match(
    override: RoutingOverride, claims: Mapping[str, object], expected: bool
) -> None:
    assert _routes_to_oauth2(override, claims) is expected


def test_an_override_needs_decodable_claims() -> None:
    settings: Final = replace(SETTINGS, jwt_enabled=True, oauth2_routing_overrides=(_override(),))
    assert PLANNER.oauth2_introspection_applies(_facts(token=JWT, claims=None), settings) is False
