from collections.abc import Iterator
from fnmatch import fnmatchcase
from typing import Final

from litellm.proxy.auth.core.credential import is_jwt
from litellm.proxy.auth.core.inputs import PlanFacts, PlanSettings, RoutingOverride
from litellm.proxy.auth.core.outcome import (
    AnonymousReason,
    AnonymousValidator,
    CredentialPlan,
    CustomValidator,
    JwtValidator,
    KeyPlan,
    KeyValidator,
    OAuth2Mode,
    OAuth2Validator,
    Plan,
    PremiumFeature,
    PremiumRequired,
)

_METRICS_ROUTE: Final = "/metrics"
_LENS_ROUTE: Final = "/lens"


def _normalized(route: str) -> str:
    return route.rstrip("/") if route != "/" and route.endswith("/") else route


def _matches_public_pattern(route: str, pattern: str) -> bool:
    return route.startswith(pattern[:-1]) if pattern.endswith("*") else route == pattern


def is_public_route(route: str, settings: PlanSettings) -> bool:
    if _normalized(route) == _METRICS_ROUTE and settings.metrics_require_auth:
        return False
    if route in settings.public_routes:
        return True
    return settings.premium_user and any(
        _matches_public_pattern(route, pattern) for pattern in settings.additional_public_routes
    )


def _claim_values(claim: object, *, split_scope: bool) -> tuple[str, ...]:
    if isinstance(claim, list):
        return tuple(str(value) for value in claim)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # JWT claims are untyped JSON
    if split_scope and isinstance(claim, str) and " " in claim.strip():
        return tuple(value for value in claim.strip().split(" ") if value)
    return (str(claim),)


def _selector_matches(selector: str, claim: str, *, split_scope: bool) -> bool:
    if "*" not in selector and "?" not in selector:
        return selector == claim
    if not split_scope and any(character.isspace() for character in claim):
        return False
    return fnmatchcase(claim, selector)


def _pairs(selectors: tuple[str, ...], claims: tuple[str, ...]) -> Iterator[tuple[str, str]]:
    for selector in selectors:
        yield from ((selector, claim) for claim in claims)


def _selectors_match(selectors: tuple[str, ...] | None, claim: object, *, split_scope: bool = False) -> bool:
    if selectors is None:
        return True
    if claim is None:
        return False
    claims: Final = _claim_values(claim, split_scope=split_scope)
    return any(
        _selector_matches(selector, value, split_scope=split_scope) for selector, value in _pairs(selectors, claims)
    )


def _override_matches(override: RoutingOverride, facts: PlanFacts) -> bool:
    token_claims: Final = facts.unverified_claims or {}
    return (
        _selectors_match(override.iss, token_claims.get("iss"))
        and _selectors_match(override.client_id, token_claims.get("client_id"))
        and _selectors_match(override.scope, token_claims.get("scope"), split_scope=True)
        and _selectors_match(override.aud, token_claims.get("aud"))
    )


def _routes_jwt_to_oauth2(facts: PlanFacts, settings: PlanSettings) -> bool:
    if facts.unverified_claims is None:
        return False
    return any(_override_matches(override, facts) for override in settings.oauth2_routing_overrides)


def _is_lens_route(route: str) -> bool:
    return route == _LENS_ROUTE or route.startswith(f"{_LENS_ROUTE}/")


def oauth2_introspection_applies(facts: PlanFacts, settings: PlanSettings) -> bool:
    request: Final = facts.request
    token_is_jwt: Final = settings.jwt_enabled and is_jwt(request.token)
    covered_route: Final = request.is_llm_api_route or facts.is_info_route or _is_lens_route(request.route)
    if not covered_route:
        return False
    if settings.oauth2_introspection and not token_is_jwt:
        return True
    return token_is_jwt and _routes_jwt_to_oauth2(facts, settings)


def _premium(plan: CredentialPlan, feature: PremiumFeature, settings: PlanSettings) -> CredentialPlan:
    return plan if settings.premium_user else PremiumRequired(feature)


def after_jwt(facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> KeyPlan:
    if not issued_key and facts.request.route in settings.unauthenticated_pass_through_routes:
        return AnonymousValidator(AnonymousReason.PASS_THROUGH_AUTH_OFF)
    return KeyValidator(pass_through_key=not issued_key)


def after_custom_auth(facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> CredentialPlan:
    if is_public_route(facts.request.route, settings):
        return AnonymousValidator(AnonymousReason.PUBLIC_ROUTE)
    if oauth2_introspection_applies(facts, settings):
        return _premium(OAuth2Validator(OAuth2Mode.INTROSPECTION), PremiumFeature.OAUTH2, settings)
    if settings.oauth2_proxy:
        return OAuth2Validator(OAuth2Mode.PROXY)
    if settings.jwt_enabled and is_jwt(facts.request.token):
        return _premium(JwtValidator(), PremiumFeature.JWT, settings)
    return after_jwt(facts, settings, issued_key=issued_key)


def plan(facts: PlanFacts, settings: PlanSettings) -> Plan:
    if settings.custom_auth:
        return CustomValidator()
    return after_custom_auth(facts, settings, issued_key=False)
