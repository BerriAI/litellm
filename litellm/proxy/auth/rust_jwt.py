"""The JWT API every Python caller uses: the proxy auth builder, ui_sso, the MCP token flows and the
pass-through endpoints. Every decision runs in gateway-auth-core; this module builds the request and
host a call needs and turns a denial into the proxy's public error."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import JWTAuthBuilderResult, LiteLLM_JWTAuth
from litellm.proxy.auth import core
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth import identity as auth_identity
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.rust_bridge._native import (
    RustAuth,
    RustAuthOutcome,
    RustAuthSettings,
    RustDenial,
    RustIdentity,
    RustJwtMapping,
    RustMappedClaims,
    jwt_claims,
    jwt_is_allowed_domain,
    jwt_is_jwt,
    jwt_unverified_claims,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

if TYPE_CHECKING:
    from opentelemetry.trace import Span

ID_JAG_TYP: Final = "oauth-id-jag+jwt"
LITELLM_JWT_ISSUER_CLAIM: Final = "_litellm_jwt_issuer"
LITELLM_INTERNAL_CLAIMS: Final = (
    LITELLM_JWT_ISSUER_CLAIM,
    "_litellm_user_id",
    "_litellm_user_email",
    "_litellm_team_id",
    "_litellm_team_ids",
    "_litellm_org_id",
    "_litellm_end_user_id",
)
TEAM_HEADER: Final = "x-litellm-team-id"
LEEWAY_SECONDS: Final = 0


@dataclass(frozen=True, slots=True)
class AssertionRejection:
    """Why a JWT assertion failed verification. The reason is for the proxy log only, since its
    wording can quote the token's claims."""

    reason: str


@dataclass(frozen=True, slots=True)
class JwtIdentity:
    user_id: str | None
    user_object: LiteLLM_UserTable | None
    agent_id: str | None


def is_jwt(token: str | None) -> bool:
    return jwt_is_jwt(token)


def unverified_claims(token: str) -> dict[str, object] | None:
    return jwt_unverified_claims(token)


def claims(verified: Mapping[str, object], litellm_jwtauth: LiteLLM_JWTAuth) -> RustMappedClaims:
    return jwt_claims(dict(verified), litellm_jwtauth)


def is_allowed_domain(user_email: str, litellm_jwtauth: LiteLLM_JWTAuth) -> bool:
    return jwt_is_allowed_domain(user_email, litellm_jwtauth)


def team_header(headers: Mapping[str, str] | None) -> str | None:
    """`x-litellm-team-id`, matched case-insensitively."""
    if not headers:
        return None
    return {name.lower(): value for name, value in headers.items()}.get(TEAM_HEADER)


def request(
    token: str,
    route: str,
    method: str | None,
    *,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
) -> core.AuthRequest:
    return core.AuthRequest(
        route=route,
        method=method,
        token=token,
        model=model,
        team_header=team_header(headers),
        is_llm_api_route=RouteChecks.is_llm_api_route(route=route),
        auth_enforced_pass_through=RouteChecks.is_auth_enforced_pass_through_route(
            route=route, method=method.upper() if isinstance(method, str) else None
        ),
        forwarded_routes=RouteChecks.forwarded_pass_through_routes(route),
        is_info_route=RouteChecks.is_info_route(route=route),
        now=time.time(),
    )


def proxy_settings() -> RustAuthSettings:
    from litellm.proxy.proxy_server import rust_auth_settings  # noqa: PLC0415  # rebound on every config load

    if rust_auth_settings is None:
        raise RuntimeError("proxy auth settings are built at startup, before the proxy serves requests")
    return rust_auth_settings


async def _drive(call: RustAuth, token: str, route: str = "") -> RustAuthOutcome:
    from litellm.proxy.auth.host.python import proxy_host  # noqa: PLC0415  # host.python imports user_api_key_auth

    return await core.drive(call, proxy_host(route, token))


def _raise_if_denied(outcome: RustAuthOutcome) -> None:
    if outcome.denied:
        auth_denial.raise_denial(outcome)


async def verify(token: str) -> dict[str, object]:
    """`auth_jwt`: the verified claims, or the proxy's error for a token it refuses."""
    outcome: Final = await _drive(RustAuth.verify_jwt(token, time.time(), proxy_settings()), token)
    _raise_if_denied(outcome)
    verified: Final = outcome.claims
    if verified is None:
        raise RuntimeError("rust auth verified a token without claims")
    return verified


async def verify_assertion(
    token: str, audience: Sequence[str], required_claims: Sequence[str]
) -> Mapping[str, object] | AssertionRejection:
    """`decode_trusted_assertion`: the verified claims or a rejection. A JWKS outage raises the 503 so
    the caller does not call the assertion bad."""
    outcome: Final = await _drive(
        RustAuth.verify_jwt_assertion(
            token,
            time.time(),
            proxy_settings(),
            audience=list(audience),
            required_claims=list(required_claims),
        ),
        token,
    )
    verified: Final = outcome.claims
    if verified is not None:
        return verified
    denial: Final = outcome.denial
    if outcome.assertion_rejected and isinstance(denial, RustDenial.Plain):
        return AssertionRejection(reason=denial.message)
    auth_denial.raise_denial(outcome)


async def authorize(
    token: str,
    route: str,
    request_method: str | None,
    *,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
    provisioning: bool,
) -> JWTAuthBuilderResult:
    """`JWTAuthManager.authorize_jwt`; `provisioning` is what `auth_builder` passes."""
    outcome: Final = await _drive(
        RustAuth.authorize_jwt(
            request(token, route, request_method, headers=headers, model=model),
            proxy_settings(),
            provisioning=provisioning,
        ),
        token,
        route,
    )
    _raise_if_denied(outcome)
    identity: Final = outcome.identity
    if not isinstance(identity, RustIdentity.Jwt):
        raise RuntimeError(f"rust auth authorized a JWT as something else: {identity!r}")
    return auth_identity.jwt_result(identity, token)


def auth_from_result(result: JWTAuthBuilderResult, parent_otel_span: Span | None = None) -> UserAPIKeyAuth:
    return auth_identity.auth_from_result(result, parent_otel_span)


async def resolve_identity(token: str) -> JwtIdentity:
    """`JWTAuthManager.resolve_identity`: the user an MCP bridge JWT names."""
    outcome: Final = await _drive(RustAuth.resolve_jwt_identity(request(token, "", None), proxy_settings()), token)
    _raise_if_denied(outcome)
    identity: Final = outcome.claim_identity
    if identity is None:
        raise RuntimeError("rust auth resolved a JWT without an identity")
    return JwtIdentity(user_id=identity.user_id, user_object=identity.user, agent_id=identity.agent_id)


async def resolve_mapping(token: str) -> UserAPIKeyAuth | RustJwtMapping:
    """The virtual key a JWT maps to without provisioning one, or why there is none."""
    outcome: Final = await _drive(RustAuth.resolve_jwt_mapping(request(token, "", None), proxy_settings()), token)
    _raise_if_denied(outcome)
    mapping: Final = outcome.mapping
    if mapping is None:
        raise RuntimeError("rust auth resolved a JWT mapping without a result")
    key: Final = outcome.mapped_key
    if mapping == RustJwtMapping.Key and key is not None:
        return key
    return mapping


@dataclass(frozen=True, slots=True)
class ClaimReading:
    """The JWT auth claim mapping for one `litellm_jwtauth`, read by gateway-auth-core."""

    litellm_jwtauth: LiteLLM_JWTAuth

    def get_user_id(self, token: dict[str, object], default_value: str | None) -> str | None:
        return claims(token, self.litellm_jwtauth).claim_user_id or default_value

    def get_user_email(self, token: dict[str, object], default_value: str | None) -> str | None:
        return claims(token, self.litellm_jwtauth).user_email or default_value

    def get_all_jwt_team_ids(self, token: dict[str, object]) -> list[str]:
        return list(claims(token, self.litellm_jwtauth).all_team_ids)

    def get_team_ids_from_jwt(self, token: dict[str, object]) -> list[str]:
        return list(claims(token, self.litellm_jwtauth).team_ids)

    def map_jwt_role_to_litellm_role(self, token: dict[str, object]) -> LitellmUserRoles | None:
        role: Final = claims(token, self.litellm_jwtauth).litellm_role
        return None if role is None else LitellmUserRoles(role)

    def is_allowed_domain(self, user_email: str) -> bool:
        return is_allowed_domain(user_email, self.litellm_jwtauth)
