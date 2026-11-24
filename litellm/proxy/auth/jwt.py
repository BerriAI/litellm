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
from litellm.proxy.auth.team_grants import team_grants
from litellm.rust_bridge._native import (
    JWT_ID_JAG_TYP,
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
    jwt_presents_claims,
    jwt_unverified_claims,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

if TYPE_CHECKING:
    from opentelemetry.trace import Span

ID_JAG_TYP: Final = JWT_ID_JAG_TYP


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


def presents_claims(token: str, verified: Mapping[str, object]) -> bool:
    """Whether `token` is the JWT `verified` came from, ignoring the claims the verifier writes."""
    return jwt_presents_claims(token, dict(verified))


def claims(verified: Mapping[str, object], litellm_jwtauth: LiteLLM_JWTAuth) -> RustMappedClaims:
    return jwt_claims(dict(verified), litellm_jwtauth)


async def _drive(call: RustAuth, token: str, route: str = "") -> RustAuthOutcome:
    from litellm.proxy.auth.host.python import proxy_host  # noqa: PLC0415  # host.python imports user_api_key_auth

    return await core.drive(call, proxy_host(route, token))


def _raise_if_denied(outcome: RustAuthOutcome) -> None:
    if outcome.denied:
        auth_denial.raise_denial(outcome)


async def verify(settings: RustAuthSettings, token: str) -> dict[str, object]:
    """`auth_jwt`: the verified claims, or the proxy's error for a token it refuses."""
    outcome: Final = await _drive(RustAuth.verify_jwt(token, time.time(), settings), token)
    _raise_if_denied(outcome)
    verified: Final = outcome.claims
    if verified is None:
        raise RuntimeError("rust auth verified a token without claims")
    return verified


async def verify_assertion(
    settings: RustAuthSettings, token: str, audience: Sequence[str], required_claims: Sequence[str]
) -> Mapping[str, object] | AssertionRejection:
    """`decode_trusted_assertion`: the verified claims or a rejection. A JWKS outage raises the 503 so
    the caller does not call the assertion bad."""
    outcome: Final = await _drive(
        RustAuth.verify_jwt_assertion(
            token,
            time.time(),
            settings,
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
    settings: RustAuthSettings,
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
            core.request(token, route, request_method, headers=headers, model=model),
            settings,
            provisioning=provisioning,
        ),
        token,
        route,
    )
    _raise_if_denied(outcome)
    identity: Final = outcome.identity
    if not isinstance(identity, RustIdentity.Jwt):
        raise RuntimeError(f"rust auth authorized a JWT as something else: {identity!r}")
    return jwt_result(identity, token)


def jwt_result(identity: RustIdentity.Jwt, token: str | None) -> JWTAuthBuilderResult:
    """The `JWTAuthBuilderResult` `authorize_jwt` returned for `token`, from the core's ids and rows."""
    end_user: Final = identity.end_user
    return JWTAuthBuilderResult(
        is_proxy_admin=identity.is_proxy_admin,
        team_id=identity.team_id,
        team_object=identity.team,
        user_id=identity.user_id,
        user_email=identity.user_email,
        user_object=identity.user,
        org_id=identity.org_id,
        org_object=identity.org,
        end_user_id=identity.end_user_id,
        end_user_object=None if end_user is None else end_user.end_user,
        token=token or "",
        team_membership=identity.membership,
        jwt_claims=dict(identity.claims),
        agent_id=identity.agent_id,
        managed_agent_context=identity.managed_agent,
    )


def auth_from_result(result: JWTAuthBuilderResult, parent_otel_span: Span | None = None) -> UserAPIKeyAuth:
    """Keep JWT identity and permission attribution identical across consumers."""
    user: Final = result["user_object"]
    admin: Final = result["is_proxy_admin"]
    auth: Final = UserAPIKeyAuth(
        api_key=None,
        user_role=(
            LitellmUserRoles.PROXY_ADMIN
            if admin
            else LitellmUserRoles.INTERNAL_USER
            if result.get("managed_agent_context") is not None
            else LitellmUserRoles(user.user_role)
            if user is not None and user.user_role is not None
            else LitellmUserRoles.INTERNAL_USER
        ),
        user_id=result["user_id"],
        user_email=result["user_email"],
        team_id=result["team_id"],
        org_id=result["org_id"],
        end_user_id=result["end_user_id"],
        parent_otel_span=parent_otel_span,
        jwt_claims=result["jwt_claims"],
        agent_id=result.get("agent_id"),
        user_tpm_limit=user.tpm_limit if user is not None and not admin else None,
        user_rpm_limit=user.rpm_limit if user is not None and not admin else None,
        user_model_max_budget=user.model_max_budget if user is not None and not admin else None,
        **team_grants(
            team_object=result["team_object"],
            team_membership=result.get("team_membership"),
            user_id=result["user_id"],
        ),
    )
    auth.managed_agent_context = result.get("managed_agent_context")
    auth._managed_delegation_verified = (  # pyright: ignore[reportPrivateUsage]  # JWT admission produces the one-shot proof consumed by managed authorization
        auth.managed_agent_context is not None and auth.managed_agent_context.mode == "delegated"
    )
    return auth


async def resolve_identity(settings: RustAuthSettings, token: str) -> JwtIdentity:
    """`JWTAuthManager.resolve_identity`: the user an MCP bridge JWT names."""
    outcome: Final = await _drive(RustAuth.resolve_jwt_identity(core.request(token, "", None), settings), token)
    _raise_if_denied(outcome)
    identity: Final = outcome.claim_identity
    if identity is None:
        raise RuntimeError("rust auth resolved a JWT without an identity")
    return JwtIdentity(user_id=identity.user_id, user_object=identity.user, agent_id=identity.agent_id)


async def resolve_mapping(settings: RustAuthSettings, token: str) -> UserAPIKeyAuth | RustJwtMapping:
    """The virtual key a JWT maps to without provisioning one, or why there is none."""
    outcome: Final = await _drive(RustAuth.resolve_jwt_mapping(core.request(token, "", None), settings), token)
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
        return jwt_is_allowed_domain(user_email, self.litellm_jwtauth)
