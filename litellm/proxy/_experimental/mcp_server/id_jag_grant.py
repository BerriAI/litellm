"""The RFC 7523 jwt-bearer grant on ``POST /token``: a requesting app that obtained an
ID-JAG (Identity Assertion Authorization Grant, the cross-app access profile) from the
customer's identity provider redeems it for the same proxy-API credential ``lite login``
stores. The assertion is proven with the JWT auth issuer and JWKS plumbing the proxy
already trusts, its claims are checked the way the ID-JAG draft prescribes, the requesting
app must be one the operator allowlisted, and the ``jti`` is single-use until the
assertion expires."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from types import MappingProxyType
from typing import Final, Protocol
from urllib.parse import unquote_plus

import jwt
from fastapi import Request
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    AssertionIdentity,
    ConsentTeam,
    SubjectIdentity,
    SubjectTokenRefusal,
)
from litellm.proxy._experimental.mcp_server.idp_token_exchange import (
    gateway_outage_refusal,
    log_line,
    read_token_exchange_prerequisites,
)
from litellm.proxy._experimental.mcp_server.oauth_utils import proxy_base_url_from_env
from litellm.proxy._types import IdJagRequestingApp
from litellm.proxy.auth.handle_jwt import ID_JAG_TYP, AssertionRejection, JWTHandler

ID_JAG_REQUIRED_CLAIMS: Final = ("iss", "sub", "aud", "exp", "iat", "jti", "client_id")
REJECTED_ASSERTION: Final = "assertion was rejected"
TEAM_HEADER: Final = "x-litellm-team-id"
_INACTIVE_SUBJECT: Final = frozenset(("no_active_key", "unresolvable"))
_REPLAY_KEY_PREFIX: Final = "mcp_gateway_dcr:id_jag:"
SCOPE_MCP: Final = "litellm:mcp"
SCOPE_LLM: Final = "litellm:llm"
_RESOURCE_CLAIM_ADAPTER: Final = TypeAdapter(list[str])
SCOPE_ROUTE_GROUPS: Final = MappingProxyType(
    {SCOPE_MCP: ("mcp_inference_routes",), SCOPE_LLM: ("llm_api_routes_without_mcp",)}
)


@dataclass(frozen=True, slots=True)
class GrantedScope:
    """What an assertion's ``scope`` claim buys: the scope echoed in the token response, exactly the
    tokens enforced, and the route groups the credential is narrowed to. No claim means the full
    credential, so both are ``None``."""

    scope: str | None
    allowed_routes: tuple[str, ...] | None


def granted_scope(claim: object, *, mcp_resource: bool) -> GrantedScope | SubjectTokenRefusal:
    """The routes the assertion buys. With ``aud`` the bare origin the ``scope`` claim decides and
    no claim is the full credential. With the MCP resource (``{base}/mcp``) as ``aud`` or as every
    ``resource``, the identity provider named that resource, so the credential is ceilinged to the
    MCP routes whatever the claim says: ``litellm:llm`` cannot be granted and is refused."""
    if claim is not None and not isinstance(claim, str):
        return SubjectTokenRefusal(error="invalid_scope", description=_SCOPE_DESCRIPTION)
    requested: Final = frozenset(claim.split()) if claim else frozenset()
    if not requested <= SCOPE_ROUTE_GROUPS.keys():
        return SubjectTokenRefusal(error="invalid_scope", description=_SCOPE_DESCRIPTION)
    if mcp_resource:
        if SCOPE_LLM in requested:
            return SubjectTokenRefusal(error="invalid_scope", description=_MCP_RESOURCE_DESCRIPTION)
        return GrantedScope(scope=SCOPE_MCP, allowed_routes=SCOPE_ROUTE_GROUPS[SCOPE_MCP])
    if not requested:
        return GrantedScope(scope=None, allowed_routes=None)
    granted: Final = tuple(scope for scope in SCOPE_ROUTE_GROUPS if scope in requested)
    return GrantedScope(scope=" ".join(granted), allowed_routes=tuple(_route_groups(granted)))


def names_mcp_resource(aud: object, base_url: str | None) -> bool:
    """Whether the verified ``aud`` names only the MCP resource, ``{base}/mcp``, and not the
    authorization server itself (the bare origin)."""
    audiences: Final = frozenset((aud,) if isinstance(aud, str) else aud if isinstance(aud, list) else ())
    return base_url is not None and f"{base_url}/mcp" in audiences and base_url not in audiences


def _resource_claim_values(claim: object) -> tuple[str, ...] | None:
    if isinstance(claim, str):
        return (claim,)
    if not isinstance(claim, list):
        return None
    try:
        values: Final = _RESOURCE_CLAIM_ADAPTER.validate_python(claim, strict=True)
    except ValidationError:
        return None
    return tuple(values) if values else None


def resource_names_mcp(claims: Mapping[str, object], base_url: str | None) -> bool | SubjectTokenRefusal:
    """The signed RFC 8707 ``resource`` claim (a URI or an array of URIs) restricts the credential like an
    ``aud`` of ``{base}/mcp`` does."""
    if "resource" not in claims:
        return False
    resources: Final = _resource_claim_values(claims["resource"])
    if resources is None:
        return SubjectTokenRefusal(error="invalid_target", description=_RESOURCE_DESCRIPTION)
    served_resources: Final = frozenset(gateway_audiences(base_url))
    if any(resource not in served_resources for resource in resources):
        return SubjectTokenRefusal(error="invalid_target", description=_RESOURCE_DESCRIPTION)
    return frozenset(resources) == frozenset((f"{base_url}/mcp",))


def _route_groups(scopes: Sequence[str]) -> Iterator[str]:
    for scope in scopes:
        yield from SCOPE_ROUTE_GROUPS[scope]


_SCOPE_DESCRIPTION: Final = (
    f"scope must be {SCOPE_MCP} (MCP gateway routes), {SCOPE_LLM} (LLM API routes), or both, space separated"
)
_RESOURCE_DESCRIPTION: Final = "the assertion's resource claim must name this gateway's origin or its /mcp resource"
_MCP_RESOURCE_DESCRIPTION: Final = (
    f"the /mcp audience or resource only grants MCP routes ({SCOPE_MCP}); use the bare origin as audience and "
    "resource for LLM routes"
)


@dataclass(frozen=True, slots=True)
class IdJagPrerequisites:
    """The deployment-level gates for redeeming an ID-JAG: the JWT auth whose issuer plumbing
    proves the assertion must be on, backed by a database, and licensed, not mapping IdP tokens to
    virtual keys (whose blocks and budgets a minted credential would sidestep), with a shared Redis
    so an assertion is redeemed once across every worker and replica (the in-memory marker is per
    process, so a second worker would mint again), ``PROXY_BASE_URL``
    must fix the gateway URL the assertion's ``aud`` has to name (a request's ``Host`` is the
    caller's to choose), and the operator must have allowlisted the requesting apps. Discovery advertises the grant only when every gate
    holds, and a redemption is refused naming the first one that does not."""

    jwt_auth_enabled: bool
    has_database: bool
    licensed: bool
    maps_jwts_to_virtual_keys: bool
    shared_replay_store: bool
    base_url: str | None
    requesting_apps: tuple[IdJagRequestingApp, ...]

    @property
    def available(self) -> bool:
        return (
            self.jwt_auth_enabled
            and self.has_database
            and self.licensed
            and not self.maps_jwts_to_virtual_keys
            and self.shared_replay_store
            and self.base_url is not None
            and bool(self.requesting_apps)
        )

    def refusal(self) -> SubjectTokenRefusal | None:
        if not self.jwt_auth_enabled:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is not enabled on this gateway, so it cannot redeem identity assertions",
            )
        if not self.has_database:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway has no database, so it cannot redeem identity assertions",
            )
        if not self.licensed:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="JWT auth is an enterprise only feature; no license is set",
            )
        if self.maps_jwts_to_virtual_keys:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway maps IdP tokens to virtual keys, which the jwt-bearer grant does not serve",
            )
        if not self.shared_replay_store:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="configure Redis (cache_params or coordination_redis) so this gateway can redeem an identity assertion exactly once across its workers",
            )
        if self.base_url is None:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="set PROXY_BASE_URL to this gateway's public URL, the audience an identity assertion must name",
            )
        if not self.requesting_apps:
            return SubjectTokenRefusal(
                error="unsupported_grant_type",
                description="this gateway has no requesting apps allowlisted for identity assertions",
            )
        return None


def read_id_jag_prerequisites() -> IdJagPrerequisites:
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # rebound after startup, so read them per call
        jwt_handler,
        redis_usage_cache,
    )

    shared: Final = read_token_exchange_prerequisites()
    return IdJagPrerequisites(
        jwt_auth_enabled=shared.jwt_auth_enabled,
        has_database=shared.has_database,
        licensed=shared.licensed,
        maps_jwts_to_virtual_keys=shared.maps_jwts_to_virtual_keys,
        shared_replay_store=redis_usage_cache is not None,
        base_url=proxy_base_url_from_env(),
        requesting_apps=_requesting_apps(jwt_handler),
    )


def _requesting_apps(jwt_handler: JWTHandler) -> tuple[IdJagRequestingApp, ...]:
    if not hasattr(jwt_handler, "litellm_jwtauth"):
        return ()
    return tuple(jwt_handler.litellm_jwtauth.id_jag_requesting_apps or ())


def jwt_bearer_available() -> bool:
    return read_id_jag_prerequisites().available


def gateway_audiences(base_url: str | None) -> tuple[str, ...]:
    """The issuer values this gateway answers to, which an ID-JAG's ``aud`` must name: its MCP
    resource and its authorization server, both under the fixed ``PROXY_BASE_URL``."""
    return () if base_url is None else (f"{base_url}/mcp", base_url)


class DecodeAssertion(Protocol):
    """Injected signature and claim verification ``(assertion, audience, required_claims)``:
    ``JWTHandler.decode_trusted_assertion`` in production, which returns the verified claims,
    a rejection, or raises when the gateway could not reach the issuer's keys."""

    def __call__(
        self, assertion: str, audience: Sequence[str], required_claims: Sequence[str], /
    ) -> Awaitable[Mapping[str, object] | AssertionRejection]: ...


class ClaimReader(Protocol):
    """The JWT auth user mapping as ``JWTHandler`` exposes it: the litellm user id and email the
    verified claims name through ``user_id_jwt_field`` and ``user_email_jwt_field``."""

    def get_user_id(self, token: dict[str, object], default_value: str | None) -> str | None: ...

    def get_user_email(self, token: dict[str, object], default_value: str | None) -> str | None: ...

    def get_all_jwt_team_ids(self, token: dict[str, object]) -> list[str]: ...

    def is_allowed_domain(self, user_email: str) -> bool: ...


class ResolveSubject(Protocol):
    """Injected subject lookup ``(claims, team_header)``: the litellm user the verified claims
    name through the JWT auth user mapping (``user_id_jwt_field`` or ``user_email_jwt_field``,
    else ``sub``) and the team the credential is minted for (see ``select_team``), ``None`` when
    the gateway knows no such user, or a refusal when the team cannot be settled. Raises on a
    database outage."""

    def __call__(
        self, claims: Mapping[str, object], team_header: str | None, /
    ) -> Awaitable[SubjectIdentity | SubjectTokenRefusal | None]: ...


class LookupUser(Protocol):
    """Injected ``(user_id, user_email)`` lookup: the stored id of the existing litellm user the
    claims name, ``None`` when the gateway knows no such user. Raises on a database outage."""

    def __call__(self, user_id: str, user_email: str | None, /) -> Awaitable[str | None]: ...


class LookupTeams(Protocol):
    """Injected lookup of the teams a user is on, or the failure text when they cannot be read."""

    def __call__(self, user_id: str, /) -> Awaitable[Sequence[ConsentTeam] | str]: ...


class GetUser(Protocol):
    def __call__(
        self, *, user_id: str, user_email: str | None, sso_user_id: str
    ) -> Awaitable[LiteLLM_UserTable | None]: ...


def proxy_user_lookup(get_user: GetUser) -> LookupUser:
    """``LookupUser`` over the proxy's user table. The assertion's subject is matched as the
    local ``user_id`` and as the ``sso_user_id`` an SSO login linked, the way the JWT and SSO
    login paths match it, so a linked account needs no email claim."""

    async def lookup_user(user_id: str, user_email: str | None) -> str | None:
        user: Final = await get_user(user_id=user_id, user_email=user_email, sso_user_id=user_id)
        return None if user is None else user.user_id

    return lookup_user


def subject_resolver(reader: ClaimReader, lookup_user: LookupUser, lookup_teams: LookupTeams) -> ResolveSubject:
    """``ResolveSubject`` over the JWT auth claim mapping and the proxy's user and team lookups."""

    async def resolve_subject(
        claims: Mapping[str, object], team_header: str | None
    ) -> SubjectIdentity | SubjectTokenRefusal | None:
        token: Final = dict(claims)
        subject: Final = claims.get("sub")
        claimed_user_id: Final = reader.get_user_id(token=token, default_value=None) or (
            subject if isinstance(subject, str) else None
        )
        if claimed_user_id is None:
            return None
        user_email: Final = reader.get_user_email(token=token, default_value=None)
        if not reader.is_allowed_domain(user_email or ""):
            return SubjectTokenRefusal(
                error="invalid_grant",
                description="the assertion's subject email is missing or outside the allowed domain",
            )
        user_id: Final = await lookup_user(claimed_user_id, user_email)
        if user_id is None:
            return None
        teams: Final = await lookup_teams(user_id)
        if teams in _INACTIVE_SUBJECT:
            return SubjectTokenRefusal(
                error="invalid_grant", description="the assertion's subject is not an active user"
            )
        if isinstance(teams, str):
            return SubjectTokenRefusal(
                error="temporarily_unavailable", description="the gateway could not read this user's teams; retry"
            )
        team: Final = select_team(team_header, tuple(reader.get_all_jwt_team_ids(token=token)), teams)
        if isinstance(team, SubjectTokenRefusal):
            return team
        return SubjectIdentity(user_id=user_id, team_id=team)

    return resolve_subject


def select_team(
    team_header: str | None, claim_teams: Sequence[str], teams: Sequence[ConsentTeam]
) -> str | None | SubjectTokenRefusal:
    """The team an ID-JAG redemption mints for, settled before the assertion is consumed since
    the minter refuses a teamless credential for a user who has a team. The ``x-litellm-team-id``
    header (a team id or alias) wins and must name one of the user's teams, then the assertion's
    mapped team claim, then the user's only team. A user on several teams must name one."""
    if team_header is not None:
        named: Final = next((team for team in teams if team_header in (team.team_id, team.team_alias)), None)
        if named is None:
            return SubjectTokenRefusal(
                error="invalid_request", description=f"{TEAM_HEADER} does not name one of this user's teams"
            )
        return named.team_id
    asserted: Final = tuple(team.team_id for team in teams if team.team_id in claim_teams)
    if len(asserted) == 1:
        return asserted[0]
    if len(asserted) > 1:
        return SubjectTokenRefusal(
            error="invalid_request",
            description=f"the assertion names several of this user's teams; send the {TEAM_HEADER} header naming the team for this credential",
        )
    if len(teams) == 1:
        return teams[0].team_id
    if teams:
        return SubjectTokenRefusal(
            error="invalid_request",
            description=f"this user belongs to several teams; send the {TEAM_HEADER} header naming the team for this credential",
        )
    return None


@dataclass(frozen=True, slots=True)
class AssertionRedemption:
    """One ``jwt-bearer`` request as the grant sees it: the assertion, how the requesting app
    identified and authenticated itself, the team the credential is asked for, the issuer
    values this gateway answers to (what the assertion's ``aud`` must name), the clock, and the
    leeway the verifier grants ``exp``, so the replay marker outlives every moment the assertion
    would still verify."""

    assertion: str
    client_id: str
    client_secret: str | None
    authorization_header: str | None
    team_id: str | None
    audiences: tuple[str, ...]
    now: float
    leeway_seconds: float


async def redeem_id_jag_assertion(
    assertion: str, client_id: str, client_secret: str | None, request: Request
) -> AssertionIdentity | SubjectTokenRefusal:
    from litellm.proxy._experimental.mcp_server.proxy_api_credentials import (  # noqa: PLC0415  # circular import at module load
        lookup_consent_teams_from_database,
    )
    from litellm.proxy.auth.auth_checks import get_user_object  # noqa: PLC0415  # circular import at module load
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # rebound after startup, so read them per call
        jwt_handler,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    resolve_subject: Final = subject_resolver(
        _claim_reader(jwt_handler),
        proxy_user_lookup(
            partial(
                get_user_object,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                proxy_logging_obj=proxy_logging_obj,
                check_db_only=True,
            )
        ),
        lookup_consent_teams_from_database,
    )
    prerequisites: Final = read_id_jag_prerequisites()
    return await identity_from_id_jag(
        AssertionRedemption(
            assertion=assertion,
            client_id=client_id,
            client_secret=client_secret,
            authorization_header=request.headers.get("authorization"),
            team_id=request.headers.get(TEAM_HEADER),
            audiences=gateway_audiences(prerequisites.base_url),
            now=datetime.now(timezone.utc).timestamp(),
            leeway_seconds=jwt_handler.leeway,
        ),
        prerequisites=prerequisites,
        decode=jwt_handler.decode_trusted_assertion,
        resolve_subject=resolve_subject,
    )


def _claim_reader(handler: ClaimReader) -> ClaimReader:
    return handler


async def identity_from_id_jag(
    redemption: AssertionRedemption,
    prerequisites: IdJagPrerequisites,
    decode: DecodeAssertion,
    resolve_subject: ResolveSubject,
) -> AssertionIdentity | SubjectTokenRefusal:
    """Prove an ID-JAG for the grant tail. The requesting app is checked first, the way a token
    endpoint authenticates its client before it looks at the grant, so an app the operator never
    allowlisted learns nothing about the assertion. Everything wrong with the assertion itself is
    ``invalid_grant`` (RFC 7523 section 3.1), a check the gateway could not complete is
    ``temporarily_unavailable``, and the reason stays in the proxy log because this endpoint is
    public and the verifier's wording can quote the assertion."""
    unmet: Final = prerequisites.refusal()
    if unmet is not None:
        return unmet
    app: Final = next((app for app in prerequisites.requesting_apps if app.client_id == redemption.client_id), None)
    if app is None:
        return SubjectTokenRefusal(error="invalid_client", description="client_id is not an allowlisted requesting app")
    if not _client_authenticated(app, redemption.client_secret, redemption.authorization_header):
        return SubjectTokenRefusal(error="invalid_client", description="client authentication failed")
    try:
        header: Final = jwt.get_unverified_header(redemption.assertion)
    except jwt.PyJWTError:
        return SubjectTokenRefusal(error="invalid_grant", description="assertion is not a JWT")
    if header.get("typ") != ID_JAG_TYP:
        return SubjectTokenRefusal(error="invalid_grant", description=f"assertion typ must be {ID_JAG_TYP}")
    try:
        verified: Final = await decode(redemption.assertion, redemption.audiences, ID_JAG_REQUIRED_CLAIMS)
    except Exception as denied:  # noqa: BLE001  # the verifier raises its 503 for a JWKS outage, and anything else is a bad assertion
        return gateway_outage_refusal(denied, denied, what="an assertion", parameter="assertion") or _rejected(
            str(denied)
        )
    if isinstance(verified, AssertionRejection):
        return _rejected(verified.reason)
    marker: Final = _replay_marker(verified, redemption)
    if isinstance(marker, SubjectTokenRefusal):
        return marker
    replay_key, replay_ttl_seconds = marker
    mcp_resource: Final = resource_names_mcp(verified, prerequisites.base_url)
    if isinstance(mcp_resource, SubjectTokenRefusal):
        return mcp_resource
    scope: Final = granted_scope(
        verified.get("scope"),
        mcp_resource=mcp_resource or names_mcp_resource(verified.get("aud"), prerequisites.base_url),
    )
    if isinstance(scope, SubjectTokenRefusal):
        return scope
    try:
        subject: Final = await resolve_subject(verified, redemption.team_id)
    except Exception as failed:  # noqa: BLE001  # the user lookup raises for a missing user and for a database outage alike
        return gateway_outage_refusal(
            failed, failed, what="an assertion's subject", parameter="assertion"
        ) or _rejected(f"subject lookup failed: {failed}")
    if subject is None:
        return _rejected("the assertion names no user the gateway knows")
    if isinstance(subject, SubjectTokenRefusal):
        return subject
    return AssertionIdentity(
        user_id=subject.user_id,
        team_id=subject.team_id,
        replay_key=replay_key,
        replay_ttl_seconds=replay_ttl_seconds,
        scope=scope.scope,
        allowed_routes=scope.allowed_routes,
    )


def _replay_marker(
    verified: Mapping[str, object], redemption: AssertionRedemption
) -> tuple[str, int] | SubjectTokenRefusal:
    """The single-use key and its lifetime for a verified assertion, once the assertion is bound to
    the presenting app, carries no proof-of-possession the gateway cannot check, and is still alive."""
    if verified.get("client_id") != redemption.client_id:
        return _rejected("the assertion's client_id is not the requesting app that presented it")
    if "cnf" in verified:
        return _rejected("the assertion is proof-bound and this gateway does not check proof of possession")
    issuer: Final = verified.get("iss")
    jti: Final = verified.get("jti")
    exp: Final = verified.get("exp")
    if not isinstance(issuer, str) or not isinstance(jti, str) or not jti or not isinstance(exp, (int, float)):
        return _rejected("the assertion's iss, jti, or exp claim has the wrong shape")
    replay_ttl_seconds: Final = math.ceil(exp + redemption.leeway_seconds - redemption.now)
    if replay_ttl_seconds <= 0:
        return _rejected("the assertion has expired")
    return _REPLAY_KEY_PREFIX + hashlib.sha256(f"{issuer}\n{jti}".encode()).hexdigest(), replay_ttl_seconds


def _rejected(reason: str) -> SubjectTokenRefusal:
    verbose_proxy_logger.warning("jwt-bearer grant refused an assertion: %s", log_line(reason))
    return SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)


def _client_authenticated(app: IdJagRequestingApp, form_secret: str | None, authorization_header: str | None) -> bool:
    """A public app (no secret configured) is identified by ``client_id`` alone. A confidential
    app must present its secret, by HTTP Basic (RFC 6749 section 2.3.1, the credentials
    form-urlencoded inside the base64) or the ``client_secret`` form field, compared in constant
    time."""
    if app.client_secret is None:
        return True
    basic: Final = _basic_credentials(authorization_header)
    if basic is not None:
        return basic[0] == app.client_id and _same_secret(basic[1], app.client_secret)
    return form_secret is not None and _same_secret(form_secret, app.client_secret)


def _same_secret(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode(), expected.encode())


def basic_client_id(authorization_header: str | None) -> str | None:
    """The ``client_id`` an HTTP Basic ``Authorization`` header carries, for a confidential app
    that leaves the form field out as RFC 6749 section 2.3.1 allows."""
    credentials: Final = _basic_credentials(authorization_header)
    return None if credentials is None else credentials[0]


def _basic_credentials(authorization_header: str | None) -> tuple[str, str] | None:
    if authorization_header is None:
        return None
    scheme, _, encoded = authorization_header.strip().partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return None
    try:
        decoded: Final = base64.b64decode(encoded.strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    client_id, separator, secret = decoded.partition(":")
    if not separator:
        return None
    return unquote_plus(client_id), unquote_plus(secret)
