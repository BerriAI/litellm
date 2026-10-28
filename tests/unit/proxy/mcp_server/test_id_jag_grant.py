import hashlib
import logging
import math
from base64 import b64encode
from collections.abc import Mapping, Sequence
from typing import Final

import jwt
import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import IdJagRequestingApp, LiteLLM_JWTAuth, LiteLLMRoutes
from litellm.proxy.auth.handle_jwt import (
    ID_JAG_TYP,
    AssertionRejection,
    JWKSUnreachableError,
    JWTHandler,
    jwks_unavailable_exception,
)
from litellm.proxy.mcp_server.gateway_dcr_flow import (
    AssertionIdentity,
    ConsentTeam,
    ReloadUserFailure,
    SubjectIdentity,
    SubjectTokenRefusal,
)
from litellm.proxy.mcp_server.id_jag_grant import (
    ID_JAG_REQUIRED_CLAIMS,
    REJECTED_ASSERTION,
    TEAM_HEADER,
    AssertionRedemption,
    GrantedScope,
    IdJagPrerequisites,
    LookupTeams,
    _claim_reader,
    basic_client_id,
    gateway_audiences,
    granted_scope,
    identity_from_id_jag,
    jwt_bearer_available,
    names_mcp_resource,
    proxy_user_lookup,
    resource_names_mcp,
    select_team,
    subject_resolver,
)

ISSUER = "https://idp.example.com"
GATEWAY = "https://gateway.example.com"
AUDIENCES = (f"{GATEWAY}/mcp", GATEWAY)
NOW = 1_800_000_000.0
PUBLIC_APP = IdJagRequestingApp(client_id="0oa-public")
CONFIDENTIAL_APP = IdJagRequestingApp(client_id="0oa-confidential", client_secret="s3cret")
SPACED_APP = IdJagRequestingApp(client_id="0oa spaced", client_secret="app secret")
EVERY_GATE_HOLDS = {
    "jwt_auth_enabled": True,
    "has_database": True,
    "licensed": True,
    "maps_jwts_to_virtual_keys": False,
    "shared_replay_store": True,
    "base_url": "https://gateway.example.com",
    "requesting_apps": (PUBLIC_APP, CONFIDENTIAL_APP, SPACED_APP),
}
JWKS_DOWN = jwks_unavailable_exception(JWKSUnreachableError("ConnectError fetching the IdP's JWKS after 3 attempts"))


def _claims(**overrides: object) -> dict[str, object]:
    return {
        "iss": ISSUER,
        "sub": "okta-user-1",
        "aud": AUDIENCES[1],
        "exp": NOW + 300,
        "iat": NOW - 5,
        "jti": "jti-1",
        "client_id": PUBLIC_APP.client_id,
        "scope": "litellm:mcp",
        **overrides,
    }


def _assertion(typ: str = ID_JAG_TYP, **claims: object) -> str:
    return jwt.encode(
        _claims(**claims),
        "unused-by-the-fake-verifier-but-long-enough-for-hs256",
        algorithm="HS256",
        headers={"typ": typ},
    )


def _unscoped_assertion(**claims: object) -> str:
    unscoped_claims: Final = {key: value for key, value in _claims(**claims).items() if key != "scope"}
    return jwt.encode(
        unscoped_claims,
        "unused-by-the-fake-verifier-but-long-enough-for-hs256",
        algorithm="HS256",
        headers={"typ": ID_JAG_TYP},
    )


class _Verifier:
    """Stands in for the injected ``DecodeAssertion`` port; records each call as a new tuple."""

    def __init__(
        self, result: Mapping[str, object] | AssertionRejection | None = None, raises: BaseException | None = None
    ) -> None:
        self.calls: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
        self.result = result
        self.raises = raises

    async def __call__(
        self, assertion: str, audience: Sequence[str], required_claims: Sequence[str]
    ) -> Mapping[str, object] | AssertionRejection:
        self.calls = (*self.calls, (assertion, tuple(audience), tuple(required_claims)))  # rebind-ok: immutable record
        if self.raises is not None:
            raise self.raises
        return self.result if self.result is not None else jwt.decode(assertion, options={"verify_signature": False})


class _Resolver:
    """Stands in for the injected ``ResolveSubject`` port; records each call as a new tuple."""

    def __init__(
        self,
        user_id: str | None = "u1",
        raises: BaseException | None = None,
        result: SubjectIdentity | SubjectTokenRefusal | None = None,
    ) -> None:
        self.calls: tuple[tuple[Mapping[str, object], str | None], ...] = ()
        self.user_id = user_id
        self.raises = raises
        self.result = result

    async def __call__(
        self, claims: Mapping[str, object], team_header: str | None
    ) -> SubjectIdentity | SubjectTokenRefusal | None:
        self.calls = (*self.calls, (dict(claims), team_header))  # rebind-ok: immutable record
        if self.raises is not None:
            raise self.raises
        if self.result is not None:
            return self.result
        return None if self.user_id is None else SubjectIdentity(user_id=self.user_id, team_id=team_header)


def _redemption(
    assertion: str,
    client_id: str = PUBLIC_APP.client_id,
    client_secret: str | None = None,
    authorization: str | None = None,
    team_id: str | None = "team-b",
    leeway: float = 0.0,
) -> AssertionRedemption:
    return AssertionRedemption(
        assertion=assertion,
        client_id=client_id,
        client_secret=client_secret,
        authorization_header=authorization,
        team_id=team_id,
        audiences=AUDIENCES,
        now=NOW,
        leeway_seconds=leeway,
    )


@pytest.mark.asyncio
async def test_the_replay_marker_outlives_the_leeway_the_verifier_grants_exp():
    """PyJWT accepts an assertion until ``exp`` plus the handler's leeway, so the marker must
    live that long too, including for an assertion already past ``exp`` but inside the leeway."""
    within_leeway = await _identity(_redemption(_assertion(), leeway=30), _Verifier(), _Resolver())
    assert isinstance(within_leeway, AssertionIdentity)
    assert within_leeway.replay_ttl_seconds == 330
    just_expired = await _identity(
        _redemption(_assertion(), leeway=30), _Verifier(result=_claims(exp=NOW - 10)), _Resolver()
    )
    assert isinstance(just_expired, AssertionIdentity)
    assert just_expired.replay_ttl_seconds == 20


async def _identity(
    redemption: AssertionRedemption,
    verifier: _Verifier | None = None,
    resolver: _Resolver | None = None,
    **unmet: object,
) -> AssertionIdentity | SubjectTokenRefusal:
    return await identity_from_id_jag(
        redemption,
        prerequisites=IdJagPrerequisites(**{**EVERY_GATE_HOLDS, **unmet}),
        decode=verifier if verifier is not None else _Verifier(),
        resolve_subject=resolver if resolver is not None else _Resolver(),
    )


@pytest.mark.asyncio
async def test_a_valid_id_jag_names_the_user_its_single_use_key_and_its_scope():
    """The verifier is asked for the gateway's own issuer values as audience and for every claim
    the ID-JAG profile requires, the subject lookup sees the verified claims, and the identity
    carries the team asked for, a replay key derived from iss and jti, a marker TTL that outlives
    the assertion, and the assertion's scope."""
    assertion, verifier, resolver = _assertion(), _Verifier(), _Resolver()
    identity = await _identity(_redemption(assertion), verifier, resolver)
    assert isinstance(identity, AssertionIdentity)
    assert verifier.calls == ((assertion, AUDIENCES, ID_JAG_REQUIRED_CLAIMS),)
    assert {"iss", "sub", "aud", "exp", "iat", "jti", "client_id"} <= set(ID_JAG_REQUIRED_CLAIMS)
    assert resolver.calls == ((_claims(), "team-b"),)
    assert (identity.user_id, identity.team_id, identity.scope) == ("u1", "team-b", "litellm:mcp")
    assert identity.allowed_routes == ("mcp_inference_routes",)
    assert identity.replay_key == "mcp_gateway_dcr:id_jag:" + hashlib.sha256(f"{ISSUER}\njti-1".encode()).hexdigest()
    assert identity.replay_ttl_seconds == math.ceil(300)
    other_jti = await _identity(_redemption(_assertion(jti="jti-2")), _Verifier(), _Resolver())
    assert isinstance(other_jti, AssertionIdentity)
    assert other_jti.replay_key != identity.replay_key
    teamless = await _identity(_redemption(_assertion(scope=""), team_id=None))
    assert isinstance(teamless, AssertionIdentity)
    assert (teamless.team_id, teamless.scope, teamless.allowed_routes) == (None, None, None)


@pytest.mark.asyncio
async def test_a_requesting_app_the_operator_did_not_allowlist_learns_nothing_about_the_assertion():
    verifier, resolver = _Verifier(), _Resolver()
    refused = await _identity(_redemption(_assertion(), client_id="0oa-stranger"), verifier, resolver)
    assert refused == SubjectTokenRefusal(
        error="invalid_client", description="client_id is not an allowlisted requesting app"
    )
    assert verifier.calls == ()
    assert resolver.calls == ()


def _basic(client_id: str, secret: str) -> str:
    return "Basic " + b64encode(f"{client_id}:{secret}".encode()).decode()


@pytest.mark.parametrize(
    "client_secret, authorization, accepted",
    [
        (None, None, False),
        ("wrong", None, False),
        ("s3cr\u00e9t", None, False),
        ("s3cret", None, True),
        (None, _basic(CONFIDENTIAL_APP.client_id, "s3cr\u00e9t"), False),
        (None, _basic(CONFIDENTIAL_APP.client_id, "s3cret"), True),
        (None, _basic(CONFIDENTIAL_APP.client_id, "wrong"), False),
        (None, _basic(PUBLIC_APP.client_id, "s3cret"), False),
        ("s3cret", _basic(CONFIDENTIAL_APP.client_id, "wrong"), False),
        (None, "Basic not-base64!", False),
        (None, "Bearer s3cret", False),
    ],
)
@pytest.mark.asyncio
async def test_a_confidential_app_must_authenticate_with_its_secret(client_secret, authorization, accepted):
    verifier = _Verifier()
    assertion = _assertion(client_id=CONFIDENTIAL_APP.client_id)
    redemption = _redemption(assertion, CONFIDENTIAL_APP.client_id, client_secret, authorization)
    identity = await _identity(redemption, verifier)
    if accepted:
        assert isinstance(identity, AssertionIdentity)
        return
    assert identity == SubjectTokenRefusal(error="invalid_client", description="client authentication failed")
    assert verifier.calls == ()


@pytest.mark.parametrize(
    "client_secret, authorization, accepted",
    [
        ("app secret", None, True),
        (None, "Basic " + b64encode(b"0oa+spaced:app+secret").decode(), True),
        (None, "Basic " + b64encode(b"0oa%20spaced:app%20secret").decode(), True),
        (None, "Basic " + b64encode(b"0oa spaced:app secret").decode(), True),
        (None, "Basic " + b64encode(b"0oa+spaced:app+wrong").decode(), False),
        (None, "Basic " + b64encode(b"0oa-spaced:app+secret").decode(), False),
    ],
)
@pytest.mark.asyncio
async def test_basic_credentials_are_form_decoded_so_a_space_in_an_id_or_secret_still_matches(
    client_secret, authorization, accepted
):
    """RFC 6749 section 2.3.1 form-encodes the id and secret before base64, so ``+`` is a space. The
    endpoint hands over the ``client_id`` it read from the header, so the header must name the same app."""
    assertion = _assertion(client_id=SPACED_APP.client_id)
    identity = await _identity(_redemption(assertion, SPACED_APP.client_id, client_secret, authorization), _Verifier())
    if accepted:
        assert isinstance(identity, AssertionIdentity)
        return
    assert identity == SubjectTokenRefusal(error="invalid_client", description="client authentication failed")


@pytest.mark.asyncio
async def test_a_public_app_is_identified_by_client_id_alone():
    identity = await _identity(_redemption(_assertion(), client_secret="anything", authorization="Bearer x"))
    assert isinstance(identity, AssertionIdentity)


@pytest.mark.parametrize(
    "assertion, description",
    [
        ("not-a-jwt", "assertion is not a JWT"),
        (_assertion(typ="JWT"), f"assertion typ must be {ID_JAG_TYP}"),
        (_assertion(typ="at+jwt"), f"assertion typ must be {ID_JAG_TYP}"),
    ],
)
@pytest.mark.asyncio
async def test_a_bearer_or_id_token_is_not_an_id_jag(assertion, description):
    verifier = _Verifier()
    refused = await _identity(_redemption(assertion), verifier)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=description)
    assert verifier.calls == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verifier, reason",
    [
        (_Verifier(result=AssertionRejection(reason="Signature verification failed")), "Signature verification"),
        (_Verifier(result=AssertionRejection(reason="Audience doesn't match")), "Audience"),
        (_Verifier(result=AssertionRejection(reason='Token is missing the "jti" claim')), "jti"),
        (_Verifier(result=AssertionRejection(reason="Signature has expired")), "expired"),
        (_Verifier(result=AssertionRejection(reason="iss is not an issuer this gateway trusts")), "iss"),
        (_Verifier(raises=Exception("Validation fails: Invalid audience")), "Invalid audience"),
        (_Verifier(result=AssertionRejection(reason="bad kid\nWARNING forged line")), "bad kid WARNING forged line"),
    ],
)
async def test_an_assertion_the_verifier_rejects_is_refused_with_the_reason_kept_in_the_log(verifier, reason, caplog):
    resolver = _Resolver()
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), verifier, resolver)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert reason not in refused.description
    assert reason in caplog.text
    assert resolver.calls == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims, reason",
    [
        ({"client_id": "0oa-confidential"}, "client_id"),
        ({"exp": NOW}, "expired"),
        ({"exp": NOW - 1}, "expired"),
        ({"jti": ""}, "jti"),
        ({"jti": 7}, "jti"),
        ({"iss": None}, "iss"),
        ({"cnf": {"jkt": "thumbprint"}}, "proof"),
    ],
)
async def test_verified_claims_that_do_not_fit_the_profile_are_refused(claims, reason, caplog):
    resolver = _Resolver()
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), _Verifier(result=_claims(**claims)), resolver)
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert reason in caplog.text
    assert resolver.calls == ()


@pytest.mark.asyncio
async def test_an_assertion_that_names_no_user_the_gateway_knows_cannot_be_redeemed(caplog):
    with caplog.at_level(logging.WARNING, logger="litellm.proxy"):
        refused = await _identity(_redemption(_assertion()), resolver=_Resolver(user_id=None))
    assert refused == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)
    assert "names no user" in caplog.text
    missing = await _identity(
        _redemption(_assertion()), resolver=_Resolver(raises=ValueError("User doesn't exist in db. 'user_id'=u1"))
    )
    assert missing == SubjectTokenRefusal(error="invalid_grant", description=REJECTED_ASSERTION)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verifier, resolver",
    [
        (_Verifier(raises=JWKS_DOWN), _Resolver()),
        (_Verifier(), _Resolver(raises=HTTPException(status_code=503, detail="database unavailable"))),
    ],
)
async def test_a_gateway_outage_is_reported_as_retryable_not_as_a_bad_assertion(verifier, resolver):
    refused = await _identity(_redemption(_assertion()), verifier, resolver)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "temporarily_unavailable"
    assert "retry" in refused.description
    assert "assertion" in refused.description
    assert "subject_token" not in refused.description


@pytest.mark.parametrize(
    "unmet, description",
    [
        ({"jwt_auth_enabled": False}, "JWT auth is not enabled"),
        ({"has_database": False}, "no database"),
        ({"licensed": False}, "enterprise only"),
        ({"maps_jwts_to_virtual_keys": True}, "maps IdP tokens to virtual keys"),
        ({"shared_replay_store": False}, "Redis"),
        ({"base_url": None}, "PROXY_BASE_URL"),
        ({"requesting_apps": ()}, "no requesting apps allowlisted"),
    ],
)
@pytest.mark.asyncio
async def test_the_deployment_gates_refuse_before_any_verification(unmet, description):
    verifier = _Verifier()
    refused = await _identity(_redemption(_assertion()), verifier, **unmet)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "unsupported_grant_type"
    assert description in refused.description
    assert verifier.calls == ()
    assert IdJagPrerequisites(**{**EVERY_GATE_HOLDS, **unmet}).available is False
    assert IdJagPrerequisites(**EVERY_GATE_HOLDS).available is True


def _running_jwt_handler(litellm_jwtauth: LiteLLM_JWTAuth | None) -> JWTHandler:
    handler: Final = JWTHandler()
    if litellm_jwtauth is not None:
        handler.update_environment(prisma_client=None, user_api_key_cache=DualCache(), litellm_jwtauth=litellm_jwtauth)
    return handler


@pytest.mark.parametrize(
    "general_settings, prisma_client, premium_user, litellm_jwtauth, expected",
    [
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), True),
        (
            {"enable_jwt_auth": True, "redis": None},
            object(),
            True,
            LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]),
            False,
        ),
        (
            {"enable_jwt_auth": True, "base_url": ""},
            object(),
            True,
            LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]),
            False,
        ),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[]), False),
        (
            {"enable_jwt_auth": True},
            object(),
            True,
            LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP], virtual_key_claim_field="vkey"),
            False,
        ),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(), False),
        ({"enable_jwt_auth": True}, object(), True, None, False),
        ({}, object(), True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
        ({"enable_jwt_auth": True}, None, True, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
        ({"enable_jwt_auth": True}, object(), False, LiteLLM_JWTAuth(id_jag_requesting_apps=[PUBLIC_APP]), False),
    ],
)
def test_availability_is_read_from_the_running_proxy(
    monkeypatch, general_settings, prisma_client, premium_user, litellm_jwtauth, expected
):
    monkeypatch.setenv("PROXY_BASE_URL", general_settings.get("base_url", "https://gateway.example.com"))
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", general_settings)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", premium_user)
    monkeypatch.setattr("litellm.proxy.proxy_server.redis_usage_cache", general_settings.get("redis", object()))
    monkeypatch.setattr("litellm.proxy.proxy_server.jwt_handler", _running_jwt_handler(litellm_jwtauth))
    assert jwt_bearer_available() is expected


def test_the_assertion_audience_is_the_fixed_gateway_url_not_the_request_host():
    """An ID-JAG must name this gateway's MCP resource or its authorization server under the
    operator-fixed PROXY_BASE_URL. Without it there is no audience, so there is no grant."""
    assert gateway_audiences("https://gateway.example.com") == (
        "https://gateway.example.com/mcp",
        "https://gateway.example.com",
    )
    assert gateway_audiences(None) == ()


@pytest.mark.asyncio
async def test_a_team_the_subject_lookup_cannot_settle_is_refused_before_the_assertion_is_consumed():
    """The resolver's refusal (a user on several teams who named none, or a header naming a team
    they are not on) comes back as is, so the client fixes its request and presents the same
    assertion again instead of finding its jti burned."""
    refusal = SubjectTokenRefusal(error="invalid_request", description=f"send the {TEAM_HEADER} header")
    refused = await _identity(_redemption(_assertion()), resolver=_Resolver(result=refusal))
    assert refused is refusal
    teamed = await _identity(
        _redemption(_assertion()), resolver=_Resolver(result=SubjectIdentity(user_id="u1", team_id="t9"))
    )
    assert isinstance(teamed, AssertionIdentity)
    assert teamed.team_id == "t9"


ENG = ConsentTeam(team_id="team-eng", team_alias="engineering")
OPS = ConsentTeam(team_id="team-ops", team_alias=None)


@pytest.mark.parametrize(
    "team_header, claim_teams, teams, expected",
    [
        (None, (), (), None),
        (None, (), (ENG,), "team-eng"),
        ("team-eng", (), (ENG, OPS), "team-eng"),
        ("engineering", (), (ENG, OPS), "team-eng"),
        ("team-ops", ("team-eng",), (ENG, OPS), "team-ops"),
        (None, ("team-ops",), (ENG, OPS), "team-ops"),
        (None, ("team-sales", "team-ops"), (ENG, OPS), "team-ops"),
        (None, ("0oa-xaa-public-app",), (ENG,), "team-eng"),
        (None, ("0oa-xaa-public-app",), (), None),
    ],
)
def test_select_team_prefers_the_header_then_the_assertion_then_the_only_team(
    team_header, claim_teams, teams, expected
):
    assert select_team(team_header, claim_teams, teams) == expected


class _Reader:
    def __init__(self, user_id=None, email=None, team_ids=(), allowed_domain=None):
        self.user_id, self.email, self.team_ids, self.allowed_domain = user_id, email, team_ids, allowed_domain

    def is_allowed_domain(self, user_email):
        return self.allowed_domain is None or user_email.split("@")[-1] == self.allowed_domain

    def get_user_id(self, token, default_value):
        return self.user_id or default_value

    def get_user_email(self, token, default_value):
        return self.email or default_value

    def get_all_jwt_team_ids(self, token):
        return list(self.team_ids)


async def _known_user(user_id: str, user_email: str | None) -> str | None:
    return user_id if user_id in {"okta-user-1", "u-mapped"} else None


async def _two_teams(user_id: str) -> tuple[ConsentTeam, ...]:
    return (ENG, OPS)


async def _teams_unreadable(user_id: str) -> ReloadUserFailure:
    return "unavailable"


def _teams_failing(failure: ReloadUserFailure) -> LookupTeams:
    async def lookup_teams(user_id: str) -> tuple[ConsentTeam, ...] | ReloadUserFailure:
        return failure

    return lookup_teams


@pytest.mark.asyncio
async def test_the_subject_resolver_binds_the_claims_to_a_known_user_and_their_asserted_team():
    """The user comes from the JWT auth user mapping, else ``sub``, and must exist. The team the
    assertion asserts settles a multi-team user, the header still wins, and unreadable teams are a
    retryable refusal rather than a credential for the wrong team."""
    asserted = subject_resolver(_Reader(team_ids=("team-ops",)), _known_user, _two_teams)
    assert await asserted(_claims(), None) == SubjectIdentity(user_id="okta-user-1", team_id="team-ops")
    assert await asserted(_claims(), "team-eng") == SubjectIdentity(user_id="okta-user-1", team_id="team-eng")
    mapped = subject_resolver(_Reader(user_id="u-mapped", email="m@example.com"), _known_user, _two_teams)
    assert await mapped(_claims(), "engineering") == SubjectIdentity(user_id="u-mapped", team_id="team-eng")
    unknown = subject_resolver(_Reader(), _known_user, _two_teams)
    assert await unknown(_claims(sub="nobody"), None) is None
    unreadable = await subject_resolver(_Reader(), _known_user, _teams_unreadable)(_claims(), None)
    assert isinstance(unreadable, SubjectTokenRefusal)
    assert unreadable.error == "temporarily_unavailable"


@pytest.mark.parametrize("failure", ["no_active_key", "unresolvable"])
@pytest.mark.asyncio
async def test_a_blocked_or_unmapped_user_is_a_bad_grant_not_an_outage_to_retry(failure):
    """A deactivated user, or one the resolver cannot map, can retry forever and never succeed, so
    the refusal is invalid_grant. Only a database the gateway could not reach is retryable."""
    refused = await subject_resolver(_Reader(), _known_user, _teams_failing(failure))(_claims(), None)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "invalid_grant"
    assert "retry" not in refused.description
    outage = await subject_resolver(_Reader(), _known_user, _teams_failing("faulted"))(_claims(), None)
    assert isinstance(outage, SubjectTokenRefusal)
    assert outage.error == "temporarily_unavailable"


def test_the_assertion_team_claims_ignore_team_id_default():
    """``team_id_default`` is how bearer JWT auth picks a team for a request, not a claim the
    assertion made, so a user outside that team must still get their only team or no team."""
    handler = JWTHandler()
    handler.litellm_jwtauth = LiteLLM_JWTAuth(team_id_default="default-team")
    claim_teams = tuple(_claim_reader(handler).get_all_jwt_team_ids(token={"iss": ISSUER, "sub": "u1"}))
    assert claim_teams == ()
    assert select_team(None, claim_teams, (ENG,)) == "team-eng"
    assert select_team(None, claim_teams, ()) is None


@pytest.mark.parametrize(
    "team_header, claim_teams, teams, error, mention",
    [
        ("team-sales", (), (ENG, OPS), "invalid_request", TEAM_HEADER),
        ("engineering", (), (), "invalid_request", TEAM_HEADER),
        (None, (), (ENG, OPS), "invalid_request", TEAM_HEADER),
        (None, ("team-eng", "team-ops"), (ENG, OPS), "invalid_request", TEAM_HEADER),
        (None, ("0oa-xaa-public-app",), (ENG, OPS), "invalid_request", TEAM_HEADER),
    ],
)
def test_select_team_refuses_a_team_the_user_is_not_on_or_an_unnamed_choice(
    team_header, claim_teams, teams, error, mention
):
    refused = select_team(team_header, claim_teams, teams)
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == error
    assert mention in refused.description


@pytest.mark.parametrize(
    "authorization, expected",
    [
        (None, None),
        ("Bearer abc", None),
        ("Basic not-base64!", None),
        (_basic("0oa-confidential", "s3cret"), "0oa-confidential"),
        ("basic " + b64encode(b"0oa%20x:s3cret").decode(), "0oa x"),
        ("Basic " + b64encode(b"0oa+x:s3cret").decode(), "0oa x"),
        ("Basic " + b64encode(b"no-separator").decode(), None),
    ],
)
def test_basic_client_id_reads_the_app_from_the_authorization_header(authorization, expected):
    assert basic_client_id(authorization) == expected


MCP_ROUTE_GROUPS = ("mcp_inference_routes",)
LLM_ROUTE_GROUPS = ("llm_api_routes_without_mcp",)


@pytest.mark.parametrize(
    "claim, scope, allowed_routes",
    [
        (None, None, None),
        ("", None, None),
        ("litellm:mcp", "litellm:mcp", MCP_ROUTE_GROUPS),
        ("litellm:llm", "litellm:llm", LLM_ROUTE_GROUPS),
        ("litellm:mcp litellm:llm", "litellm:mcp litellm:llm", MCP_ROUTE_GROUPS + LLM_ROUTE_GROUPS),
        ("litellm:llm  litellm:mcp litellm:llm", "litellm:mcp litellm:llm", MCP_ROUTE_GROUPS + LLM_ROUTE_GROUPS),
    ],
)
def test_the_scope_claim_buys_exactly_the_route_groups_it_names(claim, scope, allowed_routes):
    """``litellm:mcp`` is the MCP gateway's data-plane routes, ``litellm:llm`` the LLM API routes
    without their MCP members, both together are both, and no claim is the full credential. The
    scope echoed back is canonical: deduplicated, in vocabulary order, so it is exactly what is
    enforced. Every group named must be one the route check resolves."""
    granted = granted_scope(claim, mcp_resource=False)
    assert (granted.scope, granted.allowed_routes) == (scope, allowed_routes)
    if allowed_routes is not None:
        assert set(allowed_routes) <= set(LiteLLMRoutes._member_names_)
        mcp_members = set(LiteLLMRoutes.mcp_inference_routes.value)
        llm_members = set().union(*(LiteLLMRoutes[group].value for group in LLM_ROUTE_GROUPS))
        assert "/chat/completions" in llm_members and "/mcp" not in llm_members
        assert "/mcp" in mcp_members and "/chat/completions" not in mcp_members


@pytest.mark.parametrize("claim", ["openid", "litellm:mcp openid", "litellm:admin", 7, ["litellm:mcp"]])
@pytest.mark.asyncio
async def test_a_scope_the_gateway_does_not_know_is_refused_as_invalid_scope_before_any_lookup(claim):
    refusal = granted_scope(claim, mcp_resource=False)
    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "invalid_scope"
    assert "litellm:mcp" in refusal.description and "litellm:llm" in refusal.description
    resolver = _Resolver()
    refused = await _identity(_redemption(_assertion()), _Verifier(result=_claims(scope=claim)), resolver)
    assert refused == refusal
    assert resolver.calls == ()


@pytest.mark.asyncio
async def test_a_linked_sso_account_resolves_by_its_provider_subject_without_an_email_claim():
    """An SSO login stores the provider subject as ``sso_user_id`` on a user whose ``user_id`` is
    something else, so the lookup must offer the subject as both, with no email to fall back on."""
    from litellm.models.user import LiteLLM_UserTable

    async def get_user(*, user_id, user_email, sso_user_id):
        if sso_user_id == "00u-okta-subject" and user_email is None:
            return LiteLLM_UserTable(user_id="local-user-7", user_email=None)
        return None

    lookup = proxy_user_lookup(get_user)
    assert await lookup("00u-okta-subject", None) == "local-user-7"
    assert await lookup("nobody", None) is None


@pytest.mark.parametrize("email", ["eve@evil.example", None])
@pytest.mark.asyncio
async def test_the_allowed_email_domain_gates_the_assertions_subject_like_bearer_jwt_auth(email):
    """With ``user_allowed_email_domain`` set, bearer JWT auth refuses an identity whose email is
    outside the domain or missing. An assertion for an existing user gets the same refusal before
    any lookup, as invalid_grant, and a matching email still resolves."""
    refused = await subject_resolver(_Reader(email=email, allowed_domain="corp.example"), _known_user, _two_teams)(
        _claims(), "team-eng"
    )
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "invalid_grant"
    allowed = subject_resolver(
        _Reader(email="ann@corp.example", allowed_domain="corp.example"), _known_user, _two_teams
    )
    assert await allowed(_claims(), "team-eng") == SubjectIdentity(user_id="okta-user-1", team_id="team-eng")


@pytest.mark.parametrize("claim", [None, "", "litellm:mcp", "litellm:mcp litellm:mcp"])
def test_an_assertion_for_the_mcp_resource_is_ceilinged_to_the_mcp_routes(claim):
    """``aud = {base}/mcp`` means the identity provider named the MCP resource, so no scope or
    ``litellm:mcp`` both mint an MCP-only credential and echo ``litellm:mcp``."""
    granted = granted_scope(claim, mcp_resource=True)
    assert granted == GrantedScope(scope="litellm:mcp", allowed_routes=("mcp_inference_routes",))


@pytest.mark.parametrize("claim", ["litellm:llm", "litellm:mcp litellm:llm"])
def test_llm_routes_cannot_be_granted_under_the_mcp_resource_audience(claim):
    refusal = granted_scope(claim, mcp_resource=True)
    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "invalid_scope"
    assert "/mcp audience" in refusal.description and "bare origin" in refusal.description


@pytest.mark.parametrize(
    "aud, expected",
    [
        (AUDIENCES[0], True),
        (AUDIENCES[1], False),
        ([AUDIENCES[0]], True),
        ([AUDIENCES[0], AUDIENCES[1]], False),
        ("https://other.example", False),
        (None, False),
    ],
)
def test_the_mcp_resource_audience_is_the_mcp_path_without_the_bare_origin(aud, expected):
    assert names_mcp_resource(aud, GATEWAY) is expected
    assert names_mcp_resource(aud, None) is False


@pytest.mark.parametrize(
    "claims, expected",
    [
        ({}, False),
        ({"resource": f"{GATEWAY}/mcp"}, True),
        ({"resource": [f"{GATEWAY}/mcp"]}, True),
        ({"resource": GATEWAY}, False),
        ({"resource": [GATEWAY, f"{GATEWAY}/mcp"]}, False),
        ({"resource": "https://other.example"}, None),
        ({"resource": [f"{GATEWAY}/mcp", "https://other.example"]}, None),
        ({"resource": f"{GATEWAY}/mcp/"}, None),
        ({"resource": None}, None),
        ({"resource": 7}, None),
        ({"resource": []}, None),
        ({"resource": [7]}, None),
    ],
)
def test_the_signed_resource_claim_must_name_only_gateway_resources(
    claims: Mapping[str, object], expected: bool | None
) -> None:
    result: Final = resource_names_mcp(claims, GATEWAY)
    if isinstance(result, bool):
        assert expected is result
        return
    assert expected is None
    assert result.error == "invalid_target"


def test_the_signed_resource_claim_is_refused_without_a_gateway_base_url() -> None:
    result: Final = resource_names_mcp({"resource": f"{GATEWAY}/mcp"}, None)
    assert isinstance(result, SubjectTokenRefusal)
    assert result.error == "invalid_target"


@pytest.mark.asyncio
async def test_redeeming_for_the_mcp_resource_narrows_the_identity_and_refuses_llm_scope():
    """End to end through identity_from_id_jag: the same verified claims, with ``aud`` the MCP
    resource, come out ceilinged, and asking for LLM routes under that audience is a scope error."""
    narrowed = await _identity(_redemption(_assertion(aud=AUDIENCES[0])), _Verifier(), _Resolver())
    assert isinstance(narrowed, AssertionIdentity)
    assert (narrowed.scope, narrowed.allowed_routes) == ("litellm:mcp", ("mcp_inference_routes",))
    refused = await _identity(_redemption(_assertion(aud=AUDIENCES[0], scope="litellm:llm")), _Verifier(), _Resolver())
    assert isinstance(refused, SubjectTokenRefusal)
    assert refused.error == "invalid_scope"
    unscoped = jwt.encode(
        {key: value for key, value in _claims(aud=AUDIENCES[1]).items() if key != "scope"},
        "unused-by-the-fake-verifier-but-long-enough-for-hs256",
        algorithm="HS256",
        headers={"typ": ID_JAG_TYP},
    )
    full = await _identity(_redemption(unscoped), _Verifier(), _Resolver())
    assert isinstance(full, AssertionIdentity)
    assert (full.scope, full.allowed_routes) == (None, None)
    scoped = await _identity(_redemption(_assertion(aud=AUDIENCES[1], scope="litellm:llm")), _Verifier(), _Resolver())
    assert isinstance(scoped, AssertionIdentity)
    assert scoped.scope == "litellm:llm"


@pytest.mark.asyncio
async def test_redeeming_for_the_signed_resource_caps_the_identity_before_subject_lookup() -> None:
    mcp_claim: Final = _unscoped_assertion(aud=GATEWAY, resource=f"{GATEWAY}/mcp")
    narrowed: Final = await _identity(_redemption(mcp_claim), _Verifier(), _Resolver())
    assert isinstance(narrowed, AssertionIdentity)
    assert (narrowed.scope, narrowed.allowed_routes) == ("litellm:mcp", ("mcp_inference_routes",))

    llm_claim: Final = _assertion(aud=GATEWAY, resource=f"{GATEWAY}/mcp", scope="litellm:llm")
    refused_scope: Final = await _identity(_redemption(llm_claim), _Verifier(), _Resolver())
    assert isinstance(refused_scope, SubjectTokenRefusal)
    assert refused_scope.error == "invalid_scope"

    other_resource: Final = _unscoped_assertion(aud=GATEWAY, resource="https://other.example")
    resolver: Final = _Resolver()
    refused_resource: Final = await _identity(_redemption(other_resource), _Verifier(), resolver)
    assert isinstance(refused_resource, SubjectTokenRefusal)
    assert refused_resource.error == "invalid_target"
    assert resolver.calls == ()

    origin_resource: Final = _unscoped_assertion(aud=GATEWAY, resource=GATEWAY)
    full_resource: Final = await _identity(_redemption(origin_resource), _Verifier(), _Resolver())
    assert isinstance(full_resource, AssertionIdentity)
    assert (full_resource.scope, full_resource.allowed_routes) == (None, None)

    no_resource: Final = _unscoped_assertion(aud=GATEWAY)
    full_control: Final = await _identity(_redemption(no_resource), _Verifier(), _Resolver())
    assert isinstance(full_control, AssertionIdentity)
    assert (full_control.scope, full_control.allowed_routes) == (None, None)
