import json
import logging
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, Request
from prisma.engine.errors import BinaryNotFoundError
from prisma.errors import DataError

from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    JWTIssuerConfig,
    LiteLLM_JWTAuth,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
    ProxyException,
)
from litellm.proxy.auth.handle_jwt import JWKSUnreachableError, JWTHandler, jwks_unavailable_exception
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.gateway_dcr_flow import SubjectIdentity, SubjectTokenRefusal
from litellm.proxy.mcp_server.idp_token_exchange import (
    REJECTED_SUBJECT_TOKEN,
    SUBJECT_TOKEN_CHECK_FAULTED,
    SUBJECT_TOKEN_CHECK_UNAVAILABLE,
    TokenExchangePrerequisites,
    exchange_idp_subject_token,
    identity_from_subject_token,
    token_exchange_available,
)

IDP_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1MSJ9.idp-signature"
REQUEST_HEADERS = {"x-litellm-team-id": "team-b", "user-agent": "lite/0.1"}
EVERY_GATE_HOLDS = {
    "jwt_auth_enabled": True,
    "has_database": True,
    "licensed": True,
    "maps_jwts_to_virtual_keys": False,
}
JWKS_URL = "https://idp.example.com/.well-known/jwks.json"
JWKS_DOWN = jwks_unavailable_exception(JWKSUnreachableError(f"ConnectError fetching {JWKS_URL} after 3 attempts"))


def _authorized(user_id="u1", team_id="team-b"):
    return {
        "is_proxy_admin": False,
        "team_object": None,
        "user_object": None,
        "end_user_object": None,
        "org_object": None,
        "token": IDP_JWT,
        "team_id": team_id,
        "user_id": user_id,
        "user_email": None,
        "end_user_id": None,
        "org_id": None,
        "team_membership": None,
        "jwt_claims": {"sub": user_id},
        "agent_id": None,
    }


class _Authorizer:
    def __init__(self, result=None, raises=None):
        self.calls = []
        self.result = result if result is not None else _authorized()
        self.raises = raises

    async def __call__(self, subject_token, request_headers):
        self.calls.append((subject_token, dict(request_headers)))
        if self.raises is not None:
            raise self.raises
        return self.result


async def _identity(authorizer, subject_token=IDP_JWT, **unmet):
    return await identity_from_subject_token(
        subject_token,
        request_headers=REQUEST_HEADERS,
        prerequisites=TokenExchangePrerequisites(**{**EVERY_GATE_HOLDS, **unmet}),
        is_jwt=JWTHandler.is_jwt,
        authorize=authorizer,
    )


@pytest.mark.asyncio
async def test_a_jwt_the_proxy_accepts_names_its_user_and_team():
    """The subject token goes to the proxy's own JWT auth with the caller's headers (that is
    where the team header is read), and the identity it resolved is what gets minted."""
    authorizer = _Authorizer()
    assert await _identity(authorizer) == SubjectIdentity(user_id="u1", team_id="team-b")
    assert authorizer.calls == [(IDP_JWT, REQUEST_HEADERS)]


@pytest.mark.asyncio
async def test_a_jwt_that_resolves_no_team_names_a_teamless_identity():
    assert await _identity(_Authorizer(_authorized(team_id=None))) == SubjectIdentity(user_id="u1", team_id=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unmet, subject_token, error, mentions",
    [
        ({"jwt_auth_enabled": False}, IDP_JWT, "unsupported_grant_type", "JWT auth is not enabled"),
        ({"has_database": False}, IDP_JWT, "unsupported_grant_type", "no database"),
        ({"licensed": False}, IDP_JWT, "unsupported_grant_type", "enterprise"),
        ({"maps_jwts_to_virtual_keys": True}, IDP_JWT, "unsupported_grant_type", "virtual keys"),
        ({}, "sk-litellm-virtual-key", "invalid_request", "not a JWT"),
    ],
)
async def test_the_gates_user_api_key_auth_applies_refuse_before_any_verification(
    unmet, subject_token, error, mentions
):
    authorizer = _Authorizer()
    refusal = await _identity(authorizer, subject_token=subject_token, **unmet)
    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == error
    assert mentions in refusal.description
    assert authorizer.calls == []


@pytest.mark.parametrize(
    "unmet",
    [
        {},
        {"jwt_auth_enabled": False},
        {"has_database": False},
        {"licensed": False},
        {"maps_jwts_to_virtual_keys": True},
    ],
)
def test_the_grant_is_available_exactly_when_every_gate_holds(unmet):
    prerequisites = TokenExchangePrerequisites(**{**EVERY_GATE_HOLDS, **unmet})
    assert prerequisites.available is (unmet == {})
    assert (prerequisites.refusal() is None) is prerequisites.available


MAPPED_ISSUER = JWTIssuerConfig(
    issuer="https://idp.example.test", audience="litellm-gateway", virtual_key_claim_field="client_id"
)


def _running_jwt_handler(litellm_jwtauth):
    handler = JWTHandler()
    if litellm_jwtauth is not None:
        handler.update_environment(prisma_client=None, user_api_key_cache=DualCache(), litellm_jwtauth=litellm_jwtauth)
    return handler


@pytest.mark.parametrize(
    "general_settings, prisma_client, premium_user, litellm_jwtauth, expected",
    [
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(), True),
        ({"enable_jwt_auth": True}, object(), True, None, True),
        ({}, object(), True, LiteLLM_JWTAuth(), False),
        ({"enable_jwt_auth": True}, None, True, LiteLLM_JWTAuth(), False),
        ({"enable_jwt_auth": True}, object(), False, LiteLLM_JWTAuth(), False),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(virtual_key_claim_field="client_id"), False),
        ({"enable_jwt_auth": True}, object(), True, LiteLLM_JWTAuth(issuers=[MAPPED_ISSUER]), False),
    ],
)
def test_availability_is_read_from_the_running_proxy(
    monkeypatch, general_settings, prisma_client, premium_user, litellm_jwtauth, expected
):
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", general_settings)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", premium_user)
    monkeypatch.setattr("litellm.proxy.proxy_server.jwt_handler", _running_jwt_handler(litellm_jwtauth))
    assert token_exchange_available() is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, reason",
    [
        (HTTPException(status_code=403, detail="User not allowed to access this route"), "not allowed"),
        (ProxyException(message="Token expired", type="auth_error", param="token", code=401), "Token expired"),
        (Exception("Validation fails: signature verification failed"), "signature verification failed"),
        (Exception("Invalid JWT Submitted"), "Invalid JWT"),
        (Exception(f"Failed to fetch keys from {JWKS_URL}: 502 Bad Gateway from the IdP"), JWKS_URL),
        (ValueError("User doesn't exist in db. 'user_id'=u1. Got error - not found"), "not found"),
    ],
)
async def test_a_jwt_the_proxy_rejects_is_refused_with_the_reason_kept_in_the_log(raised, reason, caplog):
    """The endpoint is public, so the response never quotes JWT auth's wording (it can name
    the JWKS URL or relay the IdP's reply); the operator reads the reason in the proxy log."""
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
    refusal = await _identity(_Authorizer(raises=raised))
    assert refusal == SubjectTokenRefusal(error="invalid_request", description=REJECTED_SUBJECT_TOKEN)
    assert reason in caplog.text


@pytest.mark.asyncio
async def test_a_jwt_that_resolves_no_user_cannot_be_exchanged():
    refusal = await _identity(_Authorizer(_authorized(user_id=None)))
    assert refusal == SubjectTokenRefusal(
        error="invalid_request", description="subject_token names no user the gateway knows"
    )


def _user_lookup_wrapping_a_database_outage():
    p1001 = DataError(
        data={"user_facing_error": {"message": "Can't reach database server at `127.0.0.1`:`5432`", "meta": {}}}
    )
    try:
        raise p1001
    except DataError as outage:
        try:
            raise ValueError(f"User doesn't exist in db. 'user_id'=u1. Got error - {outage}")
        except ValueError as wrapped:
            return wrapped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, reason",
    [
        (JWKS_DOWN, JWKS_URL),
        (HTTPException(status_code=503, detail="the auth database is not reachable"), "not reachable"),
        (_user_lookup_wrapping_a_database_outage(), "Can't reach database server"),
    ],
)
async def test_an_idp_or_gateway_outage_is_reported_as_retryable_not_as_a_bad_token(raised, reason, caplog):
    caplog.set_level(logging.ERROR, logger="LiteLLM Proxy")
    refusal = await _identity(_Authorizer(raises=raised))
    assert refusal == SubjectTokenRefusal(error="temporarily_unavailable", description=SUBJECT_TOKEN_CHECK_UNAVAILABLE)
    assert reason in caplog.text


def _user_lookup_wrapping_a_fault_retrying_cannot_clear():
    try:
        raise BinaryNotFoundError("query engine binary not found")
    except BinaryNotFoundError as fault:
        try:
            raise ValueError(f"User doesn't exist in db. 'user_id'=u1. Got error - {fault}")
        except ValueError as wrapped:
            return wrapped


@pytest.mark.asyncio
async def test_a_database_fault_retrying_cannot_clear_is_not_reported_as_a_transient_outage(caplog):
    """The status stays 503 (the only OAuth error a client reads as the server's fault, and what
    the mint path answers to the same fault) but the wording must not tell the client to wait."""
    caplog.set_level(logging.ERROR, logger="LiteLLM Proxy")
    refusal = await _identity(_Authorizer(raises=_user_lookup_wrapping_a_fault_retrying_cannot_clear()))
    assert refusal == SubjectTokenRefusal(error="temporarily_unavailable", description=SUBJECT_TOKEN_CHECK_FAULTED)
    assert "retrying will not help" in refusal.description
    assert "faulted: " in caplog.text and "query engine binary not found" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [None, "delegating-user"])
async def test_agent_token_cannot_be_exchanged_for_a_user_identity(user_id: str | None) -> None:
    authorizer: Final = _Authorizer({**_authorized(user_id=user_id), "agent_id": "managed-agent"})
    result: Final = await _identity(authorizer)
    assert isinstance(result, SubjectTokenRefusal)
    assert result.error == "invalid_request"
    assert "direct JWT authentication" in result.description


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enforce_team", "team_id", "expected"),
    (
        (True, "exchange-team", SubjectIdentity(user_id="exchange-user", team_id="exchange-team")),
        (False, None, SubjectIdentity(user_id="exchange-user", team_id=None)),
        (True, None, SubjectTokenRefusal(error="invalid_request", description=REJECTED_SUBJECT_TOKEN)),
    ),
    ids=("native-team", "teamless-without-enforcement", "teamless-under-enforcement"),
)
async def test_subject_exchange_applies_team_admission_before_returning_an_identity(
    monkeypatch: pytest.MonkeyPatch,
    enforce_team: bool,
    team_id: str | None,
    expected: SubjectIdentity | SubjectTokenRefusal,
) -> None:
    cache: Final = UserApiKeyCache()
    database: Final = object()
    handler: Final = JWTHandler()
    handler.update_environment(
        prisma_client=database,
        user_api_key_cache=cache,
        litellm_jwtauth=LiteLLM_JWTAuth(
            user_id_jwt_field="sub",
            team_ids_jwt_field="groups",
            enforce_team_based_model_access=enforce_team,
            team_allowed_routes=["/token"],
        ),
    )
    signing_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_url: Final = "https://identity.invalid/exchange/jwks"
    cache.set_cache(
        f"litellm_jwt_auth_keys_{jwks_url}",
        [json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key()))],
    )
    cache.set_cache(
        "exchange-user",
        LiteLLM_UserTable(user_id="exchange-user", user_role="internal_user", teams=[team_id] if team_id else []),
        model_type=LiteLLM_UserTable,
    )
    if team_id is not None:
        cache.set_cache(
            f"team_id:{team_id}",
            LiteLLM_TeamTable(team_id=team_id, members_with_roles=[Member(user_id="exchange-user", role="user")]),
            model_type=LiteLLM_TeamTable,
        )
        cache.set_cache(
            f"team_membership:exchange-user:{team_id}",
            LiteLLM_TeamMembership(user_id="exchange-user", team_id=team_id),
            model_type=LiteLLM_TeamMembership,
        )
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", jwks_url)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.setattr(proxy_server, "jwt_handler", handler)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "general_settings", {"enable_jwt_auth": True})
    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "llm_model_list", None)
    token: Final = jwt.encode(
        {"sub": "exchange-user", "groups": [team_id] if team_id else []}, signing_key, algorithm="RS256"
    )
    request: Final = Request({"type": "http", "method": "POST", "path": "/token", "headers": []})
    assert await exchange_idp_subject_token(token, request) == expected
