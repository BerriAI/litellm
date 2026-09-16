import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import SubjectIdentity, SubjectTokenRefusal
from litellm.proxy._experimental.mcp_server.idp_token_exchange import identity_from_subject_token
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.handle_jwt import JWTHandler

IDP_JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1MSJ9.idp-signature"
REQUEST_HEADERS = {"x-litellm-team-id": "team-b", "user-agent": "lite/0.1"}


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


async def _identity(authorizer, subject_token=IDP_JWT, **overrides):
    arguments = {
        "request_headers": REQUEST_HEADERS,
        "jwt_auth_enabled": True,
        "has_database": True,
        "licensed": True,
        "is_jwt": JWTHandler.is_jwt,
        "authorize": authorizer,
    }
    return await identity_from_subject_token(subject_token, **{**arguments, **overrides})


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
    "overrides, subject_token, error, mentions",
    [
        ({"jwt_auth_enabled": False}, IDP_JWT, "unsupported_grant_type", "JWT auth is not enabled"),
        ({"has_database": False}, IDP_JWT, "unsupported_grant_type", "no database"),
        ({"licensed": False}, IDP_JWT, "unsupported_grant_type", "enterprise"),
        ({}, "sk-litellm-virtual-key", "invalid_request", "not a JWT"),
    ],
)
async def test_the_gates_user_api_key_auth_applies_refuse_before_any_verification(
    overrides, subject_token, error, mentions
):
    authorizer = _Authorizer()
    refusal = await _identity(authorizer, subject_token=subject_token, **overrides)
    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == error
    assert mentions in refusal.description
    assert authorizer.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, mentions",
    [
        (HTTPException(status_code=403, detail="User not allowed to access this route"), "not allowed"),
        (ProxyException(message="Token expired", type="auth_error", param="token", code=401), "Token expired"),
        (Exception("Validation fails: signature verification failed"), "signature verification failed"),
        (Exception("Invalid JWT Submitted"), "Invalid JWT"),
    ],
)
async def test_a_jwt_the_proxy_rejects_is_an_invalid_subject_token(raised, mentions):
    refusal = await _identity(_Authorizer(raises=raised))
    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "invalid_request"
    assert refusal.description.startswith("subject_token was rejected: ")
    assert mentions in refusal.description


@pytest.mark.asyncio
async def test_a_jwt_that_resolves_no_user_cannot_be_exchanged():
    refusal = await _identity(_Authorizer(_authorized(user_id=None)))
    assert refusal == SubjectTokenRefusal(
        error="invalid_request", description="subject_token names no user the gateway knows"
    )
