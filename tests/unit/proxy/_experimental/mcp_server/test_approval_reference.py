import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from litellm.constants import MCP_APPROVAL_REFERENCE_HEADER
from litellm.proxy._experimental.mcp_server.approval_reference import (
    ApprovalCheck,
    ApprovalRejected,
    ApprovalVerifierUnavailable,
    ApprovalVerified,
    JwksFetcher,
    verify_approval_reference,
)
from litellm.types.mcp_server.mcp_server_manager import MCPApprovalPolicy, parse_approval_policy

ISSUER = "https://approvals.example.com"
JWKS_URL = "https://approvals.example.com/.well-known/jwks.json"
AUDIENCE = "mcp-gateway"
SERVER_ID = "srv-123"
TOOL = "delete_records"

_now = int(time.time())

_signing_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks_for(*key_kid_pairs: tuple[rsa.RSAPrivateKey, str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key, kid in key_kid_pairs:
        doc: dict[str, object] = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        doc["kid"] = kid
        doc["use"] = "sig"
        doc["alg"] = "RS256"
        entries.append(doc)
    return entries


JWKS = _jwks_for((_signing_key, "kid-1"), (_other_key, "kid-2"))


async def _fetch_jwks(_url: str) -> list[dict[str, object]]:
    return JWKS


async def _failing_fetch(_url: str) -> list[dict[str, object]]:
    raise ConnectionError("jwks endpoint unreachable")


def _policy(*, audience: str | None = AUDIENCE) -> MCPApprovalPolicy:
    return MCPApprovalPolicy(
        tools=(TOOL,),
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        audience=audience,
    )


def _token(
    key: rsa.RSAPrivateKey | bytes | None = _signing_key,
    kid: str = "kid-1",
    alg: str = "RS256",
    **claims: object,
) -> str:
    payload = {
        "iss": ISSUER,
        "exp": _now + 600,
        "jti": "jti-1",
        "mcp_server": SERVER_ID,
        "mcp_tool": TOOL,
        "aud": AUDIENCE,
        "sub": "agent-7",
    }
    payload.update(claims)
    return jwt.encode(payload, key, algorithm=alg, headers={"kid": kid})


def _headers(token: str | None) -> dict[str, str]:
    return {MCP_APPROVAL_REFERENCE_HEADER: token} if token is not None else {}


async def _call(
    policy: MCPApprovalPolicy,
    token: str | None = _token(),
    fetch_jwks: JwksFetcher = _fetch_jwks,
    server_id: str = SERVER_ID,
) -> ApprovalCheck:
    return await verify_approval_reference(
        policy=policy,
        policy_tool=TOOL,
        server_id=server_id,
        raw_headers=_headers(token),
        fetch_jwks=fetch_jwks,
    )


@pytest.mark.asyncio
async def test_missing_header_is_rejected():
    result = await verify_approval_reference(
        policy=_policy(),
        policy_tool=TOOL,
        server_id=SERVER_ID,
        raw_headers=None,
        fetch_jwks=_fetch_jwks,
    )
    assert result == ApprovalRejected(reason="missing", message=result.message)
    assert result.reason == "missing"


@pytest.mark.asyncio
async def test_garbage_token_is_invalid():
    result = await _call(_policy(), token="not-a-jwt")
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_wrong_signing_key_is_invalid():
    attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    result = await _call(_policy(), token=_token(key=attacker_key, kid="kid-1"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_expired_token_is_rejected_expired():
    result = await _call(_policy(), token=_token(exp=_now - 10))
    assert result == ApprovalRejected(reason="expired", message=result.message)
    assert result.reason == "expired"


@pytest.mark.asyncio
async def test_wrong_issuer_is_invalid():
    result = await _call(_policy(), token=_token(iss="https://evil.example.com"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_wrong_audience_is_invalid_when_configured():
    result = await _call(_policy(audience=AUDIENCE), token=_token(aud="other-service"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_audience_absent_or_any_accepted_when_policy_audience_none():
    no_aud_token = _token()
    claims = jwt.decode(no_aud_token, options={"verify_signature": False})
    claims.pop("aud")
    token_without_aud = jwt.encode(claims, _signing_key, algorithm="RS256", headers={"kid": "kid-1"})
    result = await _call(_policy(audience=None), token=token_without_aud)
    assert isinstance(result, ApprovalVerified)

    result_with_aud = await _call(_policy(audience=None), token=_token(aud="anything"))
    assert isinstance(result_with_aud, ApprovalVerified)


@pytest.mark.asyncio
async def test_tool_binding_mismatch_is_invalid():
    result = await _call(_policy(), token=_token(mcp_tool="read_records"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_server_binding_mismatch_is_invalid():
    result = await _call(_policy(), token=_token(mcp_server="other-server"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_server_binding_rejects_alias_and_server_name():
    result_alias = await _call(_policy(), token=_token(mcp_server="server-alias"))
    assert isinstance(result_alias, ApprovalRejected)
    assert result_alias.reason == "invalid"

    result_name = await _call(_policy(), token=_token(mcp_server="records-server"))
    assert isinstance(result_name, ApprovalRejected)
    assert result_name.reason == "invalid"

    result_id = await _call(_policy(), token=_token(mcp_server=SERVER_ID))
    assert isinstance(result_id, ApprovalVerified)


@pytest.mark.asyncio
async def test_empty_server_id_fails_closed():
    result = await _call(_policy(), token=_token(mcp_server=""), server_id="")
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_alg_none_is_rejected():
    claims = jwt.decode(_token(), options={"verify_signature": False})
    unsigned = jwt.encode(claims, None, algorithm="none")
    result = await _call(_policy(), token=unsigned)
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_hs256_is_rejected():
    result = await _call(_policy(), token=_token(key=b"shared-secret", kid="kid-1", alg="HS256"))
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_missing_jti_is_invalid():
    claims = jwt.decode(_token(), options={"verify_signature": False})
    claims.pop("jti")
    token = jwt.encode(claims, _signing_key, algorithm="RS256", headers={"kid": "kid-1"})
    result = await _call(_policy(), token=token)
    assert isinstance(result, ApprovalRejected)
    assert result.reason == "invalid"


@pytest.mark.asyncio
async def test_jwks_fetch_failure_is_unavailable():
    result = await _call(_policy(), fetch_jwks=_failing_fetch)
    assert isinstance(result, ApprovalVerifierUnavailable)


@pytest.mark.asyncio
async def test_jwks_garbage_is_unavailable():
    async def _garbage_fetch(_url: str):
        return [{"kid": "kid-1", "kty": "RSA"}]

    result = await _call(_policy(), fetch_jwks=_garbage_fetch)
    assert isinstance(result, ApprovalVerifierUnavailable)


@pytest.mark.asyncio
async def test_valid_reference_verifies_with_exact_record():
    result = await _call(_policy())
    assert isinstance(result, ApprovalVerified)
    assert result.record == {
        "jti": "jti-1",
        "issuer": ISSUER,
        "subject": "agent-7",
        "expires_at": _now + 600,
    }


@pytest.mark.asyncio
async def test_record_carries_no_token_material():
    token = _token()
    result = await _call(_policy(), token=token)
    assert isinstance(result, ApprovalVerified)
    assert token not in repr(result)


@pytest.mark.asyncio
async def test_unavailable_message_does_not_leak_fetch_error_details():
    async def _leaking_fetch(_url: str):
        raise RuntimeError("secret-internal-host:9999")

    result = await _call(_policy(), fetch_jwks=_leaking_fetch)
    assert isinstance(result, ApprovalVerifierUnavailable)
    assert "secret-internal-host:9999" not in result.message


def test_policy_rejects_whitespace_only_tool_name():
    with pytest.raises(ValidationError):
        MCPApprovalPolicy(tools=(" ",), issuer=ISSUER, jwks_url=JWKS_URL)


def test_policy_rejects_empty_audience():
    with pytest.raises(ValidationError):
        MCPApprovalPolicy(tools=(TOOL,), issuer=ISSUER, jwks_url=JWKS_URL, audience="")


def test_policy_strips_surrounding_whitespace():
    policy = MCPApprovalPolicy(
        tools=(f" {TOOL} ",),
        issuer=f"  {ISSUER}  ",
        jwks_url=f" {JWKS_URL} ",
    )
    assert policy.tools == (TOOL,)
    assert policy.issuer == ISSUER
    assert policy.jwks_url == JWKS_URL


def test_parse_approval_policy_rejects_falsy_but_invalid_json():
    for stored in ("{}", "[]", "false"):
        with pytest.raises(ValidationError):
            parse_approval_policy(stored)


def test_parse_approval_policy_none_and_valid():
    assert parse_approval_policy(None) is None
    parsed = parse_approval_policy(
        json.dumps(
            {
                "tools": [TOOL],
                "issuer": ISSUER,
                "jwks_url": JWKS_URL,
            }
        )
    )
    assert parsed == MCPApprovalPolicy(tools=(TOOL,), issuer=ISSUER, jwks_url=JWKS_URL)


@pytest.mark.parametrize(
    "url",
    [
        "https://approvals.example.com/jwks.json",
        "http://127.0.0.1:8092/.well-known/jwks.json",
        "http://localhost/jwks.json",
        "http://[::1]:8092/jwks.json",
    ],
)
def test_jwks_url_accepts_https_or_loopback(url):
    policy = MCPApprovalPolicy(tools=(TOOL,), issuer=ISSUER, jwks_url=url)
    assert policy.jwks_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/jwks.json",
        "http://127.0.0.1.evil.com/jwks.json",
        "http://localhost.evil.com/jwks.json",
        "ftp://approvals.example.com/jwks.json",
        "approvals.example.com/jwks.json",
    ],
)
def test_jwks_url_rejects_plaintext_remote_and_missing_scheme(url):
    with pytest.raises(ValidationError):
        MCPApprovalPolicy(tools=(TOOL,), issuer=ISSUER, jwks_url=url)
