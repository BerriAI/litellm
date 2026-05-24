import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm_provisioning_mcp.auth import (
    JWKSValidator,
    TokenValidationError,
    _extract_scopes,
)

ISSUER = "https://idp/"
AUDIENCE = "litellm-provisioning-mcp"


def _keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = priv.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return priv, pem


def _validator(public_key, *, required_scope=None):
    v = JWKSValidator(
        jwks_url="https://idp/jwks",
        issuer=ISSUER,
        audience=AUDIENCE,
        algorithms=("RS256",),
        required_scope=required_scope,
    )
    v._jwk_client = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key=public_key)
    )
    return v


def _token(signing_pem, **claims):
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
        "sub": "agent-1",
    }
    payload.update(claims)
    return jwt.encode(payload, signing_pem, algorithm="RS256")


def test_valid_token_returns_subject_and_scopes():
    priv, pem = _keypair()
    v = _validator(priv.public_key(), required_scope="litellm:provision")
    token = _token(pem, scope="litellm:provision other:scope", client_id="cli-42")

    result = v.validate(token)

    assert result.subject == "agent-1"
    assert result.client_id == "cli-42"
    assert "litellm:provision" in result.scopes


def test_missing_required_scope_is_rejected():
    priv, pem = _keypair()
    v = _validator(priv.public_key(), required_scope="litellm:provision")
    token = _token(pem, scope="some:other")

    with pytest.raises(TokenValidationError, match="missing required scope"):
        v.validate(token)


def test_expired_token_is_rejected():
    priv, pem = _keypair()
    v = _validator(priv.public_key())
    token = _token(pem, exp=int(time.time()) - 10)

    with pytest.raises(TokenValidationError):
        v.validate(token)


def test_wrong_audience_is_rejected():
    priv, pem = _keypair()
    v = _validator(priv.public_key())
    token = _token(pem, aud="someone-else")

    with pytest.raises(TokenValidationError):
        v.validate(token)


def test_wrong_issuer_is_rejected():
    priv, pem = _keypair()
    v = _validator(priv.public_key())
    token = _token(pem, iss="https://evil/")

    with pytest.raises(TokenValidationError):
        v.validate(token)


def test_bad_signature_is_rejected():
    priv, _ = _keypair()
    attacker_priv, attacker_pem = _keypair()
    # Validator trusts `priv`'s public key, but the token is signed by attacker.
    v = _validator(priv.public_key())
    token = _token(attacker_pem)

    with pytest.raises(TokenValidationError):
        v.validate(token)


def test_empty_token_is_rejected():
    priv, _ = _keypair()
    v = _validator(priv.public_key())
    with pytest.raises(TokenValidationError):
        v.validate("")


def test_extract_scopes_across_claim_shapes():
    assert _extract_scopes({"scope": "a b"}) == ["a", "b"]
    assert _extract_scopes({"scp": ["c", "d"]}) == ["c", "d"]
    assert _extract_scopes({"permissions": ["e"], "roles": ["f"]}) == ["e", "f"]
