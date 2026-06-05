import time

import pytest
from authlib.jose import JsonWebKey, jwt

from litellm.proxy.auth.v2.jwt_verifier import JWTVerificationError, verify

ISSUER = "https://idp.example"
AUDIENCE = "litellm"


@pytest.fixture(scope="module")
def signing():
    key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
    kid = key.thumbprint()
    public = JsonWebKey.import_key(key.as_pem(is_private=False)).as_dict()
    public["kid"] = kid
    key_set = JsonWebKey.import_key_set({"keys": [public]})
    return key, kid, key_set


def _token(signing, **overrides):
    key, kid, _ = signing
    claims = {
        "sub": "user-42",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 3600,
    }
    claims.update(overrides)
    # jwt.encode returns bytes; real tokens arrive as strings off the header.
    return jwt.encode(
        {"alg": "RS256", "kid": kid}, claims, key.as_pem(is_private=True)
    ).decode("utf-8")


def test_valid_token_returns_claims(signing):
    _, _, key_set = signing
    claims = verify(_token(signing, role="admin"), key_set, ISSUER, AUDIENCE)
    assert claims["sub"] == "user-42"
    assert claims["role"] == "admin"


def test_tampered_signature_is_rejected(signing):
    _, _, key_set = signing
    bad = _token(signing)[:-4] + "AAAA"
    with pytest.raises(JWTVerificationError):
        verify(bad, key_set, ISSUER, AUDIENCE)


def test_expired_token_is_rejected(signing):
    _, _, key_set = signing
    with pytest.raises(JWTVerificationError):
        verify(_token(signing, exp=int(time.time()) - 10), key_set, ISSUER, AUDIENCE)


def test_wrong_audience_is_rejected(signing):
    _, _, key_set = signing
    with pytest.raises(JWTVerificationError):
        verify(_token(signing, aud="someone-else"), key_set, ISSUER, AUDIENCE)


def test_wrong_issuer_is_rejected(signing):
    _, _, key_set = signing
    with pytest.raises(JWTVerificationError):
        verify(_token(signing, iss="https://evil.example"), key_set, ISSUER, AUDIENCE)


def test_token_signed_by_unknown_key_is_rejected(signing):
    _, _, key_set = signing
    other = JsonWebKey.generate_key("RSA", 2048, is_private=True)
    forged = jwt.encode(
        {"alg": "RS256", "kid": "unknown"},
        {"sub": "x", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 60},
        other.as_pem(is_private=True),
    ).decode("utf-8")
    with pytest.raises(JWTVerificationError):
        verify(forged, key_set, ISSUER, AUDIENCE)


def test_garbage_is_rejected(signing):
    _, _, key_set = signing
    with pytest.raises(JWTVerificationError):
        verify("not.a.jwt", key_set, ISSUER, AUDIENCE)
