import time

import pytest
from authlib.jose import JsonWebKey, jwt

from litellm.proxy.auth.v2.authn.jwt_verifier import JWTVerificationError, verify

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


def test_rs256_to_hs256_algorithm_confusion_is_rejected(signing):
    # The JWKS holds an RSA public key. An attacker forges an HS256 token using
    # that public key as the HMAC secret. Pinning to asymmetric algorithms must
    # refuse it; accepting it would be a full authentication bypass.
    import base64
    import hashlib
    import hmac

    key, kid, key_set = signing
    public_pem = key.as_pem(is_private=False)

    def b64(raw: bytes) -> bytes:
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    header = b64(b'{"alg":"HS256","kid":"%s"}' % kid.encode())
    payload = b64(
        b'{"sub":"attacker","iss":"%s","aud":"%s","exp":9999999999}'
        % (ISSUER.encode(), AUDIENCE.encode())
    )
    signing_input = header + b"." + payload
    signature = b64(hmac.new(public_pem, signing_input, hashlib.sha256).digest())
    forged = (signing_input + b"." + signature).decode("utf-8")

    with pytest.raises(JWTVerificationError):
        verify(forged, key_set, ISSUER, AUDIENCE)
