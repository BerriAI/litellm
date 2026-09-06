import json
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from litellm.llms.base_llm.auth.identity_source import InternalIssuerSource
from litellm.llms.base_llm.auth.internal_issuer import (
    internal_issuer_assertion_source,
    internal_issuer_jwks_document,
    mint_internal_issuer_assertion,
)
from litellm.llms.base_llm.auth.jwt_signing import build_jwks, rfc7638_thumbprint

SIGNING_KEY_REF: Final = "oidc/env/ISSUER_SIGNING_KEY_PEM"
ISSUER_URL: Final = "https://issuer.internal.example"
SUBJECT: Final = "workload-a"


_PRIVATE_VALUE: Final = 90123456789012345678901234567890123456789012345678901234567890


def signing_key() -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(_PRIVATE_VALUE, ec.SECP256R1())


def pem_of(key: ec.EllipticCurvePrivateKey) -> str:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def make_config(
    issuer_url: str = ISSUER_URL,
    subject: str = SUBJECT,
    audience: str | None = None,
    ttl_seconds: int = 300,
    signing_key_ref: str = SIGNING_KEY_REF,
) -> InternalIssuerSource:
    return InternalIssuerSource(
        issuer_url=issuer_url,
        subject=subject,
        audience=audience,
        ttl_seconds=ttl_seconds,
        signing_key_ref=signing_key_ref,
    )


def key_reader_returning(pem: str | None):
    def reader(ref: str) -> str | None:
        assert ref == SIGNING_KEY_REF
        return pem

    return reader


class FakeClock:
    def __init__(self, value: float) -> None:
        self._value: Final = value

    def __call__(self) -> float:
        return self._value


def decode_ignoring_wall_clock(token: str, public_key: ec.EllipticCurvePublicKey) -> dict:
    """Tests mint with a fixed past ``FakeClock`` and no expected audience, so PyJWT's
    real-wall-clock ``exp``/``aud`` checks (irrelevant to what these tests verify) are disabled."""
    return jwt.decode(token, public_key, algorithms=["ES256"], options={"verify_exp": False, "verify_aud": False})


class TestMintInternalIssuerAssertion:
    def test_required_claims_and_asymmetric_alg(self):
        key: Final = signing_key()
        config: Final = make_config(ttl_seconds=300)

        token: Final = mint_internal_issuer_assertion(
            config, key_reader=key_reader_returning(pem_of(key)), clock=FakeClock(1_700_000_000.0)
        )
        header: Final = jwt.get_unverified_header(token)
        claims: Final = decode_ignoring_wall_clock(token, key.public_key())

        assert header["alg"] == "ES256"
        assert claims["sub"] == SUBJECT
        assert claims["iss"] == ISSUER_URL
        assert claims["iat"] == 1_700_000_000
        assert claims["exp"] == 1_700_000_300

    def test_kid_matches_the_published_jwks(self):
        key: Final = signing_key()
        config: Final = make_config()

        token: Final = mint_internal_issuer_assertion(
            config, key_reader=key_reader_returning(pem_of(key)), clock=FakeClock(1_700_000_000.0)
        )

        header_kid: Final = jwt.get_unverified_header(token)["kid"]
        published_kid: Final = build_jwks(key.public_key())["keys"][0]["kid"]
        assert header_kid == published_kid == rfc7638_thumbprint(key.public_key())

    def test_ttl_bounds_exp_minus_iat(self):
        key: Final = signing_key()
        config: Final = make_config(ttl_seconds=120)

        token: Final = mint_internal_issuer_assertion(
            config, key_reader=key_reader_returning(pem_of(key)), clock=FakeClock(1_700_000_000.0)
        )
        claims: Final = decode_ignoring_wall_clock(token, key.public_key())

        assert claims["exp"] - claims["iat"] == 120

    def test_audience_included_only_when_set(self):
        key: Final = signing_key()
        without_audience: Final = mint_internal_issuer_assertion(
            make_config(audience=None), key_reader=key_reader_returning(pem_of(key)), clock=FakeClock(1_700_000_000.0)
        )
        with_audience: Final = mint_internal_issuer_assertion(
            make_config(audience="urn:anthropic:federation"),
            key_reader=key_reader_returning(pem_of(key)),
            clock=FakeClock(1_700_000_000.0),
        )

        claims_without: Final = decode_ignoring_wall_clock(without_audience, key.public_key())
        claims_with: Final = decode_ignoring_wall_clock(with_audience, key.public_key())
        assert "aud" not in claims_without
        assert claims_with["aud"] == "urn:anthropic:federation"

    def test_jti_is_present_and_fresh_on_every_mint(self):
        key: Final = signing_key()
        config: Final = make_config()
        reader: Final = key_reader_returning(pem_of(key))

        first: Final = decode_ignoring_wall_clock(
            mint_internal_issuer_assertion(config, key_reader=reader, clock=FakeClock(1_700_000_000.0)),
            key.public_key(),
        )
        second: Final = decode_ignoring_wall_clock(
            mint_internal_issuer_assertion(config, key_reader=reader, clock=FakeClock(1_700_000_000.0)),
            key.public_key(),
        )

        assert first["jti"] and second["jti"]
        assert first["jti"] != second["jti"]

    def test_missing_signing_key_raises_value_error_naming_the_ref_not_a_secret(self):
        with pytest.raises(ValueError, match=SIGNING_KEY_REF):
            mint_internal_issuer_assertion(make_config(), key_reader=key_reader_returning(None))

    def test_malformed_signing_key_raises_value_error(self):
        with pytest.raises(ValueError, match="not a valid unencrypted PEM"):
            mint_internal_issuer_assertion(make_config(), key_reader=key_reader_returning("not-a-pem"))


class TestInternalIssuerAssertionSource:
    def test_returns_a_callable_that_mints_fresh_each_call(self):
        key: Final = signing_key()
        source: Final = internal_issuer_assertion_source(make_config(), key_reader=key_reader_returning(pem_of(key)))

        first: Final = jwt.decode(source(), key.public_key(), algorithms=["ES256"])
        second: Final = jwt.decode(source(), key.public_key(), algorithms=["ES256"])

        assert first["jti"] != second["jti"]

    def test_propagates_the_underlying_mint_failure(self):
        source: Final = internal_issuer_assertion_source(make_config(), key_reader=key_reader_returning(None))

        with pytest.raises(ValueError, match=SIGNING_KEY_REF):
            source()


class TestInternalIssuerJwksDocument:
    def test_matches_the_key_used_to_mint(self):
        key: Final = signing_key()
        config: Final = make_config()
        reader: Final = key_reader_returning(pem_of(key))

        document: Final = json.loads(internal_issuer_jwks_document(config, key_reader=reader))
        token: Final = mint_internal_issuer_assertion(config, key_reader=reader, clock=FakeClock(1_700_000_000.0))

        assert document["keys"][0]["kid"] == jwt.get_unverified_header(token)["kid"]
        assert decode_ignoring_wall_clock(token, key.public_key())

    def test_missing_signing_key_raises_value_error(self):
        with pytest.raises(ValueError, match=SIGNING_KEY_REF):
            internal_issuer_jwks_document(make_config(), key_reader=key_reader_returning(None))
