import base64
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from litellm.llms.base_llm.auth.jwt_signing import (
    MISSING_SIGNING_DEPENDENCIES_MESSAGE,
    build_jwk,
    build_jwks,
    jwks_document_json,
    load_es256_private_key,
    rfc7638_thumbprint,
    sign_es256_jwt,
)

_FIXED_PRIVATE_VALUE: Final = 55090612345678901234567890123456789012345678901234567890123456
_OTHER_PRIVATE_VALUE: Final = 1


def fixed_private_key(value: int = _FIXED_PRIVATE_VALUE) -> ec.EllipticCurvePrivateKey:
    return ec.derive_private_key(value, ec.SECP256R1())


def pem_of(key: ec.EllipticCurvePrivateKey) -> str:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def independent_thumbprint(public_key: ec.EllipticCurvePublicKey) -> str:
    """Recomputes RFC 7638 by hand, deliberately not sharing a single line of code with
    ``jwt_signing.rfc7638_thumbprint`` -- a mutation that broke the real implementation must not
    also break this reference, or the two would trivially agree by sharing the bug."""
    numbers: Final = public_key.public_numbers()
    x: Final = base64.urlsafe_b64encode(numbers.x.to_bytes(32, "big")).rstrip(b"=").decode()
    y: Final = base64.urlsafe_b64encode(numbers.y.to_bytes(32, "big")).rstrip(b"=").decode()
    canonical: Final = f'{{"crv":"P-256","kty":"EC","x":"{x}","y":"{y}"}}'
    return base64.urlsafe_b64encode(hashlib.sha256(canonical.encode()).digest()).rstrip(b"=").decode()


class TestLoadEs256PrivateKey:
    def test_valid_ec_p256_pem_loads(self):
        key: Final = load_es256_private_key(pem_of(fixed_private_key()))

        assert isinstance(key, ec.EllipticCurvePrivateKey)
        assert isinstance(key.curve, ec.SECP256R1)

    def test_garbage_pem_is_rejected(self):
        with pytest.raises(ValueError, match="not a valid unencrypted PEM"):
            load_es256_private_key("not a pem")

    def test_rsa_key_is_rejected(self):
        rsa_pem: Final = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            .private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            .decode()
        )

        with pytest.raises(ValueError, match="P-256"):
            load_es256_private_key(rsa_pem)

    def test_non_p256_curve_is_rejected(self):
        secp384_pem: Final = (
            ec.generate_private_key(ec.SECP384R1())
            .private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            .decode()
        )

        with pytest.raises(ValueError, match="P-256"):
            load_es256_private_key(secp384_pem)

    def test_error_never_echoes_key_material(self):
        pem: Final = pem_of(fixed_private_key())

        with pytest.raises(ValueError, match="P-256"):
            load_es256_private_key(pem_of(ec.generate_private_key(ec.SECP384R1())))
        with pytest.raises(ValueError, match="not a valid unencrypted PEM") as exc_info:
            load_es256_private_key("garbage-not-a-pem")

        assert pem not in str(exc_info.value)


class TestRfc7638Thumbprint:
    def test_matches_independent_recomputation(self):
        public_key: Final = fixed_private_key().public_key()

        assert rfc7638_thumbprint(public_key) == independent_thumbprint(public_key)

    def test_different_keys_have_different_thumbprints(self):
        first: Final = fixed_private_key(_FIXED_PRIVATE_VALUE).public_key()
        second: Final = fixed_private_key(_OTHER_PRIVATE_VALUE).public_key()

        assert rfc7638_thumbprint(first) != rfc7638_thumbprint(second)

    def test_thumbprint_is_deterministic(self):
        public_key: Final = fixed_private_key().public_key()

        assert rfc7638_thumbprint(public_key) == rfc7638_thumbprint(public_key)


class TestBuildJwks:
    def test_jwks_contains_one_key_matching_the_thumbprint(self):
        public_key: Final = fixed_private_key().public_key()

        jwks: Final = build_jwks(public_key)

        assert len(jwks["keys"]) == 1
        assert jwks["keys"][0]["kid"] == rfc7638_thumbprint(public_key)
        assert jwks["keys"][0]["kty"] == "EC"
        assert jwks["keys"][0]["crv"] == "P-256"
        assert jwks["keys"][0]["alg"] == "ES256"

    def test_build_jwk_stamps_the_given_kid_verbatim(self):
        jwk: Final = build_jwk(fixed_private_key().public_key(), kid="caller-supplied-kid")

        assert jwk["kid"] == "caller-supplied-kid"

    def test_jwks_document_json_round_trips_through_build_jwks(self):
        key: Final = fixed_private_key()

        document: Final = json.loads(jwks_document_json(pem_of(key)))
        jwks: Final = build_jwks(key.public_key())

        assert document == {"keys": [dict(jwk) for jwk in jwks["keys"]]}


class TestSignEs256Jwt:
    def test_minted_token_verifies_against_the_matching_public_key(self):
        key: Final = fixed_private_key()
        now: Final = int(time.time())
        claims: Final = {"sub": "workload-a", "iss": "https://issuer.example", "iat": now, "exp": now + 300}

        token: Final = sign_es256_jwt(pem_of(key), claims)
        decoded: Final = jwt.decode(token, key.public_key(), algorithms=["ES256"])

        assert decoded == claims

    def test_header_alg_is_es256(self):
        token: Final = sign_es256_jwt(pem_of(fixed_private_key()), {"sub": "x"})

        assert jwt.get_unverified_header(token)["alg"] == "ES256"

    def test_header_kid_matches_the_published_jwks(self):
        key: Final = fixed_private_key()

        token: Final = sign_es256_jwt(pem_of(key), {"sub": "x"})

        header_kid: Final = jwt.get_unverified_header(token)["kid"]
        published_kid: Final = build_jwks(key.public_key())["keys"][0]["kid"]
        assert header_kid == published_kid == rfc7638_thumbprint(key.public_key())

    def test_wrong_key_fails_verification(self):
        signing_key: Final = fixed_private_key(_FIXED_PRIVATE_VALUE)
        other_key: Final = fixed_private_key(_OTHER_PRIVATE_VALUE)

        token: Final = sign_es256_jwt(pem_of(signing_key), {"sub": "x"})

        with pytest.raises(jwt.exceptions.InvalidSignatureError):
            jwt.decode(token, other_key.public_key(), algorithms=["ES256"])


class TestBaseSdkImport:
    """A base ``pip install litellm`` has neither PyJWT nor cryptography (both are proxy extras),
    and ``litellm/__init__`` reaches this module through the Anthropic provider, so a
    module-level import of either would break ``import litellm`` for every base SDK user."""

    def test_module_imports_with_pyjwt_and_cryptography_absent(self):
        script: Final = textwrap.dedent(
            """
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in {"jwt", "cryptography"}:
                        raise ModuleNotFoundError(f"No module named {name!r}")

            sys.meta_path.insert(0, Blocker())
            import litellm
            from litellm.llms.base_llm.auth.jwt_signing import jwks_document_json
            try:
                jwks_document_json("not a key")
            except ImportError as e:
                print(e)
            """
        )
        result: Final = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip() == MISSING_SIGNING_DEPENDENCIES_MESSAGE

    def test_signing_reports_the_missing_extra(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "jwt", None)
        with pytest.raises(ImportError, match="litellm\\[proxy\\]"):
            sign_es256_jwt(pem_of(fixed_private_key()), {"sub": "x"})

