"""ES256 JWT signing primitives for Anthropic workload identity federation's
``internal_issuer`` identity source (see ``identity_source.InternalIssuerSource``).

Pure functions over an already-resolved PEM string: no I/O, no secret-manager awareness, no
caching. Given the signing key at, say, $ISSUER_SIGNING_KEY_PEM, an operator publishes the
JWKS document Anthropic's inline federation issuer needs with one line:

    python -c "from litellm.llms.base_llm.auth.jwt_signing import jwks_document_json; \\
        import os; print(jwks_document_json(os.environ['ISSUER_SIGNING_KEY_PEM']))"
"""

import base64
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric import ec

ALG: Final = "ES256"
MISSING_SIGNING_DEPENDENCIES_MESSAGE: Final = (
    "the internal_issuer identity source needs PyJWT and cryptography, which a base litellm install "
    "does not include: pip install 'litellm[proxy]'"
)
_JWK_CURVE_NAME: Final = "P-256"
_JWK_KEY_TYPE: Final = "EC"
_COORDINATE_BYTE_LENGTH: Final = 32  # P-256 field element width, RFC 7518 6.2.1.2/6.2.1.3

Jwk: TypeAlias = Mapping[str, str]
Jwks: TypeAlias = Mapping[str, tuple[Jwk, ...]]


def load_es256_private_key(pem: str) -> "ec.EllipticCurvePrivateKey":
    """Parses an unencrypted PEM EC private key. Never echoes the key material in an error."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as e:
        raise ImportError(MISSING_SIGNING_DEPENDENCIES_MESSAGE) from e
    try:
        key: Final = load_pem_private_key(pem.encode(), password=None)
    except (ValueError, TypeError) as e:
        raise ValueError("internal_issuer signing key is not a valid unencrypted PEM private key") from e
    if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
        raise ValueError(  # noqa: TRY004  # the reader classifies ValueError into a readable config error; TypeError would not
            "internal_issuer signing key must be an EC P-256 (secp256r1) private key for ES256"
        )
    return key


def _b64url_coordinate(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes(_COORDINATE_BYTE_LENGTH, "big")).rstrip(b"=").decode("ascii")


def _jwk_thumbprint_members(public_key: "ec.EllipticCurvePublicKey") -> Jwk:
    """RFC 7638 3.2's exact EC member set (crv, kty, x, y) and nothing else: an extra member
    here would change the thumbprint and desync it from the ``kid`` published in the JWKS."""
    numbers: Final = public_key.public_numbers()
    return MappingProxyType(
        {
            "crv": _JWK_CURVE_NAME,
            "kty": _JWK_KEY_TYPE,
            "x": _b64url_coordinate(numbers.x),
            "y": _b64url_coordinate(numbers.y),
        }
    )


def rfc7638_thumbprint(public_key: "ec.EllipticCurvePublicKey") -> str:
    """RFC 7638: SHA-256 over the lexicographically member-ordered, whitespace-free JSON
    rendering of the thumbprint members, base64url-encoded without padding."""
    canonical: Final = json.dumps(
        dict(sorted(_jwk_thumbprint_members(public_key).items())),  # mutable-ok: json.dumps needs a real dict
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(hashlib.sha256(canonical.encode()).digest()).rstrip(b"=").decode("ascii")


def build_jwk(public_key: "ec.EllipticCurvePublicKey", kid: str) -> Jwk:
    return MappingProxyType({**_jwk_thumbprint_members(public_key), "use": "sig", "alg": ALG, "kid": kid})


def build_jwks(public_key: "ec.EllipticCurvePublicKey") -> Jwks:
    kid: Final = rfc7638_thumbprint(public_key)
    return MappingProxyType({"keys": (build_jwk(public_key, kid),)})


def jwks_document_json(pem: str) -> str:
    """The operator-facing export: the JSON document to register as Anthropic's inline JWKS.

    ``build_jwks`` returns ``MappingProxyType``/tuple values per this repo's no-mutation
    convention; the ``json`` module only knows plain ``dict``/``list``, so those are converted
    at this one serialization boundary rather than giving up immutability throughout the module.
    """
    key: Final = load_es256_private_key(pem)
    jwks: Final = build_jwks(key.public_key())
    return json.dumps(
        {"keys": [dict(jwk) for jwk in jwks["keys"]]},  # mutable-ok: json.dumps needs real dicts/lists
        indent=2,
    )


def sign_es256_jwt(pem: str, claims: Mapping[str, object]) -> str:
    """Signs ``claims`` with the PEM key, stamping ``kid`` as its RFC 7638 thumbprint so a
    verifier can look the signing key up in the published JWKS by ``kid`` alone."""
    try:
        import jwt
    except ImportError as e:
        raise ImportError(MISSING_SIGNING_DEPENDENCIES_MESSAGE) from e
    key: Final = load_es256_private_key(pem)
    kid: Final = rfc7638_thumbprint(key.public_key())
    headers: Final = {"kid": kid}  # mutable-ok: PyJWT requires a real dict, not a Mapping
    return jwt.encode(dict(claims), key, algorithm=ALG, headers=headers)  # mutable-ok: PyJWT requires a real dict
