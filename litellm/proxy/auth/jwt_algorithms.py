from collections.abc import Collection, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

ApprovedJwtAlgorithm = Literal["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]

APPROVED_JWT_ALGORITHMS: Final[tuple[ApprovedJwtAlgorithm, ...]] = (
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
)

LEGACY_JWT_ALGORITHMS: Final = ("EdDSA",)

_KEY_TYPE_ALGORITHMS: Final = MappingProxyType(
    {
        "RSA": frozenset(APPROVED_JWT_ALGORITHMS[:6]),
        "EC": frozenset(APPROVED_JWT_ALGORITHMS[6:]),
        "OKP": frozenset(LEGACY_JWT_ALGORITHMS),
    }
)


def allowed_jwt_algorithms(fips_mode: bool) -> tuple[str, ...]:
    return APPROVED_JWT_ALGORITHMS if fips_mode else (*APPROVED_JWT_ALGORITHMS, *LEGACY_JWT_ALGORITHMS)


def jwks_keys_for(
    keys: Sequence[Mapping[str, object]], algorithms: Collection[str]
) -> tuple[Mapping[str, object], ...]:
    """Keep keys whose declared alg is allowed; keys without alg are kept only when their kty can sign with an allowed algorithm."""
    return tuple(key for key in keys if _key_allowed(key, frozenset(algorithms)))


def _key_allowed(key: Mapping[str, object], algorithms: frozenset[str]) -> bool:
    alg: Final = key.get("alg")
    if isinstance(alg, str):
        return alg in algorithms
    key_type: Final = key.get("kty")
    if not isinstance(key_type, str):
        return False
    return bool(_KEY_TYPE_ALGORITHMS.get(key_type, frozenset()) & algorithms)
