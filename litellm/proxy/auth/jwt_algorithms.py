from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache
from types import MappingProxyType
from typing import Final, TypeVar

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.auth.jwt_algorithms import APPROVED_JWT_ALGORITHMS, LEGACY_JWT_ALGORITHMS

_KEY_TYPE_ALGORITHMS: Final = MappingProxyType(
    {
        "RSA": frozenset(APPROVED_JWT_ALGORITHMS[:6]),
        "EC": frozenset(APPROVED_JWT_ALGORITHMS[6:]),
        "OKP": frozenset(LEGACY_JWT_ALGORITHMS),
    }
)


def allowed_jwt_algorithms(fips_mode: bool) -> tuple[str, ...]:
    return APPROVED_JWT_ALGORITHMS if fips_mode else (*APPROVED_JWT_ALGORITHMS, *LEGACY_JWT_ALGORITHMS)


_JWTMappingT: Final = TypeVar("_JWTMappingT", bound=Mapping[str, object])


def jwks_keys_for(keys: Sequence[_JWTMappingT], algorithms: Collection[str]) -> tuple[_JWTMappingT, ...]:
    """Keep keys whose declared alg is allowed; keys without alg are kept only when their kty can sign with an
    allowed algorithm."""
    allowed: Final = frozenset(algorithms)
    usable_keys: list[_JWTMappingT] = []
    for key in keys:
        if _key_allowed(key, allowed):
            usable_keys.append(key)
        else:
            _warn_dropped_jwks_key(key)
    return tuple(usable_keys)


def _warn_dropped_jwks_key(key: Mapping[str, object]) -> None:
    """Surface why a JWKS key was filtered out, once per distinct key (the filter re-runs on every
    JWT-authenticated request). alg/kid/kty/use are public JWKS header fields, never key material."""
    if key.get("use") == "enc":
        return  # encryption keys (e.g. Keycloak's default RSA-OAEP) are dropped by design: nothing to migrate
    alg: Final = key.get("alg")
    _warn_dropped_jwks_key_once(
        alg if isinstance(alg, str) else "undeclared",
        str(key.get("kid")),
        str(key.get("kty")),
        str(key.get("use")),
    )


@lru_cache(maxsize=256)
def _warn_dropped_jwks_key_once(alg: str, kid: str, kty: str, use: str) -> None:
    verbose_proxy_logger.warning(
        "JWT Auth: dropped a JWKS key unusable for token verification: alg=%s, kid=%s, kty=%s, use=%s; "
        "allowed algorithms follow APPROVED_JWT_ALGORITHMS (EdDSA is legacy, allowed only outside FIPS mode) — "
        "if this key signs tokens, move the IdP signing key to an approved algorithm such as RS256",
        alg,
        kid,
        kty,
        use,
    )


def _key_allowed(key: Mapping[str, object], algorithms: frozenset[str]) -> bool:
    alg: Final = key.get("alg")
    if isinstance(alg, str):
        return alg in algorithms
    key_type: Final = key.get("kty")
    if not isinstance(key_type, str):
        return False
    return bool(_KEY_TYPE_ALGORITHMS.get(key_type, frozenset()) & algorithms)
