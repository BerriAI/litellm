import hmac
from typing import Final

import jwt
from jwt.algorithms import HMACAlgorithm
from jwt.exceptions import InvalidKeyError

# SP 800-131A Rev. 2 Table 9: HMAC keys shorter than 112 bits are disallowed.
_MIN_HMAC_KEY_BYTES: Final = 14
_HASHES: Final = {"sha256": HMACAlgorithm.SHA256, "sha384": HMACAlgorithm.SHA384, "sha512": HMACAlgorithm.SHA512}


class _ProviderHMAC(HMACAlgorithm):
    """PyJWT's HS256/384/512 with the tag computed by hmac.digest, OpenSSL's one-shot HMAC, which the FIPS provider
    runs as its own MAC service. PyJWT's hmac.new assembles the HMAC outside the provider from the provider's hash."""

    def __init__(self, digest_name: str) -> None:
        super().__init__(_HASHES[digest_name])
        self._digest_name: Final = digest_name

    def prepare_key(self, key: str | bytes) -> bytes:
        key_bytes: Final = super().prepare_key(key)
        if len(key_bytes) < _MIN_HMAC_KEY_BYTES:
            raise InvalidKeyError("general_settings.fips_mode is on and needs an HMAC key of at least 112 bits")
        return key_bytes

    def sign(self, msg: bytes, key: bytes) -> bytes:
        return hmac.digest(key, msg, self._digest_name)


def compute_jwt_hmac_in_fips_provider(enabled: bool) -> None:
    """Make every jwt.encode and jwt.decode in this process compute HS256, HS384 and HS512 with the FIPS provider's
    HMAC service, or with PyJWT's own HMAC again when `enabled` is False."""
    for digest_name, hash_alg in _HASHES.items():
        name = f"HS{digest_name[3:]}"
        jwt.unregister_algorithm(name)
        jwt.register_algorithm(name, _ProviderHMAC(digest_name) if enabled else HMACAlgorithm(hash_alg))
