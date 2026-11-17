import hmac
from typing import Final

import jwt
from jwt.algorithms import Algorithm, HMACAlgorithm
from jwt.exceptions import InvalidKeyError

# SP 800-131A Rev. 2 Table 9: HMAC keys shorter than 112 bits are disallowed.
_MIN_HMAC_KEY_BYTES: Final = 14
_HASHES: Final = {"sha256": HMACAlgorithm.SHA256, "sha384": HMACAlgorithm.SHA384, "sha512": HMACAlgorithm.SHA512}


class _ProviderHMAC(HMACAlgorithm):
    """PyJWT's HS256/384/512 with the tag computed by hmac.digest, OpenSSL's one-shot HMAC, which the FIPS provider
    runs as its own MAC service. PyJWT's hmac.new assembles the HMAC outside the provider from the provider's hash."""

    def __init__(self, digest_name: str, replaced: Algorithm) -> None:
        super().__init__(_HASHES[digest_name])
        self._digest_name: Final = digest_name
        self.replaced: Final = replaced

    def prepare_key(self, key: str | bytes) -> bytes:
        key_bytes: Final = super().prepare_key(key)
        if len(key_bytes) < _MIN_HMAC_KEY_BYTES:
            raise InvalidKeyError("general_settings.fips_mode is on and needs an HMAC key of at least 112 bits")
        return key_bytes

    def sign(self, msg: bytes, key: bytes) -> bytes:
        return hmac.digest(key, msg, self._digest_name)


def _registered(name: str) -> Algorithm | None:
    try:
        return jwt.get_algorithm_by_name(name)
    except NotImplementedError:
        return None


def compute_jwt_hmac_in_fips_provider(enabled: bool) -> None:
    """Make every jwt.encode and jwt.decode in this process compute HS256, HS384 and HS512 with the FIPS provider's
    HMAC service, or put back the algorithms it replaced when `enabled` is False. PyJWT's registry is left alone
    when there is nothing to swap."""
    for digest_name in _HASHES:
        name = f"HS{digest_name[3:]}"
        current = _registered(name)
        if current is None or enabled == isinstance(current, _ProviderHMAC):
            continue
        jwt.unregister_algorithm(name)
        jwt.register_algorithm(
            name, current.replaced if isinstance(current, _ProviderHMAC) else _ProviderHMAC(digest_name, current)
        )
