import hmac
from functools import cache
from typing import Final

from jwt import PyJWT
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


class _FipsJWT(PyJWT):
    def __init__(self) -> None:
        super().__init__()
        for bits in ("256", "384", "512"):
            self._jws.unregister_algorithm(f"HS{bits}")
            self._jws.register_algorithm(f"HS{bits}", _ProviderHMAC(f"sha{bits}"))


@cache
def fips_jwt() -> PyJWT:
    """A PyJWT whose HS256, HS384 and HS512 are computed by the FIPS provider. Every other algorithm is PyJWT's own."""
    return _FipsJWT()
