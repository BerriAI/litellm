import hmac
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat
from jwt.algorithms import Algorithm, HMACAlgorithm, RSAAlgorithm, get_default_algorithms
from jwt.exceptions import InvalidKeyError

from litellm.proxy.common_utils.fips_signature import sign_message, verify_message

# SP 800-131A Rev. 2 Table 9: HMAC keys shorter than 112 bits are disallowed.
_MIN_HMAC_KEY_BYTES: Final = 14
_HASHES: Final = {"sha256": HMACAlgorithm.SHA256, "sha384": HMACAlgorithm.SHA384, "sha512": HMACAlgorithm.SHA512}
_RSA_HASHES: Final = {"sha256": RSAAlgorithm.SHA256, "sha384": RSAAlgorithm.SHA384, "sha512": RSAAlgorithm.SHA512}


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


class _ProviderRSA(RSAAlgorithm):
    """PyJWT's RS256/384/512 and PS256/384/512 with the message hashed and signed, or verified, in one call to the
    FIPS provider. The cryptography library hashes first and hands the provider a digest, which the provider's
    Security Policy counts as a non-approved signature on a pre-hashed message."""

    def __init__(self, digest_name: str, *, pss: bool) -> None:
        super().__init__(_RSA_HASHES[digest_name])
        self._digest_name: Final = digest_name
        self._pss: Final = pss

    def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
        private_key_der: Final = key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        return sign_message(private_key_der, msg, self._digest_name, pss=self._pss)

    def verify(self, msg: bytes, key: RSAPublicKey, sig: bytes) -> bool:
        public_key_der: Final = key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        return verify_message(public_key_der, msg, sig, self._digest_name, pss=self._pss)


_PROVIDER_ALGORITHMS: Final[Mapping[str, Algorithm]] = MappingProxyType(
    {
        **{f"HS{name[3:]}": _ProviderHMAC(name) for name in _HASHES},
        **{f"RS{name[3:]}": _ProviderRSA(name, pss=False) for name in _HASHES},
        **{f"PS{name[3:]}": _ProviderRSA(name, pss=True) for name in _HASHES},
    }
)


def compute_jwt_signatures_in_fips_provider(enabled: bool) -> None:
    """Make every jwt.encode and jwt.decode in this process compute HS, RS and PS signatures with the FIPS provider's
    own HMAC and signature services, or with PyJWT's own algorithms again when `enabled` is False."""
    stock: Final = get_default_algorithms()
    for name, algorithm in _PROVIDER_ALGORITHMS.items():
        jwt.unregister_algorithm(name)
        jwt.register_algorithm(name, algorithm if enabled else stock[name])
