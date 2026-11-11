import _hashlib
from typing import Final
from unittest.mock import Mock

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

from litellm.proxy.common_utils.fips_signature import openssl_signs_whole_messages, sign_message, verify_message

_needs_shared_openssl: Final = pytest.mark.skipif(
    not hasattr(_hashlib, "__file__"), reason="this Python links OpenSSL statically, out of ctypes reach"
)
_KEY: Final = rsa.generate_private_key(65537, 2048)
_PRIVATE_DER: Final = _KEY.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
_PUBLIC_DER: Final = _KEY.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
_MESSAGE: Final = b"header.payload"


def test_openssl_signs_whole_messages_only_where_python_links_a_shared_openssl() -> None:
    assert openssl_signs_whole_messages() is hasattr(_hashlib, "__file__")


@pytest.mark.parametrize("error", [AttributeError("no EVP_DigestSignInit_ex"), OSError("cannot open shared object")])
def test_openssl_signs_whole_messages_is_false_when_the_calls_cannot_be_bound(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr("litellm.proxy.common_utils.fips_signature._libcrypto", Mock(side_effect=error))
    assert openssl_signs_whole_messages() is False


@_needs_shared_openssl
def test_sign_message_makes_a_pkcs1_signature_the_cryptography_library_accepts() -> None:
    signature: Final = sign_message(_PRIVATE_DER, _MESSAGE, "sha256")

    _KEY.public_key().verify(signature, _MESSAGE, padding.PKCS1v15(), hashes.SHA256())
    with pytest.raises(InvalidSignature):
        _KEY.public_key().verify(signature, b"another message", padding.PKCS1v15(), hashes.SHA256())


@_needs_shared_openssl
def test_sign_message_makes_a_pss_signature_with_a_salt_as_long_as_the_hash() -> None:
    signature: Final = sign_message(_PRIVATE_DER, _MESSAGE, "sha384", pss=True)

    mgf: Final = padding.MGF1(hashes.SHA384())
    _KEY.public_key().verify(signature, _MESSAGE, padding.PSS(mgf=mgf, salt_length=48), hashes.SHA384())
    with pytest.raises(InvalidSignature):
        _KEY.public_key().verify(signature, _MESSAGE, padding.PSS(mgf=mgf, salt_length=20), hashes.SHA384())


@_needs_shared_openssl
@pytest.mark.parametrize("pss", [False, True])
def test_verify_message_accepts_only_the_signed_message_under_the_same_padding(pss: bool) -> None:
    signature: Final = sign_message(_PRIVATE_DER, _MESSAGE, "sha256", pss=pss)

    assert verify_message(_PUBLIC_DER, _MESSAGE, signature, "sha256", pss=pss) is True
    assert verify_message(_PUBLIC_DER, b"another message", signature, "sha256", pss=pss) is False
    assert verify_message(_PUBLIC_DER, _MESSAGE, signature, "sha256", pss=not pss) is False
    assert verify_message(_PUBLIC_DER, _MESSAGE, signature, "sha512", pss=pss) is False


@_needs_shared_openssl
def test_sign_message_and_verify_message_refuse_bytes_that_are_not_a_key() -> None:
    with pytest.raises(ValueError, match="OpenSSL refused to sign"):
        sign_message(b"not a key", _MESSAGE, "sha256")
    with pytest.raises(ValueError, match="OpenSSL refused to verify"):
        verify_message(b"not a key", _MESSAGE, b"signature", "sha256")


@_needs_shared_openssl
def test_sign_message_refuses_a_digest_openssl_does_not_know() -> None:
    with pytest.raises(ValueError, match="OpenSSL refused to sign"):
        sign_message(_PRIVATE_DER, _MESSAGE, "sha257")
