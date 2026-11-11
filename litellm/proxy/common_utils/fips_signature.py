import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

_OSSL_PARAM_UTF8_STRING: Final = 4
_OSSL_PARAM_UNMODIFIED: Final = ctypes.c_size_t(-1).value


class _OsslParam(ctypes.Structure):
    _fields_ = (
        ("key", ctypes.c_char_p),
        ("data_type", ctypes.c_uint),
        ("data", ctypes.c_char_p),
        ("data_size", ctypes.c_size_t),
        ("return_size", ctypes.c_size_t),
    )


def _text_param(name: bytes, value: bytes) -> _OsslParam:
    return _OsslParam(name, _OSSL_PARAM_UTF8_STRING, value, len(value), _OSSL_PARAM_UNMODIFIED)


@dataclass(frozen=True, slots=True)
class _Libcrypto:
    load_private_key: Callable[..., int | None]
    load_public_key: Callable[..., int | None]
    free_key: Callable[..., None]
    new_context: Callable[..., int | None]
    free_context: Callable[..., None]
    sign_init: Callable[..., int]
    sign: Callable[..., int]
    verify_init: Callable[..., int]
    verify: Callable[..., int]
    clear_errors: Callable[..., None]


@lru_cache(maxsize=1)
def _libcrypto() -> _Libcrypto:
    import _hashlib

    library: Final = ctypes.CDLL(_hashlib.__file__)
    load_key: Final = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_long)
    init: Final = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(_OsslParam),
    )
    free: Final = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    return _Libcrypto(
        load_private_key=load_key(("d2i_AutoPrivateKey", library)),
        load_public_key=load_key(("d2i_PUBKEY", library)),
        free_key=free(("EVP_PKEY_free", library)),
        new_context=ctypes.CFUNCTYPE(ctypes.c_void_p)(("EVP_MD_CTX_new", library)),
        free_context=free(("EVP_MD_CTX_free", library)),
        sign_init=init(("EVP_DigestSignInit_ex", library)),
        sign=ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_char_p,
            ctypes.c_size_t,
        )(("EVP_DigestSign", library)),
        verify_init=init(("EVP_DigestVerifyInit_ex", library)),
        verify=ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t
        )(("EVP_DigestVerify", library)),
        clear_errors=ctypes.CFUNCTYPE(None)(("ERR_clear_error", library)),
    )


def openssl_signs_whole_messages() -> bool:
    """The OpenSSL that hashlib uses is a shared library ctypes can reach and has the OpenSSL 3 message-level
    signature calls."""
    try:
        _libcrypto()
    except (AttributeError, OSError):
        return False
    return True


def _pss(digest: bytes) -> ctypes.Array[_OsslParam]:
    # RFC 7518 3.5: MGF1 over the same hash (OpenSSL's default) and a salt as long as the hash output. OpenSSL sets
    # these before the digest named in the init call, and PSS needs a digest by then, so it is named here as well.
    return (_OsslParam * 4)(
        _text_param(b"digest", digest), _text_param(b"pad-mode", b"pss"), _text_param(b"saltlen", b"digest")
    )


def sign_message(private_key_der: bytes, message: bytes, digest: str, *, pss: bool = False) -> bytes:
    """A signature over message from EVP_DigestSign, which hashes and signs in one call to the provider, so the
    provider sees the message and not a digest computed by the caller. PKCS#1 v1.5 for an RSA key unless pss.
    Raises ValueError when OpenSSL refuses the key or the digest."""
    libcrypto: Final = _libcrypto()
    name: Final = digest.encode()
    cursor: Final = ctypes.c_char_p(private_key_der)
    key: Final = libcrypto.load_private_key(None, ctypes.byref(cursor), len(private_key_der))
    context: Final = libcrypto.new_context()
    try:
        length: Final = ctypes.c_size_t(0)
        if (
            not key
            or not context
            or libcrypto.sign_init(context, None, name, None, None, key, _pss(name) if pss else None) != 1
            or libcrypto.sign(context, None, ctypes.byref(length), message, len(message)) != 1
        ):
            raise ValueError("OpenSSL refused to sign with this key and digest")
        signature: Final = ctypes.create_string_buffer(length.value)
        if libcrypto.sign(context, signature, ctypes.byref(length), message, len(message)) != 1:
            raise ValueError("OpenSSL could not sign the message")
        return signature.raw[: length.value]
    finally:
        libcrypto.free_context(context)
        libcrypto.free_key(key)
        libcrypto.clear_errors()


def verify_message(public_key_der: bytes, message: bytes, signature: bytes, digest: str, *, pss: bool = False) -> bool:
    """Whether signature is valid over message, from EVP_DigestVerify, which hashes and verifies in one call to the
    provider. Raises ValueError when OpenSSL refuses the key or the digest."""
    libcrypto: Final = _libcrypto()
    name: Final = digest.encode()
    cursor: Final = ctypes.c_char_p(public_key_der)
    key: Final = libcrypto.load_public_key(None, ctypes.byref(cursor), len(public_key_der))
    context: Final = libcrypto.new_context()
    try:
        if (
            not key
            or not context
            or libcrypto.verify_init(context, None, name, None, None, key, _pss(name) if pss else None) != 1
        ):
            raise ValueError("OpenSSL refused to verify with this key and digest")
        return libcrypto.verify(context, signature, len(signature), message, len(message)) == 1
    finally:
        libcrypto.free_context(context)
        libcrypto.free_key(key)
        libcrypto.clear_errors()
