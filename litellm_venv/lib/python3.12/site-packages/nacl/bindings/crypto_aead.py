# Copyright 2017 Donald Stufft and individual contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

"""
Implementations of authenticated encription with associated data (*AEAD*)
constructions building on the chacha20 stream cipher and the poly1305
authenticator
"""

crypto_aead_chacha20poly1305_ietf_KEYBYTES: int = (
    lib.crypto_aead_chacha20poly1305_ietf_keybytes()
)
crypto_aead_chacha20poly1305_ietf_NSECBYTES: int = (
    lib.crypto_aead_chacha20poly1305_ietf_nsecbytes()
)
crypto_aead_chacha20poly1305_ietf_NPUBBYTES: int = (
    lib.crypto_aead_chacha20poly1305_ietf_npubbytes()
)
crypto_aead_chacha20poly1305_ietf_ABYTES: int = (
    lib.crypto_aead_chacha20poly1305_ietf_abytes()
)
crypto_aead_chacha20poly1305_ietf_MESSAGEBYTES_MAX: int = (
    lib.crypto_aead_chacha20poly1305_ietf_messagebytes_max()
)
_aead_chacha20poly1305_ietf_CRYPTBYTES_MAX = (
    crypto_aead_chacha20poly1305_ietf_MESSAGEBYTES_MAX
    + crypto_aead_chacha20poly1305_ietf_ABYTES
)

crypto_aead_chacha20poly1305_KEYBYTES: int = lib.crypto_aead_chacha20poly1305_keybytes()
crypto_aead_chacha20poly1305_NSECBYTES: int = (
    lib.crypto_aead_chacha20poly1305_nsecbytes()
)
crypto_aead_chacha20poly1305_NPUBBYTES: int = (
    lib.crypto_aead_chacha20poly1305_npubbytes()
)
crypto_aead_chacha20poly1305_ABYTES: int = lib.crypto_aead_chacha20poly1305_abytes()
crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX: int = (
    lib.crypto_aead_chacha20poly1305_messagebytes_max()
)
_aead_chacha20poly1305_CRYPTBYTES_MAX = (
    crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX + crypto_aead_chacha20poly1305_ABYTES
)

crypto_aead_xchacha20poly1305_ietf_KEYBYTES: int = (
    lib.crypto_aead_xchacha20poly1305_ietf_keybytes()
)
crypto_aead_xchacha20poly1305_ietf_NSECBYTES: int = (
    lib.crypto_aead_xchacha20poly1305_ietf_nsecbytes()
)
crypto_aead_xchacha20poly1305_ietf_NPUBBYTES: int = (
    lib.crypto_aead_xchacha20poly1305_ietf_npubbytes()
)
crypto_aead_xchacha20poly1305_ietf_ABYTES: int = (
    lib.crypto_aead_xchacha20poly1305_ietf_abytes()
)
crypto_aead_xchacha20poly1305_ietf_MESSAGEBYTES_MAX: int = (
    lib.crypto_aead_xchacha20poly1305_ietf_messagebytes_max()
)
_aead_xchacha20poly1305_ietf_CRYPTBYTES_MAX = (
    crypto_aead_xchacha20poly1305_ietf_MESSAGEBYTES_MAX
    + crypto_aead_xchacha20poly1305_ietf_ABYTES
)


def crypto_aead_chacha20poly1305_ietf_encrypt(
    message: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Encrypt the given ``message`` using the IETF ratified chacha20poly1305
    construction described in RFC7539.

    :param message:
    :type message: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: authenticated ciphertext
    :rtype: bytes
    """
    ensure(
        isinstance(message, bytes),
        "Input message type must be bytes",
        raising=exc.TypeError,
    )

    mlen = len(message)

    ensure(
        mlen <= crypto_aead_chacha20poly1305_ietf_MESSAGEBYTES_MAX,
        "Message must be at most {} bytes long".format(
            crypto_aead_chacha20poly1305_ietf_MESSAGEBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_chacha20poly1305_ietf_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_ietf_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes)
        and len(key) == crypto_aead_chacha20poly1305_ietf_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_ietf_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    mxout = mlen + crypto_aead_chacha20poly1305_ietf_ABYTES

    clen = ffi.new("unsigned long long *")

    ciphertext = ffi.new("unsigned char[]", mxout)

    res = lib.crypto_aead_chacha20poly1305_ietf_encrypt(
        ciphertext, clen, message, mlen, _aad, aalen, ffi.NULL, nonce, key
    )

    ensure(res == 0, "Encryption failed.", raising=exc.CryptoError)
    return ffi.buffer(ciphertext, clen[0])[:]


def crypto_aead_chacha20poly1305_ietf_decrypt(
    ciphertext: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Decrypt the given ``ciphertext`` using the IETF ratified chacha20poly1305
    construction described in RFC7539.

    :param ciphertext:
    :type ciphertext: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: message
    :rtype: bytes
    """
    ensure(
        isinstance(ciphertext, bytes),
        "Input ciphertext type must be bytes",
        raising=exc.TypeError,
    )

    clen = len(ciphertext)

    ensure(
        clen <= _aead_chacha20poly1305_ietf_CRYPTBYTES_MAX,
        "Ciphertext must be at most {} bytes long".format(
            _aead_chacha20poly1305_ietf_CRYPTBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_chacha20poly1305_ietf_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_ietf_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes)
        and len(key) == crypto_aead_chacha20poly1305_ietf_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_ietf_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    mxout = clen - crypto_aead_chacha20poly1305_ietf_ABYTES

    mlen = ffi.new("unsigned long long *")
    message = ffi.new("unsigned char[]", mxout)

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    res = lib.crypto_aead_chacha20poly1305_ietf_decrypt(
        message, mlen, ffi.NULL, ciphertext, clen, _aad, aalen, nonce, key
    )

    ensure(res == 0, "Decryption failed.", raising=exc.CryptoError)

    return ffi.buffer(message, mlen[0])[:]


def crypto_aead_chacha20poly1305_encrypt(
    message: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Encrypt the given ``message`` using the "legacy" construction
    described in draft-agl-tls-chacha20poly1305.

    :param message:
    :type message: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: authenticated ciphertext
    :rtype: bytes
    """
    ensure(
        isinstance(message, bytes),
        "Input message type must be bytes",
        raising=exc.TypeError,
    )

    mlen = len(message)

    ensure(
        mlen <= crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX,
        "Message must be at most {} bytes long".format(
            crypto_aead_chacha20poly1305_MESSAGEBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_chacha20poly1305_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes) and len(key) == crypto_aead_chacha20poly1305_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    mlen = len(message)
    mxout = mlen + crypto_aead_chacha20poly1305_ietf_ABYTES

    clen = ffi.new("unsigned long long *")

    ciphertext = ffi.new("unsigned char[]", mxout)

    res = lib.crypto_aead_chacha20poly1305_encrypt(
        ciphertext, clen, message, mlen, _aad, aalen, ffi.NULL, nonce, key
    )

    ensure(res == 0, "Encryption failed.", raising=exc.CryptoError)
    return ffi.buffer(ciphertext, clen[0])[:]


def crypto_aead_chacha20poly1305_decrypt(
    ciphertext: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Decrypt the given ``ciphertext`` using the "legacy" construction
    described in draft-agl-tls-chacha20poly1305.

    :param ciphertext: authenticated ciphertext
    :type ciphertext: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: message
    :rtype: bytes
    """
    ensure(
        isinstance(ciphertext, bytes),
        "Input ciphertext type must be bytes",
        raising=exc.TypeError,
    )

    clen = len(ciphertext)

    ensure(
        clen <= _aead_chacha20poly1305_CRYPTBYTES_MAX,
        "Ciphertext must be at most {} bytes long".format(
            _aead_chacha20poly1305_CRYPTBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_chacha20poly1305_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes) and len(key) == crypto_aead_chacha20poly1305_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_chacha20poly1305_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    mxout = clen - crypto_aead_chacha20poly1305_ABYTES

    mlen = ffi.new("unsigned long long *")
    message = ffi.new("unsigned char[]", mxout)

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    res = lib.crypto_aead_chacha20poly1305_decrypt(
        message, mlen, ffi.NULL, ciphertext, clen, _aad, aalen, nonce, key
    )

    ensure(res == 0, "Decryption failed.", raising=exc.CryptoError)

    return ffi.buffer(message, mlen[0])[:]


def crypto_aead_xchacha20poly1305_ietf_encrypt(
    message: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Encrypt the given ``message`` using the long-nonces xchacha20poly1305
    construction.

    :param message:
    :type message: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: authenticated ciphertext
    :rtype: bytes
    """
    ensure(
        isinstance(message, bytes),
        "Input message type must be bytes",
        raising=exc.TypeError,
    )

    mlen = len(message)

    ensure(
        mlen <= crypto_aead_xchacha20poly1305_ietf_MESSAGEBYTES_MAX,
        "Message must be at most {} bytes long".format(
            crypto_aead_xchacha20poly1305_ietf_MESSAGEBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes)
        and len(key) == crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_xchacha20poly1305_ietf_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    mlen = len(message)
    mxout = mlen + crypto_aead_xchacha20poly1305_ietf_ABYTES

    clen = ffi.new("unsigned long long *")

    ciphertext = ffi.new("unsigned char[]", mxout)

    res = lib.crypto_aead_xchacha20poly1305_ietf_encrypt(
        ciphertext, clen, message, mlen, _aad, aalen, ffi.NULL, nonce, key
    )

    ensure(res == 0, "Encryption failed.", raising=exc.CryptoError)
    return ffi.buffer(ciphertext, clen[0])[:]


def crypto_aead_xchacha20poly1305_ietf_decrypt(
    ciphertext: bytes, aad: Optional[bytes], nonce: bytes, key: bytes
) -> bytes:
    """
    Decrypt the given ``ciphertext`` using the long-nonces xchacha20poly1305
    construction.

    :param ciphertext: authenticated ciphertext
    :type ciphertext: bytes
    :param aad:
    :type aad: Optional[bytes]
    :param nonce:
    :type nonce: bytes
    :param key:
    :type key: bytes
    :return: message
    :rtype: bytes
    """
    ensure(
        isinstance(ciphertext, bytes),
        "Input ciphertext type must be bytes",
        raising=exc.TypeError,
    )

    clen = len(ciphertext)

    ensure(
        clen <= _aead_xchacha20poly1305_ietf_CRYPTBYTES_MAX,
        "Ciphertext must be at most {} bytes long".format(
            _aead_xchacha20poly1305_ietf_CRYPTBYTES_MAX
        ),
        raising=exc.ValueError,
    )

    ensure(
        isinstance(aad, bytes) or (aad is None),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(nonce, bytes)
        and len(nonce) == crypto_aead_xchacha20poly1305_ietf_NPUBBYTES,
        "Nonce must be a {} bytes long bytes sequence".format(
            crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(key, bytes)
        and len(key) == crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
        "Key must be a {} bytes long bytes sequence".format(
            crypto_aead_xchacha20poly1305_ietf_KEYBYTES
        ),
        raising=exc.TypeError,
    )

    mxout = clen - crypto_aead_xchacha20poly1305_ietf_ABYTES
    mlen = ffi.new("unsigned long long *")
    message = ffi.new("unsigned char[]", mxout)

    if aad:
        _aad = aad
        aalen = len(aad)
    else:
        _aad = ffi.NULL
        aalen = 0

    res = lib.crypto_aead_xchacha20poly1305_ietf_decrypt(
        message, mlen, ffi.NULL, ciphertext, clen, _aad, aalen, nonce, key
    )

    ensure(res == 0, "Decryption failed.", raising=exc.CryptoError)

    return ffi.buffer(message, mlen[0])[:]
