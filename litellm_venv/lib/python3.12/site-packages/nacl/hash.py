# Copyright 2013 Donald Stufft and individual contributors
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
"""
The :mod:`nacl.hash` module exposes one-shot interfaces
for libsodium selected hash primitives and the constants needed
for their usage.
"""


import nacl.bindings
import nacl.encoding


BLAKE2B_BYTES = nacl.bindings.crypto_generichash_BYTES
"""Default digest size for :func:`blake2b` hash"""
BLAKE2B_BYTES_MIN = nacl.bindings.crypto_generichash_BYTES_MIN
"""Minimum allowed digest size for :func:`blake2b` hash"""
BLAKE2B_BYTES_MAX = nacl.bindings.crypto_generichash_BYTES_MAX
"""Maximum allowed digest size for :func:`blake2b` hash"""
BLAKE2B_KEYBYTES = nacl.bindings.crypto_generichash_KEYBYTES
"""Default size of the ``key`` byte array for :func:`blake2b` hash"""
BLAKE2B_KEYBYTES_MIN = nacl.bindings.crypto_generichash_KEYBYTES_MIN
"""Minimum allowed size of the ``key`` byte array for :func:`blake2b` hash"""
BLAKE2B_KEYBYTES_MAX = nacl.bindings.crypto_generichash_KEYBYTES_MAX
"""Maximum allowed size of the ``key`` byte array for :func:`blake2b` hash"""
BLAKE2B_SALTBYTES = nacl.bindings.crypto_generichash_SALTBYTES
"""Maximum allowed length of the ``salt`` byte array for
:func:`blake2b` hash"""
BLAKE2B_PERSONALBYTES = nacl.bindings.crypto_generichash_PERSONALBYTES
"""Maximum allowed length of the ``personalization``
byte array for :func:`blake2b` hash"""

SIPHASH_BYTES = nacl.bindings.crypto_shorthash_siphash24_BYTES
"""Size of the :func:`siphash24` digest"""
SIPHASH_KEYBYTES = nacl.bindings.crypto_shorthash_siphash24_KEYBYTES
"""Size of the secret ``key`` used by the :func:`siphash24` MAC"""

SIPHASHX_AVAILABLE = nacl.bindings.has_crypto_shorthash_siphashx24
"""``True`` if :func:`siphashx24` is available to be called"""

SIPHASHX_BYTES = nacl.bindings.crypto_shorthash_siphashx24_BYTES
"""Size of the :func:`siphashx24` digest"""
SIPHASHX_KEYBYTES = nacl.bindings.crypto_shorthash_siphashx24_KEYBYTES
"""Size of the secret ``key`` used by the :func:`siphashx24` MAC"""

_b2b_hash = nacl.bindings.crypto_generichash_blake2b_salt_personal
_sip_hash = nacl.bindings.crypto_shorthash_siphash24
_sip_hashx = nacl.bindings.crypto_shorthash_siphashx24


def sha256(
    message: bytes, encoder: nacl.encoding.Encoder = nacl.encoding.HexEncoder
) -> bytes:
    """
    Hashes ``message`` with SHA256.

    :param message: The message to hash.
    :type message: bytes
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes
    """
    return encoder.encode(nacl.bindings.crypto_hash_sha256(message))


def sha512(
    message: bytes, encoder: nacl.encoding.Encoder = nacl.encoding.HexEncoder
) -> bytes:
    """
    Hashes ``message`` with SHA512.

    :param message: The message to hash.
    :type message: bytes
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes
    """
    return encoder.encode(nacl.bindings.crypto_hash_sha512(message))


def blake2b(
    data: bytes,
    digest_size: int = BLAKE2B_BYTES,
    key: bytes = b"",
    salt: bytes = b"",
    person: bytes = b"",
    encoder: nacl.encoding.Encoder = nacl.encoding.HexEncoder,
) -> bytes:
    """
    Hashes ``data`` with blake2b.

    :param data: the digest input byte sequence
    :type data: bytes
    :param digest_size: the requested digest size; must be at most
                        :const:`BLAKE2B_BYTES_MAX`;
                        the default digest size is
                        :const:`BLAKE2B_BYTES`
    :type digest_size: int
    :param key: the key to be set for keyed MAC/PRF usage; if set, the key
                must be at most :data:`~nacl.hash.BLAKE2B_KEYBYTES_MAX` long
    :type key: bytes
    :param salt: an initialization salt at most
                 :const:`BLAKE2B_SALTBYTES` long;
                 it will be zero-padded if needed
    :type salt: bytes
    :param person: a personalization string at most
                   :const:`BLAKE2B_PERSONALBYTES` long;
                   it will be zero-padded if needed
    :type person: bytes
    :param encoder: the encoder to use on returned digest
    :type encoder: class
    :returns: The hashed message.
    :rtype: bytes
    """

    digest = _b2b_hash(data, digest_size=digest_size, key=key, salt=salt, person=person)
    return encoder.encode(digest)


generichash = blake2b


def siphash24(
    message: bytes,
    key: bytes = b"",
    encoder: nacl.encoding.Encoder = nacl.encoding.HexEncoder,
) -> bytes:
    """
    Computes a keyed MAC of ``message`` using the short-input-optimized
    siphash-2-4 construction.

    :param message: The message to hash.
    :type message: bytes
    :param key: the message authentication key for the siphash MAC construct
    :type key: bytes(:const:`SIPHASH_KEYBYTES`)
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes(:const:`SIPHASH_BYTES`)
    """
    digest = _sip_hash(message, key)
    return encoder.encode(digest)


shorthash = siphash24


def siphashx24(
    message: bytes,
    key: bytes = b"",
    encoder: nacl.encoding.Encoder = nacl.encoding.HexEncoder,
) -> bytes:
    """
    Computes a keyed MAC of ``message`` using the 128 bit variant of the
    siphash-2-4 construction.

    :param message: The message to hash.
    :type message: bytes
    :param key: the message authentication key for the siphash MAC construct
    :type key: bytes(:const:`SIPHASHX_KEYBYTES`)
    :param encoder: A class that is able to encode the hashed message.
    :returns: The hashed message.
    :rtype: bytes(:const:`SIPHASHX_BYTES`)
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    .. versionadded:: 1.2
    """
    digest = _sip_hashx(message, key)
    return encoder.encode(digest)
