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
from typing import Tuple

from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


__all__ = ["crypto_box_keypair", "crypto_box"]


crypto_box_SECRETKEYBYTES: int = lib.crypto_box_secretkeybytes()
crypto_box_PUBLICKEYBYTES: int = lib.crypto_box_publickeybytes()
crypto_box_SEEDBYTES: int = lib.crypto_box_seedbytes()
crypto_box_NONCEBYTES: int = lib.crypto_box_noncebytes()
crypto_box_ZEROBYTES: int = lib.crypto_box_zerobytes()
crypto_box_BOXZEROBYTES: int = lib.crypto_box_boxzerobytes()
crypto_box_BEFORENMBYTES: int = lib.crypto_box_beforenmbytes()
crypto_box_SEALBYTES: int = lib.crypto_box_sealbytes()


def crypto_box_keypair() -> Tuple[bytes, bytes]:
    """
    Returns a randomly generated public and secret key.

    :rtype: (bytes(public_key), bytes(secret_key))
    """
    pk = ffi.new("unsigned char[]", crypto_box_PUBLICKEYBYTES)
    sk = ffi.new("unsigned char[]", crypto_box_SECRETKEYBYTES)

    rc = lib.crypto_box_keypair(pk, sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return (
        ffi.buffer(pk, crypto_box_PUBLICKEYBYTES)[:],
        ffi.buffer(sk, crypto_box_SECRETKEYBYTES)[:],
    )


def crypto_box_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Returns a (public, secret) keypair deterministically generated
    from an input ``seed``.

    .. warning:: The seed **must** be high-entropy; therefore,
        its generator **must** be a cryptographic quality
        random function like, for example, :func:`~nacl.utils.random`.

    .. warning:: The seed **must** be protected and remain secret.
        Anyone who knows the seed is really in possession of
        the corresponding PrivateKey.


    :param seed: bytes
    :rtype: (bytes(public_key), bytes(secret_key))
    """
    ensure(isinstance(seed, bytes), "seed must be bytes", raising=TypeError)

    if len(seed) != crypto_box_SEEDBYTES:
        raise exc.ValueError("Invalid seed")

    pk = ffi.new("unsigned char[]", crypto_box_PUBLICKEYBYTES)
    sk = ffi.new("unsigned char[]", crypto_box_SECRETKEYBYTES)

    rc = lib.crypto_box_seed_keypair(pk, sk, seed)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return (
        ffi.buffer(pk, crypto_box_PUBLICKEYBYTES)[:],
        ffi.buffer(sk, crypto_box_SECRETKEYBYTES)[:],
    )


def crypto_box(message: bytes, nonce: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Encrypts and returns a message ``message`` using the secret key ``sk``,
    public key ``pk``, and the nonce ``nonce``.

    :param message: bytes
    :param nonce: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError("Invalid nonce size")

    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid public key")

    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    padded = (b"\x00" * crypto_box_ZEROBYTES) + message
    ciphertext = ffi.new("unsigned char[]", len(padded))

    rc = lib.crypto_box(ciphertext, padded, len(padded), nonce, pk, sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(ciphertext, len(padded))[crypto_box_BOXZEROBYTES:]


def crypto_box_open(ciphertext: bytes, nonce: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Decrypts and returns an encrypted message ``ciphertext``, using the secret
    key ``sk``, public key ``pk``, and the nonce ``nonce``.

    :param ciphertext: bytes
    :param nonce: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError("Invalid nonce size")

    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid public key")

    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    padded = (b"\x00" * crypto_box_BOXZEROBYTES) + ciphertext
    plaintext = ffi.new("unsigned char[]", len(padded))

    res = lib.crypto_box_open(plaintext, padded, len(padded), nonce, pk, sk)
    ensure(
        res == 0,
        "An error occurred trying to decrypt the message",
        raising=exc.CryptoError,
    )

    return ffi.buffer(plaintext, len(padded))[crypto_box_ZEROBYTES:]


def crypto_box_beforenm(pk: bytes, sk: bytes) -> bytes:
    """
    Computes and returns the shared key for the public key ``pk`` and the
    secret key ``sk``. This can be used to speed up operations where the same
    set of keys is going to be used multiple times.

    :param pk: bytes
    :param sk: bytes
    :rtype: bytes
    """
    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid public key")

    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    k = ffi.new("unsigned char[]", crypto_box_BEFORENMBYTES)

    rc = lib.crypto_box_beforenm(k, pk, sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(k, crypto_box_BEFORENMBYTES)[:]


def crypto_box_afternm(message: bytes, nonce: bytes, k: bytes) -> bytes:
    """
    Encrypts and returns the message ``message`` using the shared key ``k`` and
    the nonce ``nonce``.

    :param message: bytes
    :param nonce: bytes
    :param k: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError("Invalid nonce")

    if len(k) != crypto_box_BEFORENMBYTES:
        raise exc.ValueError("Invalid shared key")

    padded = b"\x00" * crypto_box_ZEROBYTES + message
    ciphertext = ffi.new("unsigned char[]", len(padded))

    rc = lib.crypto_box_afternm(ciphertext, padded, len(padded), nonce, k)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(ciphertext, len(padded))[crypto_box_BOXZEROBYTES:]


def crypto_box_open_afternm(ciphertext: bytes, nonce: bytes, k: bytes) -> bytes:
    """
    Decrypts and returns the encrypted message ``ciphertext``, using the shared
    key ``k`` and the nonce ``nonce``.

    :param ciphertext: bytes
    :param nonce: bytes
    :param k: bytes
    :rtype: bytes
    """
    if len(nonce) != crypto_box_NONCEBYTES:
        raise exc.ValueError("Invalid nonce")

    if len(k) != crypto_box_BEFORENMBYTES:
        raise exc.ValueError("Invalid shared key")

    padded = (b"\x00" * crypto_box_BOXZEROBYTES) + ciphertext
    plaintext = ffi.new("unsigned char[]", len(padded))

    res = lib.crypto_box_open_afternm(plaintext, padded, len(padded), nonce, k)
    ensure(
        res == 0,
        "An error occurred trying to decrypt the message",
        raising=exc.CryptoError,
    )

    return ffi.buffer(plaintext, len(padded))[crypto_box_ZEROBYTES:]


def crypto_box_seal(message: bytes, pk: bytes) -> bytes:
    """
    Encrypts and returns a message ``message`` using an ephemeral secret key
    and the public key ``pk``.
    The ephemeral public key, which is embedded in the sealed box, is also
    used, in combination with ``pk``, to derive the nonce needed for the
    underlying box construct.

    :param message: bytes
    :param pk: bytes
    :rtype: bytes

    .. versionadded:: 1.2
    """
    ensure(
        isinstance(message, bytes),
        "input message must be bytes",
        raising=TypeError,
    )

    ensure(isinstance(pk, bytes), "public key must be bytes", raising=TypeError)

    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid public key")

    _mlen = len(message)
    _clen = crypto_box_SEALBYTES + _mlen

    ciphertext = ffi.new("unsigned char[]", _clen)

    rc = lib.crypto_box_seal(ciphertext, message, _mlen, pk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(ciphertext, _clen)[:]


def crypto_box_seal_open(ciphertext: bytes, pk: bytes, sk: bytes) -> bytes:
    """
    Decrypts and returns an encrypted message ``ciphertext``, using the
    recipent's secret key ``sk`` and the sender's ephemeral public key
    embedded in the sealed box. The box contruct nonce is derived from
    the recipient's public key ``pk`` and the sender's public key.

    :param ciphertext: bytes
    :param pk: bytes
    :param sk: bytes
    :rtype: bytes

    .. versionadded:: 1.2
    """
    ensure(
        isinstance(ciphertext, bytes),
        "input ciphertext must be bytes",
        raising=TypeError,
    )

    ensure(isinstance(pk, bytes), "public key must be bytes", raising=TypeError)

    ensure(isinstance(sk, bytes), "secret key must be bytes", raising=TypeError)

    if len(pk) != crypto_box_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid public key")

    if len(sk) != crypto_box_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    _clen = len(ciphertext)

    ensure(
        _clen >= crypto_box_SEALBYTES,
        ("Input cyphertext must be at least {} long").format(crypto_box_SEALBYTES),
        raising=exc.TypeError,
    )

    _mlen = _clen - crypto_box_SEALBYTES

    # zero-length malloc results are implementation.dependent
    plaintext = ffi.new("unsigned char[]", max(1, _mlen))

    res = lib.crypto_box_seal_open(plaintext, ciphertext, _clen, pk, sk)
    ensure(
        res == 0,
        "An error occurred trying to decrypt the message",
        raising=exc.CryptoError,
    )

    return ffi.buffer(plaintext, _mlen)[:]
