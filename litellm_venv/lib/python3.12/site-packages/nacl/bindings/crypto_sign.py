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


crypto_sign_BYTES: int = lib.crypto_sign_bytes()
# crypto_sign_SEEDBYTES = lib.crypto_sign_seedbytes()
crypto_sign_SEEDBYTES: int = lib.crypto_sign_secretkeybytes() // 2
crypto_sign_PUBLICKEYBYTES: int = lib.crypto_sign_publickeybytes()
crypto_sign_SECRETKEYBYTES: int = lib.crypto_sign_secretkeybytes()

crypto_sign_curve25519_BYTES: int = lib.crypto_box_secretkeybytes()

crypto_sign_ed25519ph_STATEBYTES: int = lib.crypto_sign_ed25519ph_statebytes()


def crypto_sign_keypair() -> Tuple[bytes, bytes]:
    """
    Returns a randomly generated public key and secret key.

    :rtype: (bytes(public_key), bytes(secret_key))
    """
    pk = ffi.new("unsigned char[]", crypto_sign_PUBLICKEYBYTES)
    sk = ffi.new("unsigned char[]", crypto_sign_SECRETKEYBYTES)

    rc = lib.crypto_sign_keypair(pk, sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return (
        ffi.buffer(pk, crypto_sign_PUBLICKEYBYTES)[:],
        ffi.buffer(sk, crypto_sign_SECRETKEYBYTES)[:],
    )


def crypto_sign_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Computes and returns the public key and secret key using the seed ``seed``.

    :param seed: bytes
    :rtype: (bytes(public_key), bytes(secret_key))
    """
    if len(seed) != crypto_sign_SEEDBYTES:
        raise exc.ValueError("Invalid seed")

    pk = ffi.new("unsigned char[]", crypto_sign_PUBLICKEYBYTES)
    sk = ffi.new("unsigned char[]", crypto_sign_SECRETKEYBYTES)

    rc = lib.crypto_sign_seed_keypair(pk, sk, seed)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return (
        ffi.buffer(pk, crypto_sign_PUBLICKEYBYTES)[:],
        ffi.buffer(sk, crypto_sign_SECRETKEYBYTES)[:],
    )


def crypto_sign(message: bytes, sk: bytes) -> bytes:
    """
    Signs the message ``message`` using the secret key ``sk`` and returns the
    signed message.

    :param message: bytes
    :param sk: bytes
    :rtype: bytes
    """
    signed = ffi.new("unsigned char[]", len(message) + crypto_sign_BYTES)
    signed_len = ffi.new("unsigned long long *")

    rc = lib.crypto_sign(signed, signed_len, message, len(message), sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(signed, signed_len[0])[:]


def crypto_sign_open(signed: bytes, pk: bytes) -> bytes:
    """
    Verifies the signature of the signed message ``signed`` using the public
    key ``pk`` and returns the unsigned message.

    :param signed: bytes
    :param pk: bytes
    :rtype: bytes
    """
    message = ffi.new("unsigned char[]", len(signed))
    message_len = ffi.new("unsigned long long *")

    if lib.crypto_sign_open(message, message_len, signed, len(signed), pk) != 0:
        raise exc.BadSignatureError("Signature was forged or corrupt")

    return ffi.buffer(message, message_len[0])[:]


def crypto_sign_ed25519_pk_to_curve25519(public_key_bytes: bytes) -> bytes:
    """
    Converts a public Ed25519 key (encoded as bytes ``public_key_bytes``) to
    a public Curve25519 key as bytes.

    Raises a ValueError if ``public_key_bytes`` is not of length
    ``crypto_sign_PUBLICKEYBYTES``

    :param public_key_bytes: bytes
    :rtype: bytes
    """
    if len(public_key_bytes) != crypto_sign_PUBLICKEYBYTES:
        raise exc.ValueError("Invalid curve public key")

    curve_public_key_len = crypto_sign_curve25519_BYTES
    curve_public_key = ffi.new("unsigned char[]", curve_public_key_len)

    rc = lib.crypto_sign_ed25519_pk_to_curve25519(curve_public_key, public_key_bytes)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(curve_public_key, curve_public_key_len)[:]


def crypto_sign_ed25519_sk_to_curve25519(secret_key_bytes: bytes) -> bytes:
    """
    Converts a secret Ed25519 key (encoded as bytes ``secret_key_bytes``) to
    a secret Curve25519 key as bytes.

    Raises a ValueError if ``secret_key_bytes``is not of length
    ``crypto_sign_SECRETKEYBYTES``

    :param secret_key_bytes: bytes
    :rtype: bytes
    """
    if len(secret_key_bytes) != crypto_sign_SECRETKEYBYTES:
        raise exc.ValueError("Invalid curve secret key")

    curve_secret_key_len = crypto_sign_curve25519_BYTES
    curve_secret_key = ffi.new("unsigned char[]", curve_secret_key_len)

    rc = lib.crypto_sign_ed25519_sk_to_curve25519(curve_secret_key, secret_key_bytes)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(curve_secret_key, curve_secret_key_len)[:]


def crypto_sign_ed25519_sk_to_pk(secret_key_bytes: bytes) -> bytes:
    """
    Extract the public Ed25519 key from a secret Ed25519 key (encoded
    as bytes ``secret_key_bytes``).

    Raises a ValueError if ``secret_key_bytes``is not of length
    ``crypto_sign_SECRETKEYBYTES``

    :param secret_key_bytes: bytes
    :rtype: bytes
    """
    if len(secret_key_bytes) != crypto_sign_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    return secret_key_bytes[crypto_sign_SEEDBYTES:]


def crypto_sign_ed25519_sk_to_seed(secret_key_bytes: bytes) -> bytes:
    """
    Extract the seed from a secret Ed25519 key (encoded
    as bytes ``secret_key_bytes``).

    Raises a ValueError if ``secret_key_bytes``is not of length
    ``crypto_sign_SECRETKEYBYTES``

    :param secret_key_bytes: bytes
    :rtype: bytes
    """
    if len(secret_key_bytes) != crypto_sign_SECRETKEYBYTES:
        raise exc.ValueError("Invalid secret key")

    return secret_key_bytes[:crypto_sign_SEEDBYTES]


class crypto_sign_ed25519ph_state:
    """
    State object wrapping the sha-512 state used in ed25519ph computation
    """

    __slots__ = ["state"]

    def __init__(self) -> None:
        self.state: bytes = ffi.new("unsigned char[]", crypto_sign_ed25519ph_STATEBYTES)

        rc = lib.crypto_sign_ed25519ph_init(self.state)

        ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)


def crypto_sign_ed25519ph_update(
    edph: crypto_sign_ed25519ph_state, pmsg: bytes
) -> None:
    """
    Update the hash state wrapped in edph

    :param edph: the ed25519ph state being updated
    :type edph: crypto_sign_ed25519ph_state
    :param pmsg: the partial message
    :type pmsg: bytes
    :rtype: None
    """
    ensure(
        isinstance(edph, crypto_sign_ed25519ph_state),
        "edph parameter must be a ed25519ph_state object",
        raising=exc.TypeError,
    )
    ensure(
        isinstance(pmsg, bytes),
        "pmsg parameter must be a bytes object",
        raising=exc.TypeError,
    )
    rc = lib.crypto_sign_ed25519ph_update(edph.state, pmsg, len(pmsg))
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)


def crypto_sign_ed25519ph_final_create(
    edph: crypto_sign_ed25519ph_state, sk: bytes
) -> bytes:
    """
    Create a signature for the data hashed in edph
    using the secret key sk

    :param edph: the ed25519ph state for the data
                 being signed
    :type edph: crypto_sign_ed25519ph_state
    :param sk: the ed25519 secret part of the signing key
    :type sk: bytes
    :return: ed25519ph signature
    :rtype: bytes
    """
    ensure(
        isinstance(edph, crypto_sign_ed25519ph_state),
        "edph parameter must be a ed25519ph_state object",
        raising=exc.TypeError,
    )
    ensure(
        isinstance(sk, bytes),
        "secret key parameter must be a bytes object",
        raising=exc.TypeError,
    )
    ensure(
        len(sk) == crypto_sign_SECRETKEYBYTES,
        ("secret key must be {} bytes long").format(crypto_sign_SECRETKEYBYTES),
        raising=exc.TypeError,
    )
    signature = ffi.new("unsigned char[]", crypto_sign_BYTES)
    rc = lib.crypto_sign_ed25519ph_final_create(edph.state, signature, ffi.NULL, sk)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(signature, crypto_sign_BYTES)[:]


def crypto_sign_ed25519ph_final_verify(
    edph: crypto_sign_ed25519ph_state, signature: bytes, pk: bytes
) -> bool:
    """
    Verify a prehashed signature using the public key pk

    :param edph: the ed25519ph state for the data
                 being verified
    :type edph: crypto_sign_ed25519ph_state
    :param signature: the signature being verified
    :type signature: bytes
    :param pk: the ed25519 public part of the signing key
    :type pk: bytes
    :return: True if the signature is valid
    :rtype: boolean
    :raises exc.BadSignatureError: if the signature is not valid
    """
    ensure(
        isinstance(edph, crypto_sign_ed25519ph_state),
        "edph parameter must be a ed25519ph_state object",
        raising=exc.TypeError,
    )
    ensure(
        isinstance(signature, bytes),
        "signature parameter must be a bytes object",
        raising=exc.TypeError,
    )
    ensure(
        len(signature) == crypto_sign_BYTES,
        ("signature must be {} bytes long").format(crypto_sign_BYTES),
        raising=exc.TypeError,
    )
    ensure(
        isinstance(pk, bytes),
        "public key parameter must be a bytes object",
        raising=exc.TypeError,
    )
    ensure(
        len(pk) == crypto_sign_PUBLICKEYBYTES,
        ("public key must be {} bytes long").format(crypto_sign_PUBLICKEYBYTES),
        raising=exc.TypeError,
    )
    rc = lib.crypto_sign_ed25519ph_final_verify(edph.state, signature, pk)
    if rc != 0:
        raise exc.BadSignatureError("Signature was forged or corrupt")

    return True
