# Copyright 2018 Donald Stufft and individual contributors
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

__all__ = [
    "crypto_kx_keypair",
    "crypto_kx_client_session_keys",
    "crypto_kx_server_session_keys",
    "crypto_kx_PUBLIC_KEY_BYTES",
    "crypto_kx_SECRET_KEY_BYTES",
    "crypto_kx_SEED_BYTES",
    "crypto_kx_SESSION_KEY_BYTES",
]

"""
Implementations of client, server key exchange
"""
crypto_kx_PUBLIC_KEY_BYTES: int = lib.crypto_kx_publickeybytes()
crypto_kx_SECRET_KEY_BYTES: int = lib.crypto_kx_secretkeybytes()
crypto_kx_SEED_BYTES: int = lib.crypto_kx_seedbytes()
crypto_kx_SESSION_KEY_BYTES: int = lib.crypto_kx_sessionkeybytes()


def crypto_kx_keypair() -> Tuple[bytes, bytes]:
    """
    Generate a keypair.
    This is a duplicate crypto_box_keypair, but
    is included for api consistency.
    :return: (public_key, secret_key)
    :rtype: (bytes, bytes)
    """
    public_key = ffi.new("unsigned char[]", crypto_kx_PUBLIC_KEY_BYTES)
    secret_key = ffi.new("unsigned char[]", crypto_kx_SECRET_KEY_BYTES)
    res = lib.crypto_kx_keypair(public_key, secret_key)
    ensure(res == 0, "Key generation failed.", raising=exc.CryptoError)

    return (
        ffi.buffer(public_key, crypto_kx_PUBLIC_KEY_BYTES)[:],
        ffi.buffer(secret_key, crypto_kx_SECRET_KEY_BYTES)[:],
    )


def crypto_kx_seed_keypair(seed: bytes) -> Tuple[bytes, bytes]:
    """
    Generate a keypair with a given seed.
    This is functionally the same as crypto_box_seed_keypair, however
    it uses the blake2b hash primitive instead of sha512.
    It is included mainly for api consistency when using crypto_kx.
    :param seed: random seed
    :type seed: bytes
    :return: (public_key, secret_key)
    :rtype: (bytes, bytes)
    """
    public_key = ffi.new("unsigned char[]", crypto_kx_PUBLIC_KEY_BYTES)
    secret_key = ffi.new("unsigned char[]", crypto_kx_SECRET_KEY_BYTES)
    ensure(
        isinstance(seed, bytes) and len(seed) == crypto_kx_SEED_BYTES,
        "Seed must be a {} byte long bytes sequence".format(crypto_kx_SEED_BYTES),
        raising=exc.TypeError,
    )
    res = lib.crypto_kx_seed_keypair(public_key, secret_key, seed)
    ensure(res == 0, "Key generation failed.", raising=exc.CryptoError)

    return (
        ffi.buffer(public_key, crypto_kx_PUBLIC_KEY_BYTES)[:],
        ffi.buffer(secret_key, crypto_kx_SECRET_KEY_BYTES)[:],
    )


def crypto_kx_client_session_keys(
    client_public_key: bytes,
    client_secret_key: bytes,
    server_public_key: bytes,
) -> Tuple[bytes, bytes]:
    """
    Generate session keys for the client.
    :param client_public_key:
    :type client_public_key: bytes
    :param client_secret_key:
    :type client_secret_key: bytes
    :param server_public_key:
    :type server_public_key: bytes
    :return: (rx_key, tx_key)
    :rtype: (bytes, bytes)
    """
    ensure(
        isinstance(client_public_key, bytes)
        and len(client_public_key) == crypto_kx_PUBLIC_KEY_BYTES,
        "Client public key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )
    ensure(
        isinstance(client_secret_key, bytes)
        and len(client_secret_key) == crypto_kx_SECRET_KEY_BYTES,
        "Client secret key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )
    ensure(
        isinstance(server_public_key, bytes)
        and len(server_public_key) == crypto_kx_PUBLIC_KEY_BYTES,
        "Server public key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )

    rx_key = ffi.new("unsigned char[]", crypto_kx_SESSION_KEY_BYTES)
    tx_key = ffi.new("unsigned char[]", crypto_kx_SESSION_KEY_BYTES)
    res = lib.crypto_kx_client_session_keys(
        rx_key, tx_key, client_public_key, client_secret_key, server_public_key
    )
    ensure(
        res == 0,
        "Client session key generation failed.",
        raising=exc.CryptoError,
    )

    return (
        ffi.buffer(rx_key, crypto_kx_SESSION_KEY_BYTES)[:],
        ffi.buffer(tx_key, crypto_kx_SESSION_KEY_BYTES)[:],
    )


def crypto_kx_server_session_keys(
    server_public_key: bytes,
    server_secret_key: bytes,
    client_public_key: bytes,
) -> Tuple[bytes, bytes]:
    """
    Generate session keys for the server.
    :param server_public_key:
    :type server_public_key: bytes
    :param server_secret_key:
    :type server_secret_key: bytes
    :param client_public_key:
    :type client_public_key: bytes
    :return: (rx_key, tx_key)
    :rtype: (bytes, bytes)
    """
    ensure(
        isinstance(server_public_key, bytes)
        and len(server_public_key) == crypto_kx_PUBLIC_KEY_BYTES,
        "Server public key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )
    ensure(
        isinstance(server_secret_key, bytes)
        and len(server_secret_key) == crypto_kx_SECRET_KEY_BYTES,
        "Server secret key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )
    ensure(
        isinstance(client_public_key, bytes)
        and len(client_public_key) == crypto_kx_PUBLIC_KEY_BYTES,
        "Client public key must be a {} bytes long bytes sequence".format(
            crypto_kx_PUBLIC_KEY_BYTES
        ),
        raising=exc.TypeError,
    )

    rx_key = ffi.new("unsigned char[]", crypto_kx_SESSION_KEY_BYTES)
    tx_key = ffi.new("unsigned char[]", crypto_kx_SESSION_KEY_BYTES)
    res = lib.crypto_kx_server_session_keys(
        rx_key, tx_key, server_public_key, server_secret_key, client_public_key
    )
    ensure(
        res == 0,
        "Server session key generation failed.",
        raising=exc.CryptoError,
    )

    return (
        ffi.buffer(rx_key, crypto_kx_SESSION_KEY_BYTES)[:],
        ffi.buffer(tx_key, crypto_kx_SESSION_KEY_BYTES)[:],
    )
