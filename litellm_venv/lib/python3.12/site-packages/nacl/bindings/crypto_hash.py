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


from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


# crypto_hash_BYTES = lib.crypto_hash_bytes()
crypto_hash_BYTES: int = lib.crypto_hash_sha512_bytes()
crypto_hash_sha256_BYTES: int = lib.crypto_hash_sha256_bytes()
crypto_hash_sha512_BYTES: int = lib.crypto_hash_sha512_bytes()


def crypto_hash(message: bytes) -> bytes:
    """
    Hashes and returns the message ``message``.

    :param message: bytes
    :rtype: bytes
    """
    digest = ffi.new("unsigned char[]", crypto_hash_BYTES)
    rc = lib.crypto_hash(digest, message, len(message))
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)
    return ffi.buffer(digest, crypto_hash_BYTES)[:]


def crypto_hash_sha256(message: bytes) -> bytes:
    """
    Hashes and returns the message ``message``.

    :param message: bytes
    :rtype: bytes
    """
    digest = ffi.new("unsigned char[]", crypto_hash_sha256_BYTES)
    rc = lib.crypto_hash_sha256(digest, message, len(message))
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)
    return ffi.buffer(digest, crypto_hash_sha256_BYTES)[:]


def crypto_hash_sha512(message: bytes) -> bytes:
    """
    Hashes and returns the message ``message``.

    :param message: bytes
    :rtype: bytes
    """
    digest = ffi.new("unsigned char[]", crypto_hash_sha512_BYTES)
    rc = lib.crypto_hash_sha512(digest, message, len(message))
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)
    return ffi.buffer(digest, crypto_hash_sha512_BYTES)[:]
