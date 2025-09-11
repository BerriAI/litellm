# Copyright 2016 Donald Stufft and individual contributors
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


import nacl.exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


has_crypto_shorthash_siphashx24 = bool(lib.PYNACL_HAS_CRYPTO_SHORTHASH_SIPHASHX24)

BYTES: int = lib.crypto_shorthash_siphash24_bytes()
KEYBYTES: int = lib.crypto_shorthash_siphash24_keybytes()

XBYTES = 0
XKEYBYTES = 0

if has_crypto_shorthash_siphashx24:
    XBYTES = lib.crypto_shorthash_siphashx24_bytes()
    XKEYBYTES = lib.crypto_shorthash_siphashx24_keybytes()


def crypto_shorthash_siphash24(data: bytes, key: bytes) -> bytes:
    """Compute a fast, cryptographic quality, keyed hash of the input data

    :param data:
    :type data: bytes
    :param key: len(key) must be equal to
                :py:data:`.KEYBYTES` (16)
    :type key: bytes
    """
    if len(key) != KEYBYTES:
        raise exc.ValueError("Key length must be exactly {} bytes".format(KEYBYTES))
    digest = ffi.new("unsigned char[]", BYTES)
    rc = lib.crypto_shorthash_siphash24(digest, data, len(data), key)

    ensure(rc == 0, raising=exc.RuntimeError)
    return ffi.buffer(digest, BYTES)[:]


def crypto_shorthash_siphashx24(data: bytes, key: bytes) -> bytes:
    """Compute a fast, cryptographic quality, keyed hash of the input data

    :param data:
    :type data: bytes
    :param key: len(key) must be equal to
                :py:data:`.XKEYBYTES` (16)
    :type key: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_shorthash_siphashx24,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    if len(key) != XKEYBYTES:
        raise exc.ValueError("Key length must be exactly {} bytes".format(XKEYBYTES))
    digest = ffi.new("unsigned char[]", XBYTES)
    rc = lib.crypto_shorthash_siphashx24(digest, data, len(data), key)

    ensure(rc == 0, raising=exc.RuntimeError)
    return ffi.buffer(digest, XBYTES)[:]
