# Copyright 2016-2019 Donald Stufft and individual contributors
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


import binascii
from typing import NoReturn

import nacl.bindings
from nacl.utils import bytes_as_string

BYTES = nacl.bindings.crypto_generichash_BYTES
BYTES_MIN = nacl.bindings.crypto_generichash_BYTES_MIN
BYTES_MAX = nacl.bindings.crypto_generichash_BYTES_MAX
KEYBYTES = nacl.bindings.crypto_generichash_KEYBYTES
KEYBYTES_MIN = nacl.bindings.crypto_generichash_KEYBYTES_MIN
KEYBYTES_MAX = nacl.bindings.crypto_generichash_KEYBYTES_MAX
SALTBYTES = nacl.bindings.crypto_generichash_SALTBYTES
PERSONALBYTES = nacl.bindings.crypto_generichash_PERSONALBYTES

SCRYPT_AVAILABLE = nacl.bindings.has_crypto_pwhash_scryptsalsa208sha256

_b2b_init = nacl.bindings.crypto_generichash_blake2b_init
_b2b_final = nacl.bindings.crypto_generichash_blake2b_final
_b2b_update = nacl.bindings.crypto_generichash_blake2b_update


class blake2b:
    """
    :py:mod:`hashlib` API compatible blake2b algorithm implementation
    """

    MAX_DIGEST_SIZE = BYTES
    MAX_KEY_SIZE = KEYBYTES_MAX
    PERSON_SIZE = PERSONALBYTES
    SALT_SIZE = SALTBYTES

    def __init__(
        self,
        data: bytes = b"",
        digest_size: int = BYTES,
        key: bytes = b"",
        salt: bytes = b"",
        person: bytes = b"",
    ):
        """
        :py:class:`.blake2b` algorithm initializer

        :param data:
        :type data: bytes
        :param int digest_size: the requested digest size; must be
                                at most :py:attr:`.MAX_DIGEST_SIZE`;
                                the default digest size is :py:data:`.BYTES`
        :param key: the key to be set for keyed MAC/PRF usage; if set,
                    the key must be at most :py:data:`.KEYBYTES_MAX` long
        :type key: bytes
        :param salt: a initialization salt at most
                     :py:attr:`.SALT_SIZE` long; it will be zero-padded
                     if needed
        :type salt: bytes
        :param person: a personalization string at most
                       :py:attr:`.PERSONAL_SIZE` long; it will be zero-padded
                       if needed
        :type person: bytes
        """

        self._state = _b2b_init(
            key=key, salt=salt, person=person, digest_size=digest_size
        )
        self._digest_size = digest_size

        if data:
            self.update(data)

    @property
    def digest_size(self) -> int:
        return self._digest_size

    @property
    def block_size(self) -> int:
        return 128

    @property
    def name(self) -> str:
        return "blake2b"

    def update(self, data: bytes) -> None:
        _b2b_update(self._state, data)

    def digest(self) -> bytes:
        _st = self._state.copy()
        return _b2b_final(_st)

    def hexdigest(self) -> str:
        return bytes_as_string(binascii.hexlify(self.digest()))

    def copy(self) -> "blake2b":
        _cp = type(self)(digest_size=self.digest_size)
        _st = self._state.copy()
        _cp._state = _st
        return _cp

    def __reduce__(self) -> NoReturn:
        """
        Raise the same exception as hashlib's blake implementation
        on copy.copy()
        """
        raise TypeError("can't pickle {} objects".format(self.__class__.__name__))


def scrypt(
    password: bytes,
    salt: bytes = b"",
    n: int = 2**20,
    r: int = 8,
    p: int = 1,
    maxmem: int = 2**25,
    dklen: int = 64,
) -> bytes:
    """
    Derive a cryptographic key using the scrypt KDF.

    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    Implements the same signature as the ``hashlib.scrypt`` implemented
    in cpython version 3.6
    """
    return nacl.bindings.crypto_pwhash_scryptsalsa208sha256_ll(
        password, salt, n, r, p, maxmem=maxmem, dklen=dklen
    )
