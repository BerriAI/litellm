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


import os
from typing import SupportsBytes, Type, TypeVar

import nacl.bindings
from nacl import encoding

_EncryptedMessage = TypeVar("_EncryptedMessage", bound="EncryptedMessage")


class EncryptedMessage(bytes):
    """
    A bytes subclass that holds a messaged that has been encrypted by a
    :class:`SecretBox`.
    """

    _nonce: bytes
    _ciphertext: bytes

    @classmethod
    def _from_parts(
        cls: Type[_EncryptedMessage],
        nonce: bytes,
        ciphertext: bytes,
        combined: bytes,
    ) -> _EncryptedMessage:
        obj = cls(combined)
        obj._nonce = nonce
        obj._ciphertext = ciphertext
        return obj

    @property
    def nonce(self) -> bytes:
        """
        The nonce used during the encryption of the :class:`EncryptedMessage`.
        """
        return self._nonce

    @property
    def ciphertext(self) -> bytes:
        """
        The ciphertext contained within the :class:`EncryptedMessage`.
        """
        return self._ciphertext


class StringFixer:
    def __str__(self: SupportsBytes) -> str:
        return str(self.__bytes__())


def bytes_as_string(bytes_in: bytes) -> str:
    return bytes_in.decode("ascii")


def random(size: int = 32) -> bytes:
    return os.urandom(size)


def randombytes_deterministic(
    size: int, seed: bytes, encoder: encoding.Encoder = encoding.RawEncoder
) -> bytes:
    """
    Returns ``size`` number of deterministically generated pseudorandom bytes
    from a seed

    :param size: int
    :param seed: bytes
    :param encoder: The encoder class used to encode the produced bytes
    :rtype: bytes
    """
    raw_data = nacl.bindings.randombytes_buf_deterministic(size, seed)

    return encoder.encode(raw_data)
