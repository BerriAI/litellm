# Copyright 2013-2019 Donald Stufft and individual contributors
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
from typing import NoReturn, TypeVar

from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


crypto_generichash_BYTES: int = lib.crypto_generichash_blake2b_bytes()
crypto_generichash_BYTES_MIN: int = lib.crypto_generichash_blake2b_bytes_min()
crypto_generichash_BYTES_MAX: int = lib.crypto_generichash_blake2b_bytes_max()
crypto_generichash_KEYBYTES: int = lib.crypto_generichash_blake2b_keybytes()
crypto_generichash_KEYBYTES_MIN: int = lib.crypto_generichash_blake2b_keybytes_min()
crypto_generichash_KEYBYTES_MAX: int = lib.crypto_generichash_blake2b_keybytes_max()
crypto_generichash_SALTBYTES: int = lib.crypto_generichash_blake2b_saltbytes()
crypto_generichash_PERSONALBYTES: int = lib.crypto_generichash_blake2b_personalbytes()
crypto_generichash_STATEBYTES: int = lib.crypto_generichash_statebytes()

_OVERLONG = "{0} length greater than {1} bytes"
_TOOBIG = "{0} greater than {1}"


def _checkparams(digest_size: int, key: bytes, salt: bytes, person: bytes) -> None:
    """Check hash parameters"""
    ensure(
        isinstance(key, bytes),
        "Key must be a bytes sequence",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(salt, bytes),
        "Salt must be a bytes sequence",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(person, bytes),
        "Person must be a bytes sequence",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(digest_size, int),
        "Digest size must be an integer number",
        raising=exc.TypeError,
    )

    ensure(
        digest_size <= crypto_generichash_BYTES_MAX,
        _TOOBIG.format("Digest_size", crypto_generichash_BYTES_MAX),
        raising=exc.ValueError,
    )

    ensure(
        len(key) <= crypto_generichash_KEYBYTES_MAX,
        _OVERLONG.format("Key", crypto_generichash_KEYBYTES_MAX),
        raising=exc.ValueError,
    )

    ensure(
        len(salt) <= crypto_generichash_SALTBYTES,
        _OVERLONG.format("Salt", crypto_generichash_SALTBYTES),
        raising=exc.ValueError,
    )

    ensure(
        len(person) <= crypto_generichash_PERSONALBYTES,
        _OVERLONG.format("Person", crypto_generichash_PERSONALBYTES),
        raising=exc.ValueError,
    )


def generichash_blake2b_salt_personal(
    data: bytes,
    digest_size: int = crypto_generichash_BYTES,
    key: bytes = b"",
    salt: bytes = b"",
    person: bytes = b"",
) -> bytes:
    """One shot hash interface

    :param data: the input data to the hash function
    :type data: bytes
    :param digest_size: must be at most
                        :py:data:`.crypto_generichash_BYTES_MAX`;
                        the default digest size is
                        :py:data:`.crypto_generichash_BYTES`
    :type digest_size: int
    :param key: must be at most
                :py:data:`.crypto_generichash_KEYBYTES_MAX` long
    :type key: bytes
    :param salt: must be at most
                 :py:data:`.crypto_generichash_SALTBYTES` long;
                 will be zero-padded if needed
    :type salt: bytes
    :param person: must be at most
                   :py:data:`.crypto_generichash_PERSONALBYTES` long:
                   will be zero-padded if needed
    :type person: bytes
    :return: digest_size long digest
    :rtype: bytes
    """

    _checkparams(digest_size, key, salt, person)

    ensure(
        isinstance(data, bytes),
        "Input data must be a bytes sequence",
        raising=exc.TypeError,
    )

    digest = ffi.new("unsigned char[]", digest_size)

    # both _salt and _personal must be zero-padded to the correct length
    _salt = ffi.new("unsigned char []", crypto_generichash_SALTBYTES)
    _person = ffi.new("unsigned char []", crypto_generichash_PERSONALBYTES)

    ffi.memmove(_salt, salt, len(salt))
    ffi.memmove(_person, person, len(person))

    rc = lib.crypto_generichash_blake2b_salt_personal(
        digest, digest_size, data, len(data), key, len(key), _salt, _person
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)

    return ffi.buffer(digest, digest_size)[:]


_Blake2State = TypeVar("_Blake2State", bound="Blake2State")


class Blake2State:
    """
    Python-level wrapper for the crypto_generichash_blake2b state buffer
    """

    __slots__ = ["_statebuf", "digest_size"]

    def __init__(self, digest_size: int):
        self._statebuf = ffi.new("unsigned char[]", crypto_generichash_STATEBYTES)
        self.digest_size = digest_size

    def __reduce__(self) -> NoReturn:
        """
        Raise the same exception as hashlib's blake implementation
        on copy.copy()
        """
        raise TypeError("can't pickle {} objects".format(self.__class__.__name__))

    def copy(self: _Blake2State) -> _Blake2State:
        _st = self.__class__(self.digest_size)
        ffi.memmove(_st._statebuf, self._statebuf, crypto_generichash_STATEBYTES)
        return _st


def generichash_blake2b_init(
    key: bytes = b"",
    salt: bytes = b"",
    person: bytes = b"",
    digest_size: int = crypto_generichash_BYTES,
) -> Blake2State:
    """
    Create a new initialized blake2b hash state

    :param key: must be at most
                :py:data:`.crypto_generichash_KEYBYTES_MAX` long
    :type key: bytes
    :param salt: must be at most
                 :py:data:`.crypto_generichash_SALTBYTES` long;
                 will be zero-padded if needed
    :type salt: bytes
    :param person: must be at most
                   :py:data:`.crypto_generichash_PERSONALBYTES` long:
                   will be zero-padded if needed
    :type person: bytes
    :param digest_size: must be at most
                        :py:data:`.crypto_generichash_BYTES_MAX`;
                        the default digest size is
                        :py:data:`.crypto_generichash_BYTES`
    :type digest_size: int
    :return: a initialized :py:class:`.Blake2State`
    :rtype: object
    """

    _checkparams(digest_size, key, salt, person)

    state = Blake2State(digest_size)

    # both _salt and _personal must be zero-padded to the correct length
    _salt = ffi.new("unsigned char []", crypto_generichash_SALTBYTES)
    _person = ffi.new("unsigned char []", crypto_generichash_PERSONALBYTES)

    ffi.memmove(_salt, salt, len(salt))
    ffi.memmove(_person, person, len(person))

    rc = lib.crypto_generichash_blake2b_init_salt_personal(
        state._statebuf, key, len(key), digest_size, _salt, _person
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)

    return state


def generichash_blake2b_update(state: Blake2State, data: bytes) -> None:
    """Update the blake2b hash state

    :param state: a initialized Blake2bState object as returned from
                     :py:func:`.crypto_generichash_blake2b_init`
    :type state: :py:class:`.Blake2State`
    :param data:
    :type data: bytes
    """

    ensure(
        isinstance(state, Blake2State),
        "State must be a Blake2State object",
        raising=exc.TypeError,
    )

    ensure(
        isinstance(data, bytes),
        "Input data must be a bytes sequence",
        raising=exc.TypeError,
    )

    rc = lib.crypto_generichash_blake2b_update(state._statebuf, data, len(data))
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)


def generichash_blake2b_final(state: Blake2State) -> bytes:
    """Finalize the blake2b hash state and return the digest.

    :param state: a initialized Blake2bState object as returned from
                     :py:func:`.crypto_generichash_blake2b_init`
    :type state: :py:class:`.Blake2State`
    :return: the blake2 digest of the passed-in data stream
    :rtype: bytes
    """

    ensure(
        isinstance(state, Blake2State),
        "State must be a Blake2State object",
        raising=exc.TypeError,
    )

    _digest = ffi.new("unsigned char[]", crypto_generichash_BYTES_MAX)
    rc = lib.crypto_generichash_blake2b_final(
        state._statebuf, _digest, state.digest_size
    )

    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)
    return ffi.buffer(_digest, state.digest_size)[:]
