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

import nacl.bindings
import nacl.encoding

from . import _argon2

ALG = _argon2.ALG_ARGON2I13
STRPREFIX = nacl.bindings.crypto_pwhash_argon2i_STRPREFIX

SALTBYTES = _argon2.SALTBYTES

PASSWD_MIN = _argon2.PASSWD_MIN
PASSWD_MAX = _argon2.PASSWD_MAX

PWHASH_SIZE = _argon2.PWHASH_SIZE

BYTES_MIN = _argon2.BYTES_MIN
BYTES_MAX = _argon2.BYTES_MAX

verify = _argon2.verify

MEMLIMIT_MAX = nacl.bindings.crypto_pwhash_argon2i_MEMLIMIT_MAX
MEMLIMIT_MIN = nacl.bindings.crypto_pwhash_argon2i_MEMLIMIT_MIN
OPSLIMIT_MAX = nacl.bindings.crypto_pwhash_argon2i_OPSLIMIT_MAX
OPSLIMIT_MIN = nacl.bindings.crypto_pwhash_argon2i_OPSLIMIT_MIN

OPSLIMIT_INTERACTIVE = nacl.bindings.crypto_pwhash_argon2i_OPSLIMIT_INTERACTIVE
MEMLIMIT_INTERACTIVE = nacl.bindings.crypto_pwhash_argon2i_MEMLIMIT_INTERACTIVE
OPSLIMIT_SENSITIVE = nacl.bindings.crypto_pwhash_argon2i_OPSLIMIT_SENSITIVE
MEMLIMIT_SENSITIVE = nacl.bindings.crypto_pwhash_argon2i_MEMLIMIT_SENSITIVE

OPSLIMIT_MODERATE = nacl.bindings.crypto_pwhash_argon2i_OPSLIMIT_MODERATE
MEMLIMIT_MODERATE = nacl.bindings.crypto_pwhash_argon2i_MEMLIMIT_MODERATE


def kdf(
    size: int,
    password: bytes,
    salt: bytes,
    opslimit: int = OPSLIMIT_SENSITIVE,
    memlimit: int = MEMLIMIT_SENSITIVE,
    encoder: nacl.encoding.Encoder = nacl.encoding.RawEncoder,
) -> bytes:
    """
    Derive a ``size`` bytes long key from a caller-supplied
    ``password`` and ``salt`` pair using the argon2i
    memory-hard construct.

    the enclosing module provides the constants

        - :py:const:`.OPSLIMIT_INTERACTIVE`
        - :py:const:`.MEMLIMIT_INTERACTIVE`
        - :py:const:`.OPSLIMIT_MODERATE`
        - :py:const:`.MEMLIMIT_MODERATE`
        - :py:const:`.OPSLIMIT_SENSITIVE`
        - :py:const:`.MEMLIMIT_SENSITIVE`

    as a guidance for correct settings.

    :param size: derived key size, must be between
                 :py:const:`.BYTES_MIN` and
                 :py:const:`.BYTES_MAX`
    :type size: int
    :param password: password used to seed the key derivation procedure;
                     it length must be between
                     :py:const:`.PASSWD_MIN` and
                     :py:const:`.PASSWD_MAX`
    :type password: bytes
    :param salt: **RANDOM** salt used in the key derivation procedure;
                 its length must be exactly :py:const:`.SALTBYTES`
    :type salt: bytes
    :param opslimit: the time component (operation count)
                     of the key derivation procedure's computational cost;
                     it must be between
                     :py:const:`.OPSLIMIT_MIN` and
                     :py:const:`.OPSLIMIT_MAX`
    :type opslimit: int
    :param memlimit: the memory occupation component
                     of the key derivation procedure's computational cost;
                     it must be between
                     :py:const:`.MEMLIMIT_MIN` and
                     :py:const:`.MEMLIMIT_MAX`
    :type memlimit: int
    :rtype: bytes

    .. versionadded:: 1.2
    """

    return encoder.encode(
        nacl.bindings.crypto_pwhash_alg(size, password, salt, opslimit, memlimit, ALG)
    )


def str(
    password: bytes,
    opslimit: int = OPSLIMIT_INTERACTIVE,
    memlimit: int = MEMLIMIT_INTERACTIVE,
) -> bytes:
    """
    Hashes a password with a random salt, using the memory-hard
    argon2i construct and returning an ascii string that has all
    the needed info to check against a future password


    The default settings for opslimit and memlimit are those deemed
    correct for the interactive user login case.

    :param bytes password:
    :param int opslimit:
    :param int memlimit:
    :rtype: bytes

    .. versionadded:: 1.2
    """
    return nacl.bindings.crypto_pwhash_str_alg(password, opslimit, memlimit, ALG)
