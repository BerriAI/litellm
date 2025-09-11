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
from typing import cast

import nacl.bindings
import nacl.encoding
from nacl import exceptions as exc
from nacl.exceptions import ensure

_strbytes_plus_one = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_STRBYTES

AVAILABLE = nacl.bindings.has_crypto_pwhash_scryptsalsa208sha256

STRPREFIX = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_STRPREFIX

SALTBYTES = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_SALTBYTES

PASSWD_MIN = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_PASSWD_MIN
PASSWD_MAX = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_PASSWD_MAX

PWHASH_SIZE = _strbytes_plus_one - 1

BYTES_MIN = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_BYTES_MIN
BYTES_MAX = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_BYTES_MAX

MEMLIMIT_MIN = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_MIN
MEMLIMIT_MAX = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_MAX
OPSLIMIT_MIN = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_MIN
OPSLIMIT_MAX = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_MAX

OPSLIMIT_INTERACTIVE = (
    nacl.bindings.crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_INTERACTIVE
)
MEMLIMIT_INTERACTIVE = (
    nacl.bindings.crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_INTERACTIVE
)
OPSLIMIT_SENSITIVE = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_SENSITIVE
MEMLIMIT_SENSITIVE = nacl.bindings.crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_SENSITIVE

OPSLIMIT_MODERATE = 8 * OPSLIMIT_INTERACTIVE
MEMLIMIT_MODERATE = 8 * MEMLIMIT_INTERACTIVE


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
    ``password`` and ``salt`` pair using the scryptsalsa208sha256
    memory-hard construct.


    the enclosing module provides the constants

        - :py:const:`.OPSLIMIT_INTERACTIVE`
        - :py:const:`.MEMLIMIT_INTERACTIVE`
        - :py:const:`.OPSLIMIT_SENSITIVE`
        - :py:const:`.MEMLIMIT_SENSITIVE`
        - :py:const:`.OPSLIMIT_MODERATE`
        - :py:const:`.MEMLIMIT_MODERATE`

    as a guidance for correct settings respectively for the
    interactive login and the long term key protecting sensitive data
    use cases.

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
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    .. versionadded:: 1.2
    """
    ensure(
        AVAILABLE,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        len(salt) == SALTBYTES,
        "The salt must be exactly %s, not %s bytes long" % (SALTBYTES, len(salt)),
        raising=exc.ValueError,
    )

    n_log2, r, p = nacl.bindings.nacl_bindings_pick_scrypt_params(opslimit, memlimit)
    maxmem = memlimit + (2**16)

    return encoder.encode(
        nacl.bindings.crypto_pwhash_scryptsalsa208sha256_ll(
            password,
            salt,
            # Cast safety: n_log2 is a positive integer, and so 2 ** n_log2 is also
            # a positive integer. Mypy+typeshed can't deduce this, because there's no
            # way to for them to know that n_log2: int is positive.
            cast(int, 2**n_log2),
            r,
            p,
            maxmem=maxmem,
            dklen=size,
        )
    )


def str(
    password: bytes,
    opslimit: int = OPSLIMIT_INTERACTIVE,
    memlimit: int = MEMLIMIT_INTERACTIVE,
) -> bytes:
    """
    Hashes a password with a random salt, using the memory-hard
    scryptsalsa208sha256 construct and returning an ascii string
    that has all the needed info to check against a future password

    The default settings for opslimit and memlimit are those deemed
    correct for the interactive user login case.

    :param bytes password:
    :param int opslimit:
    :param int memlimit:
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    .. versionadded:: 1.2
    """
    ensure(
        AVAILABLE,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    return nacl.bindings.crypto_pwhash_scryptsalsa208sha256_str(
        password, opslimit, memlimit
    )


def verify(password_hash: bytes, password: bytes) -> bool:
    """
    Takes the output of scryptsalsa208sha256 and compares it against
    a user provided password to see if they are the same

    :param password_hash: bytes
    :param password: bytes
    :rtype: boolean
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.

    .. versionadded:: 1.2
    """
    ensure(
        AVAILABLE,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        len(password_hash) == PWHASH_SIZE,
        "The password hash must be exactly %s bytes long"
        % nacl.bindings.crypto_pwhash_scryptsalsa208sha256_STRBYTES,
        raising=exc.ValueError,
    )

    return nacl.bindings.crypto_pwhash_scryptsalsa208sha256_str_verify(
        password_hash, password
    )
