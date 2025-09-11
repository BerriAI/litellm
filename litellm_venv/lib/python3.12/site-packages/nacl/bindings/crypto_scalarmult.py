# Copyright 2013-2018 Donald Stufft and individual contributors
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


has_crypto_scalarmult_ed25519 = bool(lib.PYNACL_HAS_CRYPTO_SCALARMULT_ED25519)

crypto_scalarmult_BYTES: int = lib.crypto_scalarmult_bytes()
crypto_scalarmult_SCALARBYTES: int = lib.crypto_scalarmult_scalarbytes()

crypto_scalarmult_ed25519_BYTES = 0
crypto_scalarmult_ed25519_SCALARBYTES = 0

if has_crypto_scalarmult_ed25519:
    crypto_scalarmult_ed25519_BYTES = lib.crypto_scalarmult_ed25519_bytes()
    crypto_scalarmult_ed25519_SCALARBYTES = lib.crypto_scalarmult_ed25519_scalarbytes()


def crypto_scalarmult_base(n: bytes) -> bytes:
    """
    Computes and returns the scalar product of a standard group element and an
    integer ``n``.

    :param n: bytes
    :rtype: bytes
    """
    q = ffi.new("unsigned char[]", crypto_scalarmult_BYTES)

    rc = lib.crypto_scalarmult_base(q, n)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_SCALARBYTES)[:]


def crypto_scalarmult(n: bytes, p: bytes) -> bytes:
    """
    Computes and returns the scalar product of the given group element and an
    integer ``n``.

    :param p: bytes
    :param n: bytes
    :rtype: bytes
    """
    q = ffi.new("unsigned char[]", crypto_scalarmult_BYTES)

    rc = lib.crypto_scalarmult(q, n, p)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_SCALARBYTES)[:]


def crypto_scalarmult_ed25519_base(n: bytes) -> bytes:
    """
    Computes and returns the scalar product of a standard group element and an
    integer ``n`` on the edwards25519 curve.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_scalarmult_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(n, bytes) and len(n) == crypto_scalarmult_ed25519_SCALARBYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    q = ffi.new("unsigned char[]", crypto_scalarmult_ed25519_BYTES)

    rc = lib.crypto_scalarmult_ed25519_base(q, n)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_ed25519_BYTES)[:]


def crypto_scalarmult_ed25519_base_noclamp(n: bytes) -> bytes:
    """
    Computes and returns the scalar product of a standard group element and an
    integer ``n`` on the edwards25519 curve. The integer ``n`` is not clamped.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_scalarmult_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(n, bytes) and len(n) == crypto_scalarmult_ed25519_SCALARBYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    q = ffi.new("unsigned char[]", crypto_scalarmult_ed25519_BYTES)

    rc = lib.crypto_scalarmult_ed25519_base_noclamp(q, n)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_ed25519_BYTES)[:]


def crypto_scalarmult_ed25519(n: bytes, p: bytes) -> bytes:
    """
    Computes and returns the scalar product of a *clamped* integer ``n``
    and the given group element on the edwards25519 curve.
    The scalar is clamped, as done in the public key generation case,
    by setting to zero the bits in position [0, 1, 2, 255] and setting
    to one the bit in position 254.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :param p: a :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_scalarmult_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(n, bytes) and len(n) == crypto_scalarmult_ed25519_SCALARBYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(p, bytes) and len(p) == crypto_scalarmult_ed25519_BYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_BYTES"
        ),
        raising=exc.TypeError,
    )

    q = ffi.new("unsigned char[]", crypto_scalarmult_ed25519_BYTES)

    rc = lib.crypto_scalarmult_ed25519(q, n, p)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_ed25519_BYTES)[:]


def crypto_scalarmult_ed25519_noclamp(n: bytes, p: bytes) -> bytes:
    """
    Computes and returns the scalar product of an integer ``n``
    and the given group element on the edwards25519 curve. The integer
    ``n`` is not clamped.

    :param n: a :py:data:`.crypto_scalarmult_ed25519_SCALARBYTES` long bytes
              sequence representing a scalar
    :type n: bytes
    :param p: a :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: a point on the edwards25519 curve, represented as a
             :py:data:`.crypto_scalarmult_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_scalarmult_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(n, bytes) and len(n) == crypto_scalarmult_ed25519_SCALARBYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    ensure(
        isinstance(p, bytes) and len(p) == crypto_scalarmult_ed25519_BYTES,
        "Input must be a {} long bytes sequence".format(
            "crypto_scalarmult_ed25519_BYTES"
        ),
        raising=exc.TypeError,
    )

    q = ffi.new("unsigned char[]", crypto_scalarmult_ed25519_BYTES)

    rc = lib.crypto_scalarmult_ed25519_noclamp(q, n, p)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(q, crypto_scalarmult_ed25519_BYTES)[:]
