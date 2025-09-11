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


from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


has_crypto_core_ed25519 = bool(lib.PYNACL_HAS_CRYPTO_CORE_ED25519)

crypto_core_ed25519_BYTES = 0
crypto_core_ed25519_SCALARBYTES = 0
crypto_core_ed25519_NONREDUCEDSCALARBYTES = 0

if has_crypto_core_ed25519:
    crypto_core_ed25519_BYTES = lib.crypto_core_ed25519_bytes()
    crypto_core_ed25519_SCALARBYTES = lib.crypto_core_ed25519_scalarbytes()
    crypto_core_ed25519_NONREDUCEDSCALARBYTES = (
        lib.crypto_core_ed25519_nonreducedscalarbytes()
    )


def crypto_core_ed25519_is_valid_point(p: bytes) -> bool:
    """
    Check if ``p`` represents a point on the edwards25519 curve, in canonical
    form, on the main subgroup, and that the point doesn't have a small order.

    :param p: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :return: point validity
    :rtype: bool
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes) and len(p) == crypto_core_ed25519_BYTES,
        "Point must be a crypto_core_ed25519_BYTES long bytes sequence",
        raising=exc.TypeError,
    )

    rc = lib.crypto_core_ed25519_is_valid_point(p)
    return rc == 1


def crypto_core_ed25519_add(p: bytes, q: bytes) -> bytes:
    """
    Add two points on the edwards25519 curve.

    :param p: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type q: bytes
    :return: a point on the edwards25519 curve represented as
             a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes)
        and isinstance(q, bytes)
        and len(p) == crypto_core_ed25519_BYTES
        and len(q) == crypto_core_ed25519_BYTES,
        "Each point must be a {} long bytes sequence".format(
            "crypto_core_ed25519_BYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_BYTES)

    rc = lib.crypto_core_ed25519_add(r, p, q)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(r, crypto_core_ed25519_BYTES)[:]


def crypto_core_ed25519_sub(p: bytes, q: bytes) -> bytes:
    """
    Subtract a point from another on the edwards25519 curve.

    :param p: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
              representing a point on the edwards25519 curve
    :type q: bytes
    :return: a point on the edwards25519 curve represented as
             a :py:data:`.crypto_core_ed25519_BYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes)
        and isinstance(q, bytes)
        and len(p) == crypto_core_ed25519_BYTES
        and len(q) == crypto_core_ed25519_BYTES,
        "Each point must be a {} long bytes sequence".format(
            "crypto_core_ed25519_BYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_BYTES)

    rc = lib.crypto_core_ed25519_sub(r, p, q)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(r, crypto_core_ed25519_BYTES)[:]


def crypto_core_ed25519_scalar_invert(s: bytes) -> bytes:
    """
    Return the multiplicative inverse of integer ``s`` modulo ``L``,
    i.e an integer ``i`` such that ``s * i = 1 (mod L)``, where ``L``
    is the order of the main subgroup.

    Raises a ``exc.RuntimeError`` if ``s`` is the integer zero.

    :param s: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(s, bytes) and len(s) == crypto_core_ed25519_SCALARBYTES,
        "Integer s must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    rc = lib.crypto_core_ed25519_scalar_invert(r, s)
    ensure(rc == 0, "Unexpected library error", raising=exc.RuntimeError)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_negate(s: bytes) -> bytes:
    """
    Return the integer ``n`` such that ``s + n = 0 (mod L)``, where ``L``
    is the order of the main subgroup.

    :param s: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(s, bytes) and len(s) == crypto_core_ed25519_SCALARBYTES,
        "Integer s must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_negate(r, s)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_complement(s: bytes) -> bytes:
    """
    Return the complement of integer ``s`` modulo ``L``, i.e. an integer
    ``c`` such that ``s + c = 1 (mod L)``, where ``L`` is the order of
    the main subgroup.

    :param s: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(s, bytes) and len(s) == crypto_core_ed25519_SCALARBYTES,
        "Integer s must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_complement(r, s)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_add(p: bytes, q: bytes) -> bytes:
    """
    Add integers ``p`` and ``q`` modulo ``L``, where ``L`` is the order of
    the main subgroup.

    :param p: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type q: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes)
        and isinstance(q, bytes)
        and len(p) == crypto_core_ed25519_SCALARBYTES
        and len(q) == crypto_core_ed25519_SCALARBYTES,
        "Each integer must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_add(r, p, q)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_sub(p: bytes, q: bytes) -> bytes:
    """
    Subtract integers ``p`` and ``q`` modulo ``L``, where ``L`` is the
    order of the main subgroup.

    :param p: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type q: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes)
        and isinstance(q, bytes)
        and len(p) == crypto_core_ed25519_SCALARBYTES
        and len(q) == crypto_core_ed25519_SCALARBYTES,
        "Each integer must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_sub(r, p, q)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_mul(p: bytes, q: bytes) -> bytes:
    """
    Multiply integers ``p`` and ``q`` modulo ``L``, where ``L`` is the
    order of the main subgroup.

    :param p: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type p: bytes
    :param q: a :py:data:`.crypto_core_ed25519_SCALARBYTES`
              long bytes sequence representing an integer
    :type q: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(p, bytes)
        and isinstance(q, bytes)
        and len(p) == crypto_core_ed25519_SCALARBYTES
        and len(q) == crypto_core_ed25519_SCALARBYTES,
        "Each integer must be a {} long bytes sequence".format(
            "crypto_core_ed25519_SCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_mul(r, p, q)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]


def crypto_core_ed25519_scalar_reduce(s: bytes) -> bytes:
    """
    Reduce integer ``s`` to ``s`` modulo ``L``, where ``L`` is the order
    of the main subgroup.

    :param s: a :py:data:`.crypto_core_ed25519_NONREDUCEDSCALARBYTES`
              long bytes sequence representing an integer
    :type s: bytes
    :return: an integer represented as a
              :py:data:`.crypto_core_ed25519_SCALARBYTES` long bytes sequence
    :rtype: bytes
    :raises nacl.exceptions.UnavailableError: If called when using a
        minimal build of libsodium.
    """
    ensure(
        has_crypto_core_ed25519,
        "Not available in minimal build",
        raising=exc.UnavailableError,
    )

    ensure(
        isinstance(s, bytes) and len(s) == crypto_core_ed25519_NONREDUCEDSCALARBYTES,
        "Integer s must be a {} long bytes sequence".format(
            "crypto_core_ed25519_NONREDUCEDSCALARBYTES"
        ),
        raising=exc.TypeError,
    )

    r = ffi.new("unsigned char[]", crypto_core_ed25519_SCALARBYTES)

    lib.crypto_core_ed25519_scalar_reduce(r, s)

    return ffi.buffer(r, crypto_core_ed25519_SCALARBYTES)[:]
