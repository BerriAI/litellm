# Copyright 2013-2017 Donald Stufft and individual contributors
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


def sodium_memcmp(inp1: bytes, inp2: bytes) -> bool:
    """
    Compare contents of two memory regions in constant time
    """
    ensure(isinstance(inp1, bytes), raising=exc.TypeError)
    ensure(isinstance(inp2, bytes), raising=exc.TypeError)

    ln = max(len(inp1), len(inp2))

    buf1 = ffi.new("char []", ln)
    buf2 = ffi.new("char []", ln)

    ffi.memmove(buf1, inp1, len(inp1))
    ffi.memmove(buf2, inp2, len(inp2))

    eqL = len(inp1) == len(inp2)
    eqC = lib.sodium_memcmp(buf1, buf2, ln) == 0

    return eqL and eqC


def sodium_pad(s: bytes, blocksize: int) -> bytes:
    """
    Pad the input bytearray ``s`` to a multiple of ``blocksize``
    using the ISO/IEC 7816-4 algorithm

    :param s: input bytes string
    :type s: bytes
    :param blocksize:
    :type blocksize: int
    :return: padded string
    :rtype: bytes
    """
    ensure(isinstance(s, bytes), raising=exc.TypeError)
    ensure(isinstance(blocksize, int), raising=exc.TypeError)
    if blocksize <= 0:
        raise exc.ValueError
    s_len = len(s)
    m_len = s_len + blocksize
    buf = ffi.new("unsigned char []", m_len)
    p_len = ffi.new("size_t []", 1)
    ffi.memmove(buf, s, s_len)
    rc = lib.sodium_pad(p_len, buf, s_len, blocksize, m_len)
    ensure(rc == 0, "Padding failure", raising=exc.CryptoError)
    return ffi.buffer(buf, p_len[0])[:]


def sodium_unpad(s: bytes, blocksize: int) -> bytes:
    """
    Remove ISO/IEC 7816-4 padding from the input byte array ``s``

    :param s: input bytes string
    :type s: bytes
    :param blocksize:
    :type blocksize: int
    :return: unpadded string
    :rtype: bytes
    """
    ensure(isinstance(s, bytes), raising=exc.TypeError)
    ensure(isinstance(blocksize, int), raising=exc.TypeError)
    s_len = len(s)
    u_len = ffi.new("size_t []", 1)
    rc = lib.sodium_unpad(u_len, s, s_len, blocksize)
    if rc != 0:
        raise exc.CryptoError("Unpadding failure")
    return s[: u_len[0]]


def sodium_increment(inp: bytes) -> bytes:
    """
    Increment the value of a byte-sequence interpreted
    as the little-endian representation of a unsigned big integer.

    :param inp: input bytes buffer
    :type inp: bytes
    :return: a byte-sequence representing, as a little-endian
             unsigned big integer, the value ``to_int(inp)``
             incremented by one.
    :rtype: bytes

    """
    ensure(isinstance(inp, bytes), raising=exc.TypeError)

    ln = len(inp)
    buf = ffi.new("unsigned char []", ln)

    ffi.memmove(buf, inp, ln)

    lib.sodium_increment(buf, ln)

    return ffi.buffer(buf, ln)[:]


def sodium_add(a: bytes, b: bytes) -> bytes:
    """
    Given a couple of *same-sized* byte sequences, interpreted as the
    little-endian representation of two unsigned integers, compute
    the modular addition of the represented values, in constant time for
    a given common length of the byte sequences.

    :param a: input bytes buffer
    :type a: bytes
    :param b: input bytes buffer
    :type b: bytes
    :return: a byte-sequence representing, as a little-endian big integer,
             the integer value of ``(to_int(a) + to_int(b)) mod 2^(8*len(a))``
    :rtype: bytes
    """
    ensure(isinstance(a, bytes), raising=exc.TypeError)
    ensure(isinstance(b, bytes), raising=exc.TypeError)
    ln = len(a)
    ensure(len(b) == ln, raising=exc.TypeError)

    buf_a = ffi.new("unsigned char []", ln)
    buf_b = ffi.new("unsigned char []", ln)

    ffi.memmove(buf_a, a, ln)
    ffi.memmove(buf_b, b, ln)

    lib.sodium_add(buf_a, buf_b, ln)

    return ffi.buffer(buf_a, ln)[:]
