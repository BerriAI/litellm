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
from typing import ByteString, Optional, Tuple, cast

from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure


crypto_secretstream_xchacha20poly1305_ABYTES: int = (
    lib.crypto_secretstream_xchacha20poly1305_abytes()
)
crypto_secretstream_xchacha20poly1305_HEADERBYTES: int = (
    lib.crypto_secretstream_xchacha20poly1305_headerbytes()
)
crypto_secretstream_xchacha20poly1305_KEYBYTES: int = (
    lib.crypto_secretstream_xchacha20poly1305_keybytes()
)
crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX: int = (
    lib.crypto_secretstream_xchacha20poly1305_messagebytes_max()
)
crypto_secretstream_xchacha20poly1305_STATEBYTES: int = (
    lib.crypto_secretstream_xchacha20poly1305_statebytes()
)


crypto_secretstream_xchacha20poly1305_TAG_MESSAGE: int = (
    lib.crypto_secretstream_xchacha20poly1305_tag_message()
)
crypto_secretstream_xchacha20poly1305_TAG_PUSH: int = (
    lib.crypto_secretstream_xchacha20poly1305_tag_push()
)
crypto_secretstream_xchacha20poly1305_TAG_REKEY: int = (
    lib.crypto_secretstream_xchacha20poly1305_tag_rekey()
)
crypto_secretstream_xchacha20poly1305_TAG_FINAL: int = (
    lib.crypto_secretstream_xchacha20poly1305_tag_final()
)


def crypto_secretstream_xchacha20poly1305_keygen() -> bytes:
    """
    Generate a key for use with
    :func:`.crypto_secretstream_xchacha20poly1305_init_push`.

    """
    keybuf = ffi.new(
        "unsigned char[]",
        crypto_secretstream_xchacha20poly1305_KEYBYTES,
    )
    lib.crypto_secretstream_xchacha20poly1305_keygen(keybuf)
    return ffi.buffer(keybuf)[:]


class crypto_secretstream_xchacha20poly1305_state:
    """
    An object wrapping the crypto_secretstream_xchacha20poly1305 state.

    """

    __slots__ = ["statebuf", "rawbuf", "tagbuf"]

    def __init__(self) -> None:
        """Initialize a clean state object."""
        self.statebuf: ByteString = ffi.new(
            "unsigned char[]",
            crypto_secretstream_xchacha20poly1305_STATEBYTES,
        )

        self.rawbuf: Optional[ByteString] = None
        self.tagbuf: Optional[ByteString] = None


def crypto_secretstream_xchacha20poly1305_init_push(
    state: crypto_secretstream_xchacha20poly1305_state, key: bytes
) -> bytes:
    """
    Initialize a crypto_secretstream_xchacha20poly1305 encryption buffer.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param key: must be
                :data:`.crypto_secretstream_xchacha20poly1305_KEYBYTES` long
    :type key: bytes
    :return: header
    :rtype: bytes

    """
    ensure(
        isinstance(state, crypto_secretstream_xchacha20poly1305_state),
        "State must be a crypto_secretstream_xchacha20poly1305_state object",
        raising=exc.TypeError,
    )
    ensure(
        isinstance(key, bytes),
        "Key must be a bytes sequence",
        raising=exc.TypeError,
    )
    ensure(
        len(key) == crypto_secretstream_xchacha20poly1305_KEYBYTES,
        "Invalid key length",
        raising=exc.ValueError,
    )

    headerbuf = ffi.new(
        "unsigned char []",
        crypto_secretstream_xchacha20poly1305_HEADERBYTES,
    )

    rc = lib.crypto_secretstream_xchacha20poly1305_init_push(
        state.statebuf, headerbuf, key
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)

    return ffi.buffer(headerbuf)[:]


def crypto_secretstream_xchacha20poly1305_push(
    state: crypto_secretstream_xchacha20poly1305_state,
    m: bytes,
    ad: Optional[bytes] = None,
    tag: int = crypto_secretstream_xchacha20poly1305_TAG_MESSAGE,
) -> bytes:
    """
    Add an encrypted message to the secret stream.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param m: the message to encrypt, the maximum length of an individual
              message is
              :data:`.crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX`.
    :type m: bytes
    :param ad: additional data to include in the authentication tag
    :type ad: bytes or None
    :param tag: the message tag, usually
                :data:`.crypto_secretstream_xchacha20poly1305_TAG_MESSAGE` or
                :data:`.crypto_secretstream_xchacha20poly1305_TAG_FINAL`.
    :type tag: int
    :return: ciphertext
    :rtype: bytes

    """
    ensure(
        isinstance(state, crypto_secretstream_xchacha20poly1305_state),
        "State must be a crypto_secretstream_xchacha20poly1305_state object",
        raising=exc.TypeError,
    )
    ensure(isinstance(m, bytes), "Message is not bytes", raising=exc.TypeError)
    ensure(
        len(m) <= crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX,
        "Message is too long",
        raising=exc.ValueError,
    )
    ensure(
        ad is None or isinstance(ad, bytes),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    clen = len(m) + crypto_secretstream_xchacha20poly1305_ABYTES
    if state.rawbuf is None or len(state.rawbuf) < clen:
        state.rawbuf = ffi.new("unsigned char[]", clen)

    if ad is None:
        ad = ffi.NULL
        adlen = 0
    else:
        adlen = len(ad)

    rc = lib.crypto_secretstream_xchacha20poly1305_push(
        state.statebuf,
        state.rawbuf,
        ffi.NULL,
        m,
        len(m),
        ad,
        adlen,
        tag,
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)

    return ffi.buffer(state.rawbuf, clen)[:]


def crypto_secretstream_xchacha20poly1305_init_pull(
    state: crypto_secretstream_xchacha20poly1305_state,
    header: bytes,
    key: bytes,
) -> None:
    """
    Initialize a crypto_secretstream_xchacha20poly1305 decryption buffer.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param header: must be
                :data:`.crypto_secretstream_xchacha20poly1305_HEADERBYTES` long
    :type header: bytes
    :param key: must be
                :data:`.crypto_secretstream_xchacha20poly1305_KEYBYTES` long
    :type key: bytes

    """
    ensure(
        isinstance(state, crypto_secretstream_xchacha20poly1305_state),
        "State must be a crypto_secretstream_xchacha20poly1305_state object",
        raising=exc.TypeError,
    )
    ensure(
        isinstance(header, bytes),
        "Header must be a bytes sequence",
        raising=exc.TypeError,
    )
    ensure(
        len(header) == crypto_secretstream_xchacha20poly1305_HEADERBYTES,
        "Invalid header length",
        raising=exc.ValueError,
    )
    ensure(
        isinstance(key, bytes),
        "Key must be a bytes sequence",
        raising=exc.TypeError,
    )
    ensure(
        len(key) == crypto_secretstream_xchacha20poly1305_KEYBYTES,
        "Invalid key length",
        raising=exc.ValueError,
    )

    if state.tagbuf is None:
        state.tagbuf = ffi.new("unsigned char *")

    rc = lib.crypto_secretstream_xchacha20poly1305_init_pull(
        state.statebuf, header, key
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)


def crypto_secretstream_xchacha20poly1305_pull(
    state: crypto_secretstream_xchacha20poly1305_state,
    c: bytes,
    ad: Optional[bytes] = None,
) -> Tuple[bytes, int]:
    """
    Read a decrypted message from the secret stream.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state
    :param c: the ciphertext to decrypt, the maximum length of an individual
              ciphertext is
              :data:`.crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX` +
              :data:`.crypto_secretstream_xchacha20poly1305_ABYTES`.
    :type c: bytes
    :param ad: additional data to include in the authentication tag
    :type ad: bytes or None
    :return: (message, tag)
    :rtype: (bytes, int)

    """
    ensure(
        isinstance(state, crypto_secretstream_xchacha20poly1305_state),
        "State must be a crypto_secretstream_xchacha20poly1305_state object",
        raising=exc.TypeError,
    )
    ensure(
        state.tagbuf is not None,
        (
            "State must be initialized using "
            "crypto_secretstream_xchacha20poly1305_init_pull"
        ),
        raising=exc.ValueError,
    )
    ensure(
        isinstance(c, bytes),
        "Ciphertext is not bytes",
        raising=exc.TypeError,
    )
    ensure(
        len(c) >= crypto_secretstream_xchacha20poly1305_ABYTES,
        "Ciphertext is too short",
        raising=exc.ValueError,
    )
    ensure(
        len(c)
        <= (
            crypto_secretstream_xchacha20poly1305_MESSAGEBYTES_MAX
            + crypto_secretstream_xchacha20poly1305_ABYTES
        ),
        "Ciphertext is too long",
        raising=exc.ValueError,
    )
    ensure(
        ad is None or isinstance(ad, bytes),
        "Additional data must be bytes or None",
        raising=exc.TypeError,
    )

    mlen = len(c) - crypto_secretstream_xchacha20poly1305_ABYTES
    if state.rawbuf is None or len(state.rawbuf) < mlen:
        state.rawbuf = ffi.new("unsigned char[]", mlen)

    if ad is None:
        ad = ffi.NULL
        adlen = 0
    else:
        adlen = len(ad)

    rc = lib.crypto_secretstream_xchacha20poly1305_pull(
        state.statebuf,
        state.rawbuf,
        ffi.NULL,
        state.tagbuf,
        c,
        len(c),
        ad,
        adlen,
    )
    ensure(rc == 0, "Unexpected failure", raising=exc.RuntimeError)

    # Cast safety: we `ensure` above that `state.tagbuf is not None`.
    return (
        ffi.buffer(state.rawbuf, mlen)[:],
        int(cast(bytes, state.tagbuf)[0]),
    )


def crypto_secretstream_xchacha20poly1305_rekey(
    state: crypto_secretstream_xchacha20poly1305_state,
) -> None:
    """
    Explicitly change the encryption key in the stream.

    Normally the stream is re-keyed as needed or an explicit ``tag`` of
    :data:`.crypto_secretstream_xchacha20poly1305_TAG_REKEY` is added to a
    message to ensure forward secrecy, but this method can be used instead
    if the re-keying is controlled without adding the tag.

    :param state: a secretstream state object
    :type state: crypto_secretstream_xchacha20poly1305_state

    """
    ensure(
        isinstance(state, crypto_secretstream_xchacha20poly1305_state),
        "State must be a crypto_secretstream_xchacha20poly1305_state object",
        raising=exc.TypeError,
    )
    lib.crypto_secretstream_xchacha20poly1305_rekey(state.statebuf)
