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
from typing import Optional

import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.public import (
    PrivateKey as _Curve25519_PrivateKey,
    PublicKey as _Curve25519_PublicKey,
)
from nacl.utils import StringFixer, random


class SignedMessage(bytes):
    """
    A bytes subclass that holds a messaged that has been signed by a
    :class:`SigningKey`.
    """

    _signature: bytes
    _message: bytes

    @classmethod
    def _from_parts(
        cls, signature: bytes, message: bytes, combined: bytes
    ) -> "SignedMessage":
        obj = cls(combined)
        obj._signature = signature
        obj._message = message
        return obj

    @property
    def signature(self) -> bytes:
        """
        The signature contained within the :class:`SignedMessage`.
        """
        return self._signature

    @property
    def message(self) -> bytes:
        """
        The message contained within the :class:`SignedMessage`.
        """
        return self._message


class VerifyKey(encoding.Encodable, StringFixer):
    """
    The public key counterpart to an Ed25519 SigningKey for producing digital
    signatures.

    :param key: [:class:`bytes`] Serialized Ed25519 public key
    :param encoder: A class that is able to decode the `key`
    """

    def __init__(self, key: bytes, encoder: encoding.Encoder = encoding.RawEncoder):
        # Decode the key
        key = encoder.decode(key)
        if not isinstance(key, bytes):
            raise exc.TypeError("VerifyKey must be created from 32 bytes")

        if len(key) != nacl.bindings.crypto_sign_PUBLICKEYBYTES:
            raise exc.ValueError(
                "The key must be exactly %s bytes long"
                % nacl.bindings.crypto_sign_PUBLICKEYBYTES,
            )

        self._key = key

    def __bytes__(self) -> bytes:
        return self._key

    def __hash__(self) -> int:
        return hash(bytes(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return nacl.bindings.sodium_memcmp(bytes(self), bytes(other))

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def verify(
        self,
        smessage: bytes,
        signature: Optional[bytes] = None,
        encoder: encoding.Encoder = encoding.RawEncoder,
    ) -> bytes:
        """
        Verifies the signature of a signed message, returning the message
        if it has not been tampered with else raising
        :class:`~nacl.signing.BadSignatureError`.

        :param smessage: [:class:`bytes`] Either the original messaged or a
            signature and message concated together.
        :param signature: [:class:`bytes`] If an unsigned message is given for
            smessage then the detached signature must be provided.
        :param encoder: A class that is able to decode the secret message and
            signature.
        :rtype: :class:`bytes`
        """
        if signature is not None:
            # If we were given the message and signature separately, validate
            #   signature size and combine them.
            if not isinstance(signature, bytes):
                raise exc.TypeError(
                    "Verification signature must be created from %d bytes"
                    % nacl.bindings.crypto_sign_BYTES,
                )

            if len(signature) != nacl.bindings.crypto_sign_BYTES:
                raise exc.ValueError(
                    "The signature must be exactly %d bytes long"
                    % nacl.bindings.crypto_sign_BYTES,
                )

            smessage = signature + encoder.decode(smessage)
        else:
            # Decode the signed message
            smessage = encoder.decode(smessage)

        return nacl.bindings.crypto_sign_open(smessage, self._key)

    def to_curve25519_public_key(self) -> _Curve25519_PublicKey:
        """
        Converts a :class:`~nacl.signing.VerifyKey` to a
        :class:`~nacl.public.PublicKey`

        :rtype: :class:`~nacl.public.PublicKey`
        """
        raw_pk = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(self._key)
        return _Curve25519_PublicKey(raw_pk)


class SigningKey(encoding.Encodable, StringFixer):
    """
    Private key for producing digital signatures using the Ed25519 algorithm.

    Signing keys are produced from a 32-byte (256-bit) random seed value. This
    value can be passed into the :class:`~nacl.signing.SigningKey` as a
    :func:`bytes` whose length is 32.

    .. warning:: This **must** be protected and remain secret. Anyone who knows
        the value of your :class:`~nacl.signing.SigningKey` or it's seed can
        masquerade as you.

    :param seed: [:class:`bytes`] Random 32-byte value (i.e. private key)
    :param encoder: A class that is able to decode the seed

    :ivar: verify_key: [:class:`~nacl.signing.VerifyKey`] The verify
        (i.e. public) key that corresponds with this signing key.
    """

    def __init__(
        self,
        seed: bytes,
        encoder: encoding.Encoder = encoding.RawEncoder,
    ):
        # Decode the seed
        seed = encoder.decode(seed)
        if not isinstance(seed, bytes):
            raise exc.TypeError("SigningKey must be created from a 32 byte seed")

        # Verify that our seed is the proper size
        if len(seed) != nacl.bindings.crypto_sign_SEEDBYTES:
            raise exc.ValueError(
                "The seed must be exactly %d bytes long"
                % nacl.bindings.crypto_sign_SEEDBYTES
            )

        public_key, secret_key = nacl.bindings.crypto_sign_seed_keypair(seed)

        self._seed = seed
        self._signing_key = secret_key
        self.verify_key = VerifyKey(public_key)

    def __bytes__(self) -> bytes:
        return self._seed

    def __hash__(self) -> int:
        return hash(bytes(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return nacl.bindings.sodium_memcmp(bytes(self), bytes(other))

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    @classmethod
    def generate(cls) -> "SigningKey":
        """
        Generates a random :class:`~nacl.signing.SigningKey` object.

        :rtype: :class:`~nacl.signing.SigningKey`
        """
        return cls(
            random(nacl.bindings.crypto_sign_SEEDBYTES),
            encoder=encoding.RawEncoder,
        )

    def sign(
        self,
        message: bytes,
        encoder: encoding.Encoder = encoding.RawEncoder,
    ) -> SignedMessage:
        """
        Sign a message using this key.

        :param message: [:class:`bytes`] The data to be signed.
        :param encoder: A class that is used to encode the signed message.
        :rtype: :class:`~nacl.signing.SignedMessage`
        """
        raw_signed = nacl.bindings.crypto_sign(message, self._signing_key)

        crypto_sign_BYTES = nacl.bindings.crypto_sign_BYTES
        signature = encoder.encode(raw_signed[:crypto_sign_BYTES])
        message = encoder.encode(raw_signed[crypto_sign_BYTES:])
        signed = encoder.encode(raw_signed)

        return SignedMessage._from_parts(signature, message, signed)

    def to_curve25519_private_key(self) -> _Curve25519_PrivateKey:
        """
        Converts a :class:`~nacl.signing.SigningKey` to a
        :class:`~nacl.public.PrivateKey`

        :rtype: :class:`~nacl.public.PrivateKey`
        """
        sk = self._signing_key
        raw_private = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(sk)
        return _Curve25519_PrivateKey(raw_private)
