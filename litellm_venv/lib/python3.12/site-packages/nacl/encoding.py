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
import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type


# TODO: when the minimum supported version of Python is 3.8, we can import
# Protocol from typing, and replace Encoder with a Protocol instead.
class _Encoder(metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def encode(data: bytes) -> bytes:
        """Transform raw data to encoded data."""

    @staticmethod
    @abstractmethod
    def decode(data: bytes) -> bytes:
        """Transform encoded data back to raw data.

        Decoding after encoding should be a no-op, i.e. `decode(encode(x)) == x`.
        """


# Functions that use encoders are passed a subclass of _Encoder, not an instance
# (because the methods are all static). Let's gloss over that detail by defining
# an alias for Type[_Encoder].
Encoder = Type[_Encoder]


class RawEncoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return data

    @staticmethod
    def decode(data: bytes) -> bytes:
        return data


class HexEncoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return binascii.hexlify(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return binascii.unhexlify(data)


class Base16Encoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.b16encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.b16decode(data)


class Base32Encoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.b32encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.b32decode(data)


class Base64Encoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.b64encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.b64decode(data)


class URLSafeBase64Encoder(_Encoder):
    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.urlsafe_b64encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.urlsafe_b64decode(data)


class Encodable:
    def encode(self: SupportsBytes, encoder: Encoder = RawEncoder) -> bytes:
        return encoder.encode(bytes(self))
