# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

class AESGCM:
    def __init__(self, key: bytes) -> None: ...
    @staticmethod
    def generate_key(key_size: int) -> bytes: ...
    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...

class ChaCha20Poly1305:
    def __init__(self, key: bytes) -> None: ...
    @staticmethod
    def generate_key() -> bytes: ...
    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...

class AESCCM:
    def __init__(self, key: bytes, tag_length: int = 16) -> None: ...
    @staticmethod
    def generate_key(key_size: int) -> bytes: ...
    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...

class AESSIV:
    def __init__(self, key: bytes) -> None: ...
    @staticmethod
    def generate_key(key_size: int) -> bytes: ...
    def encrypt(
        self,
        data: bytes,
        associated_data: list[bytes] | None,
    ) -> bytes: ...
    def decrypt(
        self,
        data: bytes,
        associated_data: list[bytes] | None,
    ) -> bytes: ...

class AESOCB3:
    def __init__(self, key: bytes) -> None: ...
    @staticmethod
    def generate_key(key_size: int) -> bytes: ...
    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...

class AESGCMSIV:
    def __init__(self, key: bytes) -> None: ...
    @staticmethod
    def generate_key(key_size: int) -> bytes: ...
    def encrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
    def decrypt(
        self,
        nonce: bytes,
        data: bytes,
        associated_data: bytes | None,
    ) -> bytes: ...
