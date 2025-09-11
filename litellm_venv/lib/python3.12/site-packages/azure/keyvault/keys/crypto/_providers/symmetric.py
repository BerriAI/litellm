# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
from typing import TYPE_CHECKING

from .local_provider import LocalCryptographyProvider
from .._internal import SymmetricKey
from ... import KeyOperation, KeyType

if TYPE_CHECKING:
    from .local_provider import Algorithm
    from .._internal import Key
    from ... import JsonWebKey


class SymmetricCryptographyProvider(LocalCryptographyProvider):
    def _get_internal_key(self, key: "JsonWebKey") -> "Key":
        if key.kty not in (KeyType.oct, KeyType.oct_hsm):  # type: ignore[attr-defined]
            raise ValueError('"key" must be an oct or oct-HSM (symmetric) key')
        return SymmetricKey.from_jwk(key)

    def supports(self, operation: KeyOperation, algorithm: "Algorithm") -> bool:
        if operation in (KeyOperation.decrypt, KeyOperation.encrypt):
            return algorithm in self._internal_key.supported_encryption_algorithms
        if operation in (KeyOperation.unwrap_key, KeyOperation.wrap_key):
            return algorithm in self._internal_key.supported_key_wrap_algorithms
        return False
