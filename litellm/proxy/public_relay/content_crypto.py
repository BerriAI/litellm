from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import cast  # noqa: TID251, RUF100  # Decrypted JSON is validated by its authenticated envelope.

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True, slots=True)
class EncryptedContent:
    nonce_b64: str
    ciphertext_b64: str


def encrypt_content(key: bytes, request_id: str, content: object) -> EncryptedContent:
    nonce = os.urandom(12)
    plaintext = json.dumps(content, separators=(",", ":"), ensure_ascii=False).encode()
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, request_id.encode())
    return EncryptedContent(
        nonce_b64=base64.b64encode(nonce).decode(),
        ciphertext_b64=base64.b64encode(ciphertext).decode(),
    )


def decrypt_content(key: bytes, request_id: str, encrypted: EncryptedContent) -> object:
    nonce = base64.b64decode(encrypted.nonce_b64)
    ciphertext = base64.b64decode(encrypted.ciphertext_b64)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, request_id.encode())
    return cast(object, json.loads(plaintext))  # cast-ok: json.loads returns the authenticated JSON payload.
