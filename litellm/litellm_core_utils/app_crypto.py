import base64
import json
import os
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AppCrypto:
    def __init__(self, master_key: bytes):
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes for AES-256-GCM")
        self.key = master_key

    def encrypt_json(self, data: dict, aad: bytes | None = None) -> dict:
        aes: Final = AESGCM(self.key)
        nonce: Final = os.urandom(12)
        plaintext: Final = json.dumps(data).encode("utf-8")
        ct: Final = aes.encrypt(nonce, plaintext, aad)
        ciphertext, tag = ct[:-16], ct[-16:]
        return {
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag": base64.b64encode(tag).decode(),
        }

    def decrypt_json(self, enc: dict, aad: bytes | None = None) -> dict:
        aes: Final = AESGCM(self.key)
        nonce: Final = base64.b64decode(enc["nonce"])
        ct: Final = base64.b64decode(enc["ciphertext"])
        tag: Final = base64.b64decode(enc["tag"])
        data: Final = aes.decrypt(nonce, ct + tag, aad)
        return json.loads(data.decode())
