import base64
import json
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AppCrypto:
    def __init__(self, master_key: bytes):
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes for AES-256-GCM")
        self.key = master_key

    def encrypt_json(self, data: dict, aad: Optional[bytes] = None) -> dict:
        aes = AESGCM(self.key)
        nonce = os.urandom(12)
        plaintext = json.dumps(data).encode("utf-8")
        ct = aes.encrypt(nonce, plaintext, aad)
        ciphertext, tag = ct[:-16], ct[-16:]
        return {
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "tag": base64.b64encode(tag).decode(),
        }

    def decrypt_json(self, enc: dict, aad: Optional[bytes] = None) -> dict:
        aes = AESGCM(self.key)
        nonce = base64.b64decode(enc["nonce"])
        ct = base64.b64decode(enc["ciphertext"])
        tag = base64.b64decode(enc["tag"])
        data = aes.decrypt(nonce, ct + tag, aad)
        return json.loads(data.decode())