import os

import pytest

from litellm.litellm_core_utils.app_crypto import AppCrypto


def test_appcrypto_encrypt_decrypt_roundtrip():
    key = os.urandom(32)
    crypto = AppCrypto(key)
    data = {"event": "test", "value": 42}
    aad = b"context"
    enc = crypto.encrypt_json(data, aad=aad)
    dec = crypto.decrypt_json(enc, aad=aad)
    assert dec == data


def test_appcrypto_invalid_key_length():
    with pytest.raises(ValueError, match="32 bytes"):
        AppCrypto(b"short")
