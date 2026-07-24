import pytest
from cryptography.exceptions import InvalidTag

from litellm.proxy.public_relay.content_crypto import decrypt_content, encrypt_content


def test_content_encryption_round_trip() -> None:
    key = bytes(range(32))
    content = {"request": {"messages": [{"role": "user", "content": "secret"}]}, "response": "answer"}

    encrypted = encrypt_content(key, "request-1", content)

    assert decrypt_content(key, "request-1", encrypted) == content
    assert "secret" not in encrypted.ciphertext_b64


def test_content_is_bound_to_request_id() -> None:
    key = bytes(range(32))
    encrypted = encrypt_content(key, "request-1", {"value": "secret"})

    with pytest.raises(InvalidTag):
        decrypt_content(key, "request-2", encrypted)
