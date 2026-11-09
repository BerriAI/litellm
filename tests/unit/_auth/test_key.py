import hashlib

import pytest

from litellm._auth.key import hash_token, is_master_key


@pytest.mark.parametrize("token", ["sk-1234", "", "sk-ünïcode"])
def test_hash_token_is_sha256_hex_of_utf8(token: str):
    assert hash_token(token) == hashlib.sha256(token.encode()).hexdigest()


def test_hash_token_differs_per_token():
    assert hash_token("sk-a") != hash_token("sk-b")


@pytest.mark.parametrize(
    ("token", "master_key", "expected"),
    [
        ("sk-master", "sk-master", True),
        ("sk-master", "sk-other", False),
        ("sk-mast", "sk-master", False),
        (None, "sk-master", False),
        ("sk-master", None, False),
        (None, None, False),
        ("sk-mästér", "sk-mästér", True),
        ("sk-mästér", "sk-master", False),
    ],
)
def test_is_master_key_matches_only_the_exact_raw_key(token: str | None, master_key: str | None, expected: bool):
    assert is_master_key(token, master_key) is expected


def test_hashed_master_key_is_not_the_master_key():
    assert is_master_key(hash_token("sk-master"), "sk-master") is False
