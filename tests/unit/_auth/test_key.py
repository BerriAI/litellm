import hashlib
from typing import Final

import pytest

from litellm._auth.key import HASHED_JWT_PREFIX, hash_token, is_jwt, is_master_key, stored_api_key

JWT: Final = "header.payload.signature"


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


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        (JWT, True),
        ("a..b", True),
        ("two.parts", False),
        ("a.b.c.d", False),
        ("sk-1234", False),
        ("", False),
        (None, False),
    ],
)
def test_is_jwt_means_exactly_three_dot_separated_parts(token: str | None, expected: bool):
    assert is_jwt(token) is expected


@pytest.mark.parametrize("prefix", ["", "Bearer ", "bearer ", "BEARER "])
def test_stored_api_key_hashes_virtual_keys_with_or_without_bearer(prefix: str):
    assert stored_api_key(f"{prefix}sk-1234") == hash_token("sk-1234")


@pytest.mark.parametrize("prefix", ["", "Bearer ", "bearer "])
def test_stored_api_key_hashes_jwts_under_their_own_prefix(prefix: str):
    assert stored_api_key(f"{prefix}{JWT}") == f"{HASHED_JWT_PREFIX}{hash_token(JWT)}"


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("custom-opaque-token", "custom-opaque-token"),
        ("Bearer custom-opaque-token", "custom-opaque-token"),
        ("Basic sk-1234", "Basic sk-1234"),
        ("Bearersk-1234", "Bearersk-1234"),
        ("", ""),
    ],
)
def test_stored_api_key_keeps_other_credentials_after_stripping_bearer(api_key: str, expected: str):
    assert stored_api_key(api_key) == expected


def test_stored_api_key_is_idempotent_for_virtual_keys():
    stored = stored_api_key("sk-1234")
    assert stored_api_key(stored) == stored
