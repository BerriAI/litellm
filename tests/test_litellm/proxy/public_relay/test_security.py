import pytest

from litellm.proxy.public_relay.security import (
    hash_auth_token,
    hash_password,
    hash_rate_limit_key,
    normalize_email,
    verify_account_password,
    verify_password,
)


def test_email_is_normalized() -> None:
    assert normalize_email("  PERSON@Example.COM ") == "person@example.com"


@pytest.mark.parametrize("email", ["missing-at.example.com", "@example.com", "person@"])
def test_invalid_email_is_rejected(email: str) -> None:
    with pytest.raises(ValueError):
        normalize_email(email)


def test_argon2_password_round_trip() -> None:
    password_hash = hash_password("A-secure-password-123")

    assert password_hash.startswith("$argon2")
    assert verify_password("A-secure-password-123", password_hash)
    assert not verify_password("A-secure-password-124", password_hash)


def test_missing_account_password_runs_dummy_verification() -> None:
    assert verify_account_password("AnyPassword123", None) is False


def test_auth_and_rate_limit_hashes_are_domain_separated() -> None:
    secret = b"x" * 32
    assert hash_auth_token(secret, "same-value") != hash_rate_limit_key(secret, "same-value")
    assert hash_auth_token(secret, "same-value") == hash_auth_token(secret, "same-value")
