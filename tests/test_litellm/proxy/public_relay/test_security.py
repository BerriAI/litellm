import pytest

from litellm.proxy.public_relay.security import (
    hash_password,
    hash_verification_code,
    normalize_email,
    verification_code_matches,
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


def test_verification_hash_binds_purpose_and_email() -> None:
    secret = b"x" * 32
    expected = hash_verification_code(secret, "register", "person@example.com", "123456")

    assert verification_code_matches(secret, "register", "person@example.com", "123456", expected)
    assert not verification_code_matches(secret, "password-reset", "person@example.com", "123456", expected)
