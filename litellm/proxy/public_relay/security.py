from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PASSWORD_HASHER = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)
_DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=2$zeOv/S3K/eZaRfxASiRzzg$0tAp0M9aYEMiklP+k/Dii2ExWcKsG8JVWgn5+aDKMFg"
)


@dataclass(frozen=True, slots=True)
class SessionCredentials:
    token: str
    csrf_token: str


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 320 or _EMAIL_PATTERN.fullmatch(normalized) is None:
        raise ValueError("invalid email address")
    return normalized


def validate_password(password: str) -> str:
    if len(password) < 12 or len(password) > 128:
        raise ValueError("password must contain between 12 and 128 characters")
    if (
        password.lower() == password
        or password.upper() == password
        or not any(character.isdigit() for character in password)
    ):
        raise ValueError("password must contain upper-case, lower-case, and numeric characters")
    return password


def hash_password(password: str) -> str:
    return _PASSWORD_HASHER.hash(validate_password(password))


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (InvalidHashError, VerificationError):
        return False


def verify_account_password(password: str, password_hash: str | None) -> bool:
    matches = verify_password(password, password_hash or _DUMMY_PASSWORD_HASH)
    return password_hash is not None and matches


def new_verification_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_verification_code(secret: bytes, purpose: str, email: str, code: str) -> str:
    message = f"{purpose}:{email}:{code}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verification_code_matches(secret: bytes, purpose: str, email: str, code: str, expected: str) -> bool:
    actual = hash_verification_code(secret=secret, purpose=purpose, email=email, code=code)
    return hmac.compare_digest(actual, expected)


def new_session_credentials() -> SessionCredentials:
    return SessionCredentials(token=secrets.token_urlsafe(48), csrf_token=secrets.token_urlsafe(32))


def hash_session_token(secret: bytes, token: str) -> str:
    return hmac.new(secret, token.encode(), hashlib.sha256).hexdigest()
