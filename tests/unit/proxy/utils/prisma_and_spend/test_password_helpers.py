"""Pin password/token helper behavior.

Symbols pinned here:
  - ``hash_token``
  - ``hash_password``
  - ``verify_password``
  - ``run_off_loop_if_fips``
  - ``migrate_passwords_to_scrypt_async``
  - ``_hash_token_if_needed``
  - ``PrismaClient._is_sha256_hex`` (a nested helper inside
    ``migrate_passwords_to_scrypt_async``; the pin list labels it under the
    PrismaClient health cluster as a documentation artifact)
"""

from __future__ import annotations

import asyncio
import hashlib
import ssl
import threading
from types import SimpleNamespace
from typing import Final, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import (
    hash_token_if_needed,
    hash_password,
    hash_token,
    migrate_passwords_to_scrypt_async,
    run_off_loop_if_fips,
    verify_password,
)

PASSWORD: Final = "Seed-Lamp-4-River"
PBKDF2_ROW: Final = "pbkdf2:AAECAwQFBgcICQoLDA0OD+uxhNqRo/3ZFLEaKARqFBT+N3Kqek/SGM/5AvocSN0T"
EIGHT_CHARACTER_PBKDF2_ROW: Final = "pbkdf2:AAECAwQFBgcICQoLDA0ODxhpezknSiGV0cY8wgaSXYf/xzxcPw49uGdOt/TBvPPH"
SEVEN_CHARACTER_PBKDF2_ROW: Final = "pbkdf2:AAECAwQFBgcICQoLDA0ODxFIwl/XKIAeyWEnfqz8H9vNQX4uo7/u1HiJhB7Ro+N7"
SCRYPT_ROW: Final = "scrypt:AAECAwQFBgcICQoLDA0OD7hIwlM/k8ytnffjFDt6AuO+BvqtmZvDkWXJP+A3bf0N"
SHA256_ROW: Final = hashlib.sha256(PASSWORD.encode()).hexdigest()


def test_hash_token_returns_sha256_hex_of_input() -> None:
    token = "sk-abcDEF12345"
    result = hash_token(token)
    expected = hashlib.sha256(token.encode()).hexdigest()
    actual = {
        "len": len(result),
        "hex": all(c in "0123456789abcdef" for c in result),
        "hash": result,
        "matches_sha256": result == expected,
    }
    assert actual == {
        "len": 64,
        "hex": True,
        "hash": expected,
        "matches_sha256": True,
    }


def test_hash_token_empty_string_still_hashes() -> None:
    result = hash_token("")
    assert result == hashlib.sha256(b"").hexdigest()


def test_hash_token_raises_for_non_string() -> None:
    with pytest.raises(AttributeError):
        hash_token(None)  # type: ignore[arg-type]


def test_hash_password_uses_scrypt_prefix() -> None:
    h = hash_password("hunter2")
    fields = {
        "prefix": h[:7],
        "min_length": len(h) > 60,
        "verifies_self": verify_password("hunter2", h),
        "rejects_other": verify_password("hunter3", h),
    }
    assert fields == {
        "prefix": "scrypt:",
        "min_length": True,
        "verifies_self": True,
        "rejects_other": False,
    }


def test_hash_password_returns_distinct_hashes_per_call() -> None:
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_hash_password_error_for_non_string_raises() -> None:
    with pytest.raises(AttributeError):
        hash_password(None)  # type: ignore[arg-type]


def test_verify_password_sha256_legacy_path() -> None:
    plaintext = "legacy-pass"
    sha = hashlib.sha256(plaintext.encode()).hexdigest()
    matrix = {
        "correct": verify_password(plaintext, sha),
        "wrong": verify_password("other", sha),
        "non_hex_short": verify_password(plaintext, "not-hex"),
        "empty_stored": verify_password(plaintext, ""),
    }
    assert matrix == {
        "correct": True,
        "wrong": False,
        "non_hex_short": False,
        "empty_stored": False,
    }


def test_verify_password_scrypt_malformed_returns_false() -> None:
    assert verify_password("anything", "scrypt:not-base64") is False


def test_verify_password_unknown_format_returns_false() -> None:
    assert verify_password("x", "plaintext-not-supported") is False


def test_hash_token_if_needed_handles_sk_prefix() -> None:
    plain = "sk-secret-xyz"
    already_hashed = hashlib.sha256(plain.encode()).hexdigest()
    not_a_secret = "token-without-sk-prefix"
    actual = {
        "sk_input_is_hashed": hash_token_if_needed(plain) == already_hashed,
        "non_sk_passthrough": hash_token_if_needed(not_a_secret) == not_a_secret,
        "double_hash_stable": hash_token_if_needed(already_hashed) == already_hashed,
    }
    assert actual == {
        "sk_input_is_hashed": True,
        "non_sk_passthrough": True,
        "double_hash_stable": True,
    }


def test_hash_token_if_needed_error_on_non_string() -> None:
    with pytest.raises(AttributeError):
        hash_token_if_needed(None)  # pyright: ignore[reportArgumentType]  # intentional invalid input checks the error


# ---------------------------------------------------------------------------
# migrate_passwords_to_scrypt_async — pins behavior of the nested
# ``_is_sha256_hex`` helper too: scrypt-prefixed and sha256-hex rows are
# left alone, plaintext rows are upgraded in place.
# ---------------------------------------------------------------------------


def _make_user(user_id: str, password) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, password=password)


@pytest.mark.asyncio
async def test_migrate_passwords_skips_when_no_plaintext() -> None:
    pc = MagicMock()
    pc.db = MagicMock()
    sha = hashlib.sha256(b"already-hashed").hexdigest()
    pc.db.litellm_usertable.find_many = AsyncMock(
        return_value=[
            _make_user("a", "scrypt:abc"),
            _make_user("b", sha),
        ]
    )
    pc.db.litellm_usertable.update = AsyncMock()

    result = await migrate_passwords_to_scrypt_async(pc)
    outcome = {
        "message": result,
        "updates": pc.db.litellm_usertable.update.await_count,
        "find_called": pc.db.litellm_usertable.find_many.await_count,
        "fetch_filter": pc.db.litellm_usertable.find_many.await_args.kwargs["where"],
    }
    assert outcome == {
        "message": "No plaintext passwords found",
        "updates": 0,
        "find_called": 1,
        "fetch_filter": {"password": {"not": None}},
    }


@pytest.mark.asyncio
async def test_migrate_passwords_upgrades_only_plaintext_rows() -> None:
    pc = MagicMock()
    pc.db = MagicMock()
    users: List[SimpleNamespace] = [
        _make_user("plaintext-user-1", "plain-1"),
        _make_user("plaintext-user-2", "plain-2"),
        _make_user("scrypt-user", "scrypt:already"),
        _make_user(
            "sha-user",
            hashlib.sha256(b"alreadyhashed").hexdigest(),
        ),
        _make_user("null-pw", None),
    ]
    pc.db.litellm_usertable.find_many = AsyncMock(return_value=users)
    pc.db.litellm_usertable.update = AsyncMock()

    result = await migrate_passwords_to_scrypt_async(pc)

    updated_user_ids = sorted(
        call.kwargs["where"]["user_id"] for call in pc.db.litellm_usertable.update.await_args_list
    )
    new_password_prefixes = sorted(
        call.kwargs["data"]["password"][:7] for call in pc.db.litellm_usertable.update.await_args_list
    )
    outcome = {
        "message": result,
        "update_count": pc.db.litellm_usertable.update.await_count,
        "updated_ids": updated_user_ids,
        "all_scrypt_prefixed": new_password_prefixes,
    }
    assert outcome == {
        "message": "Migrated 2 plaintext passwords to scrypt",
        "update_count": 2,
        "updated_ids": ["plaintext-user-1", "plaintext-user-2"],
        "all_scrypt_prefixed": ["scrypt:", "scrypt:"],
    }


@pytest.mark.asyncio
async def test_migrate_passwords_raises_on_db_failure() -> None:
    pc = MagicMock()
    pc.db = MagicMock()
    pc.db.litellm_usertable.find_many = AsyncMock(side_effect=RuntimeError("db unavailable"))
    with pytest.raises(RuntimeError, match="db unavailable"):
        await migrate_passwords_to_scrypt_async(pc)


@pytest.mark.usefixtures("fips_openssl")
def test_fips_hash_is_pbkdf2_sha256_at_600k_salted_by_openssl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ssl, "RAND_bytes", lambda size: bytes(range(size)))
    assert hash_password(PASSWORD) == PBKDF2_ROW


@pytest.mark.usefixtures("fips_openssl")
def test_fips_hash_salts_every_row_and_verifies() -> None:
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second
    assert verify_password(PASSWORD, first)


@pytest.mark.usefixtures("fips_openssl")
def test_fips_verify_needs_the_right_password_of_at_least_eight_characters() -> None:
    outcome = {
        "right": verify_password(PASSWORD, PBKDF2_ROW),
        "wrong": verify_password("Seed-Lamp-4-Rivet", PBKDF2_ROW),
        "lone_surrogate": verify_password("\ud800" + PASSWORD, PBKDF2_ROW),
        "eight_characters": verify_password("Seed-Lam", EIGHT_CHARACTER_PBKDF2_ROW),
        "seven_characters": verify_password("Seed-La", SEVEN_CHARACTER_PBKDF2_ROW),
    }
    assert outcome == {
        "right": True,
        "wrong": False,
        "lone_surrogate": False,
        "eight_characters": True,
        "seven_characters": False,
    }


@pytest.mark.usefixtures("fips_openssl")
@pytest.mark.parametrize(
    "row",
    [
        SCRYPT_ROW,
        SHA256_ROW,
        PASSWORD,
        "pbkdf2:",
        PBKDF2_ROW[:-1],
        PBKDF2_ROW + "A",
        PBKDF2_ROW + "\n",
        "pbkdf2:" + "@" * 64,
        "pbkdf2:sha256:1000:AAECAwQFBgcICQoLDA0ODw==:67GE2pGj/dkUsRooBGoUFP43cqp6T9IYz/kC+hxI3RM=",
    ],
    ids=[
        "scrypt",
        "sha256",
        "plaintext",
        "prefix-only",
        "short",
        "long",
        "newline",
        "not-base64",
        "stored-iterations",
    ],
)
def test_fips_refuses_any_other_row_with_one_error_that_hides_it(caplog: pytest.LogCaptureFixture, row: str) -> None:
    with caplog.at_level("ERROR", logger="LiteLLM Proxy"):
        assert verify_password(PASSWORD, row) is False
    assert [record.levelname for record in caplog.records] == ["ERROR"]
    assert row not in caplog.text


@pytest.mark.parametrize(
    "row, verifies",
    [(SCRYPT_ROW, True), (SHA256_ROW, True), (PBKDF2_ROW, False)],
    ids=["scrypt", "sha256", "pbkdf2"],
)
def test_standard_mode_verifies_the_rows_it_always_did(row: str, verifies: bool) -> None:
    assert verify_password(PASSWORD, row) is verifies


@pytest.mark.asyncio
@pytest.mark.usefixtures("fips_openssl")
async def test_fips_boot_migration_writes_no_row() -> None:
    pc = MagicMock()
    pc.db.litellm_usertable.find_many = AsyncMock(
        return_value=[_make_user("plaintext-user", PASSWORD), _make_user("fips-user", PBKDF2_ROW)]
    )
    pc.db.litellm_usertable.update = AsyncMock()

    await migrate_passwords_to_scrypt_async(pc)

    pc.db.litellm_usertable.update.assert_not_awaited()


def _on_event_loop() -> bool:
    return asyncio._get_running_loop() is not None


@pytest.mark.asyncio
async def test_password_kdf_runs_on_the_event_loop_in_standard_mode() -> None:
    assert await run_off_loop_if_fips(_on_event_loop) is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("fips_openssl")
async def test_password_kdfs_leave_the_event_loop_and_never_overlap_in_fips_mode() -> None:
    second_started: Final = threading.Event()

    def first() -> tuple[bool, bool]:
        return _on_event_loop(), second_started.wait(timeout=0.2)

    (first_on_event_loop, overlapped), _ = await asyncio.gather(
        run_off_loop_if_fips(first), run_off_loop_if_fips(second_started.set)
    )
    assert (first_on_event_loop, overlapped) == (False, False)
