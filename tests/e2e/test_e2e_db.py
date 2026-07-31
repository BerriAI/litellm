"""Coverage for e2e_db's connection resolution.

Harness-only (no `e2e` marker): these never touch a proxy or a database.

RDS with IAM auth has no durable DATABASE_URL: the proxy assembles one at startup
from the DATABASE_* parts plus a token that rotates every ~12 minutes. A reset that
only read DATABASE_URL silently fell back to the local-docker default, so it was a
no-op against RDS and a hazard locally, which is what this resolver exists to fix.
"""

from __future__ import annotations

import pytest

from e2e_db import LOCAL_DATABASE_URL, resolve_database_url


DB_ENV_VARS = (
    "DATABASE_URL",
    "DATABASE_HOST",
    "DATABASE_PORT",
    "DATABASE_USER",
    "DATABASE_USERNAME",
    "DATABASE_NAME",
    "DATABASE_PASSWORD",
    "DATABASE_SCHEMA",
    "IAM_TOKEN_DB_AUTH",
)


@pytest.fixture
def clean_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every DB var so a developer's real `.env` (which points at a shared
    instance) cannot decide what these assert."""
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_explicit_database_url_wins(clean_db_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@explicit:5432/db")
    monkeypatch.setenv("DATABASE_HOST", "ignored.example.com")
    assert resolve_database_url() == "postgresql://u:p@explicit:5432/db"


def test_falls_back_to_local_when_nothing_is_configured(clean_db_env: None) -> None:
    assert resolve_database_url() == LOCAL_DATABASE_URL


@pytest.mark.parametrize("missing", ["DATABASE_HOST", "DATABASE_USER", "DATABASE_NAME"])
def test_incomplete_rds_parts_fall_back_rather_than_building_a_broken_url(
    clean_db_env: None, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """A half-configured environment must not produce a URL with "None" in it."""
    parts = {
        "DATABASE_HOST": "rds.example.com",
        "DATABASE_USER": "litellm",
        "DATABASE_NAME": "litellm",
    }
    del parts[missing]
    for name, value in parts.items():
        monkeypatch.setenv(name, value)
    assert resolve_database_url() == LOCAL_DATABASE_URL


def test_builds_url_from_rds_parts_with_password(
    clean_db_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_HOST", "rds.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("DATABASE_PASSWORD", "pw")
    assert resolve_database_url() == "postgresql://litellm:pw@rds.example.com:5432/litellm"


def test_honors_database_schema_and_port(
    clean_db_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_HOST", "rds.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("DATABASE_PASSWORD", "pw")
    monkeypatch.setenv("DATABASE_PORT", "6543")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    assert (
        resolve_database_url()
        == "postgresql://litellm:pw@rds.example.com:6543/litellm?schema=public"
    )


def test_mints_an_iam_token_when_iam_auth_is_enabled(
    clean_db_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RDS/IAM case this whole resolver exists for: no DATABASE_URL, no
    password, credentials come from a freshly minted token."""
    seen: dict[str, str] = {}

    def fake_token(db_host: str, db_port: str, db_user: str) -> str:
        seen.update(host=db_host, port=db_port, user=db_user)
        return "TOKEN%2FENCODED"

    monkeypatch.setattr("litellm.proxy.auth.rds_iam_token.generate_iam_auth_token", fake_token)
    monkeypatch.setenv("DATABASE_HOST", "rds.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")

    assert (
        resolve_database_url()
        == "postgresql://litellm:TOKEN%2FENCODED@rds.example.com:5432/litellm"
    )
    assert seen == {"host": "rds.example.com", "port": "5432", "user": "litellm"}


def test_password_is_used_when_iam_auth_is_off(
    clean_db_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IAM_TOKEN_DB_AUTH left unset (or "false") must not attempt to reach AWS."""

    def explode(**_kwargs: object) -> str:
        raise AssertionError("must not mint an IAM token when IAM auth is disabled")

    monkeypatch.setattr("litellm.proxy.auth.rds_iam_token.generate_iam_auth_token", explode)
    monkeypatch.setenv("DATABASE_HOST", "rds.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("DATABASE_PASSWORD", "pw")
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "false")
    assert resolve_database_url() == "postgresql://litellm:pw@rds.example.com:5432/litellm"
