"""Tests for ``DatabaseURLSettings``.

The model assembles ``DATABASE_URL`` (and optionally
``DATABASE_URL_READ_REPLICA``) from the discrete ``DATABASE_*`` env vars
emitted by the ``helm/litellm`` chart, before Prisma initializes. It covers
both IAM auth (mint a short-lived token) and password auth, for both the
writer and the read replica.

The reader URL is opt-in via ``DATABASE_HOST_READ_REPLICA`` and must not
clobber a pre-existing ``DATABASE_URL_READ_REPLICA``. A pre-existing
``DATABASE_URL`` (password auth) is likewise left untouched.
"""

import os
from unittest.mock import patch

import pytest

from litellm.proxy.db.db_url_settings import DatabaseURLSettings


def _apply() -> bool:
    """Run the production call path: load from env, write to env."""
    return DatabaseURLSettings.from_env().apply_to_env()


@pytest.fixture(autouse=True)
def _scrub_db_env(monkeypatch):
    """Remove every env var the model reads so tests start from a clean slate."""
    for var in (
        "IAM_TOKEN_DB_AUTH",
        "DATABASE_URL",
        "DATABASE_URL_READ_REPLICA",
        "DATABASE_HOST",
        "DATABASE_PORT",
        "DATABASE_USER",
        "DATABASE_USERNAME",
        "DATABASE_NAME",
        "DATABASE_SCHEMA",
        "DATABASE_PASSWORD",
        "DATABASE_HOST_READ_REPLICA",
        "DATABASE_PORT_READ_REPLICA",
        "DATABASE_USER_READ_REPLICA",
        "DATABASE_USERNAME_READ_REPLICA",
        "DATABASE_NAME_READ_REPLICA",
        "DATABASE_SCHEMA_READ_REPLICA",
        "DATABASE_PASSWORD_READ_REPLICA",
    ):
        monkeypatch.delenv(var, raising=False)


def _stub_iam_token(token: str = "FAKE_TOKEN"):
    """Patch the AWS-touching token mint so tests don't need boto3 / network."""
    return patch(
        "litellm.proxy.auth.rds_iam_token.generate_iam_auth_token",
        return_value=token,
    )


# ---------------------------------------------------------------------------
# IAM auth
# ---------------------------------------------------------------------------


def test_returns_false_when_nothing_configured(monkeypatch):
    """No env mutation, no error — just a False return."""
    assert _apply() is False
    assert "DATABASE_URL" not in os.environ


def test_assembles_writer_url_when_iam_enabled(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with _stub_iam_token("WRITER_TOKEN"):
        assert _apply() is True

    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm:WRITER_TOKEN@writer.example.com:5432/litellm_db"
    )
    # Reader was never configured, so it must not have been set.
    assert "DATABASE_URL_READ_REPLICA" not in os.environ


def test_missing_writer_envs_raises(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    # DATABASE_HOST intentionally unset.
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with pytest.raises(RuntimeError, match="DATABASE_HOST"):
        _apply()


def test_reader_url_assembled_when_host_set_and_url_unset(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    with _stub_iam_token("READER_TOKEN"):
        _apply()

    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://litellm:READER_TOKEN@reader.example.com:5432/litellm_db"
    )


def test_reader_url_not_clobbered_when_already_set(monkeypatch):
    """If the operator pinned DATABASE_URL_READ_REPLICA (e.g. a non-IAM
    reader), the model must leave it untouched even though
    DATABASE_HOST_READ_REPLICA is also set."""
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://app:secret@reader.example.com:5432/litellm_db",
    )

    with _stub_iam_token("READER_TOKEN"):
        _apply()

    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://app:secret@reader.example.com:5432/litellm_db"
    )


def test_reader_url_skipped_when_host_unset(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with _stub_iam_token("WRITER_TOKEN"):
        _apply()

    assert "DATABASE_URL_READ_REPLICA" not in os.environ


def test_reader_field_fallbacks_default_to_writer_values(monkeypatch):
    """When *_READ_REPLICA fields are unset (other than host), they fall
    back to the writer's user / name / schema."""
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    with _stub_iam_token("READER_TOKEN"):
        _apply()

    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://litellm:READER_TOKEN@reader.example.com:5432/litellm_db?schema=public"
    )


# ---------------------------------------------------------------------------
# Password auth
# ---------------------------------------------------------------------------


def test_assembles_writer_url_from_password(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db"
    )


def test_writer_password_is_percent_encoded(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss/w:rd")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm:p%40ss%2Fw%3Ard@writer.example.com:5432/litellm_db"
    )


def test_writer_url_not_clobbered_when_already_set(monkeypatch):
    """An operator-pinned DATABASE_URL (e.g. helm's $(VAR) assembly) always
    wins over the discrete fields."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://pinned:url@db.example.com:5432/litellm_db"
    )
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is False
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://pinned:url@db.example.com:5432/litellm_db"
    )


def test_writer_url_passwordless(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm@writer.example.com:5432/litellm_db"
    )


def test_database_username_alias(monkeypatch):
    """DATABASE_USERNAME is accepted as an alias for DATABASE_USER (parity
    with construct_database_url_from_env_vars)."""
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db"
    )


def test_password_reader_falls_back_to_writer_password(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://litellm:s3cr3t@reader.example.com:5432/litellm_db"
    )


def test_password_reader_uses_own_credentials(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")
    monkeypatch.setenv("DATABASE_USER_READ_REPLICA", "litellm_ro")
    monkeypatch.setenv("DATABASE_PASSWORD_READ_REPLICA", "ro_pw")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://litellm_ro:ro_pw@reader.example.com:5432/litellm_db"
    )
