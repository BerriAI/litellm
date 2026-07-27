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

from litellm.proxy.db.db_url_settings import (
    DatabaseURLSettings,
    unsupported_db_scheme,
    unsupported_db_scheme_message,
)


def _apply() -> bool:
    """Run the production call path: load from env, write to env."""
    return DatabaseURLSettings.from_env().apply_to_env()


_MANAGED_DB_ENV_VARS = (
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
    "DATABASE_URL",
    "DIRECT_URL",
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
)


@pytest.fixture(autouse=True)
def _scrub_db_env(monkeypatch):
    """Start each test from a clean slate and restore the original env afterward.

    ``apply_to_env`` writes ``DATABASE_URL`` straight into ``os.environ``.
    Registering a setenv+delenv pair per var gives ``monkeypatch`` a restore
    record even for previously unset keys, so a synthesized URL (e.g.
    ``writer.example.com``) cannot leak into later tests that read
    ``DATABASE_URL`` to decide whether to hit a real database. Restoring via
    the same ``monkeypatch`` instance the tests use also keeps undo ordering
    consistent (a hand-rolled snapshot/restore runs before ``monkeypatch``'s
    own undo and gets clobbered by it).
    """
    for var in _MANAGED_DB_ENV_VARS:
        monkeypatch.setenv(var, "scrubbed")
        monkeypatch.delenv(var)


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

    assert os.environ["DATABASE_URL"] == "postgresql://litellm:WRITER_TOKEN@writer.example.com:5432/litellm_db"
    # Reader was never configured, so it must not have been set.
    assert "DATABASE_URL_READ_REPLICA" not in os.environ


def test_assembles_azure_writer_and_reader_urls(monkeypatch):
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.postgres.database.azure.com")
    monkeypatch.setenv("DATABASE_USER", "user@example.com")
    monkeypatch.setenv("DATABASE_NAME", "litellm db")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.postgres.database.azure.com")

    with patch(
        "litellm.proxy.auth.azure_postgres_token.generate_azure_postgres_auth_token",
        side_effect=("WRITER_TOKEN", "READER_TOKEN"),
    ):
        assert _apply() is True

    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://user%40example.com:WRITER_TOKEN@writer.postgres.database.azure.com:5432/litellm%20db"
    )
    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://user%40example.com:READER_TOKEN@reader.postgres.database.azure.com:5432/litellm%20db"
    )


def test_rejects_multiple_database_token_auth_modes(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")

    with pytest.raises(RuntimeError, match="cannot both be enabled"):
        _apply()


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

    assert os.environ["DATABASE_URL_READ_REPLICA"] == "postgresql://app:secret@reader.example.com:5432/litellm_db"


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
    assert os.environ["DATABASE_URL"] == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db"


def test_writer_password_is_percent_encoded(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss/w:rd")

    assert _apply() is True
    assert os.environ["DATABASE_URL"] == "postgresql://litellm:p%40ss%2Fw%3Ard@writer.example.com:5432/litellm_db"


def test_writer_url_not_clobbered_when_already_set(monkeypatch):
    """An operator-pinned DATABASE_URL (e.g. helm's $(VAR) assembly) always
    wins over the discrete fields."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://pinned:url@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is False
    assert os.environ["DATABASE_URL"] == "postgresql://pinned:url@db.example.com:5432/litellm_db"


def test_writer_url_passwordless(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    assert _apply() is True
    assert os.environ["DATABASE_URL"] == "postgresql://litellm@writer.example.com:5432/litellm_db"


def test_database_username_alias(monkeypatch):
    """DATABASE_USERNAME is accepted as an alias for DATABASE_USER (parity
    with construct_database_url_from_env_vars)."""
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is True
    assert os.environ["DATABASE_URL"] == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db"


def test_password_reader_falls_back_to_writer_password(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    assert _apply() is True
    assert os.environ["DATABASE_URL_READ_REPLICA"] == "postgresql://litellm:s3cr3t@reader.example.com:5432/litellm_db"


def test_password_reader_uses_own_credentials(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")
    monkeypatch.setenv("DATABASE_USER_READ_REPLICA", "litellm_ro")
    monkeypatch.setenv("DATABASE_PASSWORD_READ_REPLICA", "ro_pw")

    assert _apply() is True
    assert os.environ["DATABASE_URL_READ_REPLICA"] == "postgresql://litellm_ro:ro_pw@reader.example.com:5432/litellm_db"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:p@host:5432/db",
        "postgres://u:p@host:5432/db",
        "POSTGRESQL://u:p@host:5432/db",
        "postgresql://host/db?schema=public",
    ],
)
def test_unsupported_db_scheme_accepts_postgres(url):
    assert unsupported_db_scheme(url) is None


@pytest.mark.parametrize(
    "url,scheme",
    [
        ("sqlite:///data/litellm.db", "sqlite"),
        ("sqlite:///./local.db", "sqlite"),
        ("mysql://u:p@host:3306/db", "mysql"),
        ("mssql://host/db", "mssql"),
    ],
)
def test_unsupported_db_scheme_rejects_non_postgres(url, scheme):
    assert unsupported_db_scheme(url) == scheme


def test_unsupported_db_scheme_does_not_echo_schemeless_credentials():
    """A malformed schemeless DSN must not leak its embedded credentials
    through the return value (which callers log)."""
    leaky = "litellm:s3cr3t_password@db.internal:5432/litellm"

    result = unsupported_db_scheme(leaky)

    assert result is not None
    assert "s3cr3t_password" not in result
    assert "db.internal" not in result


def test_apply_to_env_rejects_pinned_sqlite_writer(monkeypatch):
    """Componentized entrypoints pin DATABASE_URL and call apply_to_env; a
    sqlite writer must raise here rather than reach Prisma."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/litellm.db")

    with pytest.raises(RuntimeError, match="sqlite"):
        _apply()

    # The bad URL must not have been propagated as a usable connection string.
    assert os.environ["DATABASE_URL"] == "sqlite:///data/litellm.db"


def test_apply_to_env_rejects_pinned_sqlite_direct_url(monkeypatch):
    """DIRECT_URL reaches Prisma the same way DATABASE_URL does; a non-postgres
    direct URL must be rejected in apply_to_env, matching the CLI startup guard."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv("DIRECT_URL", "sqlite:///data/litellm.db")

    with pytest.raises(RuntimeError, match="DIRECT_URL.*sqlite"):
        _apply()


def test_apply_to_env_rejects_pinned_non_postgres_reader(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "mysql://u:p@reader.example.com:3306/db")

    with pytest.raises(RuntimeError, match="DATABASE_URL_READ_REPLICA.*mysql"):
        _apply()


def test_apply_to_env_accepts_pinned_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")

    # Operator-pinned URL: nothing reassembled, no error.
    assert _apply() is False


def test_unsupported_db_scheme_message_names_var_and_scheme():
    msg = unsupported_db_scheme_message("DIRECT_URL", "sqlite")
    assert "DIRECT_URL" in msg
    assert "sqlite" in msg
    assert "postgresql://" in msg
