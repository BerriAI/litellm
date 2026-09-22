"""Tests for ``DatabaseURLSettings``.

The model assembles ``DATABASE_URL`` (and optionally
``DATABASE_URL_READ_REPLICA``) from the discrete ``DATABASE_*`` env vars
emitted by the ``helm/litellm`` chart, before Prisma initializes. It covers
both token auth (mint a short-lived AWS RDS IAM or Microsoft Entra ID token)
and password auth, for both the writer and the read replica.

The reader URL is opt-in via ``DATABASE_HOST_READ_REPLICA`` and must not
clobber a pre-existing ``DATABASE_URL_READ_REPLICA``. A pre-existing
``DATABASE_URL`` (password auth) is likewise left untouched.
"""

import datetime
import hashlib
import os
import socket
import ssl
import tempfile
import threading
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import ValidationError

from litellm.proxy.db.db_url_settings import (
    PG_SSL_REQUEST,
    DatabaseURLSettings,
    token_refresh_params_from_url,
    translate_libpq_ssl_params,
    unsupported_db_scheme,
    unsupported_db_scheme_message,
)
from litellm.proxy.db.pgbouncer import PgBouncerPlan, PgBouncerSettings, plan_pgbouncer
from litellm.proxy.db.token_auth import AzureEntraTokenAuth, RdsIamTokenAuth


def _apply() -> bool:
    """Run the production call path: load from env, write to env."""
    return DatabaseURLSettings.from_env().apply_to_env()


_MANAGED_DB_ENV_VARS = (
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
    "DATABASE_DISABLE_PREPARED_STATEMENTS",
    "DATABASE_MAX_IDLE_CONNECTION_LIFETIME",
    "DATABASE_SSLMODE",
    "DATABASE_SSLROOTCERT",
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


def _stub_entra_token(token: str = "FAKE_TOKEN"):
    """Patch the Azure-touching token provider so tests don't need azure-identity."""
    return patch(
        "litellm.secret_managers.get_azure_ad_token_provider.get_azure_ad_token_provider",
        return_value=lambda: token,
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
        == "postgresql://litellm:WRITER_TOKEN@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )
    # Reader was never configured, so it must not have been set.
    assert "DATABASE_URL_READ_REPLICA" not in os.environ


def test_a_pre_encoded_iam_user_survives_url_assembly(monkeypatch):
    """This URL used to be interpolated raw, so pre-encoding ``DATABASE_USER`` was the
    only way to run IAM auth as a user whose name contains an ``@``. Encoding it again
    yields ``svc%2540corp``, which Postgres rejects with
    ``User `svc%40corp` was denied access``."""
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "svc%40corp")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with _stub_iam_token("WRITER_TOKEN"):
        assert _apply() is True

    assert os.environ["DATABASE_URL"] == (
        "postgresql://svc%40corp:WRITER_TOKEN@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_an_unreadable_toggle_fails_the_settings_model(monkeypatch):
    """Pydantic rejected `IAM_TOKEN_DB_AUTH=enabled` before token auth had its own
    parser. Reading it as 'off' instead would silently drop an operator who asked for
    token auth down to password auth, with no log line saying so."""
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "enabled")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with pytest.raises(ValidationError, match="IAM_TOKEN_DB_AUTH"):
        DatabaseURLSettings.from_env()


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
        == "postgresql://litellm:READER_TOKEN@reader.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
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
        == "postgresql://app:secret@reader.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
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
        == "postgresql://litellm:READER_TOKEN@reader.example.com:5432/litellm_db"
        "?schema=public&max_idle_connection_lifetime=60"
    )


# ---------------------------------------------------------------------------
# Azure Entra auth
# ---------------------------------------------------------------------------


def test_assembles_writer_url_when_azure_entra_enabled(monkeypatch):
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.postgres.database.azure.com")
    monkeypatch.setenv("DATABASE_USER", "litellm@contoso.onmicrosoft.com")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with _stub_entra_token("ENTRA_TOKEN"):
        assert _apply() is True

    assert os.environ["DATABASE_URL"] == (
        "postgresql://litellm%40contoso.onmicrosoft.com:ENTRA_TOKEN"
        "@writer.postgres.database.azure.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )
    assert os.environ["AZURE_POSTGRESQL_AUTH"] == "True"
    assert "IAM_TOKEN_DB_AUTH" not in os.environ


def test_azure_reader_url_assembled_from_writer_fallbacks(monkeypatch):
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.postgres.database.azure.com")
    monkeypatch.setenv("DATABASE_USER", "litellm@contoso.onmicrosoft.com")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.postgres.database.azure.com")

    with _stub_entra_token("ENTRA_TOKEN"):
        _apply()

    assert os.environ["DATABASE_URL_READ_REPLICA"] == (
        "postgresql://litellm%40contoso.onmicrosoft.com:ENTRA_TOKEN"
        "@reader.postgres.database.azure.com:5432/litellm_db?schema=public&max_idle_connection_lifetime=60"
    )


def test_azure_missing_writer_envs_names_the_azure_toggle(monkeypatch):
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    # DATABASE_HOST intentionally unset.
    monkeypatch.setenv("DATABASE_USER", "litellm@contoso.onmicrosoft.com")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with pytest.raises(RuntimeError, match="AZURE_POSTGRESQL_AUTH is enabled but"):
        _apply()


def test_both_token_toggles_is_a_startup_error(monkeypatch):
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    with pytest.raises(RuntimeError, match="can only come from one token source"):
        _apply()

    assert "DATABASE_URL" not in os.environ


@pytest.mark.parametrize(
    "env_var, expected_type",
    [("IAM_TOKEN_DB_AUTH", RdsIamTokenAuth), ("AZURE_POSTGRESQL_AUTH", AzureEntraTokenAuth)],
)
def test_token_auth_reflects_the_enabled_toggle(monkeypatch, env_var, expected_type):
    monkeypatch.setenv(env_var, "true")

    with _stub_entra_token():
        assert isinstance(DatabaseURLSettings.from_env().token_auth(), expected_type)


def test_the_toggle_agrees_with_the_refresh_loop_on_every_spelling(monkeypatch):
    """This model and `resolve_database_token_auth` (which arms the refresh loop) both
    read the same env var. When they disagreed, `AZURE_POSTGRESQL_AUTH=1` minted a token
    here and left the refresh loop convinced token auth was off."""
    from litellm.proxy.db.token_auth import resolve_database_token_auth

    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "1")

    with _stub_entra_token():
        settings_says = DatabaseURLSettings.from_env().azure_postgresql_auth
        refresh_loop_says = resolve_database_token_auth() is not None

    assert settings_says is True
    assert refresh_loop_says is True


def test_an_empty_toggle_is_off_rather_than_a_validation_error(monkeypatch):
    """`value: ""` is how a Kubernetes manifest spells 'off', and the componentized
    entrypoints build this model at import time, so a raise there is a crash loop."""
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "")
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "")

    settings = DatabaseURLSettings.from_env()

    assert (settings.azure_postgresql_auth, settings.iam_token_db_auth) == (False, False)
    assert settings.token_auth() is None


def test_apply_writer_url_to_env_leaves_the_reader_alone(monkeypatch):
    """The CLI shares the writer minting path but resolves the read replica itself, so
    it must not start writing DATABASE_URL_READ_REPLICA as a side effect."""
    monkeypatch.setenv("AZURE_POSTGRESQL_AUTH", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.postgres.database.azure.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.postgres.database.azure.com")

    with _stub_entra_token("ENTRA_TOKEN"):
        assert DatabaseURLSettings.from_env().apply_writer_url_to_env() is True

    assert "DATABASE_URL" in os.environ
    assert "DATABASE_URL_READ_REPLICA" not in os.environ


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
        == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_writer_password_is_percent_encoded(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss/w:rd")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm:p%40ss%2Fw%3Ard@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_writer_url_not_clobbered_when_already_set(monkeypatch):
    """An operator-pinned DATABASE_URL (e.g. helm's $(VAR) assembly) always
    wins over the discrete fields."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://pinned:url@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is False
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://pinned:url@db.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_writer_url_passwordless(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")

    assert _apply() is True
    assert (
        os.environ["DATABASE_URL"]
        == "postgresql://litellm@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
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
        == "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
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
        == "postgresql://litellm:s3cr3t@reader.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
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
        == "postgresql://litellm_ro:ro_pw@reader.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


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

    with pytest.raises(RuntimeError, match=r"DIRECT_URL.*sqlite"):
        _apply()


def test_apply_to_env_rejects_pinned_non_postgres_reader(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "mysql://u:p@reader.example.com:3306/db")

    with pytest.raises(RuntimeError, match=r"DATABASE_URL_READ_REPLICA.*mysql"):
        _apply()


def test_apply_to_env_accepts_pinned_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")

    # Operator-pinned URL: nothing reassembled, no error.
    assert _apply() is False


# ---------------------------------------------------------------------------
# Connection params on the read replica
# ---------------------------------------------------------------------------


def test_reader_inherits_writer_connection_params(monkeypatch):
    """The reader is a second pool: without the writer's params it sizes itself
    from Prisma's default and the operator's cap is not enforced."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@writer.example.com:5432/db?connection_limit=3&pool_timeout=20&pgbouncer=true",
    )
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db")

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["connection_limit"] == ["3"]
    assert query["pool_timeout"] == ["20"]
    assert query["pgbouncer"] == ["true"]


def test_reader_keeps_its_own_pinned_connection_params(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@writer.example.com:5432/db?connection_limit=3&pool_timeout=20",
    )
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://u:p@reader.example.com:5432/db?connection_limit=50",
    )

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["connection_limit"] == ["50"]
    assert query["pool_timeout"] == ["20"]


def test_assembled_reader_url_inherits_writer_connection_params(monkeypatch):
    """A reader assembled from the discrete DATABASE_*_READ_REPLICA vars must
    carry the params too, and must not inherit the writer's schema."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db?connection_limit=3&schema=writer_schema"
    )
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    _apply()

    reader_url = os.environ["DATABASE_URL_READ_REPLICA"]
    assert reader_url.startswith("postgresql://litellm:s3cr3t@reader.example.com:5432/litellm_db?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(reader_url).query)
    assert query["connection_limit"] == ["3"]
    assert "schema" not in query


def test_reader_does_not_inherit_writer_options(monkeypatch):
    """A writer search_path must not follow the reader, or reader queries resolve
    against the wrong schema."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@writer.example.com:5432/db?connection_limit=3&options=-c%20search_path%3Dwriter_schema",
    )
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db")

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["connection_limit"] == ["3"]
    assert "options" not in query


def test_reader_does_not_inherit_an_unvetted_writer_param(monkeypatch):
    """Inheritance is an allowlist, so a param nobody vetted for the reader stays
    on the writer. Flipping this to a denylist would let the next schema-affecting
    param leak through by default."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@writer.example.com:5432/db?connection_limit=3&application_name=writer&novel_param=x",
    )
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db")

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["connection_limit"] == ["3"]
    assert "application_name" not in query
    assert "novel_param" not in query


def test_reader_keeps_its_own_options_when_writer_params_are_appended(monkeypatch):
    """Appending the writer's pool params must leave the reader's own search_path
    intact, since that is what decides which tables its queries resolve against."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db?connection_limit=3")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://u:p@reader.example.com:5432/db?options=-c%20search_path%3Dreader_schema",
    )

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["options"] == ["-c search_path=reader_schema"]
    assert query["connection_limit"] == ["3"]


def test_reader_url_left_alone_when_nothing_is_missing(monkeypatch):
    """No params to inherit must mean the reader URL is not rewritten at all."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://u:p@reader.example.com:5432/db?options=-c%20search_path%3Dapp&max_idle_connection_lifetime=45",
    )

    _apply()

    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://u:p@reader.example.com:5432/db?options=-c%20search_path%3Dapp&max_idle_connection_lifetime=45"
    )


# ---------------------------------------------------------------------------
# DATABASE_DISABLE_PREPARED_STATEMENTS
# ---------------------------------------------------------------------------


def test_disable_prepared_statements_appends_pgbouncer_to_assembled_writer(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is True
    assert os.environ["DATABASE_URL"] == (
        "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db?pgbouncer=true&max_idle_connection_lifetime=60"
    )
    assert "DIRECT_URL" not in os.environ


def test_disable_prepared_statements_appends_pgbouncer_to_pinned_writer(monkeypatch):
    """The componentized entrypoints (gateway / backend / migrations) receive a
    pinned DATABASE_URL and call apply_to_env; without the pgbouncer param Prisma
    keeps named prepared statements and 42P05 collisions surface behind a
    transaction-pooling pgbouncer."""
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")

    assert _apply() is False
    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=true&max_idle_connection_lifetime=60"
    )


def test_disable_prepared_statements_respects_a_pinned_pgbouncer_value(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=false")

    _apply()

    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=false&max_idle_connection_lifetime=60"
    )


def test_disable_prepared_statements_applies_to_direct_url(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@direct.example.com:5432/litellm_db")

    _apply()

    assert os.environ["DIRECT_URL"] == (
        "postgresql://u:p@direct.example.com:5432/litellm_db?pgbouncer=true&max_idle_connection_lifetime=60"
    )


def test_reader_inherits_pgbouncer_from_disable_prepared_statements(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db")

    _apply()

    query = urllib.parse.parse_qs(urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query)
    assert query["pgbouncer"] == ["true"]


def test_disable_prepared_statements_off_leaves_urls_alone(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "false")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")

    _apply()

    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_disable_prepared_statements_rejects_an_unreadable_value(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "enabled")

    with pytest.raises(ValidationError, match="DATABASE_DISABLE_PREPARED_STATEMENTS"):
        DatabaseURLSettings.from_env()


def test_unsupported_db_scheme_message_names_var_and_scheme():
    msg = unsupported_db_scheme_message("DIRECT_URL", "sqlite")
    assert "DIRECT_URL" in msg
    assert "sqlite" in msg
    assert "postgresql://" in msg


def _query(url: str) -> dict[str, list[str]]:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(url).query, keep_blank_values=True)


def test_libpq_verify_full_and_sslrootcert_become_prisma_strict_sslcert(monkeypatch):
    """Prisma drops ``sslrootcert`` and treats ``verify-full`` as ``prefer``, so a
    libpq-style URL (the form the RDS docs give) connects with no certificate
    check. The URL Prisma actually receives must carry its own strict dialect."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@db.example.com:5432/litellm_db?sslmode=verify-full&sslrootcert=/certs/rds-bundle.pem",
    )

    assert _apply() is False

    assert _query(os.environ["DATABASE_URL"]) == {
        "sslmode": ["require"],
        "sslcert": ["/certs/rds-bundle.pem"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }


def _tls_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_HOST", "writer.example.com")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm_db")
    monkeypatch.setenv("DATABASE_SSLMODE", "verify-full")
    monkeypatch.setenv("DATABASE_SSLROOTCERT", "/certs/rds-bundle.pem")


def test_tls_env_vars_make_the_minted_iam_writer_url_verify_the_server(monkeypatch: pytest.MonkeyPatch):
    """The supervisor starts PgBouncer from the URL assembled here, before any
    config.yaml is read, so an IAM URL with no TLS params leaves PgBouncer on
    ``prefer`` (no SNI, no verification) and the RDS handshake fails."""
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    _tls_env(monkeypatch)

    with _stub_iam_token("WRITER_TOKEN"):
        assert _apply() is True

    url: Final = os.environ["DATABASE_URL"]
    assert url.startswith("postgresql://litellm:WRITER_TOKEN@writer.example.com:5432/litellm_db?")
    assert _query(url) == {
        "sslmode": ["require"],
        "sslcert": ["/certs/rds-bundle.pem"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }


def test_tls_env_vars_apply_to_the_password_writer_and_the_assembled_reader(monkeypatch: pytest.MonkeyPatch):
    _tls_env(monkeypatch)
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")

    assert _apply() is True

    expected: Final = {
        "schema": ["public"],
        "sslmode": ["require"],
        "sslcert": ["/certs/rds-bundle.pem"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }
    assert os.environ["DATABASE_URL"].startswith("postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db?")
    assert _query(os.environ["DATABASE_URL"]) == expected
    assert os.environ["DATABASE_URL_READ_REPLICA"].startswith(
        "postgresql://litellm:s3cr3t@reader.example.com:5432/litellm_db?"
    )
    assert _query(os.environ["DATABASE_URL_READ_REPLICA"]) == expected


def test_sslrootcert_env_var_alone_means_verify_full_for_prisma_and_pgbouncer(monkeypatch: pytest.MonkeyPatch):
    """Under libpq's default ``prefer`` a root cert is never consulted, so a URL
    carrying only ``sslrootcert`` would leave PgBouncer on ``prefer`` with the CA
    loaded but unused. Supplying a CA and nothing else must verify."""
    _tls_env(monkeypatch)
    monkeypatch.delenv("DATABASE_SSLMODE")
    monkeypatch.setenv("DATABASE_PASSWORD", "s3cr3t")

    assert _apply() is True

    url: Final = os.environ["DATABASE_URL"]
    assert _query(url) == {
        "sslmode": ["require"],
        "sslcert": ["/certs/rds-bundle.pem"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }
    plan: Final = plan_pgbouncer(url, PgBouncerSettings(enabled=True), Path("/run/pgb"), None)
    assert isinstance(plan, PgBouncerPlan), plan
    assert "server_tls_sslmode = verify-full" in plan.ini
    assert "server_tls_ca_file = /run/pgb/server-ca.pem" in plan.ini


def test_tls_env_vars_never_override_a_pinned_database_url(monkeypatch: pytest.MonkeyPatch):
    writer: Final = (
        "postgresql://pinned:url@db.example.com:5432/litellm_db?sslmode=disable&max_idle_connection_lifetime=60"
    )
    reader: Final = "postgresql://pinned:url@reader.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    monkeypatch.setenv("DATABASE_URL", writer)
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", reader)
    monkeypatch.setenv("DATABASE_HOST_READ_REPLICA", "reader.example.com")
    _tls_env(monkeypatch)

    assert _apply() is False

    assert os.environ["DATABASE_URL"] == writer
    assert os.environ["DATABASE_URL_READ_REPLICA"] == reader


def test_token_refresh_params_keep_the_prisma_tls_dialect_but_not_the_schema():
    kept: Final = token_refresh_params_from_url(
        "postgresql://u:TOKEN@db.example.com:5432/litellm_db"
        "?schema=tenant&connection_limit=5&sslmode=require&sslcert=/certs/root.pem&sslaccept=strict"
    )
    assert dict(kept) == {
        "connection_limit": "5",
        "sslmode": "require",
        "sslcert": "/certs/root.pem",
        "sslaccept": "strict",
    }


def _issue_cert(
    subject: str, issuer: x509.Certificate | None, issuer_key: ec.EllipticCurvePrivateKey | None, ca: bool
) -> tuple[x509.Certificate, ec.EllipticCurvePrivateKey]:
    key: Final = ec.generate_private_key(ec.SECP256R1())
    name: Final = x509.Name((x509.NameAttribute(x509.NameOID.COMMON_NAME, subject),))
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    builder: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer.subject if issuer else name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName((x509.DNSName("localhost"),)), critical=False)
    )
    return builder.sign(issuer_key or key, hashes.SHA256()), key


def _pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


class _TlsPostgresStub:
    """Answers one libpq ``SSLRequest`` with ``S`` and serves ``leaf + intermediate``."""

    def __init__(self, chain_pem: Path, key_pem: Path) -> None:
        self.context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(str(chain_pem), str(key_pem))
        self.listener: Final = socket.create_server(("127.0.0.1", 0))
        self.port: Final[int] = self.listener.getsockname()[1]
        self.thread: Final = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        with self.listener:
            while True:
                try:
                    conn: socket.socket = self.listener.accept()[0]
                except OSError:
                    return
                with conn:
                    try:
                        if conn.recv(8) == PG_SSL_REQUEST:
                            conn.sendall(b"S")
                            with self.context.wrap_socket(conn, server_side=True) as tls:
                                tls.recv(1)
                    except OSError:
                        continue


@dataclass(frozen=True, slots=True)
class _RdsLikePki:
    bundle: Path
    wrong_bundle: Path
    root: Path
    port: int


@pytest.fixture
def rds_like_pki(tmp_path: Path) -> Iterator[_RdsLikePki]:
    """An RDS-shaped trust setup: the server sends leaf + intermediate, the
    bundle holds only self-signed roots, and the right root is not first."""
    root, root_key = _issue_cert("Real Root CA", None, None, ca=True)
    decoys: Final = tuple(_issue_cert(f"Decoy Root CA {i}", None, None, ca=True)[0] for i in range(3))
    intermediate, intermediate_key = _issue_cert("Intermediate CA", root, root_key, ca=True)
    leaf, leaf_key = _issue_cert("localhost", intermediate, intermediate_key, ca=False)
    chain_pem: Final = tmp_path / "server-chain.pem"
    chain_pem.write_bytes(_pem(leaf) + _pem(intermediate))
    key_pem: Final = tmp_path / "server.key"
    key_pem.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    bundle: Final = tmp_path / "global-bundle.pem"
    bundle.write_bytes(b"".join(_pem(decoy) for decoy in decoys) + _pem(root))
    wrong_bundle: Final = tmp_path / "wrong-bundle.pem"
    wrong_bundle.write_bytes(b"".join(_pem(decoy) for decoy in decoys))
    root_pem: Final = tmp_path / "root.pem"
    root_pem.write_bytes(_pem(root))
    stub: Final = _TlsPostgresStub(chain_pem, key_pem)
    yield _RdsLikePki(bundle=bundle, wrong_bundle=wrong_bundle, root=root_pem, port=stub.port)
    stub.listener.close()


def _params(url: str) -> tuple[tuple[str, str], ...]:
    return tuple(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))


def test_multi_root_bundle_is_pinned_to_the_root_the_server_chains_to(
    monkeypatch: pytest.MonkeyPatch, rds_like_pki: _RdsLikePki
):
    """Prisma's ``sslcert`` loads only the first certificate of the file, so
    handing it the whole RDS bundle trusts one region's root and fails with
    "unable to get local issuer certificate" everywhere else. The URL Prisma
    receives must point at a single-certificate file holding the server's root."""
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql://u:p@localhost:{rds_like_pki.port}/litellm_db?sslmode=verify-full&sslrootcert={rds_like_pki.bundle}",
    )

    _apply()

    (sslmode, sslcert, sslaccept, _) = _params(os.environ["DATABASE_URL"])
    assert (sslmode, sslaccept) == (("sslmode", "require"), ("sslaccept", "strict"))
    assert sslcert[0] == "sslcert" and sslcert[1] != str(rds_like_pki.bundle)
    assert Path(sslcert[1]).read_bytes() == rds_like_pki.root.read_bytes()


def test_pinned_root_replaces_a_planted_symlink_instead_of_writing_through_it(
    monkeypatch: pytest.MonkeyPatch, rds_like_pki: _RdsLikePki, tmp_path: Path
):
    """The pinned file has a predictable name in a shared temp dir, so a symlink
    planted there must not redirect the write onto its target."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root_der: Final = x509.load_pem_x509_certificate(rds_like_pki.root.read_bytes()).public_bytes(
        serialization.Encoding.DER
    )
    pinned: Final = tmp_path / f"litellm-sslcert-{hashlib.sha256(root_der).hexdigest()[:16]}.pem"
    victim: Final = tmp_path / "victim.txt"
    victim.write_text("untouched")
    pinned.symlink_to(victim)
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql://u:p@localhost:{rds_like_pki.port}/litellm_db?sslmode=verify-full&sslrootcert={rds_like_pki.bundle}",
    )

    _apply()

    assert ("sslcert", str(pinned)) in _params(os.environ["DATABASE_URL"])
    assert victim.read_text() == "untouched"
    assert not pinned.is_symlink() and pinned.read_bytes() == rds_like_pki.root.read_bytes()


def test_bundle_without_the_servers_root_is_passed_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, rds_like_pki: _RdsLikePki
):
    """Nothing in the bundle verifies the server, so no root is pinned and
    Prisma keeps rejecting the connection instead of trusting a root the
    operator never shipped."""
    monkeypatch.setenv(
        "DATABASE_URL",
        f"postgresql://u:p@localhost:{rds_like_pki.port}/litellm_db"
        f"?sslmode=verify-full&sslrootcert={rds_like_pki.wrong_bundle}",
    )

    _apply()

    assert ("sslcert", str(rds_like_pki.wrong_bundle)) in _params(os.environ["DATABASE_URL"])


def test_root_cert_resolver_receives_the_urls_host_and_default_port():
    def resolver(cert_path: str, host: str, port: int) -> str:
        return f"/pinned/{host}/{port}{cert_path}"

    url: Final = translate_libpq_ssl_params(
        "postgresql://u:p@db.example.com/litellm_db?sslmode=verify-full&sslrootcert=/certs/bundle.pem", resolver
    )

    assert ("sslcert", "/pinned/db.example.com/5432/certs/bundle.pem") in _params(url)


def test_libpq_verify_ca_becomes_prisma_strict(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db?sslmode=verify-ca")

    _apply()

    assert _query(os.environ["DATABASE_URL"]) == {
        "sslmode": ["require"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }


def test_sslrootcert_alone_turns_on_strict_verification(monkeypatch):
    """libpq verifies the chain whenever a root cert is supplied under
    ``sslmode=require``; Prisma needs ``sslaccept=strict`` to do the same."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db?sslmode=require&sslrootcert=/certs/ca.pem"
    )

    _apply()

    assert _query(os.environ["DATABASE_URL"]) == {
        "sslmode": ["require"],
        "sslcert": ["/certs/ca.pem"],
        "sslaccept": ["strict"],
        "max_idle_connection_lifetime": ["60"],
    }


def test_pinned_prisma_ssl_params_win_over_libpq_translation(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://u:p@db.example.com:5432/litellm_db"
        "?sslmode=verify-full&sslrootcert=/ignored.pem&sslcert=/pinned.pem&sslaccept=accept_invalid_certs",
    )

    _apply()

    assert _query(os.environ["DATABASE_URL"]) == {
        "sslmode": ["require"],
        "sslcert": ["/pinned.pem"],
        "sslaccept": ["accept_invalid_certs"],
        "max_idle_connection_lifetime": ["60"],
    }


def test_prisma_native_ssl_url_is_left_untouched(monkeypatch):
    url = (
        "postgresql://u:p@db.example.com:5432/litellm_db"
        "?sslmode=require&sslcert=/certs/ca.pem&sslaccept=strict&max_idle_connection_lifetime=60"
    )
    monkeypatch.setenv("DATABASE_URL", url)

    _apply()

    assert os.environ["DATABASE_URL"] == url


def test_libpq_ssl_translation_covers_direct_url_and_read_replica(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db?sslmode=verify-full")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@direct.example.com:5432/db?sslmode=verify-full")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db?sslmode=verify-full")

    _apply()

    for env_var in ("DATABASE_URL", "DIRECT_URL", "DATABASE_URL_READ_REPLICA"):
        assert _query(os.environ[env_var]) == {
            "sslmode": ["require"],
            "sslaccept": ["strict"],
            "max_idle_connection_lifetime": ["60"],
        }, env_var


def test_default_idle_lifetime_applied_to_pinned_writer_and_direct_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@direct.example.com:5432/litellm_db")

    assert _apply() is False

    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )
    assert os.environ["DIRECT_URL"] == (
        "postgresql://u:p@direct.example.com:5432/litellm_db?max_idle_connection_lifetime=60"
    )


def test_url_pinned_idle_lifetime_wins_over_default_and_env_knob(monkeypatch):
    monkeypatch.setenv("DATABASE_MAX_IDLE_CONNECTION_LIFETIME", "45")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db?max_idle_connection_lifetime=300"
    )

    _apply()

    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?max_idle_connection_lifetime=300"
    )


def test_env_knob_overrides_default_idle_lifetime(monkeypatch):
    monkeypatch.setenv("DATABASE_MAX_IDLE_CONNECTION_LIFETIME", "45")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@direct.example.com:5432/litellm_db")

    _apply()

    assert os.environ["DATABASE_URL"] == (
        "postgresql://u:p@db.example.com:5432/litellm_db?max_idle_connection_lifetime=45"
    )
    assert os.environ["DIRECT_URL"] == (
        "postgresql://u:p@direct.example.com:5432/litellm_db?max_idle_connection_lifetime=45"
    )


def test_env_knob_rejects_a_non_integer_value(monkeypatch):
    monkeypatch.setenv("DATABASE_MAX_IDLE_CONNECTION_LIFETIME", "soon")

    with pytest.raises(ValidationError, match="DATABASE_MAX_IDLE_CONNECTION_LIFETIME"):
        DatabaseURLSettings.from_env()


@pytest.mark.parametrize(("knob", "expected"), [(None, "60"), ("45", "45")])
def test_reader_inherits_the_writer_idle_lifetime(monkeypatch, knob, expected):
    if knob is not None:
        monkeypatch.setenv("DATABASE_MAX_IDLE_CONNECTION_LIFETIME", knob)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db")

    _apply()

    assert os.environ["DATABASE_URL_READ_REPLICA"] == (
        f"postgresql://u:p@reader.example.com:5432/db?max_idle_connection_lifetime={expected}"
    )


def test_reader_keeps_its_own_pinned_idle_lifetime(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db?max_idle_connection_lifetime=120"
    )

    _apply()

    assert os.environ["DATABASE_URL_READ_REPLICA"] == (
        "postgresql://u:p@reader.example.com:5432/db?max_idle_connection_lifetime=120"
    )
