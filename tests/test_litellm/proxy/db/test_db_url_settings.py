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

import os
import urllib.parse
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from litellm.proxy.db.db_url_settings import (
    DatabaseURLSettings,
    unsupported_db_scheme,
    unsupported_db_scheme_message,
)
from litellm.proxy.db.token_auth import AzureEntraTokenAuth, RdsIamTokenAuth


def _apply() -> bool:
    """Run the production call path: load from env, write to env."""
    return DatabaseURLSettings.from_env().apply_to_env()


_MANAGED_DB_ENV_VARS = (
    "IAM_TOKEN_DB_AUTH",
    "AZURE_POSTGRESQL_AUTH",
    "DATABASE_DISABLE_PREPARED_STATEMENTS",
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
        == "postgresql://litellm:WRITER_TOKEN@writer.example.com:5432/litellm_db"
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

    assert os.environ["DATABASE_URL"] == "postgresql://svc%40corp:WRITER_TOKEN@writer.example.com:5432/litellm_db"


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
        "@writer.postgres.database.azure.com:5432/litellm_db"
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
        "@reader.postgres.database.azure.com:5432/litellm_db?schema=public"
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
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA", "mysql://u:p@reader.example.com:3306/db"
    )

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
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader.example.com:5432/db"
    )

    _apply()

    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query
    )
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

    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(os.environ["DATABASE_URL_READ_REPLICA"]).query
    )
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


def test_reader_url_left_alone_when_writer_has_no_params(monkeypatch):
    """No params to inherit must mean the reader URL is not rewritten at all."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer.example.com:5432/db")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://u:p@reader.example.com:5432/db?options=-c%20search_path%3Dapp",
    )

    _apply()

    assert (
        os.environ["DATABASE_URL_READ_REPLICA"]
        == "postgresql://u:p@reader.example.com:5432/db?options=-c%20search_path%3Dapp"
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
        "postgresql://litellm:s3cr3t@writer.example.com:5432/litellm_db?pgbouncer=true"
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
    assert os.environ["DATABASE_URL"] == "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=true"


def test_disable_prepared_statements_respects_a_pinned_pgbouncer_value(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=false")

    _apply()

    assert os.environ["DATABASE_URL"] == "postgresql://u:p@db.example.com:5432/litellm_db?pgbouncer=false"


def test_disable_prepared_statements_applies_to_direct_url(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/litellm_db")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@direct.example.com:5432/litellm_db")

    _apply()

    assert os.environ["DIRECT_URL"] == "postgresql://u:p@direct.example.com:5432/litellm_db?pgbouncer=true"


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

    assert os.environ["DATABASE_URL"] == "postgresql://u:p@db.example.com:5432/litellm_db"


def test_disable_prepared_statements_rejects_an_unreadable_value(monkeypatch):
    monkeypatch.setenv("DATABASE_DISABLE_PREPARED_STATEMENTS", "enabled")

    with pytest.raises(ValidationError, match="DATABASE_DISABLE_PREPARED_STATEMENTS"):
        DatabaseURLSettings.from_env()


def test_unsupported_db_scheme_message_names_var_and_scheme():
    msg = unsupported_db_scheme_message("DIRECT_URL", "sqlite")
    assert "DIRECT_URL" in msg
    assert "sqlite" in msg
    assert "postgresql://" in msg
