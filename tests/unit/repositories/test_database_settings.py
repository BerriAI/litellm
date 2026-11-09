from dataclasses import replace
from typing import Final

import pytest

from litellm.repositories.database_settings import NativeDatabaseSettings, NativeDatabaseSettingsError


def test_projects_prisma_connection_settings_without_forwarding_url() -> None:
    settings: Final = NativeDatabaseSettings.from_url(
        "postgresql://auth%40example.test:p%3Aa%2Fs%25s@db.test:5433/gateway%20db"
        "?schema=auth%20schema&connection_limit=7&pool_timeout=3&connect_timeout=4"
        "&max_idle_connection_lifetime=60&pgbouncer=true&sslmode=require&sslaccept=strict&sslcert=/ca.pem"
    )
    assert settings == NativeDatabaseSettings(
        host="db.test",
        port=5433,
        username="auth@example.test",
        password="p:a/s%s",
        database="gateway db",
        schema="auth schema",
        ssl_mode="verify-full",
        ssl_root_cert="/ca.pem",
        max_connections=7,
        acquire_timeout=3,
        idle_timeout=60,
        statement_cache_capacity=0,
        connect_timeout=4,
    )


@pytest.mark.parametrize(
    ("query", "ssl_mode", "certificate"),
    (
        ("sslmode=disable&sslaccept=strict", "disable", None),
        ("sslmode=verify-ca&sslrootcert=/root.pem", "verify-ca", "/root.pem"),
        ("sslmode=require&sslaccept=accept_invalid_certs", "require", None),
        ("sslmode=prefer", "prefer", None),
    ),
)
def test_preserves_tls_policy(query: str, ssl_mode: str, certificate: str | None) -> None:
    settings: Final = NativeDatabaseSettings.from_url(f"postgresql://user:password@db.test/gateway?{query}")
    assert (settings.ssl_mode, settings.ssl_root_cert) == (ssl_mode, certificate)


def test_database_settings_repr_redacts_password_and_provider() -> None:
    settings: Final = NativeDatabaseSettings.from_url("postgresql://user:opaque-password@db.test/gateway")
    assert "opaque-password" not in repr(settings)
    assert repr(replace(settings, password="another-password")) == repr(settings)


def test_preserves_unix_socket_and_disabled_idle_expiration() -> None:
    settings: Final = NativeDatabaseSettings.from_url(
        "postgresql://user@localhost/gateway?host=%2Fvar%2Frun%2Fpostgresql&max_idle_connection_lifetime=0"
    )
    assert (settings.host, settings.password, settings.idle_timeout) == ("/var/run/postgresql", None, None)


def test_rejects_non_postgres_without_echoing_credentials() -> None:
    with pytest.raises(NativeDatabaseSettingsError, match="PostgreSQL is required") as error:
        NativeDatabaseSettings.from_url("mysql://user:opaque-password@db.test/gateway")
    assert "opaque-password" not in str(error.value)
