import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.db.db_connection_pool import (
    DEFAULT_DB_CONNECTION_POOL_LIMIT,
    DEFAULT_DB_CONNECTION_POOL_TIMEOUT,
    append_query_params,
    apply_pool_params_to_db_urls,
    build_db_connection_url_params,
    pool_params_from_general_settings,
)


def test_build_params_defaults_only_connection_limit_and_pool_timeout():
    assert build_db_connection_url_params(connection_limit=10, pool_timeout=60) == {
        "connection_limit": 10,
        "pool_timeout": 60,
    }


def test_build_params_omits_none_timeouts_and_pgbouncer():
    params = build_db_connection_url_params(
        connection_limit=10,
        pool_timeout=None,
        connect_timeout=None,
        socket_timeout=None,
        disable_prepared_statements=False,
    )
    assert params == {"connection_limit": 10}


def test_build_params_includes_pgbouncer_and_extra_overrides():
    params = build_db_connection_url_params(
        connection_limit=10,
        pool_timeout=60,
        disable_prepared_statements=True,
        extra_params={"connection_limit": 99, "sslmode": "require"},
    )
    assert params["pgbouncer"] == "true"
    assert params["sslmode"] == "require"
    assert params["connection_limit"] == 99


def test_append_query_params_merges_and_overwrites():
    merged = append_query_params(
        "postgresql://u:p@h:5432/db?schema=public&connection_limit=1",
        {"connection_limit": 10, "pool_timeout": 60},
    )
    assert "schema=public" in merged
    assert "connection_limit=10" in merged
    assert "connection_limit=1&" not in merged and not merged.endswith("connection_limit=1")
    assert "pool_timeout=60" in merged


def test_append_query_params_missing_url_returns_empty():
    assert append_query_params(None, {"connection_limit": 10}) == ""
    assert append_query_params("", {"connection_limit": 10}) == ""


def test_append_query_params_never_logs_url(caplog):
    """Regression for #33021: the DB URL (which carries credentials) must never
    be logged, at any level."""
    secret_url = "postgresql://user:sup3rs3cr3t@h:5432/db?schema=public"
    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        append_query_params(secret_url, {"connection_limit": 10})
    assert all("sup3rs3cr3t" not in r.getMessage() for r in caplog.records)


def test_pool_params_from_general_settings_defaults_on_empty():
    assert pool_params_from_general_settings({}) == {
        "connection_limit": DEFAULT_DB_CONNECTION_POOL_LIMIT,
        "pool_timeout": DEFAULT_DB_CONNECTION_POOL_TIMEOUT,
    }


def test_pool_params_from_general_settings_reads_all_keys():
    params = pool_params_from_general_settings(
        {
            "database_connection_pool_limit": 25,
            "database_connection_pool_timeout": 45,
            "database_connect_timeout": 15,
            "database_socket_timeout": 20,
            "database_disable_prepared_statements": "true",
            "database_extra_connection_params": {"application_name": "litellm"},
        }
    )
    assert params == {
        "connection_limit": 25,
        "pool_timeout": 45,
        "connect_timeout": 15,
        "socket_timeout": 20,
        "pgbouncer": "true",
        "application_name": "litellm",
    }


def test_pool_params_connection_timeout_takes_precedence_over_pool_timeout():
    params = pool_params_from_general_settings(
        {"database_connection_timeout": 90, "database_connection_pool_timeout": 45}
    )
    assert params["pool_timeout"] == 90


def test_apply_pool_params_covers_writer_direct_and_read_replica(monkeypatch):
    """Regression for #33021 defect 1: componentized startup must apply pool
    params to all three DB URL env vars, including the read replica which was
    never receiving them."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer:5432/db?schema=public")
    monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@writer:5432/db?schema=public")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://u:p@reader:5432/db?schema=public")

    apply_pool_params_to_db_urls({"database_connection_pool_limit": 10, "database_connection_pool_timeout": 60})

    for env_var in ("DATABASE_URL", "DIRECT_URL", "DATABASE_URL_READ_REPLICA"):
        value = os.environ[env_var]
        assert "connection_limit=10" in value
        assert "pool_timeout=60" in value
        assert "schema=public" in value


def test_apply_pool_params_is_idempotent(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@writer:5432/db?schema=public")
    settings = {"database_connection_pool_limit": 10, "database_connection_pool_timeout": 60}

    apply_pool_params_to_db_urls(settings)
    once = os.environ["DATABASE_URL"]
    apply_pool_params_to_db_urls(settings)
    twice = os.environ["DATABASE_URL"]

    assert once == twice
    assert once.count("connection_limit=") == 1


def test_apply_pool_params_skips_unset_env_vars(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DIRECT_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)

    apply_pool_params_to_db_urls({})

    assert os.getenv("DATABASE_URL") is None
    assert os.getenv("DATABASE_URL_READ_REPLICA") is None
