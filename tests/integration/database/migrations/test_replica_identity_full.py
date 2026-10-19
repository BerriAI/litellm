"""Coverage for the opt-in REPLICA IDENTITY FULL post-migration step.

The DB-backed tests run against the same Postgres the migration suite uses, in
a throwaway schema so they cannot disturb the migrated tables.
"""

import os
import uuid

import pytest

from litellm_proxy_extras.replica_identity import (
    REPLICA_IDENTITY_FULL_ENV_VAR,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager

psycopg = pytest.importorskip("psycopg")

requires_db = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)


def _base_url() -> str:
    return os.environ["DATABASE_URL"].split("?")[0]


def _replica_identities(schema: str) -> dict:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        rows = conn.execute(
            "SELECT c.relname, c.relreplident FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relkind = 'r'",
            (schema,),
        ).fetchall()
    return dict(rows)


@pytest.fixture
def scratch_schema(monkeypatch):
    """A schema holding two LiteLLM tables and one foreign table, all at the default."""
    schema = f"replica_identity_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(
            f'CREATE TABLE "{schema}"."LiteLLM_ScratchTable" (id TEXT PRIMARY KEY, note TEXT)'
        )
        conn.execute(f'CREATE TABLE "{schema}"."LiteLLM_ScratchSibling" (id TEXT PRIMARY KEY)')
        conn.execute(f'CREATE TABLE "{schema}"."ScratchForeignTable" (id TEXT PRIMARY KEY)')

    monkeypatch.setenv("DATABASE_URL", f"{_base_url()}?schema={schema}")
    yield schema

    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@requires_db
def test_applies_full_to_litellm_tables_only(scratch_schema, monkeypatch):
    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")

    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is True

    identities = _replica_identities(scratch_schema)
    assert identities["LiteLLM_ScratchTable"] == "f"
    assert identities["LiteLLM_ScratchSibling"] == "f"
    assert identities["ScratchForeignTable"] == "d"


@requires_db
def test_a_locked_table_does_not_block_the_others(scratch_schema, monkeypatch):
    """ALTER TABLE needs an exclusive lock, so a table busy with a long read has
    to be skipped for the next run instead of stalling every other table behind it."""
    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")

    with psycopg.connect(_base_url()) as holder:
        holder.execute(f'SELECT * FROM "{scratch_schema}"."LiteLLM_ScratchTable"')
        assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is True

    identities = _replica_identities(scratch_schema)
    assert identities["LiteLLM_ScratchTable"] == "d"
    assert identities["LiteLLM_ScratchSibling"] == "f"


@requires_db
def test_leaves_tables_alone_when_not_requested(scratch_schema, monkeypatch):
    monkeypatch.delenv(REPLICA_IDENTITY_FULL_ENV_VAR, raising=False)

    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is False
    assert _replica_identities(scratch_schema)["LiteLLM_ScratchTable"] == "d"


@requires_db
def test_is_idempotent_across_runs(scratch_schema, monkeypatch):
    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")

    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is True
    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is True

    assert _replica_identities(scratch_schema)["LiteLLM_ScratchTable"] == "f"


@requires_db
def test_reports_failure_without_raising(scratch_schema, monkeypatch):
    """A run that cannot execute the statement must not take the migration down."""
    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")
    monkeypatch.setattr(
        ProxyExtrasDBManager,
        "_get_prisma_dir",
        staticmethod(lambda: "/nonexistent/prisma/dir"),
    )

    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is False
    assert _replica_identities(scratch_schema)["LiteLLM_ScratchTable"] == "d"






