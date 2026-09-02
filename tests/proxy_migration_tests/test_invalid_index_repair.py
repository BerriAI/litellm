import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final

import pytest
from litellm_proxy_extras.utils import INDEX_REPAIR_ADVISORY_LOCK_KEY, ProxyExtrasDBManager

psycopg = pytest.importorskip("psycopg")

pytestmark = pytest.mark.timeout(120)

requires_db: Final = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)

HEALTH_TABLE: Final = "LiteLLM_HealthCheckTable"
HEALTH_INDEX: Final = "LiteLLM_HealthCheckTable_model_id_model_name_checked_at_idx"
HEALTH_INDEX_COLUMNS: Final = '"model_id", "model_name", "checked_at" DESC'
LOOKALIKE_TABLE: Final = "LiteLLMLookalikeTable"
LOOKALIKE_INDEX: Final = "LiteLLMLookalikeTable_id_idx"
PARTITIONED_TABLE: Final = "LiteLLM_PartitionedTable"
PARTITIONED_INDEX: Final = "LiteLLM_PartitionedTable_id_idx"


def _base_url() -> str:
    return os.environ["DATABASE_URL"].split("?")[0]


def _index_validity(schema: str) -> Mapping[str, bool]:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        rows = conn.execute(
            "SELECT c.relname, i.indisvalid FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s",
            (schema,),
        ).fetchall()
    return MappingProxyType(dict(rows))


def _interrupt_concurrent_build(schema: str, table: str, statement: str) -> None:
    """Abort a CONCURRENTLY build while it waits on an older snapshot, the same
    spot the deadlock loser dies at, so it leaves its index INVALID."""
    with psycopg.connect(_base_url()) as pin:
        pin.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        pin.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
        with psycopg.connect(_base_url(), autocommit=True) as builder:
            builder.execute("SET statement_timeout = '1s'")
            with pytest.raises(psycopg.errors.QueryCanceled):
                builder.execute(statement)


def _leave_invalid_index(schema: str, table: str, index: str, columns: str) -> None:
    _interrupt_concurrent_build(
        schema, table, f'CREATE INDEX CONCURRENTLY "{index}" ON "{schema}"."{table}" ({columns})'
    )


def _leave_invalid_reindex_leftover(schema: str, table: str, index: str) -> None:
    _interrupt_concurrent_build(schema, table, f'REINDEX INDEX CONCURRENTLY "{schema}"."{index}"')


def _scratch_schema(monkeypatch: pytest.MonkeyPatch, *table_definitions: str) -> Iterator[str]:
    schema: Final = f"invalid_index_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        for definition in table_definitions:
            conn.execute(f'CREATE TABLE "{schema}".{definition}')

    monkeypatch.delenv("DIRECT_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"{_base_url()}?schema={schema}")
    yield schema

    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def scratch_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    yield from _scratch_schema(
        monkeypatch,
        f'"{HEALTH_TABLE}" (model_id TEXT, model_name TEXT, checked_at TIMESTAMPTZ)',
        f'"{LOOKALIKE_TABLE}" (id TEXT)',
    )


@pytest.fixture
def empty_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    yield from _scratch_schema(monkeypatch)


@requires_db
def test_repair_rebuilds_invalid_litellm_indexes_and_leaves_lookalike_tables_alone(scratch_schema: str) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)
    _leave_invalid_index(scratch_schema, LOOKALIKE_TABLE, LOOKALIKE_INDEX, "id")
    assert _index_validity(scratch_schema) == {HEALTH_INDEX: False, LOOKALIKE_INDEX: False}

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True

    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True, LOOKALIKE_INDEX: False}


@requires_db
def test_repair_drops_leftovers_of_interrupted_rebuilds(scratch_schema: str) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)
    _leave_invalid_reindex_leftover(scratch_schema, HEALTH_TABLE, HEALTH_INDEX)
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, f"{HEALTH_TABLE}_model_id_idx_ccold", '"model_id"')
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, f"{HEALTH_TABLE}_model_id_idx_ccnew1", '"model_id"')
    before: Final = _index_validity(scratch_schema)
    assert len(before) == 4
    assert set(before.values()) == {False}

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True

    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


@requires_db
def test_repair_is_a_no_op_when_every_index_is_valid(scratch_schema: str) -> None:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE INDEX "{HEALTH_INDEX}" ON "{scratch_schema}"."{HEALTH_TABLE}" ({HEALTH_INDEX_COLUMNS})')

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True

    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


@requires_db
def test_repair_leaves_partitioned_parent_indexes_alone(scratch_schema: str) -> None:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE TABLE "{scratch_schema}"."{PARTITIONED_TABLE}" (id INT) PARTITION BY RANGE (id)')
        conn.execute(
            f'CREATE TABLE "{scratch_schema}"."{PARTITIONED_TABLE}_p0" '
            f'PARTITION OF "{scratch_schema}"."{PARTITIONED_TABLE}" FOR VALUES FROM (0) TO (10)'
        )
        conn.execute(f'CREATE INDEX "{PARTITIONED_INDEX}" ON ONLY "{scratch_schema}"."{PARTITIONED_TABLE}" (id)')
    assert _index_validity(scratch_schema) == {PARTITIONED_INDEX: False}

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True

    assert _index_validity(scratch_schema) == {PARTITIONED_INDEX: False}


@requires_db
def test_repair_yields_to_the_replica_holding_the_repair_lock(scratch_schema: str) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)

    with psycopg.connect(_base_url(), autocommit=True) as other_replica:
        other_replica.execute("SELECT pg_advisory_lock(%s)", (INDEX_REPAIR_ADVISORY_LOCK_KEY,))
        assert ProxyExtrasDBManager.repair_invalid_indexes() is False
        assert _index_validity(scratch_schema) == {HEALTH_INDEX: False}

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True
    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


@requires_db
def test_repair_gives_up_on_a_blocked_rebuild_and_finishes_it_on_the_next_startup(scratch_schema: str) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)

    with psycopg.connect(_base_url()) as pin:
        pin.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        pin.execute(f'SELECT count(*) FROM "{scratch_schema}"."{HEALTH_TABLE}"')
        assert ProxyExtrasDBManager.repair_invalid_indexes(lock_timeout="1s") is False
        blocked: Final = _index_validity(scratch_schema)
        assert blocked[HEALTH_INDEX] is False
        assert [name for name in blocked if name.endswith("_ccnew")]

    assert ProxyExtrasDBManager.repair_invalid_indexes() is True
    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


def _hold_snapshot(schema: str, table: str, pinned: threading.Event, seconds: float) -> None:
    with psycopg.connect(_base_url()) as pin:
        pin.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        pin.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
        pinned.set()
        pin.execute("SELECT pg_sleep(%s)", (seconds,))


@requires_db
def test_repair_outlives_a_statement_timeout_passed_through_database_url_options(
    scratch_schema: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)
    monkeypatch.setenv("DATABASE_URL", f"{_base_url()}?schema={scratch_schema}&options=-c%20statement_timeout%3D2000")
    pinned: Final = threading.Event()
    holder: Final = threading.Thread(target=_hold_snapshot, args=(scratch_schema, HEALTH_TABLE, pinned, 5.0))
    holder.start()
    pinned.wait()
    try:
        assert ProxyExtrasDBManager.repair_invalid_indexes() is True
    finally:
        holder.join()

    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


@requires_db
def test_repair_defaults_to_the_public_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    table: Final = f"LiteLLM_ScratchTable_{uuid.uuid4().hex[:8]}"
    index: Final = f"{table}_id_idx"
    monkeypatch.setenv("DATABASE_URL", _base_url())
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE TABLE public."{table}" (id TEXT)')
    try:
        _leave_invalid_index("public", table, index, "id")
        assert _index_validity("public")[index] is False

        assert ProxyExtrasDBManager.repair_invalid_indexes() is True

        assert _index_validity("public")[index] is True
    finally:
        with psycopg.connect(_base_url(), autocommit=True) as conn:
            conn.execute(f'DROP TABLE public."{table}"')


def test_repair_survives_an_unreachable_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIRECT_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:9/x?schema=whatever")

    assert ProxyExtrasDBManager.repair_invalid_indexes() is False


@requires_db
def test_repair_runs_over_direct_url_when_set(scratch_schema: str) -> None:
    _leave_invalid_index(scratch_schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("DIRECT_URL", _base_url())
        env.setenv("DATABASE_URL", f"postgresql://u:p@127.0.0.1:9/x?schema={scratch_schema}")
        assert ProxyExtrasDBManager.repair_invalid_indexes() is True

    assert _index_validity(scratch_schema) == {HEALTH_INDEX: True}


def _invalidate_deployed_index(schema: str) -> None:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'DROP INDEX "{schema}"."{HEALTH_INDEX}"')
    _leave_invalid_index(schema, HEALTH_TABLE, HEALTH_INDEX, HEALTH_INDEX_COLUMNS)


@requires_db
@pytest.mark.timeout(300)
@pytest.mark.parametrize("use_v2_resolver", [True, False])
def test_setup_database_repairs_the_index_after_a_recovered_deploy(empty_schema: str, use_v2_resolver: bool) -> None:
    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True
    _invalidate_deployed_index(empty_schema)
    assert _index_validity(empty_schema)[HEALTH_INDEX] is False

    assert ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=use_v2_resolver) is True

    assert _index_validity(empty_schema)[HEALTH_INDEX] is True
