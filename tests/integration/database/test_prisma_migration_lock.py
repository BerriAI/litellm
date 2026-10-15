import os
import secrets
from typing import Final

import psycopg
import pytest
from litellm_proxy_extras.migration_lock import MIGRATION_LOCK_KEY, PRISMA_ADVISORY_LOCK_KEY, prisma_migration_lock

pytestmark = pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="needs a live Postgres")


def _can_take_prismas_session_lock(connection: psycopg.Connection[tuple[object, ...]]) -> bool:
    row: Final = connection.execute("SELECT pg_try_advisory_lock(%s)", (PRISMA_ADVISORY_LOCK_KEY,)).fetchone()
    taken: Final = row is not None and row[0] is True
    if taken:
        connection.execute("SELECT pg_advisory_unlock(%s)", (PRISMA_ADVISORY_LOCK_KEY,))
    return taken


def _granted_advisory_keys(connection: psycopg.Connection[tuple[object, ...]]) -> frozenset[int]:
    rows: Final = connection.execute(
        "SELECT (classid::bigint << 32) | objid::bigint FROM pg_locks WHERE locktype = 'advisory' AND granted"
    ).fetchall()
    return frozenset(int(row[0]) for row in rows)


def _index_builds_concurrently_now(connection: psycopg.Connection[tuple[object, ...]]) -> bool:
    table: Final = f'"lit_cic_{secrets.token_hex(4)}"'
    connection.execute(f"CREATE TABLE {table} (a int)")
    try:
        connection.execute("SET lock_timeout = '5s'")
        connection.execute(f"CREATE INDEX CONCURRENTLY ON {table} (a)")
    except psycopg.errors.LockNotAvailable:
        return False
    finally:
        connection.execute("RESET lock_timeout")
        connection.execute(f"DROP TABLE {table}")
    return True


def test_prisma_migration_lock_does_not_stall_a_concurrent_index_build():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as peer:
        with prisma_migration_lock(os.environ["DATABASE_URL"]):
            assert _index_builds_concurrently_now(peer)


def test_prisma_migration_lock_holds_prismas_key_on_a_pinned_connection_and_leaves_nothing_behind():
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as peer:
        assert _can_take_prismas_session_lock(peer)

        with prisma_migration_lock(os.environ["DATABASE_URL"]):
            held: Final = _granted_advisory_keys(peer)
            assert PRISMA_ADVISORY_LOCK_KEY in held
            assert MIGRATION_LOCK_KEY not in held
            assert not _can_take_prismas_session_lock(peer)

        assert _granted_advisory_keys(peer).isdisjoint({PRISMA_ADVISORY_LOCK_KEY, MIGRATION_LOCK_KEY})
        assert _can_take_prismas_session_lock(peer)


def test_prisma_migration_lock_does_not_wait_for_the_coordinator_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_MIGRATION_LOCK_TIMEOUT", "3")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as peer:
        peer.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))

        with prisma_migration_lock(os.environ["DATABASE_URL"]):
            assert PRISMA_ADVISORY_LOCK_KEY in _granted_advisory_keys(peer)


def test_prisma_migration_lock_times_out_when_another_prisma_migration_holds_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_MIGRATION_LOCK_TIMEOUT", "3")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as peer:
        peer.execute("SELECT pg_advisory_lock(%s)", (PRISMA_ADVISORY_LOCK_KEY,))

        with pytest.raises(RuntimeError, match="Timed out waiting for another Prisma migration"):
            with prisma_migration_lock(os.environ["DATABASE_URL"]):
                pytest.fail("Prisma migration lock was acquired while another migration held it")
