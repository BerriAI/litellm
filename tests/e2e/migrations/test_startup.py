from contextlib import ExitStack
from typing import Final

import pytest

from .checks import COMPLETE, FATAL, GATED, assert_completed, start_replicas
from .containers import Containers, failed, ready, until, waiting
from .database import PRISMA_LOCK, Database, Databases, restricted_user
from .startup_models import Migration

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


class TestMigrationStartup:
    @pytest.mark.parametrize("replicas,v2", ((1, True), (3, True), (1, False)))
    def test_fresh_database(self, containers: Containers, databases: Databases, replicas: int, v2: bool) -> None:
        with databases.create() as database, ExitStack() as stack:
            ready(tuple(stack.enter_context(containers.start(database, v2=v2)) for _ in range(replicas)), database)
            assert database.query(
                "SELECT count(*) FROM _prisma_migrations WHERE finished_at IS NULL AND rolled_back_at IS NULL"
            ) == ((0,),)
            assert database.query("SELECT count(*) > 0 FROM _prisma_migrations") == ((True,),)

    def test_concurrent_upgrade(self, containers: Containers, database: Database) -> None:
        with ExitStack() as stack:
            ready(start_replicas(stack, containers, database, (COMPLETE,)), database)
            assert_completed(database)

    def test_waiters_survive_prolonged_contention(self, containers: Containers, database: Database) -> None:
        with ExitStack() as stack:
            with database.lock():
                owner: Final = stack.enter_context(containers.start(database, (GATED,)))
                until("owner blocked in migration SQL", lambda: bool(database.blocked()))
                followers: Final = start_replicas(stack, containers, database, (GATED,), count=2)
                until("both followers attempted Prisma locking", lambda: len(database.blocked(PRISMA_LOCK)) == 2)
                waiting((owner, *followers), 120)
            ready((owner, *followers), database)
            assert_completed(database, GATED)

    def test_lock_deadline_then_restart(self, containers: Containers, database: Database) -> None:
        history: Final = database.history()
        with database.lock(PRISMA_LOCK):
            with containers.start(
                database, (COMPLETE,), environment={"LITELLM_MIGRATION_LOCK_TIMEOUT": "12"}
            ) as replica:
                until("Prisma lock contention", lambda: bool(database.blocked(PRISMA_LOCK)))
                failed((replica,), "Timed out waiting for")
                assert database.history() == history
                assert not database.exists("migration_effect")
        with containers.start(database, (COMPLETE,)) as restarted:
            ready((restarted,), database)
            assert_completed(database)

    def test_fatal_sql(self, containers: Containers, database: Database) -> None:
        with ExitStack() as stack:
            replicas: Final = start_replicas(stack, containers, database, (FATAL,))
            failed(replicas, COMPLETE.name)
            assert database.query(
                "SELECT count(*) FROM _prisma_migrations WHERE migration_name = %s AND logs LIKE %s AND finished_at IS NULL",
                (COMPLETE.name, "%MIGRATION_TEST_FATAL%"),
            ) == ((1,),)

    def test_duplicate_object_does_not_hide_incomplete_sql(self, containers: Containers, database: Database) -> None:
        database.execute(
            "CREATE TABLE migration_existing (id int PRIMARY KEY); INSERT INTO migration_existing VALUES (42)"
        )
        migration: Final = Migration(
            COMPLETE.name, "CREATE TABLE migration_existing (id int PRIMARY KEY); " + COMPLETE.script
        )
        with ExitStack() as stack:
            failed(start_replicas(stack, containers, database, (migration,)), COMPLETE.name)
            assert not database.exists("migration_effect")
            assert database.query("SELECT id FROM migration_existing") == ((42,),)
            assert database.query(
                "SELECT finished_at IS NULL FROM _prisma_migrations WHERE migration_name = %s", (COMPLETE.name,)
            ) == ((True,),)

    @pytest.mark.parametrize("v2", (True, False))
    def test_restart_preserves_history_and_data(self, containers: Containers, database: Database, v2: bool) -> None:
        history: Final = database.history()
        before: Final = database.query('SELECT token FROM "LiteLLM_VerificationToken" ORDER BY token')
        for _ in range(2):
            with containers.start(database, v2=v2) as replica:
                ready((replica,), database)
        assert database.history() == history
        assert set(before).issubset(database.query('SELECT token FROM "LiteLLM_VerificationToken" ORDER BY token'))

    def test_disabled_migrations(self, containers: Containers, database: Database) -> None:
        history: Final = database.history()
        with containers.start(database, (FATAL,), disabled=True) as replica:
            ready((replica,), database)
        assert database.history() == history

    def test_insufficient_privileges(self, containers: Containers, database: Database) -> None:
        history: Final = database.history()
        with restricted_user(database) as limited:
            with containers.start(limited, (COMPLETE,)) as replica:
                failed((replica,), "permission denied")
        assert database.history() == history
        assert not database.exists("migration_effect")
