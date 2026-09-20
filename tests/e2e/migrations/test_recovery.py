from contextlib import ExitStack
from typing import Final, Literal
from uuid import uuid4

import pytest

from .checks import (
    COMPLETE,
    FATAL,
    GATED,
    NEXT,
    assert_completed,
    confirmed_history,
    interrupt_owner,
    assert_original_proof,
    pause_completion,
    start_replicas,
    unconfirmed,
)
from .containers import Containers, failed, ready, until, waiting
from .database import COORDINATOR_LOCK, GATE_KEY, Database
from .startup_models import Migration

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


class TestMigrationRecovery:
    @pytest.mark.parametrize("after_commit", (False, True))
    def test_container_owner_crash(self, containers: Containers, database: Database, after_commit: bool) -> None:
        interrupt_owner(containers, database, after_commit, stop_database_session=False)
        history: Final = database.history()
        assert database.query(
            "SELECT applied_steps_count FROM _prisma_migrations WHERE migration_name = %s", (COMPLETE.name,)
        ) == ((int(after_commit),),)
        with ExitStack() as stack:
            successors: Final = start_replicas(stack, containers, database, (COMPLETE if after_commit else GATED,))
            if after_commit:
                ready(successors, database)
                assert_completed(database)
            else:
                unconfirmed(successors, database)
                assert database.history() == history

    @pytest.mark.parametrize("after_commit", (False, True))
    def test_owner_and_database_session_crash(
        self, containers: Containers, database: Database, after_commit: bool
    ) -> None:
        interrupt_owner(containers, database, after_commit)
        history: Final = database.history()
        with ExitStack() as stack:
            successors: Final = start_replicas(stack, containers, database, (COMPLETE,))
            if after_commit:
                ready(successors, database)
                assert_completed(database)
                return
            unconfirmed(successors, database)
            assert database.history() == history
        with containers.start(database, (COMPLETE,)) as restarted:
            unconfirmed((restarted,), database)
            assert database.history() == history

    @pytest.mark.parametrize("later_failure", (False, True))
    def test_remaining_migrations_after_recovery(
        self, containers: Containers, database: Database, later_failure: bool
    ) -> None:
        original: Final = confirmed_history(database)
        next_migration: Final = Migration(
            NEXT.name,
            f"SELECT pg_advisory_lock({GATE_KEY}); "
            + (FATAL.script if later_failure else NEXT.script)
            + f" SELECT pg_advisory_unlock({GATE_KEY});",
        )
        with ExitStack() as stack:
            with database.lock():
                owner: Final = stack.enter_context(containers.start(database, (COMPLETE, next_migration)))

                def pending() -> bool:
                    observation: Final = owner.observe()
                    assert observation.exit_code is None and not observation.ready, (
                        "Recovered owner served before pending SQL completed"
                    )
                    return bool(database.blocked())

                until("recovering owner reached the next migration", pending)
                assert_original_proof(database, original, True)
                assert not database.exists("migration_next")
                followers: Final = start_replicas(stack, containers, database, (COMPLETE, next_migration), count=2)
                replicas: Final = (owner, *followers)
                waiting(replicas, 1)
            if later_failure:
                failed(replicas, NEXT.name)
                assert database.query(
                    "SELECT finished_at IS NULL, logs LIKE %s FROM _prisma_migrations WHERE migration_name = %s",
                    ("%MIGRATION_TEST_FATAL%", NEXT.name),
                ) == ((True, True),)
            else:
                ready(replicas, database)
                assert database.query("SELECT id FROM migration_next") == ((2,),)
            assert_original_proof(database, original, True)

    def test_second_crash_during_recovery_is_atomic(self, containers: Containers, database: Database) -> None:
        original: Final = confirmed_history(database)
        pause_completion(database)
        with database.lock():
            with containers.start(database, (COMPLETE,)) as recovering:
                until("history update blocked before commit", lambda: bool(database.blocked()))
                assert_original_proof(database, original, False)
                blocked: Final = database.blocked()
                assert len(blocked) == 1
                assert database.query("SELECT pg_terminate_backend(%s)", (blocked[0][0],)) == ((True,),)
                failed((recovering,), "Lost or could not establish v2 migration coordination")
                assert_original_proof(database, original, False)
        with ExitStack() as stack:
            ready(start_replicas(stack, containers, database, (COMPLETE,)), database)
        assert_original_proof(database, original, True)

    def test_competing_recovery_rechecks_stale_failures(self, containers: Containers, database: Database) -> None:
        original: Final = confirmed_history(database)
        with ExitStack() as stack:
            with database.lock(COORDINATOR_LOCK):
                replicas: Final = start_replicas(stack, containers, database, (COMPLETE,))
                until(
                    "all replicas observed the unfinished migration",
                    lambda: all(
                        "Waiting for the v2 migration coordinator lock" in replica.logs() for replica in replicas
                    ),
                )
                assert_original_proof(database, original, False)
            ready(replicas, database)
        assert_original_proof(database, original, True)

    @pytest.mark.parametrize(
        "fault", ("no_steps", "extra_steps", "failure_logs", "checksum", "duplicate_history", "missing_script")
    )
    def test_unproven_history_is_never_repaired(
        self,
        containers: Containers,
        database: Database,
        fault: Literal["no_steps", "extra_steps", "failure_logs", "checksum", "duplicate_history", "missing_script"],
    ) -> None:
        confirmed_history(database)
        match fault:
            case "no_steps":
                database.execute(
                    "UPDATE _prisma_migrations SET applied_steps_count = 0 WHERE migration_name = %s", (COMPLETE.name,)
                )
            case "extra_steps":
                database.execute(
                    "UPDATE _prisma_migrations SET applied_steps_count = 2 WHERE migration_name = %s", (COMPLETE.name,)
                )
            case "failure_logs":
                database.execute(
                    "UPDATE _prisma_migrations SET logs = 'permission denied' WHERE migration_name = %s",
                    (COMPLETE.name,),
                )
            case "checksum":
                database.execute(
                    "UPDATE _prisma_migrations SET checksum = %s WHERE migration_name = %s", ("0" * 64, COMPLETE.name)
                )
            case "duplicate_history":
                database.execute(
                    'INSERT INTO _prisma_migrations (id, migration_name, checksum, '
                    'applied_steps_count) SELECT %s, migration_name, checksum, '
                    'applied_steps_count FROM _prisma_migrations WHERE migration_name = %s',
                    (str(uuid4()), COMPLETE.name),
                )
            case "missing_script":
                pass
        history: Final = database.history()
        with containers.start(database, () if fault == "missing_script" else (COMPLETE,)) as replica:
            unconfirmed((replica,), database)
        assert database.history() == history
        assert database.query("SELECT id FROM migration_effect") == ((1,),)

    def test_coordinator_timeout_preserves_proof(self, containers: Containers, database: Database) -> None:
        original: Final = confirmed_history(database)
        with database.lock(COORDINATOR_LOCK):
            with containers.start(
                database, (COMPLETE,), environment={"LITELLM_MIGRATION_LOCK_TIMEOUT": "3"}
            ) as replica:
                failed((replica,), "Timed out waiting for another v2 migration resolver")
                assert_original_proof(database, original, False)
        with containers.start(database, (COMPLETE,)) as replica:
            ready((replica,), database)
        assert_original_proof(database, original, True)
