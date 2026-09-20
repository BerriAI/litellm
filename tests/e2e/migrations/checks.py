import hashlib
from contextlib import ExitStack
from typing import Final
from uuid import uuid4

from psycopg import sql

from .containers import Containers, Replica, failed, until
from .database import GATE_KEY, Database
from .startup_models import Migration

COMPLETE_SQL: Final = "CREATE TABLE migration_effect (id int PRIMARY KEY); INSERT INTO migration_effect VALUES (1);"
COMPLETE: Final = Migration("20990101000000_startup_test", COMPLETE_SQL)
NEXT: Final = Migration(
    "20990102000000_next_test",
    "CREATE TABLE migration_next (id int PRIMARY KEY); INSERT INTO migration_next VALUES (2);",
)
FATAL: Final = Migration(COMPLETE.name, "DO $$ BEGIN RAISE EXCEPTION 'MIGRATION_TEST_FATAL'; END $$;")
GATED: Final = Migration(
    COMPLETE.name, f"SELECT pg_advisory_lock({GATE_KEY}); {COMPLETE.script} SELECT pg_advisory_unlock({GATE_KEY});"
)


def start_replicas(
    stack: ExitStack, containers: Containers, database: Database, migrations: tuple[Migration, ...] = (), count: int = 3
) -> tuple[Replica, ...]:
    return tuple(stack.enter_context(containers.start(database, migrations)) for _ in range(count))


def assert_completed(database: Database, migration: Migration = COMPLETE) -> None:
    assert database.query(
        'SELECT finished_at IS NOT NULL, rolled_back_at IS NULL, applied_steps_count FROM '
        '_prisma_migrations WHERE migration_name = %s',
        (migration.name,),
    ) == ((True, True, 1),), "Expected exactly one successful SQL execution"
    assert database.query("SELECT id FROM migration_effect") == ((1,),)


def confirmed_history(database: Database) -> str:
    database.execute(COMPLETE_SQL)
    row_id: Final = str(uuid4())
    database.execute(
        "INSERT INTO _prisma_migrations (id, migration_name, checksum, applied_steps_count) VALUES (%s, %s, %s, 1)",
        (row_id, COMPLETE.name, hashlib.sha256(COMPLETE.script.encode()).hexdigest()),
    )
    return row_id


def assert_original_proof(database: Database, row_id: str, finished: bool) -> None:
    assert database.query(
        'SELECT id, applied_steps_count, finished_at IS NOT NULL, rolled_back_at IS NULL FROM '
        '_prisma_migrations WHERE migration_name = %s',
        (COMPLETE.name,),
    ) == ((row_id, 1, finished, True),), "Recovery lost or replaced the original durable SQL proof"
    assert database.query("SELECT id FROM migration_effect") == ((1,),)


def pause_completion(database: Database) -> None:
    database.execute(
        sql.SQL(
            "CREATE FUNCTION migration_pause() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
            "IF NEW.migration_name = {name} AND NEW.finished_at IS NOT NULL THEN "
            "PERFORM pg_advisory_lock({gate}); PERFORM pg_advisory_unlock({gate}); END IF; RETURN NEW; END $$; "
            'CREATE TRIGGER migration_pause BEFORE UPDATE ON _prisma_migrations FOR EACH ROW '
            'EXECUTE FUNCTION migration_pause()'
        ).format(name=sql.Literal(COMPLETE.name), gate=sql.Literal(GATE_KEY))
    )


def interrupt_owner(
    containers: Containers, database: Database, after_commit: bool, *, stop_database_session: bool = True
) -> None:
    if after_commit:
        pause_completion(database)
    with database.lock():
        with containers.start(database, (COMPLETE if after_commit else GATED,)) as owner:
            until("migration at the intended crash boundary", lambda: bool(database.blocked()))
            assert database.exists("migration_effect") == after_commit
            assert database.query(
                "SELECT finished_at IS NULL FROM _prisma_migrations WHERE migration_name = %s", (COMPLETE.name,)
            ) == ((True,),)
            blocked: Final = database.blocked()
            assert len(blocked) == 1
            backend: Final = blocked[0][0]
            assert owner.state().Running
            owner.kill()
            assert owner.state().ExitCode == 137
            if stop_database_session:
                database.query("SELECT pg_terminate_backend(%s)", (backend,))
                until(
                    "terminated migration backend released",
                    lambda: not database.query("SELECT pid FROM pg_stat_activity WHERE pid = %s", (backend,)),
                )
                assert database.query(
                    "SELECT finished_at IS NULL, applied_steps_count FROM _prisma_migrations WHERE migration_name = %s",
                    (COMPLETE.name,),
                ) == ((True, int(after_commit)),)
                assert database.exists("migration_effect") == after_commit
    if not stop_database_session:
        until(
            "database backend noticed container death",
            lambda: not database.query("SELECT pid FROM pg_stat_activity WHERE pid = %s", (backend,)),
            60,
        )


def unconfirmed(replicas: tuple[Replica, ...], database: Database) -> None:
    failed(replicas, "Migration completion could not be verified")
    started: Final = str(
        database.query(
            "SELECT to_char(started_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') FROM "
            '_prisma_migrations WHERE migration_name = %s',
            (COMPLETE.name,),
        )[0][0]
    )
    for replica in replicas:
        assert_guidance(replica.logs(), started)


def assert_guidance(log: str, started: str) -> None:
    for detail in (
        COMPLETE.name,
        started,
        "cannot determine whether its SQL committed",
        "_prisma_migrations",
        "migration.sql",
        "Only after verifying every migration change is present",
        "prisma migrate resolve --applied <migration_name>",
        "Only after verifying no migration changes remain",
        "prisma migrate resolve --rolled-back <migration_name>",
        "leave migration history unchanged",
        "Repeated restarts alone",
    ):
        assert detail in log, f"Missing recovery guidance: {detail}"
