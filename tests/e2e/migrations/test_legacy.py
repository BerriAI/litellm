from contextlib import ExitStack
from dataclasses import replace
from typing import Final, Literal

import pytest

from .checks import COMPLETE, assert_completed, confirmed_history, assert_original_proof, start_replicas
from .containers import Containers, failed, ready
from .database import Database, Databases

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


def adopt_legacy(containers: Containers, database: Database) -> None:
    count: Final = database.query("SELECT count(*) FROM _prisma_migrations")[0][0]
    existing_keys: Final = database.query('SELECT token FROM "LiteLLM_VerificationToken" ORDER BY token')
    database.execute(
        'INSERT INTO "LiteLLM_ShadowEvalJob" (id, group_id, target_id, router_name, judge_model, '
        "shadow_percentage, max_turns, ends_at, stopped_at) VALUES ('migration-legacy', "
        "'migration-legacy', 'target', 'router', 'judge', 1, 1, now(), now())"
    )
    database.execute("DROP TABLE _prisma_migrations")
    with ExitStack() as stack:
        replicas: Final = start_replicas(stack, containers, database)
        ready(replicas, database)
        logs: Final = "\n".join(replica.logs() for replica in replicas)
        for detail in (
            "Legacy migration history was missing",
            "historical data backfills were not replayed or verified",
            "Continuing startup",
        ):
            assert detail in logs
    assert database.query("SELECT count(*) FROM _prisma_migrations") == ((count,),)
    assert database.query(
        'SELECT count(*) FROM _prisma_migrations WHERE finished_at IS NULL OR rolled_back_at IS '
        'NOT NULL OR applied_steps_count <> 0'
    ) == ((0,),)
    assert set(existing_keys).issubset(database.query('SELECT token FROM "LiteLLM_VerificationToken" ORDER BY token'))
    assert database.query("SELECT stopped_by FROM \"LiteLLM_ShadowEvalJob\" WHERE id = 'migration-legacy'") == (
        (None,),
    )


class TestLegacyMigrations:
    def test_matching_schema_warns_and_starts(self, containers: Containers, database: Database) -> None:
        adopt_legacy(containers, database)

    @pytest.mark.parametrize("fault", ("schema_drift", "custom_migrations", "empty_ledger"))
    def test_unrecognized_legacy_state_is_not_baselined(
        self, containers: Containers, database: Database, fault: str
    ) -> None:
        if fault == "empty_ledger":
            database.execute("TRUNCATE _prisma_migrations")
        else:
            database.execute("DROP TABLE _prisma_migrations")
        if fault == "schema_drift":
            database.execute('ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN key_alias CASCADE')
        with containers.start(database, (COMPLETE,) if fault == "custom_migrations" else ()) as replica:
            failed((replica,), "Cannot automatically baseline" if fault != "empty_ledger" else "migration")
        assert not database.exists("migration_effect")
        if database.exists("_prisma_migrations"):
            assert database.query(
                "SELECT count(*) FROM _prisma_migrations WHERE finished_at IS NOT NULL AND applied_steps_count <> 1"
            ) == ((0,),)

    @pytest.mark.parametrize("scenario", ("upgrade", "recovery", "legacy"))
    def test_non_default_schema(
        self, containers: Containers, databases: Databases, scenario: Literal["upgrade", "recovery", "legacy"]
    ) -> None:
        with databases.create(schema="migration tenant") as database:
            with containers.start(database) as seed:
                ready((seed,), database)
            match scenario:
                case "upgrade":
                    with ExitStack() as stack:
                        ready(start_replicas(stack, containers, database, (COMPLETE,)), database)
                    assert_completed(database)
                case "recovery":
                    original: Final = confirmed_history(database)
                    with ExitStack() as stack:
                        ready(start_replicas(stack, containers, database, (COMPLETE,)), database)
                    assert_original_proof(database, original, True)
                case "legacy":
                    adopt_legacy(containers, database)
            public: Final = replace(database, schema="public")
            assert not public.exists("_prisma_migrations")
            assert not public.exists('"LiteLLM_VerificationToken"')
