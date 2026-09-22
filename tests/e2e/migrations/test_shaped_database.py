from pathlib import Path
from typing import Final

import pytest

from .containers import Containers, ready
from .database import Database
from .upgrade import assert_history_clean, assert_upgraded, confirm, migration_names, provision

SPEND_ROWS: Final = 20_000
PARTITION_SCRIPT: Final = Path(__file__).parents[3] / "db_scripts" / "partition_spend_logs.sql"

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


def seed_spend_logs(database: Database, rows: int) -> None:
    database.execute(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime") '
        "SELECT 'upgrade-shape-' || g, 'acompletion', now() - (g || ' seconds')::interval, "
        "now() - (g || ' seconds')::interval FROM generate_series(1, %s) AS g",
        (rows,),
    )
    assert database.query('SELECT count(*) FROM "LiteLLM_SpendLogs"') == ((rows,),)


def partition_spend_logs(database: Database) -> None:
    database.execute(PARTITION_SCRIPT.read_bytes())
    assert database.query("SELECT relkind::text FROM pg_class WHERE oid = to_regclass('\"LiteLLM_SpendLogs\"')") == (
        ("p",),
    ), "The partition script did not leave a partitioned LiteLLM_SpendLogs behind"


class TestPopulatedDatabaseUpgrade:
    def test_upgrade_completes_and_preserves_a_populated_spend_log(
        self, containers: Containers, baseline_image: str, baseline_database: Database
    ) -> None:
        with containers.using(baseline_image).start(baseline_database) as old:
            ready((old,), baseline_database)
            key, alias = provision(old)
        seed_spend_logs(baseline_database, SPEND_ROWS)
        before: Final = migration_names(baseline_database)
        with containers.start(baseline_database) as new:
            ready((new,), baseline_database)
            assert_upgraded(before, migration_names(baseline_database))
            confirm(new, key, alias)
        assert_history_clean(baseline_database)
        assert baseline_database.query('SELECT count(*) FROM "LiteLLM_SpendLogs"') == ((SPEND_ROWS,),), (
            "The upgrade lost spend rows"
        )
        assert baseline_database.query(
            'SELECT count(*) FROM "LiteLLM_SpendLogs" WHERE "startTime" IS NULL OR "endTime" IS NULL'
        ) == ((0,),), "The upgrade nulled timestamps on existing spend rows"

    def test_upgrade_completes_on_a_partitioned_spend_log(
        self, containers: Containers, baseline_image: str, baseline_database: Database
    ) -> None:
        with containers.using(baseline_image).start(baseline_database) as old:
            ready((old,), baseline_database)
            key, alias = provision(old)
        partition_spend_logs(baseline_database)
        seed_spend_logs(baseline_database, SPEND_ROWS)
        before: Final = migration_names(baseline_database)
        with containers.start(baseline_database) as new:
            ready((new,), baseline_database)
            assert_upgraded(before, migration_names(baseline_database))
            confirm(new, key, alias)
        assert_history_clean(baseline_database)
        assert baseline_database.query(
            "SELECT relkind::text FROM pg_class WHERE oid = to_regclass('\"LiteLLM_SpendLogs\"')"
        ) == (("p",),), "The upgrade replaced the partitioned LiteLLM_SpendLogs with a plain table"
        assert baseline_database.query('SELECT count(*) FROM "LiteLLM_SpendLogs"') == ((SPEND_ROWS,),), (
            "The upgrade lost spend rows from the partitioned table"
        )
        assert baseline_database.query(
            "SELECT count(*) FROM pg_indexes WHERE tablename IN ('LiteLLM_SpendLogs', 'LiteLLM_SpendLogs_pdefault') "
            "AND indexdef LIKE %s",
            ("%(litellm_call_id)",),
        ) == ((2,),), "The litellm_call_id index was not built on the partitioned parent and propagated to its partition"
