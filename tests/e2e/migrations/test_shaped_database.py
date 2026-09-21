from typing import Final

import pytest

from .containers import Containers, ready
from .database import Database
from .upgrade import assert_history_clean, assert_upgraded, confirm, migration_names, provision

SPEND_ROWS: Final = 20_000

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


def seed_spend_logs(database: Database, rows: int) -> None:
    database.execute(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id, call_type, "startTime", "endTime") '
        "SELECT 'upgrade-shape-' || g, 'acompletion', now() - (g || ' seconds')::interval, "
        "now() - (g || ' seconds')::interval FROM generate_series(1, %s) AS g",
        (rows,),
    )
    assert database.query('SELECT count(*) FROM "LiteLLM_SpendLogs"') == ((rows,),)


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
