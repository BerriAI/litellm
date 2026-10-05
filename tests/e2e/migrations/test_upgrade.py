from contextlib import ExitStack
from typing import Final

import pytest

from .checks import start_replicas
from .containers import Containers, ready
from .database import Database
from .upgrade import assert_history_clean, assert_upgraded, confirm, migration_names, provision

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


class TestReleaseUpgrade:
    def test_candidate_applies_the_pending_release_migrations(
        self, containers: Containers, baseline_database: Database
    ) -> None:
        before: Final = migration_names(baseline_database)
        with containers.start(baseline_database) as replica:
            ready((replica,), baseline_database)
        assert_upgraded(before, migration_names(baseline_database))
        assert_history_clean(baseline_database)

    def test_upgrade_preserves_keys_minted_by_the_baseline_release(
        self, containers: Containers, baseline_image: str, baseline_database: Database
    ) -> None:
        with containers.using(baseline_image).start(baseline_database) as old:
            ready((old,), baseline_database)
            key, alias = provision(old)
            confirm(old, key, alias)
        before: Final = migration_names(baseline_database)
        with containers.start(baseline_database) as new:
            ready((new,), baseline_database)
            assert_upgraded(before, migration_names(baseline_database))
            confirm(new, key, alias)

    def test_concurrent_replicas_upgrade_a_baseline_database_once(
        self, containers: Containers, baseline_database: Database
    ) -> None:
        before: Final = migration_names(baseline_database)
        with ExitStack() as stack:
            ready(start_replicas(stack, containers, baseline_database), baseline_database)
        assert_upgraded(before, migration_names(baseline_database))
        assert_history_clean(baseline_database)
        assert baseline_database.query("SELECT count(*) FROM _prisma_migrations WHERE applied_steps_count > 1") == (
            (0,),
        ), "A migration was executed more than once across the upgrading replicas"
