from typing import Final

import pytest

from .containers import Containers, ready
from .database import Database
from .upgrade import (
    CACHED_PLAN,
    assert_history_clean,
    assert_upgraded,
    auth_traffic,
    confirm,
    keep_serving,
    migration_names,
    provision,
)

pytestmark: Final = [pytest.mark.e2e, pytest.mark.migration_startup]


class TestRollingUpgrade:
    def test_baseline_replica_keeps_serving_while_the_candidate_migrates(
        self, containers: Containers, baseline_image: str, baseline_database: Database
    ) -> None:
        with containers.using(baseline_image).start(baseline_database) as old:
            ready((old,), baseline_database)
            key, _ = provision(old)
            before: Final = migration_names(baseline_database)
            with auth_traffic(old, key) as traffic:
                keep_serving(traffic, "the baseline replica authenticating before the upgrade")
                with containers.start(baseline_database) as new:
                    ready((new,), baseline_database)
                    assert_upgraded(before, migration_names(baseline_database))
                    keep_serving(traffic, "the baseline replica authenticating after the schema moved")
                    with auth_traffic(old, provision(new)[0]) as uncached:
                        keep_serving(uncached, "the baseline replica resolving a key minted after the schema moved")
            assert_history_clean(baseline_database)
            assert CACHED_PLAN not in old.logs(), "The baseline replica hit a stale prepared statement"
            assert old.state().Running, "The baseline replica died during the upgrade"

    def test_both_releases_serve_and_share_keys_during_the_overlap(
        self, containers: Containers, baseline_image: str, baseline_database: Database
    ) -> None:
        with containers.using(baseline_image).start(baseline_database) as old:
            ready((old,), baseline_database)
            old_key, old_alias = provision(old)
            before: Final = migration_names(baseline_database)
            with containers.start(baseline_database) as new:
                ready((new,), baseline_database)
                assert_upgraded(before, migration_names(baseline_database))
                new_key, new_alias = provision(new)
                with auth_traffic(old, old_key) as old_traffic, auth_traffic(new, new_key) as new_traffic:
                    keep_serving(old_traffic, "the baseline replica serving through the overlap")
                    keep_serving(new_traffic, "the candidate replica serving through the overlap")
                confirm(old, new_key, new_alias)
                confirm(new, old_key, old_alias)
            assert CACHED_PLAN not in old.logs(), "The baseline replica hit a stale prepared statement"
