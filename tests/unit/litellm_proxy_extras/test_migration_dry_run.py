from typing import Final

import pytest
from litellm_proxy_extras.migration_dry_run import Applied, Baseline, DryRun, Rejected, report, transactional

INDEX: Final = 'CREATE INDEX IF NOT EXISTS "a_idx" ON "T"("c")'


@pytest.mark.parametrize(
    ("migration_sql", "expected"),
    (
        ('CREATE INDEX CONCURRENTLY IF NOT EXISTS "a_idx" ON "T"("c")', INDEX),
        ('create unique index concurrently "u" on "T"("d")', 'create unique index "u" on "T"("d")'),
        ('DROP INDEX CONCURRENTLY "x"', 'DROP INDEX "x"'),
        ('REINDEX INDEX CONCURRENTLY "y"', 'REINDEX INDEX "y"'),
        (INDEX, INDEX),
        ('ALTER TABLE "T" ALTER COLUMN "c" SET NOT NULL', 'ALTER TABLE "T" ALTER COLUMN "c" SET NOT NULL'),
        ("-- built CONCURRENTLY by the index job\n" + INDEX, "-- built CONCURRENTLY by the index job\n" + INDEX),
    ),
    ids=("concurrently", "unique_lowercase", "drop", "reindex", "plain_index", "alter", "word_in_comment"),
)
def test_only_concurrent_index_statements_change_so_the_migration_can_run_in_a_transaction(
    migration_sql: str, expected: str
) -> None:
    assert transactional(migration_sql) == expected


def test_a_clean_dry_run_lists_each_migration_with_the_time_the_upgrade_will_hold_its_locks() -> None:
    text: Final = report(DryRun((Applied("20270101000000_a", 0.04), Applied("20270102000000_b", 12.345)), None))
    assert text.startswith(
        "2 pending migration(s) applied one at a time in a dry run, each rolled back before the next"
    )
    assert "\n  20270101000000_a: 0.0 s\n  20270102000000_b: 12.3 s" in text


def test_no_pending_migration_is_said_plainly() -> None:
    assert report(DryRun((), None)) == "No pending migrations, this database is on this version's schema."


def test_a_rejected_migration_is_reported_with_postgres_own_words_and_what_ran_before_it() -> None:
    error: Final = 'column "team_alias" of relation "LiteLLM_TeamTable" contains null values'
    text: Final = report(DryRun((Applied("20270101000000_a", 0.5),), Rejected("20270102000000_b", error, 3.25)))
    assert text.startswith(
        "Migration 20270102000000_b fails on this database after 3.2 s, nothing was applied.\n" + error
    )
    assert text.endswith("Applied before it in the dry run:\n  20270101000000_a: 0.5 s")


def test_a_rejection_on_the_first_migration_lists_nothing_before_it() -> None:
    text: Final = report(DryRun((), Rejected("20270101000000_a", 'syntax error at or near "NOPE"', 0.0)))
    assert "Applied before it" not in text
    assert text.endswith('syntax error at or near "NOPE"')


def test_a_migration_that_needed_the_one_before_it_says_which_were_replayed_in_front() -> None:
    text: Final = report(
        DryRun((Applied("20270101000000_a", 0.1), Applied("20270102000000_b", 0.4, ("20270101000000_a",))), None)
    )
    assert text.endswith(
        "\n  20270101000000_a: 0.1 s\n  20270102000000_b: 0.4 s (depends on 20270101000000_a, replayed in front of it)"
    )


def test_a_database_without_history_that_matches_this_build_has_nothing_to_dry_run() -> None:
    result: Final = DryRun((), None, Baseline(None))
    assert result.ok is True
    assert report(result) == (
        "No migration history and the schema matches this build, the proxy baselines it at boot, nothing to dry-run."
    )


def test_a_database_without_history_the_boot_refuses_to_baseline_reports_the_refusal() -> None:
    result: Final = DryRun(
        (), None, Baseline('the schema differs from this build. prisma migrate diff:\n-- AlterTable\nALTER TABLE "x"')
    )
    assert result.ok is False
    assert report(result) == (
        "No migration history and the proxy cannot baseline it at boot: "
        'the schema differs from this build. prisma migrate diff:\n-- AlterTable\nALTER TABLE "x"'
    )
