from pathlib import Path
from typing import Final

import pytest
from litellm_proxy_extras.migration_data_checks import (
    DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS,
    MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR,
    BlockedChange,
    Check,
    CouldNotVerify,
    btree_max_entry_bytes,
    checks_for_index,
    checks_for_migration,
    checks_for_statement,
    migration_data_check_timeout_seconds,
    report,
    statements,
)

MIGRATIONS: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
API_KEY_INDEX_AS_SHIPPED_IN_V1_103_0: Final = 'CREATE INDEX IF NOT EXISTS "LiteLLM_SpendLogs_api_key_startTime_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime")'


def _shape(checks: tuple[Check, ...]) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple((check.kind, check.table, check.columns) for check in checks)


@pytest.mark.parametrize(
    ("statement", "expected"),
    (
        (
            API_KEY_INDEX_AS_SHIPPED_IN_V1_103_0,
            (("too_long", "LiteLLM_SpendLogs", ("api_key", "startTime")),),
        ),
        (
            'CREATE UNIQUE INDEX CONCURRENTLY "T_a_b_key" ON "public"."T" USING btree ("a", "b" DESC)',
            (("too_long", "T", ("a", "b")), ("duplicates", "T", ("a", "b"))),
        ),
        ('CREATE INDEX "T_a_idx" ON "T" USING gin ("a")', ()),
        ('CREATE INDEX "T_a_idx" ON "T" (lower("a"))', ()),
        ('ALTER TABLE "T" ALTER COLUMN "a" SET NOT NULL', (("nulls", "T", ("a",)),)),
        ('ALTER TABLE "T" ADD COLUMN "a" TEXT NOT NULL', (("any_row", "T", ("a",)),)),
        ('ALTER TABLE "T" ADD COLUMN "a" TEXT NOT NULL DEFAULT \'x\'', ()),
        ('ALTER TABLE "T" ADD COLUMN "a" TEXT', ()),
        (
            'ALTER TABLE "T" ADD CONSTRAINT "T_r_fkey" FOREIGN KEY ("r") REFERENCES "R"("id") ON DELETE CASCADE',
            (("orphans", "T", ("r",)),),
        ),
        ('ALTER TABLE "T" ADD CONSTRAINT "T_r_fkey" FOREIGN KEY ("r") REFERENCES "R"("id") NOT VALID', ()),
        ('ALTER TABLE "T" ADD CONSTRAINT "T_a_key" UNIQUE ("a", "b")', (("duplicates", "T", ("a", "b")),)),
        (
            'ALTER TABLE "T" ADD CONSTRAINT "T_a_check" CHECK ("a" > 0 AND "b" IS NOT NULL)',
            (("check_violations", "T", ()),),
        ),
        ('ALTER TABLE "T" ADD CONSTRAINT "T_a_check" CHECK ("a" > 0) NOT VALID', ()),
        (
            'ALTER TABLE "T" ADD COLUMN "c" TEXT, ALTER COLUMN "b" SET NOT NULL, ADD CONSTRAINT "u" UNIQUE ("c")',
            (("nulls", "T", ("b",)), ("duplicates", "T", ("c",))),
        ),
        ('CREATE TABLE "T" ("id" TEXT NOT NULL, CONSTRAINT "T_pkey" PRIMARY KEY ("id"))', ()),
        ('DROP INDEX IF EXISTS "T_a_idx"', ()),
        ("SELECT 1", ()),
    ),
)
def test_a_statement_gets_one_check_per_way_existing_rows_can_make_it_fail(
    statement: str, expected: tuple[tuple[str, str, tuple[str, ...]], ...]
) -> None:
    assert _shape(checks_for_statement(statement, "migration x")) == expected


def test_a_foreign_key_check_keeps_the_referenced_table_and_columns() -> None:
    (check,) = checks_for_statement(
        'ALTER TABLE "T" ADD CONSTRAINT "fk" FOREIGN KEY ("r", "s") REFERENCES "R"("id", "sub")', "migration x"
    )
    assert (check.referenced_table, check.referenced_columns) == ("R", ("id", "sub"))


def test_a_check_constraint_keeps_its_expression_for_the_count_query() -> None:
    (check,) = checks_for_statement('ALTER TABLE "T" ADD CONSTRAINT "c" CHECK ("a" > 0)', "migration x")
    assert check.expression == '"a" > 0'


def test_comments_and_blank_statements_are_dropped_before_matching() -> None:
    migration_sql: Final = "-- CreateIndex\n" + API_KEY_INDEX_AS_SHIPPED_IN_V1_103_0 + ";\n\n-- done; really\n"
    assert statements(migration_sql) == (API_KEY_INDEX_AS_SHIPPED_IN_V1_103_0,)
    assert _shape(checks_for_migration("x", migration_sql)) == _shape(
        checks_for_statement(API_KEY_INDEX_AS_SHIPPED_IN_V1_103_0, "x")
    )


def test_every_shipped_migration_is_readable_and_the_later_indexes_on_text_columns_get_a_check() -> None:
    derived: Final = {
        folder.name: checks_for_migration(folder.name, (folder / "migration.sql").read_text())
        for folder in MIGRATIONS.iterdir()
        if folder.is_dir()
    }
    assert len(derived) > 200
    everywhere: Final = tuple(check for checks in derived.values() for check in checks)  # comprehension-ok: flattening
    assert any(
        check.kind == "too_long" and check.table == "LiteLLM_SpendLogs" and "session_id" in check.columns
        for check in everywhere
    )


def test_a_request_log_index_gets_one_entry_size_check_over_all_its_columns() -> None:
    assert _shape(
        checks_for_index("LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime"))
    ) == (("too_long", "LiteLLM_SpendLogs", ("api_key", "startTime")),)


def test_the_btree_entry_limit_on_the_default_page_is_the_number_in_postgres_error() -> None:
    # "index row size 2976 exceeds btree version 4 maximum 2704", Postgres 14, 8 KB page (BTMaxItemSize, nbtree.h)
    assert btree_max_entry_bytes(8192) == 2704


def test_the_report_names_the_rows_the_statement_and_the_fix_and_says_nothing_was_applied() -> None:
    (check,) = checks_for_index(
        "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", ("api_key", "startTime")
    )
    text: Final = report(
        (BlockedChange(check, 2, "with an index entry over 2704 bytes, largest 3232 bytes (api_key up to 3207 bytes)"),)
    )
    assert (
        '"LiteLLM_SpendLogs".api_key, startTime: 2 row(s) with an index entry over 2704 bytes, largest 3232 bytes '
        "(api_key up to 3207 bytes)"
    ) in text
    assert 'blocks: CREATE INDEX "LiteLLM_SpendLogs_api_key_startTime_idx" (request-log index)' in text
    assert 'fix:    Shorten or hash the long "LiteLLM_SpendLogs" (api_key, startTime) values' in text
    assert text.startswith("Existing rows would make a pending schema change fail:")
    assert "No schema change was made" in text


def test_a_scan_that_timed_out_is_reported_as_unverified_with_the_knob_to_turn() -> None:
    (check,) = checks_for_statement('ALTER TABLE "T" ALTER COLUMN "a" SET NOT NULL', "migration 20270101000000_x")
    text: Final = report((CouldNotVerify(check, 600.0),))
    assert '"T".a: could not be checked within 600 s, refusing rather than guessing' in text
    assert 'blocks: ALTER TABLE "T" ALTER COLUMN "a" SET NOT NULL (migration 20270101000000_x)' in text
    assert MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        (None, DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS),
        ("30", 30.0),
        ("nope", DEFAULT_MIGRATION_DATA_CHECK_TIMEOUT_SECONDS),
    ),
)
def test_the_scan_timeout_comes_from_the_environment_and_falls_back_on_garbage(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is None:
        monkeypatch.delenv(MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(MIGRATION_DATA_CHECK_TIMEOUT_ENV_VAR, raw)
    assert migration_data_check_timeout_seconds() == expected
