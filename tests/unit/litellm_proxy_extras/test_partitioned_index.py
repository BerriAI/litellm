from pathlib import Path
from typing import Final

import pytest
from litellm_proxy_extras.partitioned_index import ConcurrentIndexMigration, parse_concurrent_index_migration

PACKAGE: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras"
CALL_ID_MIGRATION: Final = PACKAGE / "migrations" / "20260831120001_spend_logs_litellm_call_id_index" / "migration.sql"


def test_the_shipped_call_id_migration_parses_to_its_index_table_and_definition() -> None:
    assert parse_concurrent_index_migration(CALL_ID_MIGRATION.read_text()) == ConcurrentIndexMigration(
        index="LiteLLM_SpendLogs_litellm_call_id_idx",
        table="LiteLLM_SpendLogs",
        definition='("litellm_call_id")',
    )


@pytest.mark.parametrize(
    "script,expected",
    (
        (
            'CREATE INDEX CONCURRENTLY "T_a_idx" ON "T" USING btree ("a" DESC) WHERE "a" IS NOT NULL;',
            ConcurrentIndexMigration("T_a_idx", "T", 'USING btree ("a" DESC) WHERE "a" IS NOT NULL'),
        ),
        (
            "-- CreateIndex\ncreate index concurrently if not exists t_a_idx on only t (a);\n",
            ConcurrentIndexMigration("t_a_idx", "t", "(a)"),
        ),
        ('CREATE INDEX IF NOT EXISTS "T_a_idx" ON "T"("a");', None),
        ('CREATE UNIQUE INDEX CONCURRENTLY "T_a_idx" ON "T"("a");', None),
        ('CREATE INDEX CONCURRENTLY "T_a_idx" ON "T"("a"); CREATE INDEX CONCURRENTLY "T_b_idx" ON "T"("b");', None),
        ('ALTER TABLE "T" ADD COLUMN "a" TEXT;', None),
        ("-- only a comment mentioning CREATE INDEX CONCURRENTLY x ON y (z);\n", None),
    ),
)
def test_only_a_single_plain_concurrent_index_statement_is_recognised(
    script: str, expected: "ConcurrentIndexMigration | None"
) -> None:
    assert parse_concurrent_index_migration(script) == expected


@pytest.mark.parametrize(
    "index,table,partition,expected",
    (
        (
            "LiteLLM_SpendLogs_litellm_call_id_idx",
            "LiteLLM_SpendLogs",
            "LiteLLM_SpendLogs_p2026_09",
            "LiteLLM_SpendLogs_p2026_09_litellm_call_id_idx",
        ),
        (
            "LiteLLM_SpendLogs_litellm_call_id_idx",
            "LiteLLM_SpendLogs",
            "LiteLLM_SpendLogs_pdefault",
            "LiteLLM_SpendLogs_pdefault_litellm_call_id_idx",
        ),
        (
            "call_id_lookup",
            "LiteLLM_SpendLogs",
            "LiteLLM_SpendLogs_pdefault",
            "LiteLLM_SpendLogs_pdefault_call_id_lookup",
        ),
    ),
)
def test_partition_index_names_swap_the_parent_table_for_the_partition(
    index: str, table: str, partition: str, expected: str
) -> None:
    assert ConcurrentIndexMigration(index, table, "(a)").partition_index_name(partition) == expected
    assert len(expected) <= 63


def test_overlong_partition_index_names_fit_postgres_and_stay_distinct() -> None:
    migration = ConcurrentIndexMigration("t_a_idx", "t", "(a)")
    first = migration.partition_index_name("p" * 70 + "x")
    second = migration.partition_index_name("p" * 70 + "y")

    assert len(first) == len(second) == 63
    assert first.startswith("p" * 54)
    assert first != second
