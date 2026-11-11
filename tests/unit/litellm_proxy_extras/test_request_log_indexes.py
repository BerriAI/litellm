import re
from pathlib import Path
from typing import Final

import pytest
from litellm_proxy_extras.migration_recovery import is_inert_migration
from litellm_proxy_extras.request_log_indexes import (
    REQUEST_LOG_INDEXES,
    RequestLogIndex,
    filter_request_log_index_diff,
)

PACKAGE: Final = Path(__file__).resolve().parents[3] / "litellm-proxy-extras" / "litellm_proxy_extras"
SCHEMA: Final = PACKAGE / "schema.prisma"
INERT_MIGRATIONS: Final = (
    "20260823000000_add_spend_logs_api_key_starttime_index",
    "20260831120001_spend_logs_litellm_call_id_index",
)
CALL_ID_INDEX: Final = RequestLogIndex(
    "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_litellm_call_id_idx", '("litellm_call_id")'
)


def _prisma_indexes_of(schema: str, model: str) -> frozenset[str]:
    """The index names Prisma derives for a model's @@index declarations: <model>_<columns joined by _>_idx."""
    body: Final = re.search(rf"model {model} \{{(.*?)\n\}}", schema, re.DOTALL)
    assert body is not None, model
    declarations: Final[tuple[str, ...]] = tuple(
        match.group(1) for match in re.finditer(r"@@index\(\[([^\]]+)\]\)", body.group(1))
    )
    return frozenset(
        f"{model}_{'_'.join(column.strip() for column in columns.split(','))}_idx" for columns in declarations
    )


class TestTheIndexList:
    def test_every_migration_job_index_is_declared_in_the_prisma_schema_under_the_same_name(self):
        schema: Final = SCHEMA.read_text()
        for index in REQUEST_LOG_INDEXES:
            assert index.name in _prisma_indexes_of(schema, index.table), index

    @pytest.mark.parametrize("name", INERT_MIGRATIONS)
    def test_the_migrations_that_used_to_build_these_indexes_run_no_sql(self, name: str):
        assert is_inert_migration((PACKAGE / "migrations" / name / "migration.sql").read_text())


class TestIsInertMigration:
    @pytest.mark.parametrize(
        "script",
        (
            "",
            "-- only a comment\n",
            "/* block */\n-- line\n",
            "-- a semicolon; in a comment\n",
            ";\n;",
            "-- why\nSELECT 1;\n",
            "select 1",
        ),
        ids=(
            "empty",
            "line-comment",
            "both-comments",
            "semicolon-in-comment",
            "bare-separators",
            "select-1",
            "lowercase",
        ),
    )
    def test_comments_and_a_select_1_alone_are_inert(self, script: str):
        assert is_inert_migration(script) is True

    @pytest.mark.parametrize(
        "script",
        (
            "SELECT 2;",
            'SELECT 1 FROM "LiteLLM_SpendLogs";',
            '-- comment\nCREATE INDEX "ix" ON "t" ("a");',
            "/* c */ ALTER TABLE t ADD COLUMN a TEXT",
            'SELECT 1; DROP INDEX "ix";',
        ),
        ids=("select-2", "select-from", "index-after-comment", "alter-after-block-comment", "drop-after-select-1"),
    )
    def test_any_statement_is_not_inert(self, script: str):
        assert is_inert_migration(script) is False


class TestPartitionIndexName:
    def test_a_partition_gets_the_name_postgres_would_give_an_inherited_index(self):
        assert CALL_ID_INDEX.partition_index_name("LiteLLM_SpendLogs_p2026_09") == (
            "LiteLLM_SpendLogs_p2026_09_litellm_call_id_idx"
        )

    def test_an_index_not_prefixed_by_its_table_keeps_its_whole_name(self):
        index = RequestLogIndex("LiteLLM_SpendLogs", "call_id_lookup", '("litellm_call_id")')
        assert index.partition_index_name("LiteLLM_SpendLogs_pdefault") == "LiteLLM_SpendLogs_pdefault_call_id_lookup"

    def test_a_long_name_is_cut_to_63_bytes_with_a_digest_that_keeps_partitions_apart(self):
        first = CALL_ID_INDEX.partition_index_name("LiteLLM_SpendLogs_p" + "x" * 50 + "_2026_09")
        second = CALL_ID_INDEX.partition_index_name("LiteLLM_SpendLogs_p" + "x" * 50 + "_2026_10")
        assert len(first.encode()) == 63 and len(second.encode()) == 63
        assert first != second
        assert first.startswith("LiteLLM_SpendLogs_p") and first[-9] == "_"

    def test_the_byte_limit_counts_multibyte_characters(self):
        name = CALL_ID_INDEX.partition_index_name("é" * 40)
        assert len(name.encode()) <= 63 and len(name) < 63


class TestColumns:
    def test_the_columns_are_the_quoted_names_of_the_definition_in_order(self):
        index = RequestLogIndex(
            "LiteLLM_SpendLogs", "LiteLLM_SpendLogs_api_key_startTime_idx", '("api_key", "startTime")'
        )
        assert index.columns == ("api_key", "startTime")

    def test_every_migration_job_index_names_at_least_one_column(self):
        assert all(index.columns for index in REQUEST_LOG_INDEXES)


DRIFT_WITH_BOTH_INDEXES: Final = (
    "-- CreateIndex\n"
    'CREATE INDEX "LiteLLM_SpendLogs_litellm_call_id_idx" ON "LiteLLM_SpendLogs"("litellm_call_id");\n'
    "\n"
    "-- CreateIndex\n"
    'CREATE INDEX "LiteLLM_SpendLogs_api_key_startTime_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n'
)


class TestFilterRequestLogIndexDiff:
    def test_a_drift_script_that_only_creates_the_migration_job_indexes_becomes_empty(self):
        assert filter_request_log_index_diff(DRIFT_WITH_BOTH_INDEXES) == ""

    def test_other_statements_survive_with_the_migration_job_indexes_removed(self):
        other: Final = '-- AlterTable\nALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;\n'
        filtered = filter_request_log_index_diff(other + DRIFT_WITH_BOTH_INDEXES)
        assert 'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;' in filtered
        assert "LiteLLM_SpendLogs_litellm_call_id_idx" not in filtered
        assert "LiteLLM_SpendLogs_api_key_startTime_idx" not in filtered

    def test_an_index_of_another_name_on_spend_logs_is_kept(self):
        sql: Final = 'CREATE INDEX "LiteLLM_SpendLogs_end_user_idx" ON "LiteLLM_SpendLogs"("end_user");\n'
        assert filter_request_log_index_diff(sql) == sql

    def test_a_drop_of_a_migration_job_index_is_kept_for_the_operator_to_see(self):
        sql: Final = 'DROP INDEX "LiteLLM_SpendLogs_litellm_call_id_idx";\n'
        assert filter_request_log_index_diff(sql) == sql

    def test_an_empty_script_stays_empty(self):
        assert filter_request_log_index_diff("") == ""
