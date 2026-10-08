import glob
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, Optional

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../litellm-proxy-extras")
    ),
)

from litellm_proxy_extras.utils import (
    PARTITIONED_SPEND_LOGS_PUSH_ERROR,
    ProxyExtrasDBManager,
    _redact_command_error,
    _redact_credentials,
    filter_partitioned_spend_logs_diff,
)

# Path to the migrations directory
_MIGRATIONS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../litellm-proxy-extras/litellm_proxy_extras/migrations",
    )
)


def test_custom_prisma_dir(monkeypatch):
    import tempfile

    # create a temp directory
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("LITELLM_MIGRATION_DIR", temp_dir)

    ## Check if the prisma dir is the temp directory
    assert ProxyExtrasDBManager._get_prisma_dir() == temp_dir

    ## Check if the schema.prisma file is in the temp directory
    schema_path = os.path.join(temp_dir, "schema.prisma")
    assert os.path.exists(schema_path)

    ## Check if the migrations dir is in the temp directory
    migrations_dir = os.path.join(temp_dir, "migrations")
    assert os.path.exists(migrations_dir)


class TestPermissionErrorDetection:
    """Test cases for permission error detection in Prisma migrations"""

    def test_is_permission_error_postgres_42501(self):
        """Test detection of PostgreSQL 42501 error code (insufficient privilege)"""
        error_message = "Database error code: 42501 - permission denied for table users"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_must_be_owner(self):
        """Test detection of 'must be owner of table' error"""
        error_message = "ERROR: must be owner of table my_table"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_permission_denied_schema(self):
        """Test detection of 'permission denied for schema' error"""
        error_message = "permission denied for schema public"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_permission_denied_table(self):
        """Test detection of 'permission denied for table' error"""
        error_message = "permission denied for table my_table"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_must_be_owner_schema(self):
        """Test detection of 'must be owner of schema' error"""
        error_message = "must be owner of schema public"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_case_insensitive(self):
        """Test that permission error detection is case insensitive"""
        error_message = "PERMISSION DENIED FOR TABLE my_table"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True

    def test_is_permission_error_negative(self):
        """Test that non-permission errors are not detected as permission errors"""
        error_message = "column 'id' already exists"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is False


class TestIdempotentErrorDetection:
    """Test cases for idempotent error detection in Prisma migrations"""

    def test_is_idempotent_error_already_exists(self):
        """Test detection of generic 'already exists' error"""
        error_message = "object already exists"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_column_already_exists(self):
        """Test detection of 'column already exists' error"""
        error_message = "column 'email' already exists"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_duplicate_key(self):
        """Test detection of duplicate key violation error"""
        error_message = "duplicate key value violates unique constraint"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_relation_already_exists(self):
        """Test detection of 'relation already exists' error"""
        error_message = "relation 'users_pkey' already exists"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_constraint_already_exists(self):
        """Test detection of 'constraint already exists' error"""
        error_message = "constraint 'fk_user_id' already exists"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True


    def test_is_idempotent_error_case_insensitive(self):
        """Test that idempotent error detection is case insensitive"""
        error_message = "COLUMN 'ID' ALREADY EXISTS"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_does_not_exist(self):
        """Test detection of 'does not exist' error"""
        error_message = "ERROR: index 'idx' does not exist"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True

    def test_is_idempotent_error_negative(self):
        """Test that non-idempotent errors are not detected as idempotent errors"""
        error_message = "Database error code: 42501 - permission denied"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is False


class TestErrorClassificationPriority:
    """Test cases to ensure errors are correctly classified"""

    def test_permission_error_not_classified_as_idempotent(self):
        """Ensure permission errors are not mistakenly classified as idempotent"""
        error_message = "Database error code: 42501 - must be owner of table users"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is True
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is False

    def test_idempotent_error_not_classified_as_permission(self):
        """Ensure idempotent errors are not mistakenly classified as permission errors"""
        error_message = "column 'created_at' already exists"
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is True
        assert ProxyExtrasDBManager._is_permission_error(error_message) is False

    def test_unknown_error_classified_as_neither(self):
        """Ensure unknown errors are classified as neither permission nor idempotent"""
        error_message = "connection timeout"
        assert ProxyExtrasDBManager._is_permission_error(error_message) is False
        assert ProxyExtrasDBManager._is_idempotent_error(error_message) is False


def _get_all_migrations():
    """Return (migration_name, sql_content) pairs for all migrations."""
    migration_files = sorted(
        glob.glob(os.path.join(_MIGRATIONS_DIR, "*/migration.sql"))
    )
    results = []
    for path in migration_files:
        migration_name = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            results.append((migration_name, f.read()))
    return results


_LINE_COMMENT = re.compile(r"--.*$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

_PRE_GUARD_MIGRATIONS = frozenset({
    "20260331000000_add_prompt_environment_and_created_by",
    "20260418000000_add_adaptive_router_tables",
    "20260429161855_workflow_runs_tables",
    "20260605182307_add_timeout_to_mcp_server_table",
    "20260626120000_add_mcp_tool_search_enabled",
    "20260629000000_add_max_concurrent_requests_to_mcp_server_table",
    "20260710000000_add_dcr_bridge_to_mcp_server_table",
    "20260713230852_add_key_type_to_litellm_verification_token",
    "20260811172448_add_shadow_eval",
    "20260813180408_add_shadow_eval_direction",
    "20260814000000_add_proxy_worker_heartbeat",
    "20260817143646_add_daily_guardrail_usage_units",
    "20260818224500_add_shadow_eval_stopped_by",
    "20260819000000_shadow_eval_max_budget",
})


def _blanked_block_comments(sql):
    """`sql` with every `/* ... */` body blanked out, newlines kept so lines still count.

    Prisma opens a destructive migration with a `/* Warnings: You are about to drop the
    column ... */` header, which is prose about the statement rather than the statement.
    """
    return _BLOCK_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), sql)


def _statements(sql):
    """(line_number, sql) for each line, with comments removed.

    Prisma writes its own explanations as `-- CREATE INDEX CONCURRENTLY ...`, which a
    raw-line scan reads as the statement it is describing.
    """
    return [
        (number, _LINE_COMMENT.sub("", line))
        for number, line in enumerate(_blanked_block_comments(sql).splitlines(), 1)
    ]


def _guarded_migrations(all_migrations):
    """Migrations the DDL rules apply to. Prisma checksums an applied migration, so the
    ones that predate these rules cannot be edited without breaking `migrate deploy`
    for existing installs; they are named once, and the rules bind everything after.
    """
    return [
        (name, sql) for name, sql in all_migrations if name not in _PRE_GUARD_MIGRATIONS
    ]


class TestMigrationSQLIdempotency:
    """Ensure all migration SQL files use idempotent DDL (IF [NOT] EXISTS).

    Migrations on pre-existing instances can fail when DDL statements assume
    the target object doesn't already exist (or still exists for drops).
    These tests enforce that all migrations use safe, re-runnable SQL patterns.
    """

    @pytest.fixture(scope="class")
    def all_migrations(self):
        migrations = _get_all_migrations()
        assert len(migrations) > 0, (
            f"No migrations found. "
            f"Check that _MIGRATIONS_DIR ({_MIGRATIONS_DIR}) is correct."
        )
        return migrations

    def test_create_table_uses_if_not_exists(self, all_migrations):
        """CREATE TABLE statements must use IF NOT EXISTS"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            for line_num, line in _statements(sql):
                if re.search(
                    r"CREATE\s+TABLE\s+", line, re.IGNORECASE
                ) and not re.search(
                    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS", line, re.IGNORECASE
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert (
            not violations
        ), "CREATE TABLE without IF NOT EXISTS found in migrations:\n" + "\n".join(
            violations
        )

    def test_add_column_uses_if_not_exists(self, all_migrations):
        """ADD COLUMN statements must use IF NOT EXISTS"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            for line_num, line in _statements(sql):
                if re.search(r"ADD\s+COLUMN\s+", line, re.IGNORECASE) and not re.search(
                    r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS", line, re.IGNORECASE
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert not violations, (
            "ADD COLUMN without IF NOT EXISTS found in recent migrations:\n"
            + "\n".join(violations)
        )

    def test_drop_column_uses_if_exists(self, all_migrations):
        """DROP COLUMN statements must use IF EXISTS"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            for line_num, line in _statements(sql):
                if re.search(
                    r"DROP\s+COLUMN\s+", line, re.IGNORECASE
                ) and not re.search(
                    r"DROP\s+COLUMN\s+IF\s+EXISTS", line, re.IGNORECASE
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert (
            not violations
        ), "DROP COLUMN without IF EXISTS found in recent migrations:\n" + "\n".join(
            violations
        )

    _DROP_COLUMN_ALLOWLIST = {
        "20250918083359_drop_spec_version_column_from_mcp_table",
        "20260213170952_access_group_change_to_model_name",
        "20260224203854_add_agent_object_permissions_table",
    }

    def test_no_drop_column_statements(self, all_migrations):
        """Migrations must not drop columns — dropping columns is destructive
        and can break running application instances during rolling deploys."""
        violations = []
        for migration_name, sql in all_migrations:
            if migration_name in self._DROP_COLUMN_ALLOWLIST:
                continue
            for line_num, line in _statements(sql):
                if re.search(r"DROP\s+COLUMN", line, re.IGNORECASE):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert (
            not violations
        ), "DROP COLUMN found in migrations (destructive, not allowed):\n" + "\n".join(
            violations
        )

    def test_drop_index_uses_if_exists(self, all_migrations):
        """DROP INDEX statements must use IF EXISTS"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            for line_num, line in _statements(sql):
                if re.search(r"DROP\s+INDEX\s+", line, re.IGNORECASE) and not re.search(
                    r"DROP\s+INDEX\s+IF\s+EXISTS", line, re.IGNORECASE
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert (
            not violations
        ), "DROP INDEX without IF EXISTS found in recent migrations:\n" + "\n".join(
            violations
        )

    def test_create_index_uses_if_not_exists(self, all_migrations):
        """CREATE INDEX statements must use IF NOT EXISTS"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            for line_num, line in _statements(sql):
                if re.search(
                    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+", line, re.IGNORECASE
                ) and not re.search(
                    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?IF\s+NOT\s+EXISTS",
                    line,
                    re.IGNORECASE,
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert not violations, (
            "CREATE INDEX without IF NOT EXISTS found in recent migrations:\n"
            + "\n".join(violations)
        )

    def test_rename_column_is_guarded(self, all_migrations):
        """RENAME COLUMN must be inside a DO $$ IF EXISTS block"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            in_do_block = False
            for line_num, line in _statements(sql):
                if re.search(r"DO\s+\$\$", line, re.IGNORECASE):
                    in_do_block = True
                if re.search(r"END\s+\$\$", line, re.IGNORECASE):
                    in_do_block = False
                if (
                    re.search(r"RENAME\s+COLUMN\s+", line, re.IGNORECASE)
                    and not in_do_block
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert not violations, (
            "RENAME COLUMN without DO $$ IF EXISTS guard found in migrations:\n"
            + "\n".join(violations)
        )

    def test_add_constraint_is_guarded(self, all_migrations):
        """ADD CONSTRAINT must be inside a DO $$ IF NOT EXISTS block"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            in_do_block = False
            for line_num, line in _statements(sql):
                if re.search(r"DO\s+\$\$", line, re.IGNORECASE):
                    in_do_block = True
                if re.search(r"END\s+\$\$", line, re.IGNORECASE):
                    in_do_block = False
                if (
                    re.search(r"ADD\s+CONSTRAINT\s+", line, re.IGNORECASE)
                    and not in_do_block
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert not violations, (
            "ADD CONSTRAINT without DO $$ IF NOT EXISTS guard found in migrations:\n"
            + "\n".join(violations)
        )

    def test_drop_constraint_is_guarded(self, all_migrations):
        """DROP CONSTRAINT must be inside a DO $$ IF EXISTS block"""
        violations = []
        for migration_name, sql in _guarded_migrations(all_migrations):
            in_do_block = False
            for line_num, line in _statements(sql):
                if re.search(r"DO\s+\$\$", line, re.IGNORECASE):
                    in_do_block = True
                if re.search(r"END\s+\$\$", line, re.IGNORECASE):
                    in_do_block = False
                if (
                    re.search(r"DROP\s+CONSTRAINT\s+", line, re.IGNORECASE)
                    and not in_do_block
                ):
                    violations.append(f"  {migration_name}:{line_num}: {line.strip()}")
        assert not violations, (
            "DROP CONSTRAINT without DO $$ IF EXISTS guard found in migrations:\n"
            + "\n".join(violations)
        )


class TestMigrationGuardScope:
    """The guard must ignore SQL comments, exempt only the named pre-guard migrations,
    and still fail on a new migration that uses bare DDL."""

    _NEW = "20990101000000_a_new_migration"

    def _run_rules(self, migrations):
        suite = TestMigrationSQLIdempotency()
        failures = []
        for name in (
            "test_create_table_uses_if_not_exists",
            "test_add_column_uses_if_not_exists",
            "test_create_index_uses_if_not_exists",
            "test_add_constraint_is_guarded",
        ):
            try:
                getattr(suite, name)(migrations)
            except AssertionError:
                failures.append(name)
        return failures

    def test_a_comment_describing_ddl_is_not_the_ddl(self):
        sql = '-- CREATE TABLE "Foo" (id TEXT);\n-- ADD COLUMN "bar" TEXT;\n'
        assert self._run_rules([(self._NEW, sql)]) == []

    def test_a_prisma_warning_block_is_not_the_ddl_it_describes(self):
        sql = (
            "/*\n"
            "  Warnings:\n"
            "\n"
            "  - You are about to CREATE TABLE \"Foo\" and ADD COLUMN \"bar\".\n"
            "\n"
            "*/\n"
            'CREATE TABLE IF NOT EXISTS "Foo" (id TEXT);\n'
        )
        assert self._run_rules([(self._NEW, sql)]) == []

    def test_a_block_comment_does_not_shift_the_reported_line(self):
        sql = "/* filler\nfiller */\n" + 'CREATE TABLE "Foo" (id TEXT);\n'
        suite = TestMigrationSQLIdempotency()
        with pytest.raises(AssertionError) as failure:
            suite.test_create_table_uses_if_not_exists([(self._NEW, sql)])
        assert f"{self._NEW}:3:" in str(failure.value)

    def test_a_new_migration_with_bare_create_table_fails(self):
        assert "test_create_table_uses_if_not_exists" in self._run_rules(
            [(self._NEW, 'CREATE TABLE "Foo" (id TEXT);\n')]
        )

    def test_a_new_migration_with_bare_add_column_fails(self):
        assert "test_add_column_uses_if_not_exists" in self._run_rules(
            [(self._NEW, 'ALTER TABLE "Foo" ADD COLUMN "bar" TEXT;\n')]
        )

    def test_the_guarded_forms_pass(self):
        sql = (
            'CREATE TABLE IF NOT EXISTS "Foo" (id TEXT);\n'
            'ALTER TABLE "Foo" ADD COLUMN IF NOT EXISTS "bar" TEXT;\n'
            'CREATE INDEX IF NOT EXISTS "Foo_bar_idx" ON "Foo"("bar");\n'
        )
        assert self._run_rules([(self._NEW, sql)]) == []

    def test_a_pre_guard_migration_is_exempt_but_a_new_one_is_not(self):
        bare = 'CREATE TABLE "Foo" (id TEXT);\n'
        exempt = sorted(_PRE_GUARD_MIGRATIONS)[0]
        assert self._run_rules([(exempt, bare)]) == []
        assert self._run_rules([(self._NEW, bare)]) != []

    def test_every_pre_guard_migration_still_exists_on_disk(self):
        present = {name for name, _ in _get_all_migrations()}
        missing = _PRE_GUARD_MIGRATIONS - present
        assert not missing, f"pre-guard entries naming no migration: {sorted(missing)}"

    def test_no_pre_guard_entry_is_already_clean(self):
        by_name = dict(_get_all_migrations())
        redundant = [
            name
            for name in sorted(_PRE_GUARD_MIGRATIONS)
            if not self._run_rules([(TestMigrationGuardScope._NEW, by_name[name])])
        ]
        assert not redundant, f"these no longer violate and should be removed: {redundant}"


_PARTITIONED_DRIFT_SQL = """-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey",
ADD COLUMN     "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD COLUMN     "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");

-- DropTable
DROP TABLE "LiteLLM_SpendLogs_legacy";
"""


class TestPartitionedSpendLogsDriftFilter:
    """A doc-partitioned LiteLLM_SpendLogs (db_scripts/partition_spend_logs.sql) has a
    composite primary key that schema.prisma cannot express, so `prisma migrate diff`
    emits a primary-key rewrite that Postgres rejects, aborting the whole drift script
    before its legitimate statements run."""

    def test_pk_rewrite_and_runbook_artifact_drops_are_removed(self):
        filtered = filter_partitioned_spend_logs_diff(_PARTITIONED_DRIFT_SQL)
        assert 'DROP CONSTRAINT "LiteLLM_SpendLogs_pkey"' not in filtered
        assert 'PRIMARY KEY ("request_id")' not in filtered
        assert "LiteLLM_SpendLogs_legacy" not in filtered

    def test_legitimate_statements_in_the_same_script_are_kept(self):
        filtered = filter_partitioned_spend_logs_diff(_PARTITIONED_DRIFT_SQL)
        assert 'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;' in filtered
        assert 'ADD COLUMN     "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP' in filtered
        assert 'ADD COLUMN     "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP' in filtered
        assert filtered.count('ALTER TABLE "LiteLLM_SpendLogs"') == 1

    def test_an_alter_containing_only_the_pk_rewrite_is_dropped_entirely(self):
        sql = (
            'ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey",\n'
            'ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");\n'
        )
        assert filter_partitioned_spend_logs_diff(sql).strip() == ""

    def test_other_tables_pk_changes_are_untouched(self):
        sql = (
            'ALTER TABLE "LiteLLM_TeamTable" DROP CONSTRAINT "LiteLLM_TeamTable_pkey",\n'
            'ADD CONSTRAINT "LiteLLM_TeamTable_pkey" PRIMARY KEY ("team_id");\n'
        )
        filtered = filter_partitioned_spend_logs_diff(sql)
        assert 'DROP CONSTRAINT "LiteLLM_TeamTable_pkey"' in filtered
        assert 'PRIMARY KEY ("team_id")' in filtered


class _FakeCompleted:
    stdout = ""
    stderr = ""


class TestResolveAllMigrationsLedger:
    def _run(self, monkeypatch, tmp_path, partitioned, execute_fails):
        import subprocess as subprocess_module

        import litellm_proxy_extras.utils as utils_module

        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        monkeypatch.delenv("DIRECT_URL", raising=False)
        monkeypatch.setattr(
            ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: partitioned)
        )
        monkeypatch.setattr(
            ProxyExtrasDBManager,
            "_get_migration_names",
            staticmethod(lambda migrations_dir: ["20250326162113_baseline"]),
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "diff" in cmd:
                kwargs["stdout"].write(_PARTITIONED_DRIFT_SQL)
                return _FakeCompleted()
            if "execute" in cmd:
                executed_sql = open(cmd[cmd.index("--file") + 1]).read()
                calls.append(("executed_sql", executed_sql))
                if execute_fails:
                    raise subprocess_module.CalledProcessError(1, cmd, stderr="boom")
                return _FakeCompleted()
            return _FakeCompleted()

        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", fake_run)
        ProxyExtrasDBManager._resolve_all_migrations(str(tmp_path), "schema.prisma")
        return calls

    def _resolved(self, calls):
        return [c for c in calls if isinstance(c, list) and "resolve" in c]

    def _executed_sql(self, calls):
        return next(c[1] for c in calls if isinstance(c, tuple) and c[0] == "executed_sql")

    def test_failed_drift_apply_does_not_mark_migrations_applied(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, partitioned=False, execute_fails=True)
        assert self._resolved(calls) == []

    def test_successful_drift_apply_still_marks_migrations_applied(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, partitioned=False, execute_fails=False)
        assert len(self._resolved(calls)) == 1

    def test_partitioned_spend_logs_gets_the_filtered_drift_script(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, partitioned=True, execute_fails=False)
        executed_sql = self._executed_sql(calls)
        assert 'PRIMARY KEY ("request_id")' not in executed_sql
        assert "LiteLLM_SpendLogs_legacy" not in executed_sql
        assert 'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;' in executed_sql
        assert 'ADD COLUMN     "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP' in executed_sql
        assert len(self._resolved(calls)) == 1

    def test_unpartitioned_spend_logs_drift_script_is_untouched(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, partitioned=False, execute_fails=False)
        assert self._executed_sql(calls) == _PARTITIONED_DRIFT_SQL


class TestPartitionedSpendLogsPushGuard:
    def _forbid_subprocess(self, monkeypatch):
        import litellm_proxy_extras.utils as utils_module

        def fail_run(cmd, **kwargs):
            raise AssertionError(f"run_prisma should not be called, got: {cmd}")

        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", fail_run)

    def test_v1_db_push_fails_fast_with_guidance(self, monkeypatch):
        monkeypatch.setattr(
            ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: True)
        )
        self._forbid_subprocess(monkeypatch)
        with pytest.raises(RuntimeError) as err:
            ProxyExtrasDBManager._run_migrations(use_migrate=False, use_v2_resolver=False)
        assert str(err.value) == PARTITIONED_SPEND_LOGS_PUSH_ERROR

    def test_v2_db_push_fails_fast_with_guidance(self, monkeypatch):
        monkeypatch.setattr(
            ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: True)
        )
        self._forbid_subprocess(monkeypatch)
        with pytest.raises(RuntimeError) as err:
            ProxyExtrasDBManager._setup_database_v2(use_migrate=False)
        assert str(err.value) == PARTITIONED_SPEND_LOGS_PUSH_ERROR


class _FakeCursor:
    def fetchone(self):
        return (1,)


class _FakePsycopgConn:
    def __init__(self, executed):
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params):
        self._executed.append((query, params))
        return _FakeCursor()


class TestLensRenamePendingCheck:
    _DATABASE_URL: Final = "postgresql://litellm:hunter2@localhost:5432/litellm"

    def test_database_failure_text_reaches_the_raised_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psycopg

        monkeypatch.setenv("DATABASE_URL", self._DATABASE_URL)

        def refuse(conninfo: str, *, connect_timeout: int, autocommit: bool) -> NoReturn:
            raise psycopg.OperationalError("FATAL:  sorry, too many clients already")

        with pytest.raises(RuntimeError) as err:
            ProxyExtrasDBManager.raise_if_lens_rename_pending(connect=refuse)
        assert "FATAL:  sorry, too many clients already" in str(err.value)

    def test_password_libpq_echoes_is_redacted_from_the_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psycopg

        password: Final = "p%zzword"
        database_url: Final = f"postgresql://litellm:{password}@localhost:5432/litellm"
        monkeypatch.setenv("DATABASE_URL", database_url)
        monkeypatch.delenv("DIRECT_URL", raising=False)
        with pytest.raises(psycopg.Error) as libpq:
            psycopg.connect(database_url, connect_timeout=10, autocommit=True)

        with pytest.raises(RuntimeError) as err:
            ProxyExtrasDBManager.raise_if_lens_rename_pending()
        assert password not in str(err.value)
        assert str(err.value).endswith(str(libpq.value).strip().replace(password, "REDACTED"))

    def test_legacy_tables_keep_their_own_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", self._DATABASE_URL)
        executed: Final[list[tuple[str, tuple[str]]]] = []

        def legacy_tables_present(conninfo: str, *, connect_timeout: int, autocommit: bool) -> _FakePsycopgConn:
            return _FakePsycopgConn(executed)

        with pytest.raises(RuntimeError) as err:
            ProxyExtrasDBManager.raise_if_lens_rename_pending(connect=legacy_tables_present)
        assert str(err.value).startswith("Legacy Lens tables exist.")
        assert executed[0][1] == ("public",)


class TestSpendLogsPartitionDetectionSchemaScope:
    """A same-named LiteLLM_SpendLogs in another schema must not trip the
    detector: the catalog lookup has to be scoped to Prisma's target schema."""

    def _detect(self, monkeypatch, database_url):
        import sys
        import types

        executed = []
        fake_psycopg = types.ModuleType("psycopg")
        fake_psycopg.connect = lambda url, **kwargs: _FakePsycopgConn(executed)
        fake_psycopg.OperationalError = type("OperationalError", (Exception,), {})
        fake_psycopg.DatabaseError = type("DatabaseError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setenv("DATABASE_URL", database_url)
        assert ProxyExtrasDBManager.spend_logs_is_partitioned() is True
        return executed[0]

    def test_lookup_is_scoped_to_the_schema_url_param(self, monkeypatch):
        query, params = self._detect(
            monkeypatch, "postgresql://u:p@localhost:5432/db?schema=tenant_a"
        )
        assert "pg_namespace" in query
        assert "n.nspname = %s" in query
        assert params == ("tenant_a",)

    def test_lookup_falls_back_to_public_without_a_schema_param(self, monkeypatch):
        query, params = self._detect(monkeypatch, "postgresql://u:p@localhost:5432/db")
        assert "n.nspname = %s" in query
        assert params == ("public",)

    def test_only_partitioned_relations_match(self, monkeypatch):
        query, _ = self._detect(monkeypatch, "postgresql://u:p@localhost:5432/db")
        assert "pg_partitioned_table" in query


class TestSpendLogsPartitionDetectionMissingPsycopg:
    """psycopg ships in the `extra_proxy` install, but a stripped-down image
    can still lack it. When it does, detection must fail closed to False
    (never crash the migration path) and say so loudly, because a silent
    False here is what let a genuinely partitioned LiteLLM_SpendLogs hit the
    unfiltered primary-key rewrite in production."""

    def test_missing_psycopg_returns_false(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psycopg", None)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        assert ProxyExtrasDBManager.spend_logs_is_partitioned() is False

    def test_missing_psycopg_logs_a_warning(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "psycopg", None)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
            ProxyExtrasDBManager.spend_logs_is_partitioned()
        assert any(
            "psycopg is not installed" in record.message for record in caplog.records
        )


_ATTEMPT_BUDGET = 4
_P3009_MIGRATION_NAME = "20260415120000_health_check_latest_per_model_index"
_P3009_STARTED_AT = "2026-10-02 23:20:56.439594 UTC"
_P3009_DEADLOCK_LOGS = "ERROR: deadlock detected\nDETAIL: Process 72 waits for ShareLock on transaction 991"

_P3005_STDERR = """Error: P3005

The database schema is not empty. Read more about how to baseline an existing production database: https://pris.ly/d/migrate-baseline
"""


def _p3018_stderr(migration_name):
    return f"""Error: P3018

A migration failed to apply. New migrations cannot be applied before the error is recovered from.

Migration name: {migration_name}

Database error code: 42P07

Database error:
ERROR: relation "SomeTable" already exists
"""


def _p3009_stderr(migration_name: str, started_at: str) -> str:
    return (
        "Error: P3009\n\n"
        "migrate found failed migrations in the target database, new migrations will not be applied. "
        "Read more about how to resolve migration issues in a production database: "
        "https://pris.ly/d/migrate-resolve\n"
        f"The `{migration_name}` migration started at {started_at} failed\n"
    )


@dataclass(frozen=True, slots=True)
class _LedgerRow:
    migration_name: str
    started_at: str
    finished: bool = False
    rolled_back: bool = False
    logs: str | None = None


@dataclass(frozen=True, slots=True)
class _LedgerCursor:
    row: tuple[object, ...] | None = None

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row

    def fetchall(self) -> tuple[tuple[object, ...], ...]:
        return ()


class _LedgerConnection:
    def __init__(self, ledger: "_FakeLedger") -> None:
        self.ledger = ledger

    def __enter__(self) -> "_LedgerConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: object, params: tuple[object, ...] = ()) -> _LedgerCursor:
        return self.ledger.execute(query, params)


class _FakeLedger:
    def __init__(self, at_error: tuple[_LedgerRow, ...], after_peer: tuple[_LedgerRow, ...]) -> None:
        self.rows = at_error
        self.after_peer = after_peer
        self._peer_observed = False

    def connect(self, *args: object, **kwargs: object) -> _LedgerConnection:
        return _LedgerConnection(self)

    def execute(self, query: object, params: tuple[object, ...]) -> _LedgerCursor:
        text: Final = str(query)
        if "WHERE migration_name = %s" not in text or not params:
            return _LedgerCursor()
        if not self._peer_observed:
            self.rows = self.after_peer
            self._peer_observed = True
        matching: Final = tuple(
            row
            for row in self.rows
            if row.migration_name == params[0] and (len(params) == 1 or row.started_at == params[1])
        )
        if "rolled_back_at IS NULL" in text:
            unresolved: Final = next((row for row in matching if not row.finished and not row.rolled_back), None)
            return _LedgerCursor((unresolved.logs,) if unresolved else None)
        if "IS NOT NULL" in text:
            resolved: Final = next((row for row in matching if row.finished or row.rolled_back), None)
            return _LedgerCursor((1,) if resolved else None)
        return _LedgerCursor()


@pytest.mark.parametrize(
    "pooled,direct,expected",
    (
        ("postgresql://pool/db?pgbouncer=true", None, "postgresql://pool/db?pgbouncer=true"),
        ("postgresql://pool/db?pgbouncer=true", "postgresql://writer/db", "postgresql://writer/db?schema=public"),
        (
            "postgresql://pool/db?schema=tenant%20one&pgbouncer=true",
            "postgresql://writer/db?sslmode=require&schema=wrong",
            "postgresql://writer/db?sslmode=require&schema=tenant+one",
        ),
    ),
)
def test_v2_migrations_use_the_direct_connection_with_the_runtime_schema(pooled, direct, expected):
    from litellm_proxy_extras.migration_lock import migration_environment

    environment = {"DATABASE_URL": pooled, "PRISMA_OFFLINE_MODE": "true"}
    configured = {**environment, **({"DIRECT_URL": direct} if direct else {})}
    migrated = migration_environment(configured)

    assert migrated["DATABASE_URL"] == expected
    assert migrated["PRISMA_OFFLINE_MODE"] == "true"
    assert configured["DATABASE_URL"] == pooled


class _MigrateDeployHarness:
    """Drives _setup_database_v2 with a scripted sequence of
    `prisma migrate deploy` outcomes, with every recovery command faked out so
    nothing touches a database or the packaged migrations directory."""

    def __init__(
        self,
        monkeypatch,
        tmp_path,
        outcomes,
        repeat_last=False,
        confirmed_migrations=(),
        ledger: "_FakeLedger | None" = None,
    ):
        import subprocess as subprocess_module

        import litellm_proxy_extras.utils as utils_module

        self.deploy_calls = []
        self.resolved = []
        self.baselines = 0
        self._outcomes = list(outcomes)
        self._repeat_last = repeat_last
        self._subprocess_module = subprocess_module
        self.confirmed_migrations = set(confirmed_migrations)

        if ledger is None:
            monkeypatch.delenv("DATABASE_URL", raising=False)
        else:
            monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
            monkeypatch.setattr("psycopg.connect", ledger.connect)
        monkeypatch.setenv("LITELLM_MIGRATION_DIR", str(tmp_path))
        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", self._fake_run)
        monkeypatch.setattr(utils_module, "_get_prisma_env", lambda: {})
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)

        self.baseline_succeeds = True

    def _fake_baseline(self, *args, **kwargs):
        self.baselines += 1
        if not self.baseline_succeeds:
            raise RuntimeError("The existing schema was not verified")

    def _next_outcome(self):
        if self._outcomes:
            if self._repeat_last and len(self._outcomes) == 1:
                return self._outcomes[0]
            return self._outcomes.pop(0)
        raise AssertionError("prisma migrate deploy called more times than scripted")

    def _fake_run(self, cmd, **kwargs):
        assert cmd[1:] == ["migrate", "deploy"], f"unexpected prisma command: {cmd}"
        self.deploy_calls.append(cmd)
        outcome = self._next_outcome()
        if outcome == "ok":
            return _FakeCompleted()
        if outcome == "timeout":
            raise self._subprocess_module.TimeoutExpired(cmd, 1)
        raise self._subprocess_module.CalledProcessError(1, cmd, stderr=outcome)

    def run(self):
        while not ProxyExtrasDBManager._run_database_v2(
            use_migrate=True,
            recover_completed=self._fake_recovery,
            baseline_existing=self._fake_baseline,
        ):
            continue
        return True

    def _fake_recovery(self, name):
        if name not in self.confirmed_migrations:
            return False
        self.confirmed_migrations.remove(name)
        self.resolved.append(name)
        return True


class TestConcurrentP3009Recovery:
    @pytest.mark.parametrize(
        "after_peer",
        (
            (_LedgerRow(_P3009_MIGRATION_NAME, _P3009_STARTED_AT, rolled_back=True, logs=_P3009_DEADLOCK_LOGS),),
            (
                _LedgerRow(_P3009_MIGRATION_NAME, _P3009_STARTED_AT, rolled_back=True, logs=_P3009_DEADLOCK_LOGS),
                _LedgerRow(_P3009_MIGRATION_NAME, "2026-10-02 23:21:11.539224 UTC"),
            ),
            (_LedgerRow(_P3009_MIGRATION_NAME, _P3009_STARTED_AT, finished=True, logs=_P3009_DEADLOCK_LOGS),),
        ),
        ids=("rolled-back", "rolled-back-beside-a-fresh-in-flight-row", "finished"),
    )
    def test_a_p3009_row_a_peer_already_recovered_is_retried(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        after_peer: tuple[_LedgerRow, ...],
    ) -> None:
        deadlocked_row: Final = _LedgerRow(
            _P3009_MIGRATION_NAME,
            _P3009_STARTED_AT,
            logs=_P3009_DEADLOCK_LOGS,
        )
        harness: Final = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_p3009_stderr(_P3009_MIGRATION_NAME, _P3009_STARTED_AT), "ok"],
            ledger=_FakeLedger(at_error=(deadlocked_row,), after_peer=after_peer),
        )

        assert harness.run() is True
        assert len(harness.deploy_calls) == 2

    @pytest.mark.parametrize(
        "ledger_rows",
        (
            (
                _LedgerRow(
                    _P3009_MIGRATION_NAME,
                    _P3009_STARTED_AT,
                    logs='ERROR: syntax error at or near "SLECT"',
                ),
            ),
            (
                _LedgerRow(
                    _P3009_MIGRATION_NAME,
                    _P3009_STARTED_AT,
                    logs='ERROR: syntax error at or near "SLECT"',
                ),
                _LedgerRow(
                    _P3009_MIGRATION_NAME,
                    "2026-10-02 23:19:40.120000 UTC",
                    rolled_back=True,
                    logs=_P3009_DEADLOCK_LOGS,
                ),
            ),
        ),
        ids=("only-row", "beside-a-recovered-earlier-attempt"),
    )
    def test_an_unresolved_p3009_row_without_the_deadlock_marker_stops(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        ledger_rows: tuple[_LedgerRow, ...],
    ) -> None:
        harness: Final = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_p3009_stderr(_P3009_MIGRATION_NAME, _P3009_STARTED_AT)],
            ledger=_FakeLedger(at_error=ledger_rows, after_peer=ledger_rows),
        )

        with pytest.raises(RuntimeError, match="Migration completion could not be verified"):
            harness.run()
        assert len(harness.deploy_calls) == 1


class TestMigrateDeployAttemptAccounting:
    def test_a_push_created_database_finishes_bootstrapping(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_P3005_STDERR, "ok"],
        )

        assert harness.run() is True
        assert harness.baselines == 1
        assert harness.resolved == []
        assert len(harness.deploy_calls) == 2

    def test_repeated_recovery_of_one_migration_still_gives_up(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_p3018_stderr("20250329084805_new_cron_job_table")],
            repeat_last=True,
            confirmed_migrations=("20250329084805_new_cron_job_table",),
        )

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == 2
        assert harness.resolved == ["20250329084805_new_cron_job_table"]

    def test_timeouts_still_spend_the_budget(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(monkeypatch, tmp_path, ["timeout"], repeat_last=True)

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == _ATTEMPT_BUDGET

    def test_an_unverified_baseline_stops_without_replaying_migrations(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(monkeypatch, tmp_path, [_P3005_STDERR], repeat_last=True)
        harness.baseline_succeeds = False

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == 1

    def test_lock_contention_does_not_spend_the_failure_budget(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            ["Error: P1002\nTimed out waiting for the advisory lock"] * 6 + ["ok"],
        )
        assert harness.run() is True
        assert len(harness.deploy_calls) == 7

    def test_duplicate_object_error_without_completion_proof_is_fatal(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(monkeypatch, tmp_path, [_p3018_stderr("20260101000000_x")])
        with pytest.raises(RuntimeError, match="cannot be auto-recovered"):
            harness.run()
        assert harness.resolved == []
        assert len(harness.deploy_calls) == 1

    @pytest.mark.parametrize("name", ("20260101000000_x", "20260101000000_migration with spaces"))
    def test_an_interrupted_migration_with_confirmed_sql_can_finish(self, monkeypatch, tmp_path, name):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [f"Error: P3009\nThe `{name}` migration failed", "ok"],
            confirmed_migrations=(name,),
        )
        assert harness.run() is True
        assert harness.resolved == [name]

    def test_an_interrupted_migration_without_confirmation_stops(self, monkeypatch, tmp_path):
        name = "20260101000000_x"
        started = "2026-09-12 20:15:06.694553 UTC"
        report = f"Error: P3009\nThe `{name}` migration started at {started} failed"
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [report],
        )
        with pytest.raises(RuntimeError, match="Migration completion could not be verified") as failure:
            harness.run()
        message = str(failure.value)
        assert name in message
        assert started in message
        assert "start record but no successful completion record" in message
        assert "cannot determine whether its SQL committed" in message
        assert "avoid repeating or skipping database changes" in message
        assert "_prisma_migrations" in message
        assert "migration.sql" in message
        assert "same database" in message
        assert "Only after verifying every migration change is present" in message
        assert "prisma migrate resolve --applied <migration_name>" in message
        assert "Only after verifying no migration changes remain" in message
        assert "prisma migrate resolve --rolled-back <migration_name>" in message
        assert "leave migration history unchanged" in message
        assert "Repeated restarts alone" in message
        assert report in message
        assert len(harness.deploy_calls) == 1
        assert harness.resolved == []

    def test_an_unrecoverable_error_is_not_retried(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            ['Error: P3018\n\nMigration name: 20260101000000_x\n\nERROR: syntax error at or near "SLECT"\n'],
            repeat_last=True,
        )

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == 1
        assert harness.resolved == []


@pytest.mark.parametrize(
    "steps,logs,script,expected",
    (
        (1, "", b"CREATE TABLE item (id int);", True),
        (0, "", b"CREATE TABLE item (id int);", False),
        (0, "already exists", b"CREATE TABLE item (id int);", False),
        (1, "permission denied", b"CREATE TABLE item (id int);", False),
        (1, "", b"CREATE TABLE item (id text);", False),
        (2, "", b"CREATE TABLE item (id int);", False),
    ),
)
def test_migration_completion_requires_a_matching_successful_script(steps, logs, script, expected):
    import hashlib

    from litellm_proxy_extras.migration_recovery import MigrationProgress

    progress = MigrationProgress(hashlib.sha256(b"CREATE TABLE item (id int);").hexdigest(), steps, logs)
    assert progress.confirms_completion(script) is expected


def test_prisma_lock_waiting_has_its_own_deadline():
    from litellm_proxy_extras.utils import _MigrateAttemptBudget

    budget = _MigrateAttemptBudget(attempts_left=4, contention_seconds_left=2)
    waiting = budget.after_contention(1)
    assert waiting.attempts_left == 4
    with pytest.raises(RuntimeError, match="advisory lock"):
        waiting.after_contention(2)


class TestJWTKeyMappingCascade:
    """Regression tests for issue #33702.

    A virtual key referenced by a LiteLLM_JWTKeyMapping row could not be deleted
    because LiteLLM_JWTKeyMapping_token_fkey was created ON DELETE RESTRICT, so
    deleting the key (Admin UI, /key/delete, team delete, ...) raised a foreign
    key violation. The mapping must be removed automatically when its key is
    deleted, which the FK now enforces via ON DELETE CASCADE.
    """

    _FK_NAME = "LiteLLM_JWTKeyMapping_token_fkey"

    def _effective_on_delete(self):
        """Replay every migration in order and return the last ON DELETE action
        declared for the JWT key mapping FK."""
        action = None
        for _migration_name, sql in _get_all_migrations():
            for match in re.finditer(
                rf'ADD\s+CONSTRAINT\s+"{re.escape(self._FK_NAME)}".*?'
                r"ON\s+DELETE\s+(CASCADE|RESTRICT|SET\s+NULL|NO\s+ACTION|SET\s+DEFAULT)",
                sql,
                re.IGNORECASE | re.DOTALL,
            ):
                action = re.sub(r"\s+", " ", match.group(1).upper())
        return action

    def test_fk_effective_on_delete_is_cascade(self):
        """The final FK definition across all migrations must cascade deletes."""
        assert self._effective_on_delete() == "CASCADE", (
            f"{self._FK_NAME} must end up ON DELETE CASCADE so deleting a "
            "virtual key removes its JWT key mapping (issue #33702)"
        )

    def test_schema_declares_cascade_on_relation(self):
        """schema.prisma must declare onDelete: Cascade on the mapping relation
        so the generated client and DB agree."""
        schema_paths = glob.glob(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "../../../**/schema.prisma"
                )
            ),
            recursive=True,
        )
        declaring = tuple(
            (path, schema)
            for path, schema in ((p, Path(p).read_text()) for p in schema_paths)
            if "model LiteLLM_JWTKeyMapping" in schema
        )
        assert declaring, "No schema.prisma declaring LiteLLM_JWTKeyMapping found"
        for path, schema in declaring:
            match = re.search(
                r"litellm_verification_token\s+LiteLLM_VerificationToken\s+@relation\(([^)]*)\)",
                schema,
            )
            assert match is not None, (
                f"{path} declares LiteLLM_JWTKeyMapping but its verification token "
                "relation could not be parsed, so this test cannot vouch for it "
                "(issue #33702)"
            )
            assert "onDelete: Cascade" in match.group(1), (
                f"{path} must declare onDelete: Cascade on the JWT key mapping "
                "relation (issue #33702)"
            )



class TestStripPrismaQueryParams:
    """The psycopg URL the job connects with is derived from the Prisma-dialect
    DATABASE_URL, whose TLS params mean something else to libpq."""

    @staticmethod
    def _query(url: str) -> dict[str, str]:
        from urllib.parse import parse_qsl, urlparse

        return dict(parse_qsl(urlparse(url).query))

    def test_prisma_ca_sslcert_becomes_sslrootcert_with_verify_full(self):
        url = "postgresql://u:p@writer:5432/db?schema=public&sslmode=require&sslcert=/tmp/pinned.pem&sslaccept=strict"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {"sslmode": "verify-full", "sslrootcert": "/tmp/pinned.pem"}
        assert cleaned.startswith("postgresql://u:p@writer:5432/db?")

    @pytest.mark.parametrize("sslmode", ["prefer", "require"])
    @pytest.mark.parametrize("sslaccept", ["strict", "unknown-mode-prisma-treats-as-strict"])
    def test_strict_verifies_chain_and_hostname_whatever_sslmode_prisma_was_given(self, sslmode, sslaccept):
        url = f"postgresql://writer/db?sslmode={sslmode}&sslcert=/certs/ca.pem&sslaccept={sslaccept}"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {"sslmode": "verify-full", "sslrootcert": "/certs/ca.pem"}

    def test_strict_with_tls_disabled_stays_off(self):
        url = "postgresql://writer/db?sslmode=disable&sslcert=/certs/ca.pem&sslaccept=strict"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {"sslmode": "disable"}

    @pytest.mark.parametrize("sslaccept", ["&sslaccept=accept_invalid_certs", ""])
    def test_without_strict_the_ca_is_dropped_so_libpq_checks_nothing_like_prisma(self, sslaccept):
        url = f"postgresql://writer/db?sslmode=require&sslcert=/certs/ca.pem{sslaccept}"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {"sslmode": "require"}

    def test_a_ca_alone_without_strict_or_sslmode_leaves_libpq_its_defaults(self):
        cleaned = ProxyExtrasDBManager._strip_prisma_query_params("postgresql://writer/db?sslcert=/certs/ca.pem")

        assert cleaned == "postgresql://writer/db"

    def test_a_libpq_client_certificate_pair_is_left_alone(self):
        url = "postgresql://writer/db?sslmode=verify-full&sslrootcert=/ca.pem&sslcert=/client.crt&sslkey=/client.key"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {
            "sslmode": "verify-full",
            "sslrootcert": "/ca.pem",
            "sslcert": "/client.crt",
            "sslkey": "/client.key",
        }

    def test_an_explicit_sslrootcert_wins_over_the_prisma_sslcert(self):
        url = "postgresql://writer/db?sslmode=require&sslrootcert=/ca.pem&sslcert=/pinned.pem&sslaccept=strict"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert self._query(cleaned) == {"sslmode": "verify-full", "sslrootcert": "/ca.pem"}

    def test_prisma_only_params_are_dropped_and_plain_urls_pass_through(self):
        url = "postgresql://u:p@pooler:6543/db?schema=tenant&pgbouncer=true&connection_limit=5&connect_timeout=3"

        cleaned = ProxyExtrasDBManager._strip_prisma_query_params(url)

        assert cleaned == "postgresql://u:p@pooler:6543/db?connect_timeout=3"
        assert (
            ProxyExtrasDBManager._strip_prisma_query_params("postgresql://u:p@writer/db")
            == "postgresql://u:p@writer/db"
        )


class TestBuildRequestLogIndexes:
    """The migration job hands the index build the direct database URL and the schema
    the migrations target, waits for it, and reports its result."""

    @pytest.fixture
    def builds(self):
        return []

    @pytest.fixture
    def build(self, builds):
        def record(database_url: str, schema: str) -> bool:
            builds.append((database_url, schema))
            return True

        return record

    def test_the_build_gets_the_direct_url_without_prisma_params_and_the_prisma_schema(self, monkeypatch, builds, build):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@pooler:6543/db?schema=tenant&pgbouncer=true")
        monkeypatch.setenv("DIRECT_URL", "postgresql://u:p@primary:5432/db?connection_limit=1")

        assert ProxyExtrasDBManager.build_request_log_indexes(build=build) is True

        assert builds == [("postgresql://u:p@primary:5432/db", "tenant")]

    def test_the_build_defaults_to_the_database_url_and_the_public_schema(self, monkeypatch, builds, build):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@primary:5432/db")
        monkeypatch.delenv("DIRECT_URL", raising=False)

        assert ProxyExtrasDBManager.build_request_log_indexes(build=build) is True

        assert builds == [("postgresql://u:p@primary:5432/db", "public")]

    def test_a_build_that_leaves_indexes_missing_is_reported_so_the_job_reruns(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@primary:5432/db")

        assert ProxyExtrasDBManager.build_request_log_indexes(build=lambda url, schema: False) is False

    def test_without_a_database_url_nothing_is_built(self, monkeypatch, builds, build):
        monkeypatch.delenv("DATABASE_URL", raising=False)

        assert ProxyExtrasDBManager.build_request_log_indexes(build=build) is True

        assert builds == []


class TestStartRequestLogIndexBuild:
    """A serving proxy that ran the migrations starts the index build on a daemon thread
    and goes on to serve while it runs."""

    def test_the_build_runs_on_a_daemon_thread_that_does_not_hold_up_the_caller(self):
        release: Final = threading.Event()
        builds: Final[list[str]] = []  # mutable-ok: the builder thread hands back the thread it ran on

        def build() -> bool:
            assert release.wait(5), "the caller never came back from start_request_log_index_build"
            builds.append(threading.current_thread().name)
            return True

        thread: Final = ProxyExtrasDBManager.start_request_log_index_build(build=build)

        assert builds == [], "the build ran before start_request_log_index_build returned"
        assert thread.daemon is True
        release.set()
        thread.join(5)
        assert builds == ["litellm-request-log-indexes"]


class TestRunMigrationJob:
    """`run_migration_job` is `setup_database` followed by the index build, each step's
    result deciding whether the job reports success."""

    @pytest.fixture
    def calls(self):
        return []

    @pytest.fixture
    def setup(self, calls):
        def record(result: bool):
            def setup_database(use_migrate: bool, use_v2_resolver: bool) -> bool:
                calls.append(("setup", use_migrate, use_v2_resolver))
                return result

            return setup_database

        return record

    @pytest.fixture
    def build(self, calls):
        def record(result: bool):
            def build_request_log_indexes() -> bool:
                calls.append(("build",))
                return result

            return build_request_log_indexes

        return record

    def test_the_job_builds_the_indexes_after_the_migrations_succeed(self, calls, setup, build):
        assert ProxyExtrasDBManager.run_migration_job(True, False, setup=setup(True), build=build(True)) is True

        assert calls == [("setup", True, False), ("build",)]

    def test_the_job_fails_without_building_when_the_migrations_fail(self, calls, setup, build):
        assert ProxyExtrasDBManager.run_migration_job(True, True, setup=setup(False), build=build(True)) is False

        assert calls == [("setup", True, True)]

    def test_the_job_fails_when_an_index_could_not_be_built(self, calls, setup, build):
        assert ProxyExtrasDBManager.run_migration_job(True, True, setup=setup(True), build=build(False)) is False

        assert calls == [("setup", True, True), ("build",)]


class TestMigrationJobOwnedDrift:
    JOB_INDEXES = (
        "-- CreateIndex\n"
        'CREATE INDEX "LiteLLM_SpendLogs_litellm_call_id_idx" ON "LiteLLM_SpendLogs"("litellm_call_id");\n'
        "\n-- CreateIndex\n"
        'CREATE INDEX "LiteLLM_SpendLogs_api_key_startTime_idx" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n'
    )

    def test_a_plain_spend_logs_table_only_loses_the_migration_job_indexes(self):
        filtered = ProxyExtrasDBManager._filter_migration_job_owned_drift(
            _PARTITIONED_DRIFT_SQL + self.JOB_INDEXES, partitioned=False
        )
        assert "LiteLLM_SpendLogs_litellm_call_id_idx" not in filtered
        assert "LiteLLM_SpendLogs_api_key_startTime_idx" not in filtered
        assert 'PRIMARY KEY ("request_id")' in filtered

    def test_a_partitioned_spend_logs_table_also_loses_its_partitioning_artifacts(self):
        filtered = ProxyExtrasDBManager._filter_migration_job_owned_drift(
            _PARTITIONED_DRIFT_SQL + self.JOB_INDEXES, partitioned=True
        )
        assert "LiteLLM_SpendLogs_litellm_call_id_idx" not in filtered
        assert 'PRIMARY KEY ("request_id")' not in filtered
        assert "LiteLLM_SpendLogs_legacy" not in filtered
        assert 'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;' in filtered


_P3018_UNCLASSIFIED_STDERR: Final = (
    "Error: P3018\n\n"
    "A migration failed to apply. New migrations cannot be applied before the error is "
    "recovered from.\n\n"
    "Migration name: 20260921190000_agent_identity\n\n"
    "Database error code: 23505\n\n"
    "Database error:\n"
    'ERROR: could not create unique index "agent_identity_key"\n'
    "DETAIL: Key (agent_id)=(agent-1) is duplicated.\n"
)


_FAKE_PRISMA_PID: Final = 424242


class TestV1MigrationFailuresLogAtError:
    @staticmethod
    def _run_v1_migrations(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        deploy_stderr: Optional[str] = None,
        deploy_timeout: bool = False,
        diff_stderr: Optional[str] = None,
        database_url: Optional[str] = None,
    ) -> tuple[bool, list[list[str]], list[int]]:
        import litellm_proxy_extras.utils as utils_module

        calls: Final[list[list[str]]] = []
        killed_pids: Final[list[int]] = []

        class _FakePrismaPopen:
            def __init__(
                self,
                argv: tuple[str, ...],
                *,
                env: Optional[dict[str, str]] = None,
                stdout: object = None,
                stderr: object = None,
                text: object = None,
                start_new_session: object = None,
            ) -> None:
                self.args: Final = argv
                self.argv: Final = argv
                self.pid: Final = _FAKE_PRISMA_PID
                self.returncode: Optional[int] = None
                calls.append(list(argv))

            def __enter__(self) -> "_FakePrismaPopen":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def _subcommand(self) -> tuple[str, str]:
                known: Final = (
                    ("migrate", "deploy"),
                    ("migrate", "diff"),
                    ("migrate", "resolve"),
                    ("db", "execute"),
                )
                for index in range(len(self.argv) - 1):
                    pair: Final = tuple(self.argv[index : index + 2])
                    if pair in known:
                        return pair
                return ("", "")

            def communicate(self, timeout: Optional[float] = None) -> tuple[str, str]:
                subcommand: Final = self._subcommand()
                if subcommand == ("migrate", "deploy"):
                    if deploy_timeout:
                        raise subprocess.TimeoutExpired(self.argv, timeout)
                    if deploy_stderr is not None:
                        self.returncode = 1
                        return "", deploy_stderr
                    self.returncode = 0
                    return "No pending migrations to apply", ""
                if subcommand == ("migrate", "diff") and diff_stderr is not None:
                    self.returncode = 1
                    return "", diff_stderr
                self.returncode = 0
                return "", ""

        migration_dir: Final = tmp_path / "migration_dir"
        migration_dir.mkdir()
        if database_url is None:
            monkeypatch.delenv("DATABASE_URL", raising=False)
        else:
            monkeypatch.setenv("DATABASE_URL", database_url)
        monkeypatch.setenv("LITELLM_MIGRATION_DIR", str(migration_dir))
        monkeypatch.setattr(
            utils_module.prisma_toolchain.subprocess, "Popen", _FakePrismaPopen
        )
        monkeypatch.setattr(
            utils_module.prisma_toolchain.os, "killpg", lambda pid, sig: killed_pids.append(pid)
        )
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)

        succeeded: Final = ProxyExtrasDBManager._run_migrations(use_migrate=True, use_v2_resolver=False)
        return succeeded, calls, killed_pids

    @staticmethod
    def _deploy_call_count(calls: list[list[str]]) -> int:
        return sum(1 for call in calls if tuple(call[-2:]) == ("migrate", "deploy"))

    @staticmethod
    def _error_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.ERROR and record.name.startswith("litellm_proxy_extras")
        ]

    def test_an_unrecognized_prisma_error_logs_its_stderr_at_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        stderr: Final = "Error: P1001: Can't reach database server at db:5432"
        with caplog.at_level(logging.ERROR, logger="litellm_proxy_extras"):
            succeeded, calls, _ = self._run_v1_migrations(
                monkeypatch, tmp_path, deploy_stderr=stderr
            )

        assert succeeded is False
        assert self._deploy_call_count(calls) == 4
        assert any(stderr in message for message in self._error_messages(caplog))

    def test_an_unclassified_p3018_logs_its_stderr_and_retry_failure_at_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="litellm_proxy_extras"):
            succeeded, calls, _ = self._run_v1_migrations(
                monkeypatch, tmp_path, deploy_stderr=_P3018_UNCLASSIFIED_STDERR
            )

        assert succeeded is False
        assert self._deploy_call_count(calls) == 4
        messages: Final = self._error_messages(caplog)
        assert any(
            "20260921190000_agent_identity" in message and "is duplicated" in message
            for message in messages
        )
        assert any(
            "The process failed to execute" in message and "Retrying... (3 attempts left)" in message
            for message in messages
        )

    def test_called_process_error_with_no_command_retries_all_v1_attempts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import litellm_proxy_extras.utils as utils_module

        migration_dir: Final = tmp_path / "migration_dir"
        migration_dir.mkdir()
        monkeypatch.setenv("LITELLM_MIGRATION_DIR", str(migration_dir))
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DIRECT_URL", raising=False)
        monkeypatch.setenv("PRISMA_OFFLINE_MODE", "true")
        monkeypatch.setenv("PRISMA_CLI_PATH", sys.executable)
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)
        calls: Final[list[None]] = []

        class _FakePrismaPopen:
            def __init__(
                self,
                argv: tuple[str, ...],
                *,
                env: Optional[dict[str, str]] = None,
                stdout: object = None,
                stderr: object = None,
                text: object = None,
                start_new_session: object = None,
            ) -> None:
                self.args: Final = None
                self.returncode: Final = 1
                calls.append(None)

            def __enter__(self) -> "_FakePrismaPopen":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def communicate(self, timeout: Optional[float] = None) -> tuple[str, str]:
                return "", "Error: P3018 unclassified"

        monkeypatch.setattr(
            utils_module.prisma_toolchain.subprocess, "Popen", _FakePrismaPopen
        )

        try:
            succeeded: Final = ProxyExtrasDBManager._run_migrations(
                use_migrate=True, use_v2_resolver=False
            )
        except TypeError as error:
            pytest.fail(
                f"_run_migrations raised TypeError after {len(calls)} Popen calls: {error}",
                pytrace=False,
            )

        assert succeeded is False
        assert len(calls) == 4

    def test_a_timeout_logs_at_error_naming_the_migrate_deploy_timeout_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from litellm_proxy_extras.prisma_toolchain import PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR

        with caplog.at_level(logging.ERROR, logger="litellm_proxy_extras"):
            succeeded, calls, killed_pids = self._run_v1_migrations(
                monkeypatch, tmp_path, deploy_timeout=True
            )

        assert succeeded is False
        assert self._deploy_call_count(calls) == 4
        assert killed_pids == [_FAKE_PRISMA_PID] * 4
        assert any(
            "timed out" in message and PRISMA_MIGRATE_DEPLOY_TIMEOUT_ENV_VAR in message
            for message in self._error_messages(caplog)
        )

    def test_a_recovered_baseline_logs_nothing_at_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.ERROR, logger="litellm_proxy_extras"):
            succeeded, calls, _ = self._run_v1_migrations(
                monkeypatch,
                tmp_path,
                deploy_stderr=_P3005_STDERR,
                database_url="postgresql://user:pass@db:5432/litellm",
            )

        assert succeeded is True
        assert tuple(calls[0][-2:]) == ("migrate", "deploy")
        assert self._error_messages(caplog) == []

    def test_a_failed_baseline_recovery_logs_its_stderr_at_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        database_url: Final = "postgresql://llmproxy:s3cr3t 'p\"w@db:5432/litellm"
        monkeypatch.delenv("DIRECT_URL", raising=False)
        with caplog.at_level(logging.DEBUG, logger="litellm_proxy_extras"):
            succeeded, calls, _ = self._run_v1_migrations(
                monkeypatch,
                tmp_path,
                deploy_stderr=_P3005_STDERR,
                diff_stderr=f"baseline diff failed: XYZ-7731 for {database_url}",
                database_url=database_url,
            )

        assert succeeded is False
        assert self._deploy_call_count(calls) == 4
        assert [
            record.getMessage()
            for record in caplog.records
            if "s3cr3t" in record.getMessage() or 'p"w' in record.getMessage()
        ] == []
        messages: Final = self._error_messages(caplog)
        assert any("postgresql://REDACTED@db:5432/litellm" in message for message in messages)
        assert any("XYZ-7731" in message for message in messages)


@pytest.mark.parametrize(
    "database_url,direct_url,text,expected",
    (
        (
            "postgresql://u:pa ss@db:5432/litellm",
            None,
            'Error: P1000: Authentication failed against database server at "postgresql://u:pa ss@db:5432/litellm"',
            'Error: P1000: Authentication failed against database server at "postgresql://REDACTED@db:5432/litellm"',
        ),
        (
            "postgresql://u:pa'ss@db:5432/litellm",
            None,
            "postgresql://u:pa'ss@db:5432/litellm",
            "postgresql://REDACTED@db:5432/litellm",
        ),
        (
            'postgresql://u:pa"ss@db:5432/litellm',
            None,
            'postgresql://u:pa"ss@db:5432/litellm',
            "postgresql://REDACTED@db:5432/litellm",
        ),
        (
            "postgresql://u:p@ss@db:5432/litellm",
            None,
            "postgresql://u:p@ss@db:5432/litellm",
            "postgresql://REDACTED@db:5432/litellm",
        ),
        (
            "postgresql://u:p%20ss@db:5432/litellm",
            None,
            "postgresql://u:p ss@db:5432/litellm",
            "postgresql://REDACTED@db:5432/litellm",
        ),
        (
            "postgresql://db/litellm?password=a b&sslmode=require",
            None,
            "postgresql://db/litellm?password=a b&sslmode=require",
            "postgresql://db/litellm?REDACTED&sslmode=require",
        ),
        (
            "postgresql://db/litellm?sslpassword=zq'7x",
            None,
            "postgresql://db/litellm?sslpassword=zq'7x",
            "postgresql://db/litellm?REDACTED",
        ),
        (
            None,
            "postgresql://u:pa ss@db:5432/litellm",
            "postgresql://u:pa ss@db:5432/litellm",
            "postgresql://REDACTED@db:5432/litellm",
        ),
        (
            None,
            None,
            "postgresql://u:pw@db/x",
            "postgresql://REDACTED@db/x",
        ),
        (
            "postgresql://u:p@db:5432/litellm",
            None,
            "Error: P1001: Can't reach database server at db:5432",
            "Error: P1001: Can't reach database server at db:5432",
        ),
        (
            "postgresql://u:p@db:5432/litellm",
            None,
            "Error:P1001: Can't reach database server at db:5432",
            "Error:P1001: Can't reach database server at db:5432",
        ),
        (
            None,
            None,
            "plain text with no URL",
            "plain text with no URL",
        ),
    ),
)
def test_redact_credentials_masks_passwords_in_embedded_urls(
    database_url: str | None,
    direct_url: str | None,
    text: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if database_url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database_url)
    if direct_url is None:
        monkeypatch.delenv("DIRECT_URL", raising=False)
    else:
        monkeypatch.setenv("DIRECT_URL", direct_url)
    assert _redact_credentials(text) == expected


@pytest.mark.parametrize("password", ("zq'7x", 'zq"7x', "zq'\"7x", "zq 7x", "zq@7x"))
def test_redact_command_error_masks_url_arguments(password: str, monkeypatch: pytest.MonkeyPatch) -> None:
    database_url: Final = f"postgresql://u:{password}@db:5432/litellm"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("DIRECT_URL", raising=False)
    error: Final = subprocess.CalledProcessError(1, ["prisma", "migrate", "diff", "--to-url", database_url])

    message: Final = _redact_command_error(error)

    assert "zq" not in message
    assert "7x" not in message
    assert "postgresql://REDACTED@db:5432/litellm" in message
    assert "returned non-zero exit status 1" in message


@pytest.mark.parametrize(
    "command",
    (
        None,
        Path("/usr/bin/prisma"),
        7,
        ("prisma", "migrate", "deploy"),
        ["prisma", "migrate", "deploy"],
        "prisma migrate deploy",
    ),
)
def test_redact_command_error_preserves_unredacted_command_format(
    command: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DIRECT_URL", raising=False)
    error: Final = subprocess.CalledProcessError(1, command)

    assert _redact_command_error(error) == str(error)


def test_redact_command_error_masks_password_in_tuple_url_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password: Final = "zq 7x"
    database_url: Final = f"postgresql://u:{password}@db:5432/litellm"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("DIRECT_URL", raising=False)
    error: Final = subprocess.CalledProcessError(1, ("prisma", "migrate", "deploy", "--to-url", database_url))

    message: Final = _redact_command_error(error)

    assert message.startswith("Command '('")
    assert password not in message
    assert "postgresql://REDACTED@db:5432/litellm" in message
