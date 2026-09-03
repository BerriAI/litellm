import glob
import os
import re
import sys
import types

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../litellm-proxy-extras")
    ),
)

from litellm_proxy_extras.utils import (
    PARTITIONED_SPEND_LOGS_PUSH_ERROR,
    SKIP_INDEX_MIGRATIONS_ENV_VAR,
    ProxyExtrasDBManager,
    _MigrationLedger,
    filter_partitioned_spend_logs_diff,
    filter_skipped_index_statements,
    index_names_created_by,
    is_index_only_migration,
)

# Path to the migrations directory
_MIGRATIONS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../litellm-proxy-extras/litellm_proxy_extras/migrations",
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


class _MigrateDeployHarness:
    """Drives _setup_database_v2 with a scripted sequence of
    `prisma migrate deploy` outcomes, with every recovery command faked out so
    nothing touches a database or the packaged migrations directory."""

    def __init__(self, monkeypatch, tmp_path, outcomes, repeat_last=False):
        import subprocess as subprocess_module

        import litellm_proxy_extras.utils as utils_module

        self.deploy_calls = []
        self.resolved = []
        self.baselines = 0
        self._outcomes = list(outcomes)
        self._repeat_last = repeat_last
        self._subprocess_module = subprocess_module

        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(
            ProxyExtrasDBManager, "_get_prisma_dir", staticmethod(lambda: str(tmp_path))
        )
        monkeypatch.setattr(
            ProxyExtrasDBManager,
            "_create_baseline_migration",
            staticmethod(self._fake_baseline),
        )
        monkeypatch.setattr(
            ProxyExtrasDBManager,
            "_roll_back_migration",
            staticmethod(lambda name: None),
        )
        monkeypatch.setattr(
            ProxyExtrasDBManager,
            "_resolve_specific_migration",
            staticmethod(self.resolved.append),
        )
        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", self._fake_run)
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)

        self.baseline_succeeds = True

    def _fake_baseline(self, *args, **kwargs):
        self.baselines += 1
        return self.baseline_succeeds

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
        return ProxyExtrasDBManager._setup_database_v2(use_migrate=True)


class TestMigrateDeployAttemptAccounting:
    """A `prisma db push` database has a full schema and no ledger, so the v2
    resolver baselines it and then works through every migration whose objects
    already exist. Those recoveries make progress, so they must not spend the
    retry budget, which is there to stop a run that is getting nowhere."""

    def test_a_push_created_database_finishes_bootstrapping(
        self, monkeypatch, tmp_path
    ):
        already_there = [
            "20250329084805_new_cron_job_table",
            "20250806095134_rename_alias_to_server_name_mcp_table",
            "20260224203854_add_agent_object_permissions_table",
            "20260301120000_fourth_table",
            "20260302120000_fifth_table",
            "20260303120000_sixth_table",
        ]
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_P3005_STDERR]
            + [_p3018_stderr(name) for name in already_there]
            + ["ok"],
        )

        assert harness.run() is True
        assert harness.baselines == 1
        assert harness.resolved == already_there
        assert len(harness.deploy_calls) == len(already_there) + 2

    def test_repeated_recovery_of_one_migration_still_gives_up(
        self, monkeypatch, tmp_path
    ):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            [_p3018_stderr("20250329084805_new_cron_job_table")],
            repeat_last=True,
        )

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) <= _ATTEMPT_BUDGET + 1

    def test_timeouts_still_spend_the_budget(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch, tmp_path, ["timeout"], repeat_last=True
        )

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == _ATTEMPT_BUDGET

    def test_a_baseline_that_never_lands_stops_after_the_budget(
        self, monkeypatch, tmp_path
    ):
        harness = _MigrateDeployHarness(
            monkeypatch, tmp_path, [_P3005_STDERR], repeat_last=True
        )
        harness.baseline_succeeds = False

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == _ATTEMPT_BUDGET

    def test_an_unrecoverable_error_is_not_retried(self, monkeypatch, tmp_path):
        harness = _MigrateDeployHarness(
            monkeypatch,
            tmp_path,
            ["Error: P3018\n\nMigration name: 20260101000000_x\n\nERROR: syntax error at or near \"SLECT\"\n"],
            repeat_last=True,
        )

        with pytest.raises(RuntimeError):
            harness.run()
        assert len(harness.deploy_calls) == 1
        assert harness.resolved == []


_INDEX_ONLY_MIGRATION = "20260823000000_add_spend_logs_api_key_starttime_index"
_SKIPPED_INDEX_NAME = "LiteLLM_SpendLogs_api_key_startTime_idx"
_INDEX_ONLY_MIGRATION_SQL = (
    "-- CreateIndex\n"
    f'CREATE INDEX IF NOT EXISTS "{_SKIPPED_INDEX_NAME}" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n'
)
_OTHER_INDEX_ONLY_MIGRATION = "20260826000000_add_team_alias_index"
_OTHER_INDEX_NAME = "LiteLLM_TeamTable_team_alias_idx"
_OTHER_INDEX_ONLY_MIGRATION_SQL = (
    f'CREATE INDEX IF NOT EXISTS "{_OTHER_INDEX_NAME}" ON "LiteLLM_TeamTable"("team_alias");\n'
)
_COLUMN_MIGRATION = "20260824000000_add_budget_updated_by"
_COLUMN_MIGRATION_SQL = '-- AlterTable\nALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN "updated_by" TEXT;\n'
_MIXED_MIGRATION = "20260825000000_add_widget_table"
_MIXED_MIGRATION_SQL = (
    'CREATE TABLE IF NOT EXISTS "LiteLLM_Widget" ("id" TEXT NOT NULL);\n'
    'CREATE INDEX IF NOT EXISTS "LiteLLM_Widget_id_idx" ON "LiteLLM_Widget"("id");\n'
)
_DRIFT_WITH_SKIPPED_INDEX_SQL = (
    "-- AlterTable\n"
    'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;\n\n'
    "-- CreateIndex\n"
    f'CREATE INDEX "{_SKIPPED_INDEX_NAME}" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n\n'
    "-- CreateIndex\n"
    'CREATE INDEX "LiteLLM_SpendLogs_session_id_idx" ON "LiteLLM_SpendLogs"("session_id");\n'
)
_ONLY_SKIPPED_INDEX_DRIFT_SQL = (
    f'-- CreateIndex\nCREATE INDEX "{_SKIPPED_INDEX_NAME}" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n'
)
_DRIFT_WITH_HISTORICAL_INDEX_SQL = (
    "-- AlterTable\n"
    'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;\n\n'
    "-- CreateIndex\n"
    f'CREATE INDEX "{_OTHER_INDEX_NAME}" ON "LiteLLM_TeamTable"("team_alias");\n'
)
_DRIFT_DROPPING_SKIPPED_INDEX_SQL = (
    "-- DropIndex\n"
    f'DROP INDEX "{_SKIPPED_INDEX_NAME}";\n\n'
    "-- AlterTable\n"
    'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;\n'
)


def _ship_migration(prisma_dir, name, sql):
    migration_dir = prisma_dir / "migrations" / name
    migration_dir.mkdir(parents=True)
    (migration_dir / "migration.sql").write_text(sql)


def _ship_release(prisma_dir):
    _ship_migration(prisma_dir, _INDEX_ONLY_MIGRATION, _INDEX_ONLY_MIGRATION_SQL)
    _ship_migration(prisma_dir, _OTHER_INDEX_ONLY_MIGRATION, _OTHER_INDEX_ONLY_MIGRATION_SQL)
    _ship_migration(prisma_dir, _COLUMN_MIGRATION, _COLUMN_MIGRATION_SQL)
    _ship_migration(prisma_dir, _MIXED_MIGRATION, _MIXED_MIGRATION_SQL)


def _skip_index_migrations(monkeypatch, tmp_path):
    _ship_release(tmp_path)
    monkeypatch.setenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, "true")


def _ledger(exists=True, applied=()):
    return _MigrationLedger(exists=exists, applied=frozenset(applied))


class _CompletedWithStdout:
    stderr = ""

    def __init__(self, stdout):
        self.stdout = stdout


class TestIndexOnlyMigrationDetection:
    def test_create_index_only_is_index_only(self):
        assert is_index_only_migration(_INDEX_ONLY_MIGRATION_SQL) is True

    def test_drop_and_create_index_is_index_only(self):
        sql = (
            'DROP INDEX IF EXISTS "LiteLLM_TagTable_name_key";\n'
            'CREATE INDEX "LiteLLM_TagTable_name_idx" ON "LiteLLM_TagTable"("name");\n'
        )
        assert is_index_only_migration(sql) is True

    def test_a_migration_that_only_drops_indexes_is_not(self):
        assert is_index_only_migration('DROP INDEX IF EXISTS "LiteLLM_DailyTagSpend_tag_key";\n') is False

    def test_a_table_or_column_change_is_not(self):
        assert is_index_only_migration(_MIXED_MIGRATION_SQL) is False
        assert is_index_only_migration(_COLUMN_MIGRATION_SQL) is False

    def test_an_empty_migration_is_not(self):
        assert is_index_only_migration("-- nothing here\n") is False

    def test_index_names_created_by_handles_every_create_index_form(self):
        sql = (
            'CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "a" ON "t"("x");\n'
            "-- CreateIndex\n"
            'create index "b" on "t"("y");\n'
            'ALTER TABLE "t" ADD COLUMN "z" TEXT;\n'
        )
        assert index_names_created_by(sql) == frozenset({"a", "b"})


class TestSkipIndexMigrationsEnv:
    def test_unset_skips_nothing(self, monkeypatch, tmp_path):
        monkeypatch.delenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, raising=False)
        _ship_release(tmp_path)
        assert ProxyExtrasDBManager.skipped_migrations(str(tmp_path)) == ()
        assert ProxyExtrasDBManager.skipped_index_names(str(tmp_path)) == frozenset()

    def test_only_index_only_migrations_are_skipped(self, monkeypatch, tmp_path):
        _skip_index_migrations(monkeypatch, tmp_path)
        assert ProxyExtrasDBManager.skipped_migrations(str(tmp_path)) == (
            _INDEX_ONLY_MIGRATION,
            _OTHER_INDEX_ONLY_MIGRATION,
        )
        assert ProxyExtrasDBManager.skipped_index_names(str(tmp_path)) == frozenset(
            {_SKIPPED_INDEX_NAME, _OTHER_INDEX_NAME}
        )

    def test_shipped_index_only_migrations_are_recognized_and_the_baseline_is_not(self, monkeypatch):
        monkeypatch.setenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, "true")
        skipped = ProxyExtrasDBManager.skipped_migrations(os.path.dirname(_MIGRATIONS_DIR))
        assert "20260228100000_add_spend_logs_composite_index" in skipped
        assert "20250326162113_baseline" not in skipped
        assert "20250416151339_drop_tag_uniqueness_requirement" not in skipped


class TestSkippedIndexDriftFilter:
    def test_only_the_skipped_index_is_removed(self):
        filtered = filter_skipped_index_statements(_DRIFT_WITH_SKIPPED_INDEX_SQL, frozenset({_SKIPPED_INDEX_NAME}))
        assert _SKIPPED_INDEX_NAME not in filtered
        assert 'CREATE INDEX "LiteLLM_SpendLogs_session_id_idx"' in filtered
        assert 'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;' in filtered

    def test_a_drop_of_the_skipped_index_is_removed_too(self):
        sql = (
            f'DROP INDEX "{_SKIPPED_INDEX_NAME}";\n\n'
            f'CREATE INDEX "{_SKIPPED_INDEX_NAME}" ON "LiteLLM_SpendLogs"("api_key", "startTime");\n\n'
            'CREATE INDEX "LiteLLM_SpendLogs_session_id_idx" ON "LiteLLM_SpendLogs"("session_id");\n'
        )
        filtered = filter_skipped_index_statements(sql, frozenset({_SKIPPED_INDEX_NAME}))
        assert _SKIPPED_INDEX_NAME not in filtered
        assert 'CREATE INDEX "LiteLLM_SpendLogs_session_id_idx"' in filtered

    def test_nothing_to_skip_leaves_the_script_byte_identical(self):
        unfiltered = filter_skipped_index_statements(_DRIFT_WITH_SKIPPED_INDEX_SQL, frozenset())
        assert unfiltered == _DRIFT_WITH_SKIPPED_INDEX_SQL

    def test_a_script_that_is_only_the_skipped_index_becomes_empty(self):
        only_skipped = filter_skipped_index_statements(_ONLY_SKIPPED_INDEX_DRIFT_SQL, frozenset({_SKIPPED_INDEX_NAME}))
        assert only_skipped.strip() == ""


class TestResolveAllMigrationsHonorsSkippedIndexes:
    def _run(self, monkeypatch, tmp_path, diff_sql, flag=True):
        import litellm_proxy_extras.utils as utils_module

        if flag:
            _skip_index_migrations(monkeypatch, tmp_path)
        else:
            monkeypatch.delenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, raising=False)
            _ship_release(tmp_path)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        monkeypatch.delenv("DIRECT_URL", raising=False)
        monkeypatch.setattr(ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: False))
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "diff" in cmd:
                kwargs["stdout"].write(diff_sql)
            if "execute" in cmd:
                calls.append(("executed_sql", open(cmd[cmd.index("--file") + 1]).read()))
            return _FakeCompleted()

        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", fake_run)
        ProxyExtrasDBManager._resolve_all_migrations(str(tmp_path), "schema.prisma")
        return calls

    def test_the_skipped_index_is_kept_out_of_the_executed_drift_script(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, _DRIFT_WITH_SKIPPED_INDEX_SQL)
        executed = next(c[1] for c in calls if isinstance(c, tuple))
        assert _SKIPPED_INDEX_NAME not in executed
        assert 'CREATE INDEX "LiteLLM_SpendLogs_session_id_idx"' in executed
        assert 'ADD COLUMN     "updated_by" TEXT' in executed

    def test_a_drift_that_is_only_the_skipped_index_executes_nothing_and_still_marks_applied(
        self, monkeypatch, tmp_path
    ):
        calls = self._run(monkeypatch, tmp_path, _ONLY_SKIPPED_INDEX_DRIFT_SQL)
        assert not any(isinstance(c, tuple) for c in calls)
        resolved = {c[-1] for c in calls if isinstance(c, list) and "resolve" in c}
        assert resolved == {_INDEX_ONLY_MIGRATION, _OTHER_INDEX_ONLY_MIGRATION, _COLUMN_MIGRATION, _MIXED_MIGRATION}

    def test_an_index_owned_by_a_migration_applied_long_ago_is_kept_out_while_the_flag_is_set(
        self, monkeypatch, tmp_path
    ):
        calls = self._run(monkeypatch, tmp_path, _DRIFT_WITH_HISTORICAL_INDEX_SQL)
        executed = next(c[1] for c in calls if isinstance(c, tuple))
        assert _OTHER_INDEX_NAME not in executed
        assert 'ADD COLUMN     "updated_by" TEXT' in executed

    def test_without_the_flag_the_drift_script_rebuilds_that_index(self, monkeypatch, tmp_path):
        calls = self._run(monkeypatch, tmp_path, _DRIFT_WITH_HISTORICAL_INDEX_SQL, flag=False)
        executed = next(c[1] for c in calls if isinstance(c, tuple))
        assert f'CREATE INDEX "{_OTHER_INDEX_NAME}"' in executed

    def test_the_log_names_only_the_indexes_actually_removed(self, monkeypatch, tmp_path, caplog):
        with caplog.at_level("INFO", logger="litellm_proxy_extras"):
            self._run(monkeypatch, tmp_path, _DRIFT_WITH_SKIPPED_INDEX_SQL)
        removed_lines = [r.message for r in caplog.records if "removed the index statements" in r.message]
        assert removed_lines == [
            f"{SKIP_INDEX_MIGRATIONS_ENV_VAR}: removed the index statements for {_SKIPPED_INDEX_NAME} "
            "from the drift script"
        ]

    def test_a_drift_without_a_guarded_index_logs_no_removal(self, monkeypatch, tmp_path, caplog):
        with caplog.at_level("INFO", logger="litellm_proxy_extras"):
            self._run(monkeypatch, tmp_path, _PARTITIONED_DRIFT_SQL)
        assert not any("removed the index statements" in r.message for r in caplog.records)

    def test_the_log_names_a_skipped_index_that_is_only_dropped(self, monkeypatch, tmp_path, caplog):
        with caplog.at_level("INFO", logger="litellm_proxy_extras"):
            self._run(monkeypatch, tmp_path, _DRIFT_DROPPING_SKIPPED_INDEX_SQL)
        assert any(f"removed the index statements for {_SKIPPED_INDEX_NAME} from" in r.message for r in caplog.records)


class _V2SkipHarness:
    def __init__(self, monkeypatch, tmp_path, outcomes, ledgers):
        import subprocess as subprocess_module

        import litellm_proxy_extras.utils as utils_module

        self.events = []
        self.baselines = 0
        self.resolve_error = None
        self._outcomes = list(outcomes)
        self._subprocess = subprocess_module
        ledger_reads = iter(ledgers)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", staticmethod(lambda: str(tmp_path)))
        monkeypatch.setattr(ProxyExtrasDBManager, "_read_migration_ledger", staticmethod(lambda: next(ledger_reads)))
        monkeypatch.setattr(ProxyExtrasDBManager, "_create_baseline_migration", staticmethod(self._baseline))
        monkeypatch.setattr(ProxyExtrasDBManager, "_roll_back_migration", staticmethod(lambda name: None))
        monkeypatch.setattr(ProxyExtrasDBManager, "_resolve_specific_migration", staticmethod(self._resolve))
        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", self._run_prisma)
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)

    def _baseline(self, *args, **kwargs):
        self.baselines += 1
        return True

    def _resolve(self, name):
        self.events.append(("resolve", name))
        if self.resolve_error is not None:
            raise self.resolve_error

    def _run_prisma(self, cmd, **kwargs):
        assert cmd[1:] == ["migrate", "deploy"], cmd
        self.events.append(("deploy",))
        outcome = self._outcomes.pop(0)
        if outcome == "ok":
            return _FakeCompleted()
        raise self._subprocess.CalledProcessError(1, cmd, stderr=outcome)

    def run(self):
        return ProxyExtrasDBManager._setup_database_v2(use_migrate=True)


class TestV2RecordsSkippedIndexMigrationsBeforeDeploy:
    def test_the_skip_is_recorded_before_the_first_deploy(self, monkeypatch, tmp_path):
        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [_ledger()])
        assert harness.run() is True
        assert harness.events == [
            ("resolve", _INDEX_ONLY_MIGRATION),
            ("resolve", _OTHER_INDEX_ONLY_MIGRATION),
            ("deploy",),
        ]

    def test_unset_env_never_touches_the_ledger(self, monkeypatch, tmp_path):
        monkeypatch.delenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, raising=False)
        _ship_release(tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [])
        assert harness.run() is True
        assert harness.events == [("deploy",)]

    def test_an_index_migration_the_ledger_already_has_is_left_alone(self, monkeypatch, tmp_path):
        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [_ledger(applied=[_INDEX_ONLY_MIGRATION])])
        assert harness.run() is True
        assert harness.events == [("resolve", _OTHER_INDEX_ONLY_MIGRATION), ("deploy",)]

    def test_a_database_without_a_ledger_is_baselined_before_the_skip_is_recorded(self, monkeypatch, tmp_path):
        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, [_P3005_STDERR, "ok"], [_ledger(exists=False), _ledger()])
        assert harness.run() is True
        assert harness.baselines == 1
        assert harness.events == [
            ("deploy",),
            ("resolve", _INDEX_ONLY_MIGRATION),
            ("resolve", _OTHER_INDEX_ONLY_MIGRATION),
            ("deploy",),
        ]

    def test_an_already_recorded_skip_is_not_an_error(self, monkeypatch, tmp_path):
        import subprocess as subprocess_module

        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [_ledger()])
        harness.resolve_error = subprocess_module.CalledProcessError(
            1,
            ["prisma", "migrate", "resolve"],
            stderr=(
                f"Error: P3008\n\nThe migration `{_INDEX_ONLY_MIGRATION}` "
                "is already recorded as applied in the database.\n"
            ),
        )
        assert harness.run() is True
        assert harness.events == [
            ("resolve", _INDEX_ONLY_MIGRATION),
            ("resolve", _OTHER_INDEX_ONLY_MIGRATION),
            ("deploy",),
        ]

    def test_any_other_resolve_failure_stops_boot_before_deploy(self, monkeypatch, tmp_path):
        import subprocess as subprocess_module

        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [_ledger()])
        harness.resolve_error = subprocess_module.CalledProcessError(
            1, ["prisma", "migrate", "resolve"], stderr="Error: P1001\n\nCan't reach database server\n"
        )
        with pytest.raises(RuntimeError, match="migrate resolve --applied"):
            harness.run()
        assert harness.events == [("resolve", _INDEX_ONLY_MIGRATION)]

    def test_a_resolve_timeout_stops_boot_before_deploy(self, monkeypatch, tmp_path):
        import subprocess as subprocess_module

        _skip_index_migrations(monkeypatch, tmp_path)
        harness = _V2SkipHarness(monkeypatch, tmp_path, ["ok"], [_ledger()])
        harness.resolve_error = subprocess_module.TimeoutExpired(["prisma", "migrate", "resolve"], 60)
        with pytest.raises(RuntimeError, match=r"(?s)migrate resolve --applied .*timed out after 60s"):
            harness.run()
        assert harness.events == [("resolve", _INDEX_ONLY_MIGRATION)]


class TestV1RecordsSkippedIndexMigrationsBeforeDeploy:
    def _arrange(self, monkeypatch, tmp_path, env_set, resolve_error=None):
        import litellm_proxy_extras.utils as utils_module

        if env_set:
            _skip_index_migrations(monkeypatch, tmp_path)
        else:
            monkeypatch.delenv(SKIP_INDEX_MIGRATIONS_ENV_VAR, raising=False)
            _ship_release(tmp_path)
        monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", staticmethod(lambda: str(tmp_path)))
        monkeypatch.setattr(ProxyExtrasDBManager, "_read_migration_ledger", staticmethod(lambda: _ledger()))
        monkeypatch.setattr(utils_module.time, "sleep", lambda seconds: None)
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd[1:])
            if resolve_error is not None and "resolve" in cmd:
                raise resolve_error
            return _CompletedWithStdout("No pending migrations to apply.")

        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", fake_run)
        return commands

    def _prisma_commands(self, monkeypatch, tmp_path, env_set):
        commands = self._arrange(monkeypatch, tmp_path, env_set)
        assert ProxyExtrasDBManager._run_migrations(use_migrate=True, use_v2_resolver=False) is True
        return commands

    def test_the_skip_is_recorded_before_deploy(self, monkeypatch, tmp_path):
        assert self._prisma_commands(monkeypatch, tmp_path, env_set=True) == [
            ["migrate", "resolve", "--applied", _INDEX_ONLY_MIGRATION],
            ["migrate", "resolve", "--applied", _OTHER_INDEX_ONLY_MIGRATION],
            ["migrate", "deploy"],
        ]

    def test_unset_env_only_deploys(self, monkeypatch, tmp_path):
        assert self._prisma_commands(monkeypatch, tmp_path, env_set=False) == [["migrate", "deploy"]]

    def test_any_other_resolve_failure_stops_boot_before_deploy(self, monkeypatch, tmp_path):
        import subprocess as subprocess_module

        commands = self._arrange(
            monkeypatch,
            tmp_path,
            env_set=True,
            resolve_error=subprocess_module.CalledProcessError(
                1, ["prisma", "migrate", "resolve"], stderr="Error: P1001\n\nCan't reach database server\n"
            ),
        )
        with pytest.raises(RuntimeError, match="migrate resolve --applied"):
            ProxyExtrasDBManager._run_migrations(use_migrate=True, use_v2_resolver=False)
        assert commands == [["migrate", "resolve", "--applied", _INDEX_ONLY_MIGRATION]]

    def test_a_resolve_timeout_stops_boot_before_deploy(self, monkeypatch, tmp_path):
        import subprocess as subprocess_module

        commands = self._arrange(
            monkeypatch,
            tmp_path,
            env_set=True,
            resolve_error=subprocess_module.TimeoutExpired(["prisma", "migrate", "resolve"], 60),
        )
        with pytest.raises(RuntimeError, match=r"(?s)migrate resolve --applied .*timed out after 60s"):
            ProxyExtrasDBManager._run_migrations(use_migrate=True, use_v2_resolver=False)
        assert commands == [["migrate", "resolve", "--applied", _INDEX_ONLY_MIGRATION]]


class TestSkipUnderDbPush:
    def test_v2_db_push_warns_that_the_skip_has_no_effect(self, monkeypatch, tmp_path, caplog):
        import litellm_proxy_extras.utils as utils_module

        _skip_index_migrations(monkeypatch, tmp_path)
        monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", staticmethod(lambda: str(tmp_path)))
        monkeypatch.setattr(ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(lambda: False))
        monkeypatch.setattr(utils_module.prisma_toolchain, "run_prisma", lambda cmd, **kwargs: _FakeCompleted())
        with caplog.at_level("WARNING", logger="litellm_proxy_extras"):
            assert ProxyExtrasDBManager._setup_database_v2(use_migrate=False) is True
        assert any("no effect under prisma db push" in r.message for r in caplog.records)


class _FakeLedgerConn:
    def __init__(self, rows, undefined_table, executed):
        self._rows = rows
        self._undefined_table = undefined_table
        self._executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self._executed.append(query.as_string(None))
        if self._undefined_table is not None:
            raise self._undefined_table()
        return self

    def fetchall(self):
        return self._rows


def _fake_psycopg(rows=(), undefined_table=False, connect_error=False, executed=None):
    import psycopg.sql

    module = types.ModuleType("psycopg")
    module.sql = psycopg.sql

    class UndefinedTable(Exception):
        pass

    class OperationalError(Exception):
        pass

    class DatabaseError(Exception):
        pass

    def connect(*args, **kwargs):
        if connect_error:
            raise OperationalError("unreachable")
        return _FakeLedgerConn(
            list(rows), UndefinedTable if undefined_table else None, executed if executed is not None else []
        )

    module.errors = types.SimpleNamespace(UndefinedTable=UndefinedTable)
    module.OperationalError = OperationalError
    module.DatabaseError = DatabaseError
    module.connect = connect
    return module


class TestReadMigrationLedger:
    def _read(self, monkeypatch, psycopg_module):
        monkeypatch.setitem(sys.modules, "psycopg", psycopg_module)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        return ProxyExtrasDBManager._read_migration_ledger()

    def test_applied_rows_are_read(self, monkeypatch):
        ledger = self._read(monkeypatch, _fake_psycopg(rows=[("a",), ("b",)]))
        assert ledger == _ledger(applied=["a", "b"])

    def test_the_ledger_is_read_from_the_url_schema(self, monkeypatch):
        executed = []
        monkeypatch.setitem(sys.modules, "psycopg", _fake_psycopg(rows=[("a",)], executed=executed))
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db?schema=custom")
        assert ProxyExtrasDBManager._read_migration_ledger() == _ledger(applied=["a"])
        assert 'FROM "custom"."_prisma_migrations"' in executed[0]

    def test_the_ledger_defaults_to_the_public_schema(self, monkeypatch):
        executed = []
        self._read(monkeypatch, _fake_psycopg(executed=executed))
        assert 'FROM "public"."_prisma_migrations"' in executed[0]

    def test_a_missing_ledger_table_reads_as_absent(self, monkeypatch):
        assert self._read(monkeypatch, _fake_psycopg(undefined_table=True)) == _ledger(exists=False)

    def test_no_psycopg_reads_as_an_existing_empty_ledger(self, monkeypatch):
        assert self._read(monkeypatch, None) == _ledger()

    def test_a_connection_failure_reads_as_an_existing_empty_ledger(self, monkeypatch):
        assert self._read(monkeypatch, _fake_psycopg(connect_error=True)) == _ledger()
