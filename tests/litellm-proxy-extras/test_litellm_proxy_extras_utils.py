import glob
import os
import re
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../litellm-proxy-extras")
    ),
)

from litellm_proxy_extras.utils import ProxyExtrasDBManager

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

    def test_is_idempotent_error_does_not_exist(self):
        """Test detection of 'does not exist' error"""
        error_message = "ERROR: index 'idx' does not exist"
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
        for migration_name, sql in all_migrations:
            for line_num, line in enumerate(sql.splitlines(), 1):
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
        for migration_name, sql in all_migrations:
            for line_num, line in enumerate(sql.splitlines(), 1):
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
        for migration_name, sql in all_migrations:
            for line_num, line in enumerate(sql.splitlines(), 1):
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
            for line_num, line in enumerate(sql.splitlines(), 1):
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
        for migration_name, sql in all_migrations:
            for line_num, line in enumerate(sql.splitlines(), 1):
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
        for migration_name, sql in all_migrations:
            for line_num, line in enumerate(sql.splitlines(), 1):
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
        for migration_name, sql in all_migrations:
            lines = sql.splitlines()
            in_do_block = False
            for line_num, line in enumerate(lines, 1):
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
        for migration_name, sql in all_migrations:
            lines = sql.splitlines()
            in_do_block = False
            for line_num, line in enumerate(lines, 1):
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
        for migration_name, sql in all_migrations:
            lines = sql.splitlines()
            in_do_block = False
            for line_num, line in enumerate(lines, 1):
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
