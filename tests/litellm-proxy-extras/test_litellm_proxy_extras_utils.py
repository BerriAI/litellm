import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../litellm-proxy-extras")
    ),
)

from litellm_proxy_extras.utils import ProxyExtrasDBManager


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


class TestMarkAllMigrationsApplied:
    """Test that _mark_all_migrations_applied only marks migrations without applying diffs"""

    @patch("litellm_proxy_extras.utils.subprocess.run")
    @patch.object(
        ProxyExtrasDBManager,
        "_get_migration_names",
        return_value=["20250326162113_baseline", "20250329084805_new_cron_job_table"],
    )
    def test_marks_each_migration_as_applied(self, mock_get_names, mock_run):
        """Verify each migration is marked as applied via prisma migrate resolve"""
        ProxyExtrasDBManager._mark_all_migrations_applied("/fake/migrations/dir")

        assert mock_run.call_count == 2
        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            assert "migrate" in cmd
            assert "resolve" in cmd
            assert "--applied" in cmd

    @patch("litellm_proxy_extras.utils.subprocess.run")
    @patch.object(
        ProxyExtrasDBManager,
        "_get_migration_names",
        return_value=["20250326162113_baseline"],
    )
    def test_does_not_generate_or_apply_diffs(self, mock_get_names, mock_run):
        """Verify no diff generation or db execute commands are run"""
        ProxyExtrasDBManager._mark_all_migrations_applied("/fake/migrations/dir")

        for call_args in mock_run.call_args_list:
            cmd = call_args[0][0]
            # Should never run diff or db execute
            assert "diff" not in cmd
            assert "execute" not in cmd
            assert "push" not in cmd

    @patch("litellm_proxy_extras.utils.subprocess.run")
    @patch.object(
        ProxyExtrasDBManager,
        "_get_migration_names",
        return_value=["20250326162113_baseline"],
    )
    def test_skips_already_applied_migration(self, mock_get_names, mock_run):
        """Verify already-applied migrations are silently skipped"""
        mock_run.side_effect = subprocess.CalledProcessError(
            1,
            "prisma",
            stderr="Migration `20250326162113_baseline` is already recorded as applied in the database.",
        )
        # Should not raise
        ProxyExtrasDBManager._mark_all_migrations_applied("/fake/migrations/dir")


class TestSetupDatabaseFailFast:
    """Test that setup_database fails fast on non-recoverable migration errors"""

    @patch("litellm_proxy_extras.utils.os.chdir")
    @patch("litellm_proxy_extras.utils.os.getcwd", return_value="/original")
    @patch.object(
        ProxyExtrasDBManager, "_get_prisma_dir", return_value="/fake/prisma/dir"
    )
    @patch("litellm_proxy_extras.utils.subprocess.run")
    def test_p3009_non_idempotent_raises_runtime_error(
        self, mock_run, mock_dir, mock_getcwd, mock_chdir
    ):
        """P3009 with non-idempotent error should raise RuntimeError, not silently retry"""
        error = subprocess.CalledProcessError(
            1,
            "prisma",
            stderr="P3009: migrate found failed migrations in the target database, `20250329084805_new_cron_job_table` migration. Error: syntax error at or near 'ALTR'",
            output="",
        )
        mock_run.side_effect = error

        with pytest.raises(RuntimeError, match="requires manual intervention"):
            ProxyExtrasDBManager.setup_database(use_migrate=True)

    @patch("litellm_proxy_extras.utils.os.chdir")
    @patch("litellm_proxy_extras.utils.os.getcwd", return_value="/original")
    @patch.object(
        ProxyExtrasDBManager, "_get_prisma_dir", return_value="/fake/prisma/dir"
    )
    @patch("litellm_proxy_extras.utils.subprocess.run")
    def test_p3009_unmatched_regex_raises_runtime_error(
        self, mock_run, mock_dir, mock_getcwd, mock_chdir
    ):
        """P3009 with unparseable migration name should fail fast, not silently retry"""
        error = subprocess.CalledProcessError(
            1,
            "prisma",
            stderr="P3009: migrate found failed migrations in the target database, unexpected format",
            output="",
        )
        mock_run.side_effect = error

        with pytest.raises(RuntimeError, match="could not extract migration name"):
            ProxyExtrasDBManager.setup_database(use_migrate=True)

        # Should fail on first attempt, not retry
        assert mock_run.call_count == 1

    @patch("litellm_proxy_extras.utils.os.chdir")
    @patch("litellm_proxy_extras.utils.os.getcwd", return_value="/original")
    @patch.object(
        ProxyExtrasDBManager, "_get_prisma_dir", return_value="/fake/prisma/dir"
    )
    @patch("litellm_proxy_extras.utils.subprocess.run")
    def test_successful_deploy_does_not_call_resolve_all(
        self, mock_run, mock_dir, mock_getcwd, mock_chdir
    ):
        """After successful prisma migrate deploy, no diff/resolve should be called"""
        mock_run.return_value = MagicMock(stdout="All migrations applied", returncode=0)

        result = ProxyExtrasDBManager.setup_database(use_migrate=True)

        assert result is True
        # Only one subprocess call: prisma migrate deploy
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert cmd == ["prisma", "migrate", "deploy"]
