import os
import sys
import time
from unittest.mock import Mock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm_proxy_extras.utils import ProxyExtrasDBManager, MigrationLockManager


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


class TestMigrationLockManager:
    """Test cases for MigrationLockManager"""

    def test_acquire_lock_success(self):
        """Test successful lock acquisition using Redis SET NX"""
        mock_redis = Mock()
        # FIX VERIFICATION: redis_client.set() returns True for successful SET NX
        mock_redis.redis_client.set.return_value = True

        lock_manager = MigrationLockManager(mock_redis)
        result = lock_manager.acquire_lock()

        assert result is True
        assert lock_manager.lock_acquired is True
        # Verify SET NX was called with correct parameters
        mock_redis.redis_client.set.assert_called_once()
        call_args = mock_redis.redis_client.set.call_args
        assert call_args[1]["nx"] is True  # SET NX parameter
        assert call_args[1]["ex"] == 300  # TTL

    def test_acquire_lock_failure(self):
        """Test lock acquisition failure when lock is already held"""
        mock_redis = Mock()
        # FIX VERIFICATION: redis_client.set() returns False when key exists
        mock_redis.redis_client.set.return_value = False

        lock_manager = MigrationLockManager(mock_redis)
        result = lock_manager.acquire_lock()

        assert result is False
        assert lock_manager.lock_acquired is False

    def test_acquire_lock_without_redis(self):
        """Test lock acquisition without Redis (graceful fallback)"""
        lock_manager = MigrationLockManager(redis_cache=None)
        result = lock_manager.acquire_lock()

        assert result is True
        assert lock_manager.lock_acquired is True

    def test_release_lock_success(self):
        """Test successful lock release using Lua script"""
        mock_redis = Mock()
        # FIX VERIFICATION: Lua script returns 1 for successful delete
        mock_redis.redis_client.eval.return_value = 1

        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"
        lock_manager.lock_acquired = True

        lock_manager.release_lock()

        # Verify Lua script was called for atomic compare-and-delete
        mock_redis.redis_client.eval.assert_called_once()
        call_args = mock_redis.redis_client.eval.call_args
        # Verify Lua script contains atomic compare-and-delete logic
        lua_script = call_args[0][0]
        assert "GET" in lua_script
        assert "DEL" in lua_script
        assert lock_manager.lock_acquired is False

    def test_release_lock_wrong_owner(self):
        """Test releasing lock when not the owner"""
        mock_redis = Mock()
        # FIX VERIFICATION: Lua script returns 0 when ownership check fails
        mock_redis.redis_client.eval.return_value = 0

        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"
        lock_manager.lock_acquired = True

        lock_manager.release_lock()

        mock_redis.redis_client.eval.assert_called_once()
        assert lock_manager.lock_acquired is False

    def test_context_manager(self):
        """Test MigrationLockManager as context manager"""
        mock_redis = Mock()
        mock_redis.redis_client.set.return_value = True
        mock_redis.redis_client.eval.return_value = 1

        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"

        with lock_manager:
            assert lock_manager.lock_acquired is True

        # Should call release_lock when exiting context
        mock_redis.redis_client.eval.assert_called_once()

    def test_wait_for_lock_release_success(self):
        """Test waiting for lock release and acquiring it"""
        mock_redis = Mock()
        # First call fails, second call succeeds
        mock_redis.redis_client.set.side_effect = [False, True]

        lock_manager = MigrationLockManager(mock_redis)
        result = lock_manager.wait_for_lock_release(check_interval=0.1, max_wait=1)

        assert result is True
        assert lock_manager.lock_acquired is True
        assert mock_redis.redis_client.set.call_count == 2

    def test_wait_for_lock_release_timeout(self):
        """Test timeout when waiting for lock release"""
        mock_redis = Mock()
        # Always fails to acquire lock
        mock_redis.redis_client.set.return_value = False

        lock_manager = MigrationLockManager(mock_redis)
        start_time = time.time()
        result = lock_manager.wait_for_lock_release(check_interval=0.1, max_wait=0.5)
        end_time = time.time()

        assert result is False
        assert lock_manager.lock_acquired is False
        assert end_time - start_time >= 0.5  # Should wait at least max_wait time
        assert mock_redis.redis_client.set.call_count > 1  # Multiple attempts


class TestProxyExtrasDBManagerMigrationLock:
    """Test cases for ProxyExtrasDBManager with migration locking"""

    @patch("litellm_proxy_extras.utils.ProxyExtrasDBManager._execute_migration")
    def test_setup_database_with_redis_lock_success(
        self, mock_execute_migration, monkeypatch
    ):
        """Test successful database setup with Redis lock"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        mock_execute_migration.return_value = True

        # Mock Redis cache
        mock_redis = Mock()
        mock_redis.redis_client.set.return_value = True  # Lock acquired
        mock_redis.redis_client.eval.return_value = 1  # Lock released

        result = ProxyExtrasDBManager.setup_database(
            use_migrate=True, redis_cache=mock_redis
        )

        assert result is True
        mock_execute_migration.assert_called_once()
        # Verify lock was acquired
        mock_redis.redis_client.set.assert_called_once()

    @patch("litellm_proxy_extras.utils.ProxyExtrasDBManager._execute_migration")
    def test_setup_database_with_redis_lock_wait_and_acquire(
        self, mock_execute_migration, monkeypatch
    ):
        """Test database setup when lock is held, then acquired after waiting"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        mock_execute_migration.return_value = True

        # Mock Redis cache - first call fails, second call succeeds
        mock_redis = Mock()
        mock_redis.redis_client.set.side_effect = [False, True]
        mock_redis.redis_client.eval.return_value = 1

        result = ProxyExtrasDBManager.setup_database(
            use_migrate=True, redis_cache=mock_redis
        )

        assert result is True
        mock_execute_migration.assert_called_once()
        # Should have tried to acquire lock twice
        assert mock_redis.redis_client.set.call_count == 2

    @patch("litellm_proxy_extras.utils.ProxyExtrasDBManager._execute_migration")
    def test_setup_database_without_redis(self, mock_execute_migration, monkeypatch):
        """Test database setup without Redis cache (single instance mode)"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        mock_execute_migration.return_value = True

        result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=None)

        assert result is True
        mock_execute_migration.assert_called_once()

    def test_setup_database_no_database_url(self):
        """Test database setup without DATABASE_URL"""
        with patch.dict(os.environ, {}, clear=True):
            result = ProxyExtrasDBManager.setup_database(
                use_migrate=True, redis_cache=None
            )

        assert result is False

    @patch("litellm_proxy_extras.utils.ProxyExtrasDBManager._execute_migration")
    def test_setup_database_lock_timeout(self, mock_execute_migration, monkeypatch):
        """Test database setup when lock acquisition times out"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")

        # Mock Redis cache - always fails to acquire lock
        mock_redis = Mock()
        mock_redis.redis_client.set.return_value = False

        # Patch wait_for_lock_release to simulate timeout
        with patch.object(MigrationLockManager, "wait_for_lock_release") as mock_wait:
            mock_wait.return_value = False  # Simulate timeout

            result = ProxyExtrasDBManager.setup_database(
                use_migrate=True, redis_cache=mock_redis
            )

        assert result is False
        mock_execute_migration.assert_not_called()  # Should not run migration
        mock_wait.assert_called_once()
