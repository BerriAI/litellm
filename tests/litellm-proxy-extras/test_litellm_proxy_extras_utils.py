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

class TestMigrationLockManager:
    """Test cases for MigrationLockManager"""

    def test_acquire_lock_without_redis(self):
        """Test lock acquisition when Redis is not available"""
        lock_manager = MigrationLockManager()
        result = lock_manager.acquire_lock()
        assert result is True  # Should return True when Redis is not available
        assert lock_manager.lock_acquired is True  # Redis 없을 때도 lock_acquired는 True

    def test_acquire_lock_with_redis_success(self):
        """Test successful lock acquisition with Redis"""
        mock_redis = Mock()
        mock_redis.set_cache.return_value = True
        lock_manager = MigrationLockManager(mock_redis)

        result = lock_manager.acquire_lock()

        assert result is True
        assert lock_manager.lock_acquired is True
        mock_redis.set_cache.assert_called_once()

    def test_acquire_lock_with_redis_failure(self):
        """Test failed lock acquisition with Redis"""
        mock_redis = Mock()
        mock_redis.set_cache.return_value = False
        lock_manager = MigrationLockManager(mock_redis)

        result = lock_manager.acquire_lock()

        assert result is False
        assert lock_manager.lock_acquired is False
        mock_redis.set_cache.assert_called_once()

    def test_acquire_lock_with_redis_exception(self):
        """Test lock acquisition with Redis exception"""
        mock_redis = Mock()
        mock_redis.set_cache.side_effect = Exception("Redis error")
        lock_manager = MigrationLockManager(mock_redis)

        result = lock_manager.acquire_lock()

        assert result is False
        assert lock_manager.lock_acquired is False

    def test_wait_for_lock_release_success(self):
        """Test successful waiting for lock release"""
        mock_redis = Mock()
        # First call returns False (lock held), second call returns True (lock acquired)
        mock_redis.set_cache.side_effect = [False, True]
        lock_manager = MigrationLockManager(mock_redis)

        result = lock_manager.wait_for_lock_release(check_interval=0.1, max_wait=1)

        assert result is True
        assert lock_manager.lock_acquired is True
        assert mock_redis.set_cache.call_count == 2

    def test_wait_for_lock_release_timeout(self):
        """Test timeout while waiting for lock release"""
        mock_redis = Mock()
        mock_redis.set_cache.return_value = False  # Lock always held
        lock_manager = MigrationLockManager(mock_redis)

        result = lock_manager.wait_for_lock_release(check_interval=0.1, max_wait=0.2)

        assert result is False
        assert lock_manager.lock_acquired is False

    def test_release_lock_not_acquired(self):
        """Test releasing lock when not acquired"""
        mock_redis = Mock()
        lock_manager = MigrationLockManager(mock_redis)

        lock_manager.release_lock()

        mock_redis.get_cache.assert_not_called()
        mock_redis.delete_cache.assert_not_called()

    def test_release_lock_success(self):
        """Test successful lock release"""
        mock_redis = Mock()
        mock_redis.get_cache.return_value = "pod_123_456"
        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"
        lock_manager.lock_acquired = True

        lock_manager.release_lock()

        mock_redis.get_cache.assert_called_once()
        mock_redis.delete_cache.assert_called_once()
        assert lock_manager.lock_acquired is False

    def test_release_lock_wrong_owner(self):
        """Test releasing lock when not the owner"""
        mock_redis = Mock()
        mock_redis.get_cache.return_value = "pod_999_999"  # Different pod
        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"
        lock_manager.lock_acquired = True

        lock_manager.release_lock()

        mock_redis.get_cache.assert_called_once()
        mock_redis.delete_cache.assert_not_called()
        assert lock_manager.lock_acquired is False


    def test_context_manager(self):
        """Test MigrationLockManager as context manager"""
        mock_redis = Mock()
        mock_redis.set_cache.return_value = True
        # Mock get_cache to return the same pod_id for successful release
        mock_redis.get_cache.return_value = "pod_123_456"

        lock_manager = MigrationLockManager(mock_redis)
        lock_manager.pod_id = "pod_123_456"  # Set consistent pod_id

        with lock_manager:
            assert lock_manager.lock_acquired is True

        # Should call release_lock when exiting context
        mock_redis.get_cache.assert_called_once()
        mock_redis.delete_cache.assert_called_once()


class TestProxyExtrasDBManagerMigrationLock:
    """Test cases for ProxyExtrasDBManager with migration locking"""

    @patch('litellm_proxy_extras.utils.subprocess.run')
    @patch('litellm_proxy_extras.utils.ProxyExtrasDBManager._get_prisma_dir')
    @patch('os.chdir')
    def test_setup_database_with_redis_lock_success(self, mock_chdir, mock_get_prisma_dir, mock_subprocess):
        """Test successful database setup with Redis lock"""
        # Setup mocks
        mock_get_prisma_dir.return_value = "/test/prisma"
        mock_subprocess.return_value = Mock(stdout="Migration completed", stderr="")

        # Mock Redis cache
        mock_redis = Mock()
        mock_redis.set_cache.return_value = True  # Lock acquired successfully
        mock_redis.get_cache.return_value = "pod_123_456"

        # Set DATABASE_URL
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
            result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=mock_redis)

        assert result is True
        # set_cache is called once in acquire_lock (__enter__ calls acquire_lock)
        assert mock_redis.set_cache.call_count == 1
        mock_subprocess.assert_called_once()

    @patch('litellm_proxy_extras.utils.subprocess.run')
    @patch('litellm_proxy_extras.utils.ProxyExtrasDBManager._get_prisma_dir')
    @patch('os.chdir')
    def test_setup_database_with_redis_lock_wait_and_skip(self, mock_chdir, mock_get_prisma_dir, mock_subprocess):
        """Test database setup when lock is held by another pod, then acquired after waiting"""
        # Setup mocks
        mock_get_prisma_dir.return_value = "/test/prisma"

        # Mock Redis cache - first call fails, second call succeeds
        mock_redis = Mock()
        mock_redis.set_cache.side_effect = [False, True]  # First fails, then succeeds
        mock_redis.get_cache.return_value = "pod_123_456"

        # Set DATABASE_URL
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
            result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=mock_redis)

        assert result is True  # Should return True after waiting and acquiring lock
        # set_cache is called 2 times: once in __enter__, once in wait_for_lock_release
        assert mock_redis.set_cache.call_count == 2
        # Proceed for case handling in case of migration failure
        mock_subprocess.assert_called_once()

    @patch('litellm_proxy_extras.utils.subprocess.run')
    @patch('litellm_proxy_extras.utils.ProxyExtrasDBManager._get_prisma_dir')
    @patch('os.chdir')
    def test_setup_database_without_redis(self, mock_chdir, mock_get_prisma_dir, mock_subprocess):
        """Test database setup without Redis cache"""
        # Setup mocks
        mock_get_prisma_dir.return_value = "/test/prisma"
        mock_subprocess.return_value = Mock(stdout="Migration completed", stderr="")

        # Set DATABASE_URL
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
            result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=None)

        assert result is True
        # Redis가 없을 때는 락 보호 없이 마이그레이션을 실행해야 함
        mock_subprocess.assert_called_once()

    def test_setup_database_no_database_url(self):
        """Test database setup without DATABASE_URL"""
        with patch.dict(os.environ, {}, clear=True):
            result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=None)

        assert result is False

    @patch('litellm_proxy_extras.utils.subprocess.run')
    @patch('litellm_proxy_extras.utils.ProxyExtrasDBManager._get_prisma_dir')
    @patch('os.chdir')
    @patch.object(MigrationLockManager, 'LOCK_TTL_SECONDS', 1)  # Set short TTL for testing
    def test_setup_database_lock_timeout(self, mock_chdir, mock_get_prisma_dir, mock_subprocess):
        """Test database setup when lock acquisition times out"""
        # Setup mocks
        mock_get_prisma_dir.return_value = "/test/prisma"

        # Mock Redis cache - always fails to acquire lock
        mock_redis = Mock()
        mock_redis.set_cache.return_value = False  # Always fails

        # Set DATABASE_URL
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://test:test@localhost/test'}):
            # Patch the wait_for_lock_release method to use shorter timeout
            with patch.object(MigrationLockManager, 'wait_for_lock_release') as mock_wait:
                mock_wait.return_value = False  # Simulate timeout

                result = ProxyExtrasDBManager.setup_database(use_migrate=True, redis_cache=mock_redis)

        assert result is False  # Should return False after timeout
        mock_subprocess.assert_not_called()  # Should not run migration
        # Verify that wait_for_lock_release was called with default parameters
        mock_wait.assert_called_once_with()

    def test_wait_for_lock_release_actual_timeout(self):
        """Test actual timeout behavior of wait_for_lock_release with real timing"""
        mock_redis = Mock()
        mock_redis.set_cache.return_value = False  # Always fails to acquire lock
        lock_manager = MigrationLockManager(mock_redis)

        # Test with very short timeout to verify actual timeout behavior
        start_time = time.time()
        result = lock_manager.wait_for_lock_release(check_interval=0.1, max_wait=0.5)
        end_time = time.time()

        assert result is False  # Should timeout
        assert end_time - start_time >= 0.5  # Should wait at least the max_wait time
        assert end_time - start_time < 1.0   # But not too much longer
        # Should have called set_cache multiple times during the wait
        assert mock_redis.set_cache.call_count > 1

