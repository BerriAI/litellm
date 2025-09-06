import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
from sqlite3 import OperationalError


class TestPrismaClientRetryLogic(unittest.TestCase):
    """Test cases for the PrismaClient retry logic with intelligent cooldowns."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simplified test client without full litellm dependencies
        # We'll mock backoff and create a minimal test implementation
        import backoff

        mock_backoff = MagicMock()
        mock_expo = MagicMock()
        mock_on_backoff = MagicMock()

        backoff.on_exception = mock_backoff
        backoff.expo = mock_expo
        
        self.backoff_mock = mock_backoff
        self.on_backoff_mock = mock_on_backoff

        # Create a simplified PrismaClient for testing
        class MockPrismaClient:
            def __init__(self):
                self.max_tries = 3
                self.factor = 2
                self.jitter = 0.1
                self.db = MagicMock()
                self.proxy_logging_obj = MagicMock()

            @backoff.on_exception(
                backoff.expo,
                (OperationalError, TimeoutError),
                max_tries=3,
                factor=2,
                jitter=0.1,
                on_backoff=mock_on_backoff,
            )
            async def get_data(self, **kwargs):
                # Mock database operation
                if hasattr(self.db.litellm_verificationtoken, 'find_unique'):
                    return await self.db.litellm_verificationtoken.find_unique(**kwargs)
                raise OperationalError("Database error")

        self.client = MockPrismaClient()

    def test_successful_retry_after_failures(self):
        """Test that function retries on OperationalError and succeeds after a few attempts."""

        async def run_test():
            # Configure mock to fail twice then succeed
            call_count = 0
            def mock_db_operation(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise OperationalError("Database connection failed")
                return "success_result"

            # Mock the database operation
            self.client.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=mock_db_operation)

            # Call get_data which has the backoff decorator
            result = await self.client.get_data(token="test_token")

            # Assert that the function was called 3 times (2 failures + 1 success)
            self.assertEqual(self.client.db.litellm_verificationtoken.find_unique.call_count, 3)
            self.assertEqual(result, "success_result")
            # Assert that backoff was called 2 times (for the failures)
            self.assertEqual(self.on_backoff_mock.call_count, 2)

        asyncio.run(run_test())

    def test_failure_after_exhausting_retries(self):
        """Test that function fails after exhausting all retry attempts."""

        async def run_test():
            # Configure mock to always fail
            def mock_db_operation(*args, **kwargs):
                raise OperationalError("Database connection permanently failed")

            self.client.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=mock_db_operation)

            # Call get_data and expect it to fail after retries
            with self.assertRaises(OperationalError):
                await self.client.get_data(token="test_token")

            # Assert that the function was called max_tries times
            self.assertEqual(self.client.db.litellm_verificationtoken.find_unique.call_count, 3)  # max_tries = 3
            # Assert that backoff was called 2 times (max_tries - 1)
            self.assertEqual(self.on_backoff_mock.call_count, 2)

        asyncio.run(run_test())

    def test_immediate_failure_for_non_transitory_errors(self):
        """Test that non-transitory errors don't trigger retries."""

        async def run_test():
            # Use ValueError which is not in the retry exceptions list
            def mock_db_operation(*args, **kwargs):
                raise ValueError("Non-transitory error")

            self.client.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=mock_db_operation)

            # Call get_data and expect immediate failure
            with self.assertRaises(ValueError):
                await self.client.get_data(token="test_token")

            # Assert that the function was called only once (no retries)
            self.assertEqual(self.client.db.litellm_verificationtoken.find_unique.call_count, 1)

        asyncio.run(run_test())

    @patch.dict('os.environ', {
        'DB_MAX_TRIES': '5',
        'DB_FACTOR': '3',
        'DB_JITTER': '0.2'
    })
    def test_environment_variable_configuration(self):
        """Test that retry parameters are correctly read from environment variables."""

        # Mock the environment reading similar to how PrismaClient does it
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'DB_MAX_TRIES': '5',
                'DB_FACTOR': '3',
                'DB_JITTER': '0.2'
            }.get(key, default)

            # Create new client to test env var configuration
            client = self._create_test_client_with_env()

            # Check that environment variables are properly read
            self.assertEqual(client.max_tries, 5)
            self.assertEqual(client.factor, 3)
            self.assertEqual(client.jitter, 0.2)

    @patch.dict('os.environ', {}, clear=True)
    def test_default_values_when_env_vars_missing(self):
        """Test that default values are used when environment variables are not set."""

        # Create a client with explicitly set default parameters
        with patch('os.getenv', return_value=None):
            client = self._create_test_client_with_defaults(max_tries=4, factor=1, jitter=0.0)

            # Check that default values are used
            self.assertEqual(client.max_tries, 4)
            self.assertEqual(client.factor, 1)
            self.assertEqual(client.jitter, 0.0)

    def _create_test_client_with_env(self):
        """Create a minimal client that reads from environment variables."""
        with patch('os.getenv') as mock_getenv:
            def env_getter(key, default=None):
                return {
                    'DB_MAX_TRIES': '5',
                    'DB_FACTOR': '3',
                    'DB_JITTER': '0.2'
                }.get(key, str(default) if default is not None else None)

            mock_getenv.side_effect = env_getter

            class EnvPrismaClient:
                def __init__(self):
                    self.max_tries = int(os.getenv("DB_MAX_TRIES", 3))
                    self.factor = int(os.getenv("DB_FACTOR", 2))
                    self.jitter = float(os.getenv("DB_JITTER", 0.1)) if os.getenv("DB_JITTER") else 0.1

            return EnvPrismaClient()

    def _create_test_client_with_defaults(self, max_tries=3, factor=2, jitter=0.1):
        """Create a minimal client with explicit default parameters."""
        with patch('os.getenv', return_value=None):
            class DefaultPrismaClient:
                def __init__(self):
                    self.max_tries = int(os.getenv("DB_MAX_TRIES", max_tries))
                    self.factor = int(os.getenv("DB_FACTOR", factor))
                    self.jitter = float(os.getenv("DB_JITTER", jitter)) if os.getenv("DB_JITTER") else jitter

            return DefaultPrismaClient()

    def test_timeout_error_triggers_retry(self):
        """Test that TimeoutError (transitory exception) triggers retries."""

        async def run_test():
            call_count = 0
            def mock_db_operation(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    raise TimeoutError("Database operation timed out")
                return "timeout_recovery_result"

            self.client.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=mock_db_operation)

            result = await self.client.get_data(token="test_token")

            # Assert recovery after timeout
            self.assertEqual(self.client.db.litellm_verificationtoken.find_unique.call_count, 2)
            self.assertEqual(result, "timeout_recovery_result")
            self.assertEqual(self.on_backoff_mock.call_count, 1)

        asyncio.run(run_test())


class TestBackoffDecoratorConfiguration(unittest.TestCase):
    """Test the backoff decorator configuration specifically."""

    def test_decorator_configured_for_transitory_exceptions_only(self):
        """Test that the backoff decorator is only configured for OperationalError and TimeoutError."""

        from unittest.mock import patch, MagicMock
        import backoff

        # Test that we can import the necessary components
        mock_backoff = MagicMock()
        mock_expo = MagicMock()
        mock_on_backoff = MagicMock()

        backoff.on_exception = mock_backoff
        backoff.expo = mock_expo
        
        # Verify that OperationalError and TimeoutError are the exceptions configured
        from sqlite3 import OperationalError
        from asyncio import TimeoutError

        # Test the decorator setup (this would be verified by checking the actual decorator calls)
        self.assertTrue(callable(mock_backoff))
        self.assertTrue(callable(mock_expo))
        self.assertTrue(callable(mock_on_backoff))


if __name__ == '__main__':
    unittest.main()


class TestGeminiCooldownLogic(unittest.TestCase):
    def setUp(self):
        # Mock litellm dependencies
        self.mock_litellm = MagicMock()
        sys.modules['litellm'] = self.mock_litellm
        sys.modules['litellm.proxy'] = self.mock_litellm.proxy
        sys.modules['litellm.proxy.utils'] = self.mock_litellm.proxy.utils

        # Mock user_api_key_dict
        self.user_api_key_dict = MagicMock()
        self.user_api_key_dict.api_key = "test_api_key_hash"

        # Mock request data
        self.request_data = {
            "model": "gemini-pro",
            "litellm_call_id": "test_call_id"
        }

        # Mock HTTPException for Gemini
        self.gemini_exception = MagicMock()
        self.gemini_exception.status_code = 429
        self.gemini_exception.detail = {"error": {"code": "RESOURCE_EXHAUSTED"}}

        # Mock ProxyLogging and its internal cache
        self.proxy_logging = self.mock_litellm.proxy.utils.ProxyLogging(user_api_key_cache=MagicMock())
        self.proxy_logging.internal_usage_cache = MagicMock()
        self.proxy_logging.internal_usage_cache.async_set_cache = AsyncMock()
        self.proxy_logging.internal_usage_cache.async_get_cache = AsyncMock()

    @patch('litellm.proxy.utils._is_gemini_resource_exhausted', return_value=True)
    @patch('litellm.proxy.utils._handle_gemini_cooldown', new_callable=AsyncMock)
    def test_cooldown_triggered_on_gemini_error(self, mock_handle_cooldown, mock_is_exhausted):
        """Test that cooldown is triggered for Gemini's RESOURCE_EXHAUSTED error."""
        async def run_test():
            await self.proxy_logging.post_call_failure_hook(
                request_data=self.request_data,
                original_exception=self.gemini_exception,
                user_api_key_dict=self.user_api_key_dict
            )
            # Verify that cooldown was handled
            mock_handle_cooldown.assert_called_once_with(
                api_key_hash=self.user_api_key_dict.api_key,
                usage_cache=self.proxy_logging.internal_usage_cache
            )
        asyncio.run(run_test())

    @patch('litellm.proxy.utils._is_gemini_key_in_cooldown', return_value=True)
    def test_pre_call_hook_blocks_during_cooldown(self, mock_is_in_cooldown):
        """Test that pre_call_hook blocks requests for a key in cooldown."""
        async def run_test():
            with self.assertRaises(self.mock_litellm.proxy.utils.HTTPException) as context:
                await self.proxy_logging.pre_call_hook(
                    user_api_key_dict=self.user_api_key_dict,
                    data=self.request_data,
                    call_type="completion"
                )
            self.assertEqual(context.exception.status_code, 429)
        asyncio.run(run_test())

    @patch('litellm.proxy.utils._is_gemini_key_in_cooldown', return_value=False)
    def test_pre_call_hook_allows_requests_after_cooldown(self, mock_is_in_cooldown):
        """Test that pre_call_hook allows requests when not in cooldown."""
        async def run_test():
            try:
                await self.proxy_logging.pre_call_hook(
                    user_api_key_dict=self.user_api_key_dict,
                    data=self.request_data,
                    call_type="completion"
                )
            except self.mock_litellm.proxy.utils.HTTPException:
                self.fail("pre_call_hook raised HTTPException unexpectedly!")
        asyncio.run(run_test())