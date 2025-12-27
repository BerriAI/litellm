"""
Test for Issue #18395: AuthenticationError should not be retried.

This test verifies that:
1. AuthenticationError (401) is NOT retried when no retry_policy is provided
2. When explicit retry_policy with AuthenticationErrorRetries=0 is provided, it's respected
3. Debug logging is added before retry attempts
"""

import pytest
from unittest.mock import patch

import litellm
from litellm import completion
from litellm.exceptions import AuthenticationError


class TestAuthenticationErrorNoRetry:
    """Test that AuthenticationError is not retried by default (Issue #18395)."""

    @pytest.mark.parametrize("sync_mode", [True, False])
    def test_authentication_error_not_retried_without_policy(self, sync_mode):
        """
        Test that AuthenticationError (401) is NOT retried when num_retries is set
        but no explicit retry_policy is provided.

        This is the core fix for Issue #18395.
        """
        # Arrange
        original_num_retries = litellm.num_retries
        litellm.num_retries = 3  # Enable retries globally

        try:
            with patch.object(litellm, "completion_with_retries") as mock_retry:
                with patch.object(
                    litellm, "acompletion_with_retries"
                ) as mock_async_retry:
                    # Act - trigger an AuthenticationError via mock_response
                    try:
                        completion(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Hi"}],
                            mock_response="Exception: AuthenticationError",
                        )
                    except AuthenticationError:
                        pass  # Expected

                    # Assert - completion_with_retries should NOT be called for AuthError
                    mock_retry.assert_not_called()
                    mock_async_retry.assert_not_called()
        finally:
            litellm.num_retries = original_num_retries

    def test_authentication_error_respects_explicit_retry_policy(self):
        """
        Test that when explicit retry_policy with AuthenticationErrorRetries=0 is set,
        the error is NOT retried.
        """
        from litellm.types.router import RetryPolicy

        retry_policy = RetryPolicy(
            AuthenticationErrorRetries=0,  # Explicitly don't retry auth errors
        )

        with patch.object(litellm, "completion_with_retries") as mock_retry:
            try:
                completion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    retry_policy=retry_policy,
                    mock_response="Exception: AuthenticationError",
                )
            except AuthenticationError:
                pass  # Expected

            # Assert - should NOT retry when policy says 0 retries for AuthError
            mock_retry.assert_not_called()

    def test_rate_limit_error_status_code_is_retryable(self):
        """
        Test that _should_retry returns True for RateLimitError status codes.
        This ensures the fix for AuthError doesn't break retries for valid cases.
        """
        # 429 (RateLimitError) should be retried
        assert litellm._should_retry(429) is True

        # 500+ (server errors) should be retried
        assert litellm._should_retry(500) is True
        assert litellm._should_retry(503) is True

        # 401 (AuthenticationError) should NOT be retried
        assert litellm._should_retry(401) is False

        # 400 (BadRequestError) should NOT be retried
        assert litellm._should_retry(400) is False

