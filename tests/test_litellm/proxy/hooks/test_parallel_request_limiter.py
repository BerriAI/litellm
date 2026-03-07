"""
Unit Tests for the max parallel request limiter for the proxy
"""

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)


@pytest.fixture
def parallel_request_handler():
    """Create a parallel request handler with mocked internal_usage_cache."""
    mock_cache = MagicMock()
    return _PROXY_MaxParallelRequestsHandler(internal_usage_cache=mock_cache)


class TestRaiseRateLimitError:
    """Tests for the raise_rate_limit_error method."""

    def test_error_message_without_additional_details(self, parallel_request_handler):
        """
        Test that when additional_details is None, the error message does not contain 'None'.

        This is a regression test for a bug where the error message would include
        the literal string 'None' when additional_details was not provided.
        """
        with pytest.raises(HTTPException) as exc_info:
            parallel_request_handler.raise_rate_limit_error(additional_details=None)

        assert exc_info.value.status_code == 429
        assert "None" not in exc_info.value.detail
        assert exc_info.value.detail == "Max parallel request limit reached"

    def test_error_message_with_additional_details(self, parallel_request_handler):
        """
        Test that when additional_details is provided, it is included in the error message.
        """
        additional_info = "for api_key: sk-1234"

        with pytest.raises(HTTPException) as exc_info:
            parallel_request_handler.raise_rate_limit_error(additional_details=additional_info)

        assert exc_info.value.status_code == 429
        assert additional_info in exc_info.value.detail
        assert exc_info.value.detail == f"Max parallel request limit reached {additional_info}"

    def test_error_includes_retry_after_header(self, parallel_request_handler):
        """
        Test that the HTTPException includes a retry-after header.
        """
        with pytest.raises(HTTPException) as exc_info:
            parallel_request_handler.raise_rate_limit_error()

        assert "retry-after" in exc_info.value.headers
        # retry-after should be a string representing seconds
        assert exc_info.value.headers["retry-after"].isdigit() or \
               float(exc_info.value.headers["retry-after"]) >= 0
