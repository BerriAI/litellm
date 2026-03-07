"""
Unit Tests for the max parallel request limiter for the proxy
"""

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)


class TestRaiseRateLimitError:
    """Tests for the raise_rate_limit_error method"""

    def test_raise_rate_limit_error_without_additional_details(self):
        """
        Test that error message does not contain 'None' when additional_details is not provided.

        This is a regression test for issue #18460 where the error message would
        show "Max parallel request limit reached None" instead of just
        "Max parallel request limit reached".
        """
        mock_cache = MagicMock(spec=DualCache)
        mock_internal_cache = MagicMock()
        mock_internal_cache.dual_cache = mock_cache

        handler = _PROXY_MaxParallelRequestsHandler(
            internal_usage_cache=mock_internal_cache
        )

        with pytest.raises(HTTPException) as exc_info:
            handler.raise_rate_limit_error()

        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "Max parallel request limit reached"
        assert "None" not in exc_info.value.detail
        assert "retry-after" in exc_info.value.headers

    def test_raise_rate_limit_error_with_additional_details(self):
        """
        Test that error message includes additional_details when provided.
        """
        mock_cache = MagicMock(spec=DualCache)
        mock_internal_cache = MagicMock()
        mock_internal_cache.dual_cache = mock_cache

        handler = _PROXY_MaxParallelRequestsHandler(
            internal_usage_cache=mock_internal_cache
        )

        additional_info = "Hit limit for api_key. Current limits: max_parallel_requests: 5"

        with pytest.raises(HTTPException) as exc_info:
            handler.raise_rate_limit_error(additional_details=additional_info)

        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == f"Max parallel request limit reached {additional_info}"
        assert "retry-after" in exc_info.value.headers
