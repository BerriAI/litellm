"""
Tests for enforce_model_rate_limits feature.

This feature allows users to enforce TPM/RPM limits set on model deployments
regardless of the routing strategy being used.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm import Router
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)


class TestModelRateLimitingCheck:
    """Test the ModelRateLimitingCheck class directly."""

    def test_get_deployment_limits_from_top_level(self):
        """Test extracting limits from top-level deployment config."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "tpm": 1000,
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 1000
        assert rpm == 10

    def test_get_deployment_limits_from_litellm_params(self):
        """Test extracting limits from litellm_params."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4", "tpm": 2000, "rpm": 20},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 2000
        assert rpm == 20

    def test_get_deployment_limits_from_model_info(self):
        """Test extracting limits from model_info."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id", "tpm": 3000, "rpm": 30},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm == 3000
        assert rpm == 30

    def test_get_deployment_limits_none_when_not_set(self):
        """Test that None is returned when limits are not set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        tpm, rpm = check._get_deployment_limits(deployment)
        assert tpm is None
        assert rpm is None

    def test_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set."""
        check = ModelRateLimitingCheck(dual_cache=MagicMock())

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = 10  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)
        assert "current usage=10" in str(exc_info.value)

    def test_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit."""
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = 5
        mock_cache.increment_cache.return_value = 6

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = check.pre_call_check(deployment)
        assert result == deployment

    def test_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded."""
        mock_cache = MagicMock()
        mock_cache.get_cache.return_value = 1000  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            check.pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)
        assert "current usage=1000" in str(exc_info.value)

    def test_log_success_event_increments_cache(self):
        """Test that log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        check.log_success_event(kwargs, None, None, None)

        # Verify increment_cache was called
        mock_cache.increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


class TestModelRateLimitingCheckAsync:
    """Test async methods of ModelRateLimitingCheck."""

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_when_no_limits(self):
        """Test that requests are allowed when no limits are set (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=None)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_rpm(self):
        """Test that RateLimitError is raised when RPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=10)  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "RPM limit=10" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_pre_call_check_allows_request_under_limit(self):
        """Test that requests are allowed when under the limit (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=5)
        mock_cache.async_increment_cache = AsyncMock(return_value=6)

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "rpm": 10,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        result = await check.async_pre_call_check(deployment)
        assert result == deployment

    @pytest.mark.asyncio
    async def test_async_pre_call_check_raises_rate_limit_error_when_over_tpm(self):
        """Test that RateLimitError is raised when TPM limit is exceeded (async)."""
        mock_cache = MagicMock()
        mock_cache.async_get_cache = AsyncMock(return_value=1000)  # Already at limit

        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        deployment = {
            "tpm": 1000,
            "litellm_params": {"model": "gpt-4"},
            "model_info": {"id": "test-id"},
            "model_name": "test-model",
        }

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await check.async_pre_call_check(deployment)

        assert "TPM limit=1000" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_log_success_event_increments_cache(self):
        """Test that async_log_success_event correctly increments the cache."""
        mock_cache = MagicMock()
        mock_cache.async_increment_cache = AsyncMock()
        check = ModelRateLimitingCheck(dual_cache=mock_cache)

        kwargs = {
            "standard_logging_object": {
                "model_id": "test-id",
                "total_tokens": 50,
                "hidden_params": {"litellm_model_name": "gpt-4"},
            }
        }

        await check.async_log_success_event(kwargs, None, None, None)

        # Verify async_increment_cache was called
        mock_cache.async_increment_cache.assert_called_once()
        _, kwarg_params = mock_cache.async_increment_cache.call_args
        assert "test-id:gpt-4:tpm:" in kwarg_params["key"]
        assert kwarg_params["value"] == 50


class TestRouterWithEnforceModelRateLimits:
    """Test Router integration with enforce_model_rate_limits."""

    def test_router_initializes_with_enforce_model_rate_limits(self):
        """Test that Router properly initializes the ModelRateLimitingCheck."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        router = Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Check that the callback was added
        assert router.optional_callbacks is not None
        assert len(router.optional_callbacks) == 1
        assert isinstance(router.optional_callbacks[0], ModelRateLimitingCheck)

    def test_router_optional_callbacks_contains_model_rate_limiting(self):
        """Test that ModelRateLimitingCheck is in the callbacks list."""
        model_list = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "test"},
                "rpm": 10,
            }
        ]

        Router(
            model_list=model_list,
            optional_pre_call_checks=["enforce_model_rate_limits"],
        )

        # Find the ModelRateLimitingCheck in litellm.callbacks
        found = False
        for callback in litellm.callbacks:
            if isinstance(callback, ModelRateLimitingCheck):
                found = True
                break

        assert found, "ModelRateLimitingCheck should be in litellm.callbacks"
