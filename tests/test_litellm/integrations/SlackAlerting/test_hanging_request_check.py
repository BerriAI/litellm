import json
import os
import sys
import time
import types
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
from litellm.types.integrations.slack_alerting import (
    HANGING_ALERT_BUFFER_TIME_SECONDS,
    HangingRequestData,
)


class TestAlertingHangingRequestCheck:
    """Test suite for AlertingHangingRequestCheck class"""

    @pytest.fixture
    def proxy_server_module_factory(self):
        """Factory for a dummy litellm.proxy.proxy_server module used by tests."""

        _UNSET = object()

        def _factory(
            internal_usage_cache_return_value: Optional[dict] = None,
            internal_usage_cache_side_effect: Optional[list] = None,
            internal_usage_cache=_UNSET,
        ):
            dummy_proxy_server = types.ModuleType("litellm.proxy.proxy_server")
            proxy_logging_obj = MagicMock()

            # Distinguish between:
            # - internal_usage_cache is None: litellm should return early
            # - internal_usage_cache exists but async_get_cache returns None: request is still hanging
            if internal_usage_cache is _UNSET:
                mock_internal_cache = AsyncMock()
                if internal_usage_cache_side_effect is not None:
                    mock_internal_cache.async_get_cache.side_effect = internal_usage_cache_side_effect
                else:
                    mock_internal_cache.async_get_cache.return_value = internal_usage_cache_return_value
                proxy_logging_obj.internal_usage_cache = mock_internal_cache
            else:
                proxy_logging_obj.internal_usage_cache = internal_usage_cache

            setattr(dummy_proxy_server, "proxy_logging_obj", proxy_logging_obj)
            return dummy_proxy_server

        return _factory

    @pytest.fixture
    def mock_slack_alerting(self):
        """Create a mock SlackAlerting object for testing"""
        mock_slack = MagicMock()
        mock_slack.alerting_threshold = 300  # 5 minutes
        mock_slack.send_alert = AsyncMock()
        return mock_slack

    @pytest.fixture
    def hanging_request_checker(self, mock_slack_alerting):
        """Create an AlertingHangingRequestCheck instance for testing"""
        return AlertingHangingRequestCheck(slack_alerting_object=mock_slack_alerting)

    @pytest.mark.asyncio
    async def test_init_creates_cache_with_correct_ttl(self, mock_slack_alerting):
        """
        Test that initialization creates a hanging request cache with correct TTL.
        The TTL should be alerting_threshold + buffer time.
        """
        checker = AlertingHangingRequestCheck(slack_alerting_object=mock_slack_alerting)

        # The cache should be created with TTL = alerting_threshold + buffer time
        # NOTE: checker runs every alerting_threshold/2, so we keep entries long enough
        # to guarantee at least one post-threshold check can occur.
        expected_ttl = (
            mock_slack_alerting.alerting_threshold
            + (mock_slack_alerting.alerting_threshold / 2)
            + HANGING_ALERT_BUFFER_TIME_SECONDS
        )
        assert checker.hanging_request_cache.default_ttl == expected_ttl

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_success(self, hanging_request_checker):
        """
        Test successfully adding a request to the hanging request cache.
        Should extract metadata and store HangingRequestData in cache.
        """
        request_data = {
            "litellm_call_id": "test_request_123",
            "model": "gpt-4",
            "deployment": {"litellm_params": {"api_base": "https://api.openai.com/v1"}},
            "metadata": {
                "user_api_key_alias": "test_key",
                "user_api_key_team_alias": "test_team",
            },
        }

        with patch("litellm.get_api_base", return_value="https://api.openai.com/v1"):
            await hanging_request_checker.add_request_to_hanging_request_check(request_data)

        # Verify the request was added to cache
        cached_data = await hanging_request_checker.hanging_request_cache.async_get_cache(key="test_request_123")

        assert cached_data is not None
        assert isinstance(cached_data, HangingRequestData)
        assert cached_data.request_id == "test_request_123"
        assert cached_data.model == "gpt-4"
        assert cached_data.api_base == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_none_request_data(self, hanging_request_checker):
        """
        Test that passing None request_data returns early without error.
        Should handle gracefully when no request data is provided.
        """
        result = await hanging_request_checker.add_request_to_hanging_request_check(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_minimal_data(self, hanging_request_checker):
        """
        Test adding request with minimal required data.
        Should handle cases where optional fields are missing.
        """
        request_data = {
            "litellm_call_id": "minimal_request_456",
            "model": "gpt-3.5-turbo",
        }

        await hanging_request_checker.add_request_to_hanging_request_check(request_data)

        cached_data = await hanging_request_checker.hanging_request_cache.async_get_cache(key="minimal_request_456")

        assert cached_data is not None
        assert cached_data.request_id == "minimal_request_456"
        assert cached_data.model == "gpt-3.5-turbo"
        assert cached_data.api_base is None
        assert cached_data.key_alias == ""
        assert cached_data.team_alias == ""

    @pytest.mark.asyncio
    async def test_send_hanging_request_alert(self, hanging_request_checker):
        """
        Test sending a hanging request alert.
        Should format the alert message correctly and call slack alerting.
        """
        hanging_request_data = HangingRequestData(
            request_id="test_hanging_request",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            key_alias="test_key",
            team_alias="test_team",
        )

        await hanging_request_checker.send_hanging_request_alert(hanging_request_data)

        # Verify slack alert was called
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

        # Check the alert message format
        call_args = hanging_request_checker.slack_alerting_object.send_alert.call_args
        message = call_args[1]["message"]

        assert "Requests are hanging - 300s+ request time" in message
        assert "Request Model: `gpt-4`" in message
        assert "API Base: `https://api.openai.com/v1`" in message
        assert "Key Alias: `test_key`" in message
        assert "Team Alias: `test_team`" in message
        assert call_args[1]["level"] == "Medium"

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_no_proxy_logging(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """
        Test send_alerts_for_hanging_requests when proxy_logging_obj.internal_usage_cache is None.
        Should return early without processing when internal usage cache is unavailable.
        """
        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache=None)

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            result = await hanging_request_checker.send_alerts_for_hanging_requests()
            assert result is None

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_completed_request(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """
        Test send_alerts_for_hanging_requests when request has completed (not hanging).
        Should remove completed requests from cache and not send alerts.
        """
        # Add a request to the hanging cache
        hanging_data = HangingRequestData(
            request_id="completed_request_789",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="completed_request_789", value=hanging_data, ttl=300
        )

        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_return_value={"status": "success"})

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            # Mock the cache method to return our test request
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
                return_value=["completed_request_789"]
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Verify no alert was sent since request completed
        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_actual_hanging_request(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """
        Test send_alerts_for_hanging_requests when request is actually hanging.
        Should send alert for requests that haven't completed within threshold.
        """
        # Add a hanging request to the cache
        hanging_data = HangingRequestData(
            request_id="hanging_request_999",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            key_alias="test_key",
            team_alias="test_team",
        )
        # Ensure this request is older than alerting_threshold to avoid false positives
        alerting_threshold = hanging_request_checker.slack_alerting_object.alerting_threshold
        safety_margin_seconds = 10
        hanging_data.start_time = time.time() - (alerting_threshold + safety_margin_seconds)
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_request_999", value=hanging_data, ttl=300
        )

        # Mock internal usage cache to return None (meaning request is still hanging)
        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_return_value=None)

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            # Mock the cache method to return our test request
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
                return_value=["hanging_request_999"]
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Verify alert was sent for hanging request
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

        # Verify the cache was updated to mark alert_sent=True
        cached_data = await hanging_request_checker.hanging_request_cache.async_get_cache(key="hanging_request_999")
        assert cached_data is not None
        assert cached_data.alert_sent is True

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_does_not_alert_before_threshold(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """Should not send hanging alert before alerting_threshold has elapsed."""
        hanging_data = HangingRequestData(
            request_id="not_yet_hanging_request",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        hanging_data.start_time = time.time()  # just started
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="not_yet_hanging_request", value=hanging_data, ttl=300
        )

        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_return_value=None)

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
                return_value=["not_yet_hanging_request"]
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_prevents_duplicate_alerts(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """Should not send duplicate alerts for the same request_id after alert_sent is True."""
        hanging_data = HangingRequestData(
            request_id="duplicate_alert_test",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        alerting_threshold = hanging_request_checker.slack_alerting_object.alerting_threshold
        safety_margin_seconds = 10
        hanging_data.start_time = time.time() - (alerting_threshold + safety_margin_seconds)
        hanging_data.alert_sent = True

        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="duplicate_alert_test", value=hanging_data, ttl=300
        )

        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_return_value=None)

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
                return_value=["duplicate_alert_test"]
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_rechecks_status_before_alert(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """Should avoid false alerts if the request completes between status checks."""
        request_id = "race_completion_request"
        hanging_data = HangingRequestData(
            request_id=request_id,
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        alerting_threshold = hanging_request_checker.slack_alerting_object.alerting_threshold
        hanging_data.start_time = time.time() - (alerting_threshold + 10)

        await hanging_request_checker.hanging_request_cache.async_set_cache(key=request_id, value=hanging_data, ttl=300)

        # First call -> still hanging (None), second call -> completed.
        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_side_effect=[None, {"status": "success"}])

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(return_value=[request_id])
            await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

        # Request should be removed from hanging cache after re-check detects completion.
        cached_data = await hanging_request_checker.hanging_request_cache.async_get_cache(key=request_id)
        assert cached_data is None

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_missing_hanging_data(
        self, hanging_request_checker, proxy_server_module_factory
    ):
        """
        Test send_alerts_for_hanging_requests when hanging request data is missing from cache.
        Should continue processing other requests when individual request data is missing.
        """
        dummy_proxy_server = proxy_server_module_factory(internal_usage_cache_return_value=None)

        with patch.dict(sys.modules, {"litellm.proxy.proxy_server": dummy_proxy_server}):
            # Mock cache to return request ID but no data (simulating expired or missing data)
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
                return_value=["missing_request_111"]
            )
            hanging_request_checker.hanging_request_cache.async_get_cache = AsyncMock(return_value=None)

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Should not crash and should not send any alerts
        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()
