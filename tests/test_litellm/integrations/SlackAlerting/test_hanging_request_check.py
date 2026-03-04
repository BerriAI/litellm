import json
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
from litellm.types.integrations.slack_alerting import HangingRequestData


class TestAlertingHangingRequestCheck:
    """Test suite for AlertingHangingRequestCheck class"""

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
        expected_ttl = (
            mock_slack_alerting.alerting_threshold + 60
        )  # HANGING_ALERT_BUFFER_TIME_SECONDS
        assert checker.hanging_request_cache.default_ttl == expected_ttl

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_success(
        self, hanging_request_checker
    ):
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
            await hanging_request_checker.add_request_to_hanging_request_check(
                request_data
            )

        # Verify the request was added to cache
        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="test_request_123"
            )
        )

        assert cached_data is not None
        assert isinstance(cached_data, HangingRequestData)
        assert cached_data.request_id == "test_request_123"
        assert cached_data.model == "gpt-4"
        assert cached_data.api_base == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_none_request_data(
        self, hanging_request_checker
    ):
        """
        Test that passing None request_data returns early without error.
        Should handle gracefully when no request data is provided.
        """
        result = await hanging_request_checker.add_request_to_hanging_request_check(
            None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_minimal_data(
        self, hanging_request_checker
    ):
        """
        Test adding request with minimal required data.
        Should handle cases where optional fields are missing.
        """
        request_data = {
            "litellm_call_id": "minimal_request_456",
            "model": "gpt-3.5-turbo",
        }

        await hanging_request_checker.add_request_to_hanging_request_check(request_data)

        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="minimal_request_456"
            )
        )

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
        self, hanging_request_checker
    ):
        """
        Test send_alerts_for_hanging_requests when proxy_logging_obj.internal_usage_cache is None.
        Should return early without processing when internal usage cache is unavailable.
        """
        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            mock_proxy.internal_usage_cache = None

            result = await hanging_request_checker.send_alerts_for_hanging_requests()
            assert result is None

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_completed_request(
        self, hanging_request_checker
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

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            # Mock internal usage cache to return a request status (meaning request completed)
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = {"status": "success"}
            mock_proxy.internal_usage_cache = mock_internal_cache

            # Mock the cache method to return our test request
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["completed_request_789"])
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Verify no alert was sent since request completed
        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_actual_hanging_request(
        self, hanging_request_checker
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
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_request_999", value=hanging_data, ttl=300
        )

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            # Mock internal usage cache to return None (meaning request is still hanging)
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = None
            mock_proxy.internal_usage_cache = mock_internal_cache

            # Mock the cache method to return our test request
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["hanging_request_999"])
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Verify alert was sent for hanging request
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_missing_hanging_data(
        self, hanging_request_checker
    ):
        """
        Test send_alerts_for_hanging_requests when hanging request data is missing from cache.
        Should continue processing other requests when individual request data is missing.
        """
        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            mock_internal_cache = AsyncMock()
            mock_proxy.internal_usage_cache = mock_internal_cache

            # Mock cache to return request ID but no data (simulating expired or missing data)
            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["missing_request_111"])
            )
            hanging_request_checker.hanging_request_cache.async_get_cache = AsyncMock(
                return_value=None
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Should not crash and should not send any alerts
        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_hanging_request_removed_from_cache_after_alert(
        self, hanging_request_checker
    ):
        """
        Test that a hanging request is removed from cache after sending an alert.
        This prevents duplicate alerts on subsequent loop iterations and is
        essential for digest mode to work correctly (issue #22753).
        """
        # Add a hanging request to the cache
        hanging_data = HangingRequestData(
            request_id="hanging_dedup_001",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            key_alias="test_key",
            team_alias="test_team",
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_dedup_001", value=hanging_data, ttl=300
        )

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = None  # request still hanging
            mock_proxy.internal_usage_cache = mock_internal_cache

            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["hanging_dedup_001"])
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # Alert should have been sent once
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

        # Request should be removed from cache after alerting
        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="hanging_dedup_001"
            )
        )
        assert cached_data is None, (
            "Hanging request should be removed from cache after alert is sent "
            "to prevent duplicate alerts on subsequent loop iterations"
        )

    @pytest.mark.asyncio
    async def test_no_duplicate_alerts_across_loop_iterations(
        self, hanging_request_checker
    ):
        """
        Test that running send_alerts_for_hanging_requests twice does not
        produce duplicate alerts for the same request (issue #22753).
        """
        hanging_data = HangingRequestData(
            request_id="hanging_nodup_002",
            model="gemini-2.5-flash",
            api_base="https://generativelanguage.googleapis.com",
            key_alias="prod_key",
            team_alias="ml_team",
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_nodup_002", value=hanging_data, ttl=300
        )

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = None
            mock_proxy.internal_usage_cache = mock_internal_cache

            # First iteration: should find and alert
            await hanging_request_checker.send_alerts_for_hanging_requests()

            # Second iteration: cache should be empty, no new alerts
            await hanging_request_checker.send_alerts_for_hanging_requests()

        # send_alert should have been called exactly once, not twice
        assert hanging_request_checker.slack_alerting_object.send_alert.call_count == 1, (
            "send_alert should only be called once per hanging request, "
            "not on every loop iteration"
        )
