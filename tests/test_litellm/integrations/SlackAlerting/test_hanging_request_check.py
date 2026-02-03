import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    def mock_slack_alerting(self):
        """Create a mock SlackAlerting object for testing"""
        mock_slack = MagicMock()
        mock_slack.alerting_threshold = 300  # 5 minutes
        mock_slack.send_alert = AsyncMock()

        # The hanging-request checker reads request status from this cache.
        mock_slack.internal_usage_cache = MagicMock()
        mock_slack.internal_usage_cache.async_get_cache = AsyncMock()
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
            "litellm_params": {
                "metadata": {
                    "user_api_key_alias": "test_key",
                    "user_api_key_team_alias": "test_team",
                    "user_api_key_org_id": "org_123",
                    "user_api_key_team_id": "team_123",
                    "model_info": {"id": "deployment_abc"},
                    "alerting_metadata": {"trace_id": "trace_1"},
                }
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
        assert cached_data.organization_id == "org_123"
        assert cached_data.team_id == "team_123"
        assert cached_data.deployment_id == "deployment_abc"
        assert cached_data.alerting_metadata == {"trace_id": "trace_1"}

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
        assert cached_data.organization_id is None
        assert cached_data.team_id is None
        assert cached_data.deployment_id is None
        assert cached_data.alerting_metadata is None

    @pytest.mark.asyncio
    async def test_add_request_to_hanging_request_check_uses_litellm_params_fallback_for_api_base(
        self, hanging_request_checker
    ):
        request_data = {
            "litellm_call_id": "fallback_api_base_1",
            "model": "gpt-4",
            "litellm_params": {"api_base": "https://example.com/v1"},
        }

        with patch(
            "litellm.get_api_base", return_value="https://example.com/v1"
        ) as mock_get_api_base:
            await hanging_request_checker.add_request_to_hanging_request_check(
                request_data
            )

        mock_get_api_base.assert_called_once()
        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="fallback_api_base_1"
            )
        )
        assert cached_data is not None
        assert cached_data.api_base == "https://example.com/v1"

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
            organization_id="org_123",
            team_id="team_123",
            deployment_id="deployment_abc",
            key_alias="test_key",
            team_alias="test_team",
        )

        await hanging_request_checker.send_hanging_request_alert(
            hanging_request_data,
            elapsed_seconds=301.23,
            threshold_seconds=hanging_request_checker.slack_alerting_object.alerting_threshold,
        )

        # Verify slack alert was called
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

        # Check the alert message format
        call_args = hanging_request_checker.slack_alerting_object.send_alert.call_args
        message = call_args[1]["message"]

        assert "Requests are hanging - 300s+ request time" in message
        assert "Request ID: `test_hanging_request`" in message
        assert "Request Model: `gpt-4`" in message
        assert "Elapsed: `301.23s`" in message
        assert "API Base: `https://api.openai.com/v1`" in message
        assert "Deployment ID: `deployment_abc`" in message
        assert "Organization ID: `org_123`" in message
        assert "Team ID: `team_123`" in message
        assert "Key Alias: `test_key`" in message
        assert "Team Alias: `test_team`" in message
        assert call_args[1]["level"] == "Medium"

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_no_proxy_logging(
        self, hanging_request_checker
    ):
        """
        Test send_alerts_for_hanging_requests when internal_usage_cache is None.
        Should return early without processing when internal usage cache is unavailable.
        """
        hanging_request_checker.slack_alerting_object.internal_usage_cache = None
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

        # Mock internal usage cache to return a request status (meaning request completed)
        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.return_value = (
            "success"
        )

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
        # Ensure this request is older than alerting_threshold to avoid false positives
        alerting_threshold = (
            hanging_request_checker.slack_alerting_object.alerting_threshold
        )
        safety_margin_seconds = 10
        hanging_data.start_time = time.time() - (
            alerting_threshold + safety_margin_seconds
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_request_999", value=hanging_data, ttl=300
        )

        # Mock internal usage cache to return None (meaning request is still hanging)
        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.return_value = (
            None
        )

        # Mock the cache method to return our test request
        hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
            AsyncMock(return_value=["hanging_request_999"])
        )

        await hanging_request_checker.send_alerts_for_hanging_requests()

        # Verify alert was sent for hanging request
        hanging_request_checker.slack_alerting_object.send_alert.assert_called_once()

        # Verify the cache was updated to mark alert_sent=True
        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="hanging_request_999"
            )
        )
        assert cached_data is not None
        assert cached_data.alert_sent is True

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_does_not_alert_before_threshold(
        self, hanging_request_checker
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

        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.return_value = (
            None
        )

        hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
            AsyncMock(return_value=["not_yet_hanging_request"])
        )

        await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_prevents_duplicate_alerts(
        self, hanging_request_checker
    ):
        """Should not send duplicate alerts for the same request_id after alert_sent is True."""
        hanging_data = HangingRequestData(
            request_id="duplicate_alert_test",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        alerting_threshold = (
            hanging_request_checker.slack_alerting_object.alerting_threshold
        )
        safety_margin_seconds = 10
        hanging_data.start_time = time.time() - (
            alerting_threshold + safety_margin_seconds
        )
        hanging_data.alert_sent = True

        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="duplicate_alert_test", value=hanging_data, ttl=300
        )

        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.return_value = (
            None
        )

        hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
            AsyncMock(return_value=["duplicate_alert_test"])
        )

        await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_rechecks_status_before_alert(
        self, hanging_request_checker
    ):
        """Should avoid false alerts if the request completes between status checks."""
        request_id = "race_completion_request"
        hanging_data = HangingRequestData(
            request_id=request_id,
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        alerting_threshold = (
            hanging_request_checker.slack_alerting_object.alerting_threshold
        )
        hanging_data.start_time = time.time() - (alerting_threshold + 10)

        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key=request_id, value=hanging_data, ttl=300
        )

        # First call -> still hanging (None), second call -> completed.
        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.side_effect = [
            None,
            "success",
        ]

        hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
            AsyncMock(return_value=[request_id])
        )
        await hanging_request_checker.send_alerts_for_hanging_requests()

        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()

        # Request should be removed from hanging cache after re-check detects completion.
        cached_data = (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key=request_id
            )
        )
        assert cached_data is None

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_with_missing_hanging_data(
        self, hanging_request_checker
    ):
        """
        Test send_alerts_for_hanging_requests when hanging request data is missing from cache.
        Should continue processing other requests when individual request data is missing.
        """
        hanging_request_checker.slack_alerting_object.internal_usage_cache.async_get_cache.return_value = (
            None
        )

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
