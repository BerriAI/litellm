import asyncio
import time
import unittest
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules
from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
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
        The TTL should be 1.5x alerting_threshold + buffer time, so entries
        survive long enough to be checked after crossing the threshold.
        """
        checker = AlertingHangingRequestCheck(slack_alerting_object=mock_slack_alerting)

        expected_ttl = int(
            mock_slack_alerting.alerting_threshold * 1.5 + 60
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
        # Add a hanging request that is older than the alerting threshold
        hanging_data = HangingRequestData(
            request_id="hanging_request_999",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            key_alias="test_key",
            team_alias="test_team",
            created_at=time.time() - 301,
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
    async def test_send_alerts_for_hanging_requests_alerts_once_per_hang(
        self, hanging_request_checker
    ):
        """
        A single hanging request must alert exactly once even though the
        checker tick revisits it on every run within the cache TTL.
        """
        hanging_data = HangingRequestData(
            request_id="hanging_once_555",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
            created_at=time.time() - 301,
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="hanging_once_555", value=hanging_data, ttl=300
        )

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = None
            mock_proxy.internal_usage_cache = mock_internal_cache

            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["hanging_once_555"])
            )

            for _ in range(3):
                await hanging_request_checker.send_alerts_for_hanging_requests()

        assert hanging_request_checker.slack_alerting_object.send_alert.call_count == 1
        cached = await hanging_request_checker.hanging_request_cache.async_get_cache(
            key="hanging_once_555"
        )
        assert cached is not None
        assert cached.alerted is True

    @pytest.mark.asyncio
    async def test_send_alerts_for_hanging_requests_skips_request_younger_than_threshold(
        self, hanging_request_checker
    ):
        """
        Test that an in-flight request younger than the alerting threshold
        does not trigger an alert and stays in the cache for later checks.
        """
        hanging_data = HangingRequestData(
            request_id="young_request_123",
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        await hanging_request_checker.hanging_request_cache.async_set_cache(
            key="young_request_123", value=hanging_data, ttl=300
        )

        with patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy:
            # Mock internal usage cache to return None (request still in flight)
            mock_internal_cache = AsyncMock()
            mock_internal_cache.async_get_cache.return_value = None
            mock_proxy.internal_usage_cache = mock_internal_cache

            hanging_request_checker.hanging_request_cache.async_get_oldest_n_keys = (
                AsyncMock(return_value=["young_request_123"])
            )

            await hanging_request_checker.send_alerts_for_hanging_requests()

        # No alert for a request below the threshold, and it must remain
        # cached so a later check can alert if it never completes
        hanging_request_checker.slack_alerting_object.send_alert.assert_not_called()
        assert (
            await hanging_request_checker.hanging_request_cache.async_get_cache(
                key="young_request_123"
            )
            is not None
        )

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


class HangingRequestLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.now = 1000.0
        self.clock_patch = patch("time.time", side_effect=lambda: self.now)
        self.clock_patch.start()
        self.logger = ProxyLogging(user_api_key_cache=UserApiKeyCache())
        self.logger.alerting = ["slack"]
        self.slack = self.logger.slack_alerting_instance
        self.slack.alerting = ["slack"]
        self.slack.default_webhook_url = "https://example.invalid/hanging-alerts"
        self.slack.batch_size = 1000
        self.checker = self.slack.hanging_request_check
        self.proxy_patch = patch.object(proxy_server, "proxy_logging_obj", self.logger)
        self.proxy_patch.start()

    async def asyncTearDown(self) -> None:
        if self.slack._periodic_flush_task is not None:
            self.slack._periodic_flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.slack._periodic_flush_task
        self.proxy_patch.stop()
        self.clock_patch.stop()

    async def add_request(self, request_id: str, *, alerted: bool = False) -> None:
        data: Final = HangingRequestData(request_id=request_id, model=request_id, created_at=self.now, alerted=alerted)
        await self.checker.hanging_request_cache.async_set_cache(
            key=request_id, value=data, ttl=self.checker.hanging_request_cache_ttl
        )

    async def test_success_removes_request_immediately(self) -> None:
        await self.add_request("completed")
        await self.logger.update_request_status("completed", "success")
        self.assertIsNone(await self.checker.hanging_request_cache.async_get_cache(key="completed"))

    async def test_failure_removes_request_immediately(self) -> None:
        await self.add_request("failed")
        await self.logger.update_request_status("failed", "fail")
        self.assertIsNone(await self.checker.hanging_request_cache.async_get_cache(key="failed"))

    async def test_finished_burst_does_not_alert_after_status_expires(self) -> None:
        for index in range(61):
            request_id: Final = f"finished-{index}"
            await self.add_request(request_id)
            await self.logger.update_request_status(request_id, "success")
        await self.add_request("pending")
        for elapsed in (150, 300, 450):
            self.now = 1000.0 + elapsed
            await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 1)
        self.assertIn("Request Model: `pending`", self.slack.log_queue[0]["payload"]["text"])

    async def test_more_than_twenty_hangs_are_checked_in_one_tick(self) -> None:
        for index in range(61):
            await self.add_request(f"pending-{index}")
        self.now = 1300.0
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 61)
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 61)

    async def test_alerted_entries_do_not_hide_new_hangs(self) -> None:
        for index in range(20):
            await self.add_request(f"already-alerted-{index}", alerted=True)
        await self.add_request("pending")
        self.now = 1300.0
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 1)
        self.assertIn("Request Model: `pending`", self.slack.log_queue[0]["payload"]["text"])

    async def test_young_request_remains_tracked_until_threshold(self) -> None:
        await self.add_request("pending")
        self.now = 1299.0
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 0)
        self.assertIsNotNone(await self.checker.hanging_request_cache.async_get_cache(key="pending"))
        self.now = 1300.0
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 1)

    async def test_missing_webhook_does_not_mark_request_alerted(self) -> None:
        await self.add_request("pending")
        self.now = 1300.0
        with patch.dict("os.environ", {}, clear=True):
            self.slack.default_webhook_url = None
            with self.assertRaises(ValueError):
                await self.checker.send_alerts_for_hanging_requests()
        self.slack.default_webhook_url = "https://example.invalid/hanging-alerts"
        await self.checker.send_alerts_for_hanging_requests()
        self.assertEqual(len(self.slack.log_queue), 1)

    async def stream(self) -> AsyncGenerator[str, None]:
        yield "data: first\n\n"
        yield "data: second\n\n"

    async def test_client_closing_stream_removes_request(self) -> None:
        await self.add_request("stream")
        stream: Final = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=self.stream(),
            user_api_key_dict=UserAPIKeyAuth(),
            request_data={"litellm_call_id": "stream", "model": "stream"},
            proxy_logging_obj=self.logger,
        )
        self.assertEqual(await anext(stream), "data: first\n\n")
        await stream.aclose()
        self.assertIsNone(await self.checker.hanging_request_cache.async_get_cache(key="stream"))

    async def test_cancelled_stream_removes_request_and_propagates_cancellation(self) -> None:
        await self.add_request("stream")
        stream: Final = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
            response=self.stream(),
            user_api_key_dict=UserAPIKeyAuth(),
            request_data={"litellm_call_id": "stream", "model": "stream"},
            proxy_logging_obj=self.logger,
        )
        self.assertEqual(await anext(stream), "data: first\n\n")
        with self.assertRaises(asyncio.CancelledError):
            await stream.athrow(asyncio.CancelledError())
        self.assertIsNone(await self.checker.hanging_request_cache.async_get_cache(key="stream"))
