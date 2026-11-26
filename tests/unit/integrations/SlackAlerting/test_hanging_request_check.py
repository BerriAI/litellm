import asyncio
import json
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Final, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adds the grandparent directory to sys.path to allow importing project modules

from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging
from litellm.types.integrations.slack_alerting import (
    MAX_OLDEST_HANGING_REQUESTS_TO_CHECK,
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


START: Final = 1000.0
DEADLOCK_GUARD_SECONDS: Final = 30.0


@dataclass(slots=True)
class ControlledClock:
    now: float

    def __call__(self) -> float:
        return self.now


@dataclass(frozen=True, slots=True)
class AlertingProxy:
    logger: ProxyLogging
    clock: ControlledClock

    @property
    def checker(self) -> AlertingHangingRequestCheck:
        return self.logger.slack_alerting_instance.hanging_request_check

    async def track(self, request_id: str, *, alerted: bool = False) -> None:
        await self.checker.hanging_request_cache.async_set_cache(
            key=request_id,
            value=HangingRequestData(
                request_id=request_id, model=request_id, created_at=self.clock.now, alerted=alerted
            ),
            ttl=self.checker.hanging_request_cache_ttl,
        )

    async def tick_at(self, elapsed: float) -> None:
        self.clock.now = START + elapsed
        await self.checker.send_alerts_for_hanging_requests()

    def alerted_models(self) -> tuple[str, ...]:
        return tuple(
            text.split("Request Model: `", 1)[1].split("`", 1)[0]
            for text in (item["payload"]["text"] for item in self.logger.slack_alerting_instance.log_queue)
        )


@pytest.fixture
def alerting_proxy() -> Iterator[AlertingProxy]:
    logger: Final = ProxyLogging(user_api_key_cache=UserApiKeyCache())
    logger.alerting = ["slack"]
    slack: Final = logger.slack_alerting_instance
    slack.alerting = ["slack"]
    slack.default_webhook_url = "https://example.invalid/hanging-alerts"
    slack.batch_size = 10_000
    slack.periodic_started = True
    clock: Final = ControlledClock(now=START)
    with patch("time.time", new=clock), patch("litellm.proxy.proxy_server.proxy_logging_obj", new=logger):
        yield AlertingProxy(logger=logger, clock=clock)


@pytest.mark.asyncio
async def test_completed_burst_never_alerts_after_its_status_markers_expire(alerting_proxy):
    for index in range(61):
        await alerting_proxy.track(f"finished-{index}")
        await alerting_proxy.logger.update_request_status(f"finished-{index}", "success")
    await alerting_proxy.track("pending")

    for elapsed in (150, 300, 450):
        await alerting_proxy.tick_at(elapsed)

    assert alerting_proxy.alerted_models() == ("pending",)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["success", "fail"])
async def test_status_written_before_registration_still_clears_the_request(alerting_proxy, status):
    await alerting_proxy.logger.update_request_status("late", status)
    await alerting_proxy.track("late")

    await alerting_proxy.tick_at(150)
    await alerting_proxy.tick_at(450)

    assert alerting_proxy.alerted_models() == ()
    assert await alerting_proxy.checker.hanging_request_cache.async_get_cache(key="late") is None


@pytest.mark.asyncio
async def test_every_real_hang_beyond_one_batch_alerts_exactly_once(alerting_proxy):
    hang_ids: Final = tuple(f"hang-{index}" for index in range(2 * MAX_OLDEST_HANGING_REQUESTS_TO_CHECK + 1))
    for hang_id in hang_ids:
        await alerting_proxy.track(hang_id)

    await alerting_proxy.tick_at(300)
    await alerting_proxy.tick_at(450)

    assert alerting_proxy.alerted_models() == hang_ids


@pytest.mark.asyncio
async def test_already_alerted_oldest_requests_do_not_hide_a_newer_hang(alerting_proxy):
    for index in range(2 * MAX_OLDEST_HANGING_REQUESTS_TO_CHECK):
        await alerting_proxy.track(f"already-alerted-{index}", alerted=True)
    alerting_proxy.clock.now = START + 1
    await alerting_proxy.track("newer")

    await alerting_proxy.tick_at(301)

    assert alerting_proxy.alerted_models() == ("newer",)


@pytest.mark.asyncio
async def test_a_hang_alerts_at_exactly_the_threshold_and_never_again(alerting_proxy):
    await alerting_proxy.track("pending")

    await alerting_proxy.tick_at(299.999)
    assert alerting_proxy.alerted_models() == ()
    assert await alerting_proxy.checker.hanging_request_cache.async_get_cache(key="pending") is not None

    await alerting_proxy.tick_at(300)
    await alerting_proxy.tick_at(450)
    assert alerting_proxy.alerted_models() == ("pending",)


@pytest.mark.asyncio
async def test_scan_lets_other_tasks_run_between_batches_and_sees_their_completions(alerting_proxy):
    hang_ids: Final = tuple(f"hang-{index}" for index in range(2 * MAX_OLDEST_HANGING_REQUESTS_TO_CHECK + 1))
    for hang_id in hang_ids:
        await alerting_proxy.track(hang_id)
    completed_id: Final = hang_ids[-1]
    ready: Final = asyncio.Event()
    permission: Final = asyncio.Event()

    async def complete_a_later_request_when_the_scan_yields() -> int:
        ready.set()
        await permission.wait()
        alerts_already_sent: Final = len(alerting_proxy.logger.slack_alerting_instance.log_queue)
        await alerting_proxy.logger.update_request_status(completed_id, "success")
        return alerts_already_sent

    alerting_proxy.clock.now = START + 300
    worker: Final = asyncio.create_task(complete_a_later_request_when_the_scan_yields())
    try:
        await asyncio.wait_for(ready.wait(), timeout=DEADLOCK_GUARD_SECONDS)
        permission.set()
        await alerting_proxy.checker.send_alerts_for_hanging_requests()
        alerts_sent_when_the_worker_ran: Final = await asyncio.wait_for(worker, timeout=DEADLOCK_GUARD_SECONDS)
    finally:
        if not worker.done():
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

    assert alerts_sent_when_the_worker_ran == MAX_OLDEST_HANGING_REQUESTS_TO_CHECK
    assert alerting_proxy.alerted_models() == hang_ids[:-1]
