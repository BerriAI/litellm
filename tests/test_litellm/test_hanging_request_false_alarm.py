"""
Tests for fix of issue #12355: False "Requests are hanging" Slack alerts.

Root cause: send_alerts_for_hanging_requests() had no elapsed-time check.
If a request hadn't completed yet (request_status cache miss), it was
reported as hanging even when it was still within the expected time window.
Additionally, alerted requests were never removed from the cache, so the
same request triggered duplicate alerts on every check cycle.

Fix:
1. Record `start_time` (monotonic) in HangingRequestData when a request
   is added.
2. Only alert when `elapsed >= alerting_threshold`.
3. Remove the entry from the hanging_request_cache after alerting to
   prevent duplicate alerts.
"""

import asyncio
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# proxy_server requires apscheduler; mock the module for the entire test file.
# The patch is applied once and cleaned up when the test process tears down.
_mock_proxy_module = MagicMock()
_patcher = patch.dict(sys.modules, {"litellm.proxy.proxy_server": _mock_proxy_module})
_patcher.start()

from litellm.integrations.SlackAlerting.hanging_request_check import (  # noqa: E402
    AlertingHangingRequestCheck,
)
from litellm.types.integrations.slack_alerting import HangingRequestData  # noqa: E402


def _make_checker(threshold: float = 10.0) -> AlertingHangingRequestCheck:
    """Create an AlertingHangingRequestCheck with a mocked SlackAlerting."""
    slack = MagicMock()
    slack.alerting_threshold = threshold
    slack.send_alert = AsyncMock()
    checker = AlertingHangingRequestCheck(slack_alerting_object=slack)
    return checker


def _setup_proxy_mock(request_status_map=None):
    """Configure the mock proxy_logging_obj for the test."""
    if request_status_map is None:
        request_status_map = {}

    async def _async_get_cache(key, litellm_parent_otel_span=None, local_only=True):
        for pattern, value in request_status_map.items():
            if pattern in key:
                return value
        return None

    mock_cache = MagicMock()
    mock_cache.async_get_cache = _async_get_cache
    _mock_proxy_module.proxy_logging_obj = MagicMock()
    _mock_proxy_module.proxy_logging_obj.internal_usage_cache = mock_cache


class TestHangingRequestStartTime(unittest.TestCase):
    """HangingRequestData records start_time."""

    def test_start_time_stored_on_add(self):
        checker = _make_checker(threshold=300.0)
        before = time.monotonic()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                checker.add_request_to_hanging_request_check(
                    request_data={
                        "litellm_call_id": "req-1",
                        "model": "gpt-4",
                        "metadata": {},
                    }
                )
            )
            cached = loop.run_until_complete(
                checker.hanging_request_cache.async_get_cache(key="req-1")
            )
        finally:
            loop.close()
        after = time.monotonic()
        self.assertIsInstance(cached, HangingRequestData)
        self.assertGreaterEqual(cached.start_time, before)
        self.assertLessEqual(cached.start_time, after)


class TestNoFalseAlarm(unittest.TestCase):
    """Requests within the alerting threshold must NOT trigger alerts."""

    def test_young_request_not_alerted(self):
        """A request that just started should not be flagged as hanging."""
        checker = _make_checker(threshold=300.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="young-1",
                model="gpt-4",
                start_time=time.monotonic(),
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="young-1", value=data, ttl=360
                )
            )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_not_called()


class TestAlertAfterThreshold(unittest.TestCase):
    """Requests exceeding the alerting threshold SHOULD trigger one alert."""

    def test_old_request_alerted(self):
        """A request older than alerting_threshold should be flagged."""
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="old-1",
                model="deepseek-r1",
                start_time=time.monotonic() - 15,
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="old-1", value=data, ttl=70
                )
            )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_called_once()


class TestNoDuplicateAlerts(unittest.TestCase):
    """After sending an alert, the entry must be removed to prevent dups."""

    def test_entry_removed_after_alert(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="dup-1",
                model="gpt-4",
                start_time=time.monotonic() - 20,
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="dup-1", value=data, ttl=70
                )
            )
            # first check → alert fires
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
            self.assertEqual(checker.slack_alerting_object.send_alert.call_count, 1)

            # second check → entry removed, no second alert
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
            self.assertEqual(checker.slack_alerting_object.send_alert.call_count, 1)
        finally:
            loop.close()


class TestCompletedRequestCleared(unittest.TestCase):
    """Completed requests must be cleared without alerting."""

    def test_completed_request_not_alerted(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock(request_status_map={"done-1": "success"})
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="done-1",
                model="gpt-4",
                start_time=time.monotonic() - 20,
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="done-1", value=data, ttl=70
                )
            )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_not_called()


class TestMixedRequests(unittest.TestCase):
    """Mix of young, old, and completed requests."""

    def test_only_old_incomplete_alerted(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock(request_status_map={"mix-done": "success"})
        loop = asyncio.new_event_loop()
        try:
            now = time.monotonic()
            entries = [
                HangingRequestData(
                    request_id="mix-old", model="deepseek-r1", start_time=now - 20
                ),
                HangingRequestData(
                    request_id="mix-young", model="gpt-4", start_time=now - 2
                ),
                HangingRequestData(
                    request_id="mix-done", model="claude", start_time=now - 30
                ),
            ]
            for entry in entries:
                loop.run_until_complete(
                    checker.hanging_request_cache.async_set_cache(
                        key=entry.request_id, value=entry, ttl=70
                    )
                )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        self.assertEqual(checker.slack_alerting_object.send_alert.call_count, 1)
        call_kwargs = checker.slack_alerting_object.send_alert.call_args
        self.assertIn("deepseek-r1", str(call_kwargs))


class TestBoundaryElapsedTime(unittest.TestCase):
    """Request exactly at the threshold boundary."""

    def test_exactly_at_threshold_alerts(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="boundary-1",
                model="gpt-4",
                start_time=time.monotonic() - 10.0,
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="boundary-1", value=data, ttl=70
                )
            )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_called_once()

    def test_just_below_threshold_no_alert(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            data = HangingRequestData(
                request_id="below-1",
                model="gpt-4",
                start_time=time.monotonic() - 9.5,
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="below-1", value=data, ttl=70
                )
            )
            loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_not_called()


class TestNoneRequestData(unittest.TestCase):
    """None request_data should be a no-op."""

    def test_none_request_data(self):
        checker = _make_checker()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                checker.add_request_to_hanging_request_check(request_data=None)
            )
        finally:
            loop.close()
        self.assertIsNone(result)


class TestLegacyDataBackcompat(unittest.TestCase):
    """Entries created before fix (start_time defaults to 0.0 in a long-running
    process) should still be detected as hanging."""

    def test_legacy_entry_treated_as_hanging(self):
        checker = _make_checker(threshold=10.0)
        _setup_proxy_mock()
        loop = asyncio.new_event_loop()
        try:
            # Legacy entries have no start_time; default 0.0 makes them
            # appear extremely old, so they should always be flagged.
            data = HangingRequestData(
                request_id="legacy-1",
                model="gpt-4",
            )
            loop.run_until_complete(
                checker.hanging_request_cache.async_set_cache(
                    key="legacy-1", value=data, ttl=70
                )
            )
            # Simulate a long-running process where monotonic() >> 0.0
            with patch(
                "litellm.integrations.SlackAlerting.hanging_request_check.time"
            ) as mock_time:
                mock_time.monotonic.return_value = 1000.0
                loop.run_until_complete(checker.send_alerts_for_hanging_requests())
        finally:
            loop.close()

        checker.slack_alerting_object.send_alert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
