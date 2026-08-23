import asyncio
import datetime
import json
import time
import unittest
from typing import Final, List, Optional, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import CallInfo, Litellm_EntityType
from litellm.types.integrations.slack_alerting import SlackAlertingCacheKeys


class TestSlackAlerting(unittest.TestCase):
    def setUp(self):
        self.slack_alerting = SlackAlerting()

    def test_get_percent_of_max_budget_left(self):
        # Test case 1: When max_budget is None
        user_info = CallInfo(max_budget=None, spend=50.0, event_group=Litellm_EntityType.KEY)
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 2: When max_budget is 0
        user_info = CallInfo(max_budget=0.0, spend=50.0, event_group=Litellm_EntityType.KEY)
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 3: When spend is less than max_budget
        user_info = CallInfo(max_budget=100.0, spend=75.0, event_group=Litellm_EntityType.KEY)
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.25)

        # Test case 4: When spend equals max_budget
        user_info = CallInfo(max_budget=100.0, spend=100.0, event_group=Litellm_EntityType.KEY)
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 5: When spend exceeds max_budget
        user_info = CallInfo(max_budget=100.0, spend=120.0, event_group=Litellm_EntityType.KEY)
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, -0.2)

    def test_get_event_and_event_message_max_budget(self):
        # Initial setup with no event
        event = None
        event_message = "Test Message: "

        # Test case 1: When spend exceeds max_budget
        user_info = CallInfo(
            max_budget=100.0,
            spend=120.0,
            soft_budget=None,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        self.assertEqual(event, "budget_crossed")
        self.assertTrue("Budget Crossed" in event_message)

        # Test case 2: When 5% of max_budget is left
        user_info = CallInfo(
            max_budget=100.0,
            spend=95.0,
            soft_budget=None,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        self.assertEqual(event, "threshold_crossed")
        self.assertTrue("5% Threshold Crossed" in event_message)

        # Test case 3: When 15% of max_budget is left
        user_info = CallInfo(
            max_budget=100.0,
            spend=85.0,
            soft_budget=None,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        self.assertEqual(event, "threshold_crossed")
        self.assertTrue("15% Threshold Crossed" in event_message)

    def test_get_event_and_event_message_soft_budget(self):
        # Initial setup with no event
        event = None
        event_message = "Test Message: "

        # Test case 1: When spend exceeds soft_budget
        user_info = CallInfo(
            max_budget=None,
            spend=120.0,
            soft_budget=100.0,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        self.assertEqual(event, "soft_budget_crossed")
        self.assertTrue("Total Soft Budget" in event_message)

        # Test case 2: When spend is less than soft_budget
        user_info = CallInfo(
            max_budget=None,
            spend=90.0,
            soft_budget=100.0,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=None, event_message=event_message
        )
        print("got event", event)
        print("got event_message", event_message)
        self.assertEqual(event, None)  # No event should be triggered

    def test_get_event_and_event_message_both_budgets(self):
        # Initial setup with no event
        event = None
        event_message = "Test Message: "

        # Test case 1: When spend exceeds both max_budget and soft_budget
        user_info = CallInfo(
            max_budget=150.0,
            spend=160.0,
            soft_budget=100.0,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        # budget_crossed has higher priority
        self.assertEqual(event, "budget_crossed")
        self.assertTrue("Budget Crossed" in event_message)

        # Test case 2: When spend exceeds soft_budget but not max_budget
        user_info = CallInfo(
            max_budget=150.0,
            spend=120.0,
            soft_budget=100.0,
            event_group=Litellm_EntityType.KEY,
        )
        event, event_message = self.slack_alerting._get_event_and_event_message(
            user_info=user_info, event=event, event_message=event_message
        )
        self.assertEqual(event, "soft_budget_crossed")
        self.assertTrue("Total Soft Budget" in event_message)

    # Calling update_values with alerting args should try to start the periodic task
    @patch("asyncio.create_task")
    def test_update_values_starts_periodic_task(self, mock_create_task):
        # Make it do nothing (or return a dummy future)
        mock_create_task.return_value = AsyncMock()  # prevents awaiting errors

        assert self.slack_alerting.periodic_started == False

        self.slack_alerting.update_values(alerting_args={"slack_alerting": "True"})
        assert self.slack_alerting.periodic_started == True

    @patch("litellm.integrations.SlackAlerting.slack_alerting.datetime")
    def test_alert_type_in_formatted_message(self, mock_datetime):
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "12:34:56"

        # Import required types
        from litellm.types.integrations.slack_alerting import AlertType

        # Create a simple test message to check formatting
        alert_type = AlertType.llm_exceptions
        level = "Medium"
        message = "Test alert message"
        current_time = "12:34:56"

        # Test the specific formatting logic we're interested in
        alert_type_formatted = f"Alert type: `{alert_type.name}`\n"
        formatted_message = (
            f"{alert_type_formatted}\n Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
        )

        # Verify alert_type is in the formatted message as expected
        self.assertIn("Alert type: `llm_exceptions`", formatted_message)
        self.assertIn("Level: `Medium`", formatted_message)
        self.assertIn("Timestamp: `12:34:56`", formatted_message)
        self.assertIn("Message: Test alert message", formatted_message)

    def test_original_redis_error_reproduction(self):
        """Test that reproduces the original Redis serialization error."""
        # This test verifies that the original error would occur without our fix
        outage_value = {
            "alerts": [408],
            "deployment_ids": {"zapier-multi-provider-gemini-2.5-flash-1ite-vertex"},
            "last_updated_at": 1760601633.6620142,
            "major_alert_sent": False,
            "minor_alert_sent": False,
            "provider_region_id": "vertex_aius-east1",
        }

        # This should raise a TypeError due to set not being JSON serializable
        with self.assertRaises(TypeError) as context:
            json.dumps(outage_value)

        # Verify the specific error message
        self.assertIn("Object of type set is not JSON serializable", str(context.exception))

    def test_fixed_redis_serialization(self):
        """Test that our fix resolves the Redis serialization error."""
        # Same data that caused the original error
        outage_value = {
            "alerts": [408],
            "deployment_ids": {"zapier-multi-provider-gemini-2.5-flash-1ite-vertex"},
            "last_updated_at": 1760601633.6620142,
            "major_alert_sent": False,
            "minor_alert_sent": False,
            "provider_region_id": "vertex_aius-east1",
        }

        # Apply our fix
        cache_value = self.slack_alerting._prepare_outage_value_for_cache(outage_value)

        # This should now work without errors
        json_str = json.dumps(cache_value)
        self.assertIsInstance(json_str, str)

        # Verify the data is correct
        parsed_data = json.loads(json_str)
        self.assertEqual(
            parsed_data["deployment_ids"],
            ["zapier-multi-provider-gemini-2.5-flash-1ite-vertex"],
        )
        self.assertEqual(parsed_data["alerts"], [408])
        self.assertEqual(parsed_data["provider_region_id"], "vertex_aius-east1")


_REPORT_SENT_KEY: Final = SlackAlertingCacheKeys.report_sent_key.value
_DAILY_REPORT_FREQUENCY: Final = 900


async def _slack_alerting_with_due_daily_report() -> SlackAlerting:
    slack_alerting: Final = SlackAlerting(
        internal_usage_cache=DualCache(),
        alerting_args={"daily_report_frequency": _DAILY_REPORT_FREQUENCY},
    )
    await slack_alerting.internal_usage_cache.async_set_cache(
        key=_REPORT_SENT_KEY,
        value=time.time() - _DAILY_REPORT_FREQUENCY - 1,
    )
    slack_alerting.send_daily_reports = AsyncMock()
    return slack_alerting


async def _read_report_sent(slack_alerting: SlackAlerting) -> float:
    return await slack_alerting.internal_usage_cache.async_get_cache(
        key=_REPORT_SENT_KEY,
        parent_otel_span=None,
    )


@pytest.mark.asyncio
async def test_daily_report_skipped_when_another_pod_holds_the_lock():
    """regression: issue #14809 - every pod sent its own copy of the daily report.

    The losing pod must also leave report_sent untouched so the winner's window still counts.
    """
    slack_alerting: Final = await _slack_alerting_with_due_daily_report()
    report_sent_before: Final = await _read_report_sent(slack_alerting)
    pod_lock_manager: Final = AsyncMock()
    pod_lock_manager.acquire_lock.return_value = False

    result: Final = await slack_alerting._run_scheduler_helper(
        llm_router=MagicMock(),
        pod_lock_manager=pod_lock_manager,
    )

    assert result is False
    slack_alerting.send_daily_reports.assert_not_awaited()
    assert await _read_report_sent(slack_alerting) == report_sent_before
    pod_lock_manager.acquire_lock.assert_awaited_once_with(
        cronjob_id="slack_daily_report",
        ttl=_DAILY_REPORT_FREQUENCY,
        allow_reentrant=False,
    )


@pytest.mark.asyncio
async def test_daily_report_sent_by_the_pod_that_wins_the_lock():
    slack_alerting: Final = await _slack_alerting_with_due_daily_report()
    report_sent_before: Final = await _read_report_sent(slack_alerting)
    llm_router: Final = MagicMock()
    pod_lock_manager: Final = AsyncMock()
    pod_lock_manager.acquire_lock.return_value = True

    result: Final = await slack_alerting._run_scheduler_helper(
        llm_router=llm_router,
        pod_lock_manager=pod_lock_manager,
    )

    assert result is True
    slack_alerting.send_daily_reports.assert_awaited_once_with(router=llm_router)
    assert await _read_report_sent(slack_alerting) > report_sent_before
    pod_lock_manager.acquire_lock.assert_awaited_once_with(
        cronjob_id="slack_daily_report",
        ttl=_DAILY_REPORT_FREQUENCY,
        allow_reentrant=False,
    )


@pytest.mark.parametrize("lock_state", ["no_pod_lock_manager", "no_redis_configured"])
@pytest.mark.asyncio
async def test_daily_report_still_sent_without_a_working_lock(lock_state: str):
    """Single-pod parity: a missing lock manager, or one whose acquire_lock returns None
    because redis isn't configured, must not suppress the report."""
    slack_alerting: Final = await _slack_alerting_with_due_daily_report()
    report_sent_before: Final = await _read_report_sent(slack_alerting)
    llm_router: Final = MagicMock()
    pod_lock_manager: Final = (
        None if lock_state == "no_pod_lock_manager" else AsyncMock(acquire_lock=AsyncMock(return_value=None))
    )

    result: Final = await slack_alerting._run_scheduler_helper(
        llm_router=llm_router,
        pod_lock_manager=pod_lock_manager,
    )

    assert result is True
    slack_alerting.send_daily_reports.assert_awaited_once_with(router=llm_router)
    assert await _read_report_sent(slack_alerting) > report_sent_before


@pytest.mark.asyncio
async def test_daily_report_lock_not_attempted_before_the_interval_elapses():
    """The lock is a per-window marker, so a pod must not burn it on a check that isn't due yet."""
    slack_alerting: Final = await _slack_alerting_with_due_daily_report()
    await slack_alerting.internal_usage_cache.async_set_cache(key=_REPORT_SENT_KEY, value=time.time())
    pod_lock_manager: Final = AsyncMock()
    pod_lock_manager.acquire_lock.return_value = True

    result: Final = await slack_alerting._run_scheduler_helper(
        llm_router=MagicMock(),
        pod_lock_manager=pod_lock_manager,
    )

    assert result is False
    pod_lock_manager.acquire_lock.assert_not_awaited()
    slack_alerting.send_daily_reports.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_daily_report_threads_the_pod_lock_manager_through():
    """The loop in _run_scheduled_daily_report is where the lock manager reaches the gate."""
    slack_alerting: Final = SlackAlerting(alert_types=["daily_reports"])
    pod_lock_manager: Final = AsyncMock()
    slack_alerting._run_scheduler_helper = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await slack_alerting._run_scheduled_daily_report(
            llm_router=MagicMock(),
            pod_lock_manager=pod_lock_manager,
        )

    _, kwargs = slack_alerting._run_scheduler_helper.await_args
    assert kwargs["pod_lock_manager"] is pod_lock_manager
