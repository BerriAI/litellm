import asyncio
from datetime import datetime
import io
import json
import os
import time
import unittest
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIError
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._internal_context import current_service_target
from litellm.caching.caching import DualCache
from litellm.integrations.SlackAlerting.budget_alert_types import get_budget_alert_type
from litellm.integrations.SlackAlerting.slack_alerting import DeploymentMetrics, SlackAlerting
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CallInfo, Litellm_EntityType
from litellm.router import Router
from litellm.types.integrations.slack_alerting import AlertQueueItem, AlertType, SlackAlertingCacheKeys
from litellm.utils import get_api_base
from datetime import timedelta
from typing import Optional
from litellm.proxy._types import WebhookEvent


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

    def test_get_user_info_str_omits_absent_token_for_user_alert(self):
        user_info = CallInfo(
            spend=85.0,
            max_budget=100.0,
            user_id="user-1",
            user_email="person@example.com",
            event_group=Litellm_EntityType.USER,
        )

        result = self.slack_alerting._get_user_info_str(user_info)

        self.assertIn("*user_id:* `user-1`", result)
        self.assertIn("*user_email:* `person@example.com`", result)
        self.assertNotIn("*token:*", result)

    def test_get_event_and_event_message_max_budget(self):
        event = None
        event_message = get_budget_alert_type("user_budget").get_event_message()

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

        event_message = get_budget_alert_type("user_budget").get_event_message()
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
        self.assertEqual(event_message, "User Budget: 5% or less of budget remaining")

        event_message = get_budget_alert_type("user_budget").get_event_message()
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
        self.assertEqual(event_message, "User Budget: 15% or less of budget remaining")

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


def _slack_alerting_with_env_resolution() -> SlackAlerting:
    slack_alerting: Final = SlackAlerting(alerting=["slack"], internal_usage_cache=DualCache())
    slack_alerting.periodic_started = True
    return slack_alerting


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_group, expected_prefix",
    [
        (Litellm_EntityType.TEAM_MEMBER, "Team Member Budget: Budget Crossed"),
        (Litellm_EntityType.KEY, "Key Budget: Budget Crossed"),
    ],
)
async def test_max_budget_alert_labels_team_member_budget(event_group, expected_prefix):
    slack_alerting: Final = _slack_alerting_with_env_resolution()
    slack_alerting.send_alert = AsyncMock()

    await slack_alerting.budget_alerts(
        type="max_budget_alert",
        user_info=CallInfo(
            spend=10.5,
            max_budget=10.0,
            token="hashed_key",
            user_id="member_1",
            team_id="team_a",
            event_group=event_group,
        ),
    )

    assert slack_alerting.send_alert.await_args.kwargs["message"].startswith(expected_prefix)


@pytest.mark.asyncio
async def test_send_alert_falls_back_to_alerting_webhook_url_env(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERTING_WEBHOOK_URL", "https://chat.example.com/hooks/abc")
    slack_alerting: Final = _slack_alerting_with_env_resolution()

    await slack_alerting.send_alert(
        message="budget crossed",
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
    )

    assert slack_alerting.log_queue[0]["url"] == "https://chat.example.com/hooks/abc"


@pytest.mark.asyncio
async def test_send_alert_prefers_slack_webhook_url_over_fallback(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T0/B0/X0")
    monkeypatch.setenv("ALERTING_WEBHOOK_URL", "https://chat.example.com/hooks/abc")
    slack_alerting: Final = _slack_alerting_with_env_resolution()

    await slack_alerting.send_alert(
        message="budget crossed",
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
    )

    assert slack_alerting.log_queue[0]["url"] == "https://hooks.slack.com/services/T0/B0/X0"


@pytest.mark.asyncio
async def test_send_alert_raises_when_no_webhook_url_configured(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERTING_WEBHOOK_URL", raising=False)
    slack_alerting: Final = _slack_alerting_with_env_resolution()

    with pytest.raises(ValueError, match="SLACK_WEBHOOK_URL / ALERTING_WEBHOOK_URL"):
        await slack_alerting.send_alert(
            message="budget crossed",
            level="High",
            alert_type=AlertType.budget_alerts,
            alerting_metadata={},
        )


SLACK_WEBHOOK_URL: Final = "https://hooks.slack.com/services/test"
THRESHOLD_ALERT: Final = "User Budget: 15% or less of budget remaining\n\n*user_id:* `user-a`"
CROSSED_ALERT: Final = "User Budget: Budget Crossed\n\n*user_id:* `user-b`"


class _SlackWebhookBody(TypedDict):
    text: ReadOnly[str]


_SLACK_WEBHOOK_BODY: Final = TypeAdapter(_SlackWebhookBody)


def _webhook_accepting_posts() -> AsyncMock:
    response: Final = MagicMock(spec=httpx.Response)
    response.status_code = 200
    http_handler: Final = AsyncMock(spec=AsyncHTTPHandler)
    http_handler.post.return_value = response
    return http_handler


def _slack_alerting_flushing_to(http_handler: AsyncHTTPHandler) -> SlackAlerting:
    slack_alerting: Final = SlackAlerting(alerting=["slack"], async_http_handler=http_handler)
    slack_alerting.periodic_started = True
    return slack_alerting


def _queued_slack_alert(text: str) -> AlertQueueItem:
    return {
        "url": SLACK_WEBHOOK_URL,
        "headers": {"Content-type": "application/json"},
        "payload": {"text": text},
        "alert_type": AlertType.budget_alerts,
    }


def _posted_slack_bodies(http_handler: AsyncMock) -> tuple[_SlackWebhookBody, ...]:
    return tuple(_SLACK_WEBHOOK_BODY.validate_json(call.kwargs["data"]) for call in http_handler.post.call_args_list)


async def _send_budget_alert(slack_alerting: SlackAlerting, message: str) -> None:
    await slack_alerting.send_alert(
        message=message,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
    )


@pytest.mark.asyncio
async def test_async_send_batch_delivers_every_distinct_alert_queued_in_one_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_URL", SLACK_WEBHOOK_URL)
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = _slack_alerting_flushing_to(http_handler)
    await _send_budget_alert(slack_alerting, THRESHOLD_ALERT)
    await _send_budget_alert(slack_alerting, CROSSED_ALERT)

    await slack_alerting.async_send_batch()

    posted_texts: Final = tuple(body["text"] for body in _posted_slack_bodies(http_handler))
    assert len(posted_texts) == 2
    assert THRESHOLD_ALERT in posted_texts[0]
    assert CROSSED_ALERT in posted_texts[1]
    assert not any(text.startswith("[Num Alerts") for text in posted_texts)
    assert slack_alerting.log_queue == []


@pytest.mark.asyncio
async def test_async_send_batch_collapses_only_identical_alerts() -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = _slack_alerting_flushing_to(http_handler)
    slack_alerting.log_queue.extend(
        (
            _queued_slack_alert(THRESHOLD_ALERT),
            _queued_slack_alert(CROSSED_ALERT),
            _queued_slack_alert(THRESHOLD_ALERT),
        )
    )

    await slack_alerting.async_send_batch()

    assert _posted_slack_bodies(http_handler) == (
        {"text": f"[Num Alerts: 2]\n\n{THRESHOLD_ALERT}"},
        {"text": CROSSED_ALERT},
    )


def _periodic_flush_tasks() -> list[asyncio.Task[object]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t.get_coro() is not None and t.get_coro().__qualname__ == "SlackAlerting.periodic_flush"
    ]


@pytest.mark.asyncio
async def test_update_values_repeated_alerting_reload_keeps_single_periodic_flush_task() -> None:
    slack_alerting: Final = SlackAlerting(alerting=["slack"])
    try:
        for _ in range(5):
            slack_alerting.update_values(alerting=["slack"])
        await asyncio.sleep(0)
        flush_tasks: Final = _periodic_flush_tasks()
        assert len(flush_tasks) == 1, f"expected 1 periodic_flush task, found {len(flush_tasks)}"
    finally:
        for t in _periodic_flush_tasks():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_daily_report_schedule_cache_calls_declare_their_key_family():
    """The report_sent read and write run inside ``service_target("daily_report_schedule")``
    so the background spans read ``redis.get daily_report_schedule`` rather than a bare
    ``redis.get`` with no owner."""
    slack_alerting: Final = await _slack_alerting_with_due_daily_report()
    cache: Final = slack_alerting.internal_usage_cache
    seen: list[tuple[str, str | None]] = []
    real_get, real_set = cache.async_get_cache, cache.async_set_cache

    async def _get(*args, **kwargs):
        seen.append(("get", current_service_target()))
        return await real_get(*args, **kwargs)

    async def _set(*args, **kwargs):
        seen.append(("set", current_service_target()))
        return await real_set(*args, **kwargs)

    with (
        patch.object(cache, "async_get_cache", side_effect=_get),
        patch.object(cache, "async_set_cache", side_effect=_set),
    ):
        result: Final = await slack_alerting._run_scheduler_helper(llm_router=MagicMock(), pod_lock_manager=None)

    assert result is True
    assert seen == [("get", "daily_report_schedule"), ("set", "daily_report_schedule")]
    assert current_service_target() is None


@pytest.fixture
def slack_alerting() -> SlackAlerting:
    return SlackAlerting(
        alerting_threshold=1, internal_usage_cache=DualCache(), alerting=["slack"]
    )

key_info: Final = CallInfo(
    token="test_token",
    spend=81,
    soft_budget=80,
    max_budget=100,
    user_id="test@test.com",
    user_email="test@test.com",
    key_alias="test-key",
    event_group=Litellm_EntityType.KEY,
)

team_info: Final = CallInfo(
    token="test_token",
    spend=160,
    soft_budget=150,
    max_budget=200,
    team_id="team-123",
    team_alias="engineering-team",
    event_group=Litellm_EntityType.TEAM,
)

user_info: Final = CallInfo(
    token="test_token",
    spend=45,
    soft_budget=40,
    max_budget=50,
    user_id="user123",
    event_group=Litellm_EntityType.USER,
)

key_no_max_budget_info: Final = CallInfo(
    token="test_token",
    spend=90,
    soft_budget=85,
    user_id="dev@test.com",
    user_email="dev@test.com",
    key_alias="dev-key",
    event_group=Litellm_EntityType.KEY,
)

@pytest.mark.parametrize(
    "model, optional_params, expected_api_base",
    [
        ("openai/my-fake-model", {"api_base": "my-fake-api-base"}, "my-fake-api-base"),
        ("gpt-5-mini", {}, "https://api.openai.com"),
    ],
)
def test_get_api_base_unit_test(model, optional_params, expected_api_base):
    api_base = get_api_base(model=model, optional_params=optional_params)

    assert api_base == expected_api_base


def test_init():
    slack_alerting = SlackAlerting(
        alerting_threshold=32,
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )
    assert slack_alerting.alerting_threshold == 32
    assert slack_alerting.alerting == ["slack"]
    assert slack_alerting.alert_types == ["llm_exceptions"]

    slack_no_alerting = SlackAlerting()
    assert slack_no_alerting.alerting == []

    print("passed testing slack alerting init")


@pytest.mark.asyncio
async def test_response_taking_too_long_callback(slack_alerting):
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=301)
    kwargs = {"model": "test_model", "messages": "test_messages", "litellm_params": {}}
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.response_taking_too_long_callback(
            kwargs, None, start_time, end_time
        )
        mock_send_alert.assert_awaited_once()

@pytest.mark.asyncio
async def test_alerting_metadata(slack_alerting):
    """
    Test alerting_metadata is propogated correctly for response taking too long
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=301)
    kwargs = {
        "model": "test_model",
        "messages": "test_messages",
        "litellm_params": {"metadata": {"alerting_metadata": {"hello": "world"}}},
    }
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:

        ## RESPONSE TAKING TOO LONG
        await slack_alerting.response_taking_too_long_callback(
            kwargs, None, start_time, end_time
        )
        mock_send_alert.assert_awaited_once()

        assert "hello" in mock_send_alert.call_args[1]["alerting_metadata"]

@pytest.mark.asyncio
async def test_budget_alerts_crossed(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_awaited_once()

@pytest.mark.asyncio
async def test_budget_alerts_crossed_again(slack_alerting):
    user_max_budget = 100
    user_current_spend = 101
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_awaited_once()
        mock_send_alert.reset_mock()
        await slack_alerting.budget_alerts(
            "user_budget",
            user_info=CallInfo(
                token="",
                spend=user_current_spend,
                max_budget=user_max_budget,
                event_group=Litellm_EntityType.USER,
            ),
        )
        mock_send_alert.assert_not_awaited()

@pytest.mark.asyncio
async def test_daily_reports_unit_test(slack_alerting):
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "test-gpt",
                    "litellm_params": {"model": "gpt-5-mini"},
                    "model_info": {"id": "1234"},
                }
            ]
        )
        deployment_metrics = DeploymentMetrics(
            id="1234",
            failed_request=False,
            latency_per_output_token=20.3,
            updated_at=litellm.utils.get_utc_datetime(),
        )

        updated_val = await slack_alerting.async_update_daily_reports(
            deployment_metrics=deployment_metrics
        )

        assert updated_val == 1

        await slack_alerting.send_daily_reports(router=router)

        mock_send_alert.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_daily_reports_ignores_zero_values():
    router = MagicMock()
    router.get_model_ids.return_value = ["model1", "model2", "model3"]

    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    # model1:failed=None, model2:failed=0, model3:failed=10, model1:latency=0; model2:latency=0; model3:latency=None
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(
        return_value=[None, 0, 10, 0, 0, None]
    )
    slack_alerting.internal_usage_cache.async_set_cache_pipeline = AsyncMock()

    router.get_model_info.side_effect = lambda x: {"litellm_params": {"model": x}}

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)

        # Check that the send_alert method was called
        mock_send_alert.assert_called_once()
        message = mock_send_alert.call_args[1]["message"]

        # Ensure the message includes only the non-zero, non-None metrics
        assert "model3" in message
        assert "model2" not in message
        assert "model1" not in message

    assert result == True

@pytest.mark.asyncio
async def test_send_daily_reports_all_zero_or_none():
    router = MagicMock()
    router.get_model_ids.return_value = ["model1", "model2", "model3"]

    slack_alerting = SlackAlerting(internal_usage_cache=MagicMock())
    slack_alerting.internal_usage_cache.async_batch_get_cache = AsyncMock(
        return_value=[None, 0, None, 0, None, 0]
    )

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        result = await slack_alerting.send_daily_reports(router)

        # Check that the send_alert method was not called
        mock_send_alert.assert_not_called()

    assert result == False

@pytest.mark.parametrize(
    "alerting_type",
    [
        "token_budget",
        "user_budget",
        "team_budget",
        "organization_budget",
        "proxy_budget",
        "projected_limit_exceeded",
    ],
)
@pytest.mark.asyncio
async def test_send_token_budget_crossed_alerts(alerting_type):
    slack_alerting = SlackAlerting()

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        user_info = {
            "token": "sk-test-mock-token-606",
            "spend": 86,
            "max_budget": 100,
            "user_id": "ishaan@berri.ai",
            "user_email": "ishaan@berri.ai",
            "key_alias": "my-test-key",
            "projected_exceeded_date": "10/20/2024",
            "projected_spend": 200,
            "event_group": Litellm_EntityType.KEY,
        }

        user_info = CallInfo(**user_info)

        for _ in range(50):
            await slack_alerting.budget_alerts(
                type=alerting_type,
                user_info=user_info,
            )
        mock_send_alert.assert_awaited_once()

@pytest.mark.parametrize(
    "alerting_type",
    [
        "token_budget",
        "user_budget",
        "team_budget",
        "organization_budget",
        "proxy_budget",
        "projected_limit_exceeded",
    ],
)
@pytest.mark.asyncio
async def test_webhook_alerting(alerting_type):
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(
        slack_alerting, "send_webhook_alert", new=AsyncMock()
    ) as mock_send_alert:
        user_info = {
            "token": "sk-test-mock-token-606",
            "spend": 1,
            "max_budget": 0,
            "user_id": "ishaan@berri.ai",
            "user_email": "ishaan@berri.ai",
            "key_alias": "my-test-key",
            "projected_exceeded_date": "10/20/2024",
            "projected_spend": 200,
            "event_group": Litellm_EntityType.KEY,
        }

        user_info = CallInfo(**user_info)
        for _ in range(50):
            await slack_alerting.budget_alerts(
                type=alerting_type,
                user_info=user_info,
            )
        mock_send_alert.assert_awaited_once()

@pytest.mark.parametrize(
    "model, api_base, llm_provider, vertex_project, vertex_location",
    [
        ("gpt-5-mini", None, "openai", None, None),
        (
            "azure/gpt-5-mini",
            "https://openai-gpt-4-test-v-1.openai.azure.com",
            "azure",
            None,
            None,
        ),
        ("gemini-3.8-flash", None, "vertex_ai", "hardy-device-38811", "us-central1"),
    ],
)
@pytest.mark.parametrize("error_code", [500, 408, 400])
@pytest.mark.asyncio
async def test_outage_alerting_called(
    model, api_base, llm_provider, vertex_project, vertex_location, error_code
):
    """
    If call fails, outage alert is called

    If multiple calls fail, outage alert is sent
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    litellm.callbacks = [slack_alerting]

    error_to_raise: Optional[APIError] = None

    if error_code == 400:
        print("RAISING 400 ERROR CODE")
        error_to_raise = litellm.BadRequestError(
            message="this is a bad request",
            model=model,
            llm_provider=llm_provider,
        )
    elif error_code == 408:
        print("RAISING 408 ERROR CODE")
        error_to_raise = litellm.Timeout(
            message="A timeout occurred", model=model, llm_provider=llm_provider
        )
    elif error_code == 500:
        print("RAISING 500 ERROR CODE")
        error_to_raise = litellm.ServiceUnavailableError(
            message="API is unavailable",
            model=model,
            llm_provider=llm_provider,
            response=httpx.Response(
                status_code=503,
                request=httpx.Request(
                    method="completion",
                    url="https://github.com/BerriAI/litellm",
                ),
            ),
        )

    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_AI_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": vertex_project,
                },
            }
        ],
        num_retries=0,
        allowed_fails=100,
    )

    slack_alerting.update_values(llm_router=router)
    with patch.object(
        slack_alerting, "outage_alerts", new=AsyncMock()
    ) as mock_outage_alert:
        try:
            await router.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hey!"}],
                mock_response=error_to_raise,
            )
        except Exception as e:
            pass

        mock_outage_alert.assert_called_once()

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        for _ in range(6):
            try:
                await router.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "Hey!"}],
                    mock_response=error_to_raise,
                )
            except Exception as e:
                pass
        await asyncio.sleep(3)
        if error_code == 500 or error_code == 408:
            mock_send_alert.assert_called_once()
        else:
            mock_send_alert.assert_not_called()

@pytest.mark.parametrize(
    "model, api_base, llm_provider, vertex_project, vertex_location",
    [
        ("gpt-5-mini", None, "openai", None, None),
        (
            "azure/gpt-5-mini",
            "https://openai-gpt-4-test-v-1.openai.azure.com",
            "azure",
            None,
            None,
        ),
        ("gemini-3.8-flash", None, "vertex_ai", "hardy-device-38811", "us-central1"),
    ],
)
@pytest.mark.parametrize("error_code", [500, 408, 400])
@pytest.mark.asyncio
async def test_region_outage_alerting_called(
    model, api_base, llm_provider, vertex_project, vertex_location, error_code
):
    """
    If call fails, outage alert is called

    If multiple calls fail, outage alert is sent
    """
    slack_alerting = SlackAlerting(
        alerting=["webhook"], alert_types=[AlertType.region_outage_alerts]
    )

    litellm.callbacks = [slack_alerting]

    error_to_raise: Optional[APIError] = None

    if error_code == 400:
        print("RAISING 400 ERROR CODE")
        error_to_raise = litellm.BadRequestError(
            message="this is a bad request",
            model=model,
            llm_provider=llm_provider,
        )
    elif error_code == 408:
        print("RAISING 408 ERROR CODE")
        error_to_raise = litellm.Timeout(
            message="A timeout occurred", model=model, llm_provider=llm_provider
        )
    elif error_code == 500:
        print("RAISING 500 ERROR CODE")
        error_to_raise = litellm.ServiceUnavailableError(
            message="API is unavailable",
            model=model,
            llm_provider=llm_provider,
            response=httpx.Response(
                status_code=503,
                request=httpx.Request(
                    method="completion",
                    url="https://github.com/BerriAI/litellm",
                ),
            ),
        )

    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_AI_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": vertex_project,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": os.getenv("AZURE_AI_API_KEY"),
                    "api_base": api_base,
                    "vertex_location": vertex_location,
                    "vertex_project": "vertex_project-2",
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        allowed_fails=100,
    )

    slack_alerting.update_values(llm_router=router)
    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        for idx in range(6):
            if idx % 2 == 0:
                deployment_id = "1"
            else:
                deployment_id = "2"
            await slack_alerting.region_outage_alerts(
                exception=error_to_raise, deployment_id=deployment_id  # type: ignore
            )
        if model == "gemini-3.8-flash" and (error_code == 500 or error_code == 408):
            mock_send_alert.assert_called_once()
        else:
            mock_send_alert.assert_not_called()

@pytest.mark.asyncio
async def test_print_alerting_payload_warning():
    """
    Test if alerts are printed to verbose logger when log_to_console=True
    """
    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_proxy_logger
    from litellm.integrations.SlackAlerting.batching_handler import send_to_webhook

    # Create a string buffer to capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    verbose_proxy_logger.addHandler(handler)
    verbose_proxy_logger.setLevel(logging.WARNING)

    # Create SlackAlerting instance with log_to_console=True
    slack_alerting = SlackAlerting(
        alerting_threshold=0.0000001,
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
        internal_usage_cache=DualCache(),
    )
    slack_alerting.alerting_args.log_to_console = True

    test_payload = {"text": "Test alert message"}

    # Send an alert
    with patch.object(
        slack_alerting.async_http_handler, "post", new=AsyncMock()
    ) as mock_post:
        await send_to_webhook(
            slackAlertingInstance=slack_alerting,
            item={
                "url": "https://example.com",
                "headers": {"Content-Type": "application/json"},
                "payload": {"text": "Test alert message"},
            },
            count=1,
        )

    # Check if the payload was logged
    log_output = log_stream.getvalue()
    print(log_output)
    assert "Test alert message" in log_output

    # Clean up
    verbose_proxy_logger.removeHandler(handler)
    log_stream.close()

@pytest.mark.asyncio
async def test_soft_budget_alerts():
    """
    Test if soft budget alerts (warnings when approaching budget limit) work correctly
    - Test alert is sent when spend reaches 80% of budget
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        # Test 80% threshold
        user_info = CallInfo(
            token="test_token",
            spend=80,  # $80 spent
            soft_budget=80,
            user_id="test@test.com",
            user_email="test@test.com",
            key_alias="test-key",
            event_group=Litellm_EntityType.KEY,
        )

        await slack_alerting.budget_alerts(
            type="soft_budget",
            user_info=user_info,
        )
        mock_send_alert.assert_called_once()

        # Verify alert message contains correct percentage
        alert_message = mock_send_alert.call_args[1]["message"]

        print("GOT MESSAGE\n\n", alert_message)

        expected_message = (
            "Soft Budget Crossed: Total Soft Budget:`80.0`\n"
            "\n"
            "*spend:* `80.0`\n"
            "*soft_budget:* `80.0`\n"
            "*user_id:* `test@test.com`\n"
            "*user_email:* `test@test.com`\n"
            "*key_alias:* `test-key`\n"
            "*event_group:* `key`\n"
        )
        assert alert_message == expected_message

@pytest.mark.parametrize(
    "entity_info",
    [
        key_info,
        team_info,
        user_info,
        key_no_max_budget_info,
    ],
)
@pytest.mark.asyncio
async def test_soft_budget_alerts_webhook(entity_info):
    """
    Tests that soft budget alerts are triggered for different entity types.

    Tests:
    - Key with max budget
    - Team
    - User
    - Key without max budget
    """
    slack_alerting = SlackAlerting(alerting=["webhook"])

    with patch.object(slack_alerting, "send_alert", new=AsyncMock()) as mock_send_alert:
        # Test entity hit soft budget limit
        await slack_alerting.budget_alerts(
            type="soft_budget",
            user_info=entity_info,
        )
        mock_send_alert.assert_called_once()

        # Verify the webhook event
        call_args = mock_send_alert.call_args[1]
        logged_webhook_event: WebhookEvent = call_args["user_info"]

        # Validate the webhook event has all expected fields
        assert logged_webhook_event.spend == entity_info.spend
        assert logged_webhook_event.soft_budget == entity_info.soft_budget
        assert logged_webhook_event.max_budget == entity_info.max_budget
        assert logged_webhook_event.user_id == entity_info.user_id
        assert logged_webhook_event.user_email == entity_info.user_email
        assert logged_webhook_event.key_alias == entity_info.key_alias
        assert logged_webhook_event.event_group == entity_info.event_group
