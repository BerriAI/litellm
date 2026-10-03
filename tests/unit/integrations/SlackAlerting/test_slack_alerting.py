import asyncio
import datetime
import json
import time
import unittest
from collections.abc import Mapping
from typing import Final, List, Literal, Optional, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._internal_context import current_service_target
from litellm.caching.caching import DualCache
from litellm.integrations.SlackAlerting.budget_alert_types import get_budget_alert_type
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import CallInfo, Litellm_EntityType, WebhookEvent
from litellm.types.integrations.slack_alerting import (
    AlertQueueItem,
    AlertType,
    SlackAlertingArgs,
    SlackAlertingCacheKeys,
)


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


def _budget_webhook_event(
    key_alias: str | None, event_group: Litellm_EntityType = Litellm_EntityType.KEY
) -> WebhookEvent:
    return WebhookEvent(
        spend=85.0,
        max_budget=100.0,
        token="hashed_key",
        key_alias=key_alias,
        event="threshold_crossed",
        event_message="15% or less of budget remaining",
        event_group=event_group,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alerting_args, key_alias, event_group, delivered",
    [
        ({}, None, None, True),
        ({}, None, Litellm_EntityType.KEY, True),
        ({"slack_budget_alert_key_aliases": None}, None, None, True),
        ({"slack_budget_alert_key_aliases": None}, None, Litellm_EntityType.TEAM, True),
        ({"slack_budget_alert_key_aliases": []}, "github-example-api", Litellm_EntityType.KEY, False),
        (
            {"slack_budget_alert_key_aliases": ["github-example-api"]},
            "github-example-api",
            Litellm_EntityType.KEY,
            True,
        ),
        (
            {"slack_budget_alert_key_aliases": ["github-example-api"]},
            "github-example-api-extra",
            Litellm_EntityType.KEY,
            False,
        ),
        ({"slack_budget_alert_key_aliases": ["github-example-*"]}, "github-example-api", Litellm_EntityType.KEY, True),
        ({"slack_budget_alert_key_aliases": ["github-example-*"]}, "other-api", Litellm_EntityType.KEY, False),
        ({"slack_budget_alert_key_aliases": ["github-example-*"]}, "GitHub-example-api", Litellm_EntityType.KEY, False),
        ({"slack_budget_alert_key_aliases": ["github-example-*"]}, None, Litellm_EntityType.KEY, False),
        ({"slack_budget_alert_key_aliases": ["*"]}, "", Litellm_EntityType.KEY, False),
        ({"slack_budget_alert_key_aliases": ["*"]}, None, None, False),
        (
            {"slack_budget_alert_key_aliases": ["other-*", "github-example-?"]},
            "github-example-a",
            Litellm_EntityType.KEY,
            True,
        ),
        (
            {"slack_budget_alert_key_aliases": ["github-example-?", "other-*"]},
            "github-example-a",
            Litellm_EntityType.KEY,
            True,
        ),
        (
            {"slack_budget_alert_key_aliases": ["github-example-?", "other-*"]},
            "github-example-ab",
            Litellm_EntityType.KEY,
            False,
        ),
        ({"slack_budget_alert_key_aliases": ["github-example-[ab]"]}, "github-example-b", Litellm_EntityType.KEY, True),
    ],
)
async def test_slack_budget_key_alias_filter_delivery(
    alerting_args: Mapping[str, object],
    key_alias: str | None,
    event_group: Litellm_EntityType | None,
    delivered: bool,
) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args=dict(alerting_args),
        async_http_handler=http_handler,
    )
    event: Final = _budget_webhook_event(key_alias, event_group) if event_group is not None else None
    await slack_alerting.send_alert(
        message=THRESHOLD_ALERT,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={"key_alias": "github-example-api"},
        user_info=event,
    )

    assert len(slack_alerting.log_queue) == int(delivered)
    await slack_alerting.flush_queue()
    assert http_handler.post.await_count == int(delivered)
    if delivered:
        assert THRESHOLD_ALERT in _posted_slack_bodies(http_handler)[0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("event_group", tuple(group for group in Litellm_EntityType if group != Litellm_EntityType.KEY))
async def test_slack_budget_key_alias_filter_excludes_non_key_entities(event_group: Litellm_EntityType) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": ["github-example-*"]},
        async_http_handler=http_handler,
    )
    await slack_alerting.send_alert(
        message=THRESHOLD_ALERT,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
        user_info=_budget_webhook_event("github-example-api", event_group),
    )

    assert slack_alerting.log_queue == []
    await slack_alerting.flush_queue()
    http_handler.post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alert_type",
    (
        AlertType.llm_exceptions,
        AlertType.failed_tracking_spend,
        AlertType.user_spend_thresholds,
        AlertType.user_spend_anomalies,
    ),
)
async def test_slack_budget_key_alias_filter_leaves_other_alert_types_unchanged(alert_type: AlertType) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        alert_types=[alert_type],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": []},
        async_http_handler=http_handler,
    )
    await slack_alerting.send_alert(
        message="other alert",
        level="High",
        alert_type=alert_type,
        alerting_metadata={},
    )
    await slack_alerting.flush_queue()

    http_handler.post.assert_awaited_once()
    assert "other alert" in _posted_slack_bodies(http_handler)[0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", (1, 100))
async def test_slack_budget_key_alias_filter_preserves_webhook_and_teams_delivery(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://webhook.example/budget")
    monkeypatch.setenv("MS_TEAMS_WEBHOOK_URL", "https://teams.example/budget")
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack", "webhook", "ms_teams"],
        alerting_args={"slack_budget_alert_key_aliases": []},
        async_http_handler=http_handler,
        batch_size=batch_size,
    )
    event: Final = _budget_webhook_event("github-example-api")
    await slack_alerting.send_alert(
        message=THRESHOLD_ALERT,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
        user_info=event,
    )
    assert tuple(call.kwargs["url"] for call in http_handler.post.call_args_list) == (
        ("https://webhook.example/budget", "https://teams.example/budget")
        if batch_size == 1
        else ("https://webhook.example/budget",)
    )
    await slack_alerting.flush_queue()

    assert tuple(call.kwargs["url"] for call in http_handler.post.call_args_list) == (
        "https://webhook.example/budget",
        "https://teams.example/budget",
    )
    assert WebhookEvent.model_validate_json(http_handler.post.call_args_list[0].kwargs["data"]) == event
    teams_body: Final = json.loads(http_handler.post.call_args_list[1].kwargs["data"])
    assert THRESHOLD_ALERT in teams_body["attachments"][0]["content"]["body"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key_alias, delivered", (("github-example-api", True), ("other-api", False)))
async def test_slack_budget_key_alias_filter_retains_native_thresholds_and_dedup(
    key_alias: str, delivered: bool
) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": ["github-example-*"]},
        async_http_handler=http_handler,
    )
    below_threshold: Final = CallInfo(
        spend=50.0, max_budget=100.0, token="hashed_key", key_alias=key_alias, event_group=Litellm_EntityType.KEY
    )
    await slack_alerting.budget_alerts(type="token_budget", user_info=below_threshold)
    assert slack_alerting.log_queue == []
    assert (
        await slack_alerting.internal_usage_cache.async_get_cache("budget_alerts:threshold_crossed:hashed_key") is None
    )

    at_threshold: Final = below_threshold.model_copy(update={"spend": 85.0})
    await slack_alerting.budget_alerts(type="token_budget", user_info=at_threshold)
    assert len(slack_alerting.log_queue) == int(delivered)
    assert (
        await slack_alerting.internal_usage_cache.async_get_cache("budget_alerts:threshold_crossed:hashed_key")
        == "SENT_WITH_SLACK_DEDUP"
    )
    assert (
        await slack_alerting.internal_usage_cache.async_get_cache("budget_alerts:slack:threshold_crossed:hashed_key")
    ) == ("SENT" if delivered else None)
    await slack_alerting.budget_alerts(type="token_budget", user_info=at_threshold)
    assert len(slack_alerting.log_queue) == int(delivered)
    await slack_alerting.flush_queue()
    assert http_handler.post.await_count == int(delivered)
    if delivered:
        assert "15% or less of budget remaining" in _posted_slack_bodies(http_handler)[0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key_alias, delivered", (("github-example-api", True), ("other-api", False)))
async def test_slack_budget_key_alias_filter_applies_before_digest(key_alias: str, delivered: bool) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": ["github-example-*"]},
        alert_type_config={"budget_alerts": {"digest": True, "digest_interval": 0}},
        async_http_handler=http_handler,
    )
    await slack_alerting.send_alert(
        message=THRESHOLD_ALERT,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
        user_info=_budget_webhook_event(key_alias),
    )
    assert slack_alerting.log_queue == []
    assert len(slack_alerting.digest_buckets) == int(delivered)
    await slack_alerting._flush_digest_buckets()
    await slack_alerting.flush_queue()
    assert http_handler.post.await_count == int(delivered)
    if delivered:
        assert THRESHOLD_ALERT in _posted_slack_bodies(http_handler)[0]["text"]


@pytest.mark.parametrize("patterns", (None, [], ["github-example-*", "exact-alias", "?"]))
def test_slack_budget_key_alias_patterns_validate_and_round_trip(patterns: object) -> None:
    args: Final = SlackAlertingArgs.model_validate({"slack_budget_alert_key_aliases": patterns})
    assert args.slack_budget_alert_key_aliases == patterns
    assert args.model_dump()["slack_budget_alert_key_aliases"] == patterns


@pytest.mark.parametrize(
    "patterns", ("github-example-*", 123, {}, ("alias",), [""], [None], [123], [True], [["alias"]])
)
def test_slack_budget_key_alias_patterns_reject_invalid_config(patterns: object) -> None:
    with pytest.raises(ValidationError, match="slack_budget_alert_key_aliases"):
        SlackAlertingArgs.model_validate({"slack_budget_alert_key_aliases": patterns})


@pytest.mark.asyncio
@pytest.mark.parametrize("patterns, delivered", (([], False), (["github-example-*"], True)))
async def test_slack_budget_key_alias_filter_reload_validates_and_changes_delivery(
    patterns: list[str], delivered: bool
) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        async_http_handler=http_handler,
    )
    slack_alerting.update_values(alerting_args={"slack_budget_alert_key_aliases": patterns})
    with pytest.raises(ValidationError, match="slack_budget_alert_key_aliases"):
        slack_alerting.update_values(alerting_args={"slack_budget_alert_key_aliases": "github-example-*"})
    await slack_alerting.send_alert(
        message=THRESHOLD_ALERT,
        level="High",
        alert_type=AlertType.budget_alerts,
        alerting_metadata={},
        user_info=_budget_webhook_event("github-example-api"),
    )
    await slack_alerting.flush_queue()
    assert http_handler.post.await_count == int(delivered)
    if delivered:
        assert THRESHOLD_ALERT in _posted_slack_bodies(http_handler)[0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", (False, True))
@pytest.mark.parametrize("patterns", (["github-example-*"], None))
@pytest.mark.parametrize(
    "budget_type, spend, max_budget, soft_budget, event",
    (
        ("token_budget", 85.0, 100.0, None, "threshold_crossed"),
        ("max_budget_alert", 100.0, 100.0, None, "budget_crossed"),
        ("soft_budget", 50.0, None, 40.0, "soft_budget_crossed"),
        ("projected_limit_exceeded", 50.0, 100.0, None, "projected_limit_exceeded"),
    ),
)
async def test_budget_filter_reload_sends_slack_without_repeating_other_destinations(
    monkeypatch: pytest.MonkeyPatch,
    digest: bool,
    patterns: list[str] | None,
    budget_type: Literal["token_budget", "max_budget_alert", "soft_budget", "projected_limit_exceeded"],
    spend: float,
    max_budget: float | None,
    soft_budget: float | None,
    event: str,
) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://webhook.example/budget")
    monkeypatch.setenv("MS_TEAMS_WEBHOOK_URL", "https://teams.example/budget")
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack", "webhook", "ms_teams"],
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": []},
        alert_type_config={"budget_alerts": {"digest": digest, "digest_interval": 0}},
        async_http_handler=http_handler,
    )
    info: Final = CallInfo(
        spend=spend,
        max_budget=max_budget,
        soft_budget=soft_budget,
        token="hashed_key",
        key_alias="github-example-api",
        event_group=Litellm_EntityType.KEY,
    )
    await slack_alerting.budget_alerts(type=budget_type, user_info=info)
    await slack_alerting.flush_queue()
    assert tuple(c.kwargs["url"] for c in http_handler.post.call_args_list) == (
        "https://webhook.example/budget",
        "https://teams.example/budget",
    )
    slack_alerting.update_values(alerting_args={"slack_budget_alert_key_aliases": patterns})
    await slack_alerting.budget_alerts(type=budget_type, user_info=info)
    await slack_alerting.budget_alerts(type=budget_type, user_info=info)
    await slack_alerting._flush_digest_buckets()
    await slack_alerting.flush_queue()
    assert tuple(c.kwargs["url"] for c in http_handler.post.call_args_list) == (
        "https://webhook.example/budget",
        "https://teams.example/budget",
        SLACK_WEBHOOK_URL,
    )
    assert (
        "github-example-api"
        in _SLACK_WEBHOOK_BODY.validate_json(http_handler.post.call_args_list[-1].kwargs["data"])["text"]
    )
    assert (
        await slack_alerting.internal_usage_cache.async_get_cache(f"budget_alerts:slack:{event}:hashed_key") == "SENT"
    )


@pytest.mark.asyncio
async def test_budget_slack_and_other_destination_windows_expire_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://webhook.example/budget")
    http_handler: Final = _webhook_accepting_posts()
    cache: Final = DualCache()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack", "webhook"],
        internal_usage_cache=cache,
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": []},
        async_http_handler=http_handler,
    )
    info: Final = CallInfo(
        spend=85.0,
        max_budget=100.0,
        token="hashed_key",
        key_alias="github-example-api",
        event_group=Litellm_EntityType.KEY,
    )
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    slack_alerting.update_values(alerting_args={"slack_budget_alert_key_aliases": None})
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    await slack_alerting.flush_queue()
    cache.delete_cache("budget_alerts:threshold_crossed:hashed_key")
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    await slack_alerting.flush_queue()
    cache.delete_cache("budget_alerts:slack:threshold_crossed:hashed_key")
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    await slack_alerting.flush_queue()
    assert tuple(c.kwargs["url"] for c in http_handler.post.call_args_list) == (
        "https://webhook.example/budget",
        SLACK_WEBHOOK_URL,
        "https://webhook.example/budget",
        SLACK_WEBHOOK_URL,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", (False, True))
async def test_budget_empty_channel_mapping_does_not_consume_slack_dedup(digest: bool) -> None:
    http_handler: Final = _webhook_accepting_posts()
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        alert_to_webhook_url={AlertType.budget_alerts: []},
        alert_type_config={"budget_alerts": {"digest": digest, "digest_interval": 0}},
        async_http_handler=http_handler,
    )
    info: Final = CallInfo(
        spend=85.0,
        max_budget=100.0,
        token="hashed_key",
        key_alias="github-example-api",
        event_group=Litellm_EntityType.KEY,
    )
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    assert slack_alerting.log_queue == []
    assert slack_alerting.digest_buckets == {}
    slack_alerting.update_values(alert_to_webhook_url={AlertType.budget_alerts: SLACK_WEBHOOK_URL})
    await slack_alerting.budget_alerts(type="token_budget", user_info=info)
    await slack_alerting._flush_digest_buckets()
    await slack_alerting.flush_queue()
    assert "github-example-api" in _posted_slack_bodies(http_handler)[0]["text"]
    http_handler.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_budget_sent_marker_does_not_repeat_slack() -> None:
    http_handler: Final = _webhook_accepting_posts()
    cache: Final = DualCache()
    await cache.async_set_cache("budget_alerts:threshold_crossed:hashed_key", "SENT", ttl=86400)
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        internal_usage_cache=cache,
        default_webhook_url=SLACK_WEBHOOK_URL,
        alerting_args={"slack_budget_alert_key_aliases": ["github-example-*"]},
        async_http_handler=http_handler,
    )
    await slack_alerting.budget_alerts(
        type="token_budget",
        user_info=CallInfo(
            spend=85.0,
            max_budget=100.0,
            token="hashed_key",
            key_alias="github-example-api",
            event_group=Litellm_EntityType.KEY,
        ),
    )
    await slack_alerting.flush_queue()
    http_handler.post.assert_not_awaited()


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
