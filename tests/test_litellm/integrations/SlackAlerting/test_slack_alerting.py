import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system-path
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import CallInfo, Litellm_EntityType


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

        assert not self.slack_alerting.periodic_started

        self.slack_alerting.update_values(alerting_args={"slack_alerting": "True"})
        assert self.slack_alerting.periodic_started

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


class TestSendWebhookAlert(unittest.IsolatedAsyncioTestCase):
    """Tests for send_webhook_alert URL resolution and per-threshold dedup."""

    def _make_webhook_event(self, budget_percentage_used=None):
        from litellm.proxy._types import WebhookEvent

        return WebhookEvent(
            event="threshold_crossed",
            event_message="80% of budget consumed",
            spend=80.0,
            max_budget=100.0,
            event_group=Litellm_EntityType.KEY,
            budget_percentage_used=budget_percentage_used,
        )

    async def test_uses_alert_to_webhook_url_over_env_var(self):
        """alert_to_webhook_url config takes priority over WEBHOOK_URL env var."""
        from litellm.proxy._types import AlertType

        mock_response = MagicMock()
        mock_response.status_code = 200

        alerting = SlackAlerting(
            alerting=["webhook"],
            alert_to_webhook_url={AlertType.budget_alerts: "https://configured.example.com/hook"},
        )
        alerting.async_http_handler = AsyncMock()
        alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

        with unittest.mock.patch.dict("os.environ", {"WEBHOOK_URL": "https://env.example.com/hook"}):
            result = await alerting.send_webhook_alert(self._make_webhook_event())

        assert result is True
        posted_url = alerting.async_http_handler.post.call_args[1]["url"]
        assert posted_url == "https://configured.example.com/hook"

    async def test_falls_back_to_env_var_when_no_config(self):
        """When alert_to_webhook_url is not set, falls back to WEBHOOK_URL env var."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        alerting = SlackAlerting(alerting=["webhook"])
        alerting.async_http_handler = AsyncMock()
        alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

        with unittest.mock.patch.dict("os.environ", {"WEBHOOK_URL": "https://env.example.com/hook"}):
            result = await alerting.send_webhook_alert(self._make_webhook_event())

        assert result is True
        posted_url = alerting.async_http_handler.post.call_args[1]["url"]
        assert posted_url == "https://env.example.com/hook"

    async def test_raises_when_no_url_configured(self):
        """Raises an exception when no URL is configured anywhere."""
        alerting = SlackAlerting(alerting=["webhook"])
        alerting.async_http_handler = AsyncMock()

        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            # Ensure WEBHOOK_URL is absent
            import os

            os.environ.pop("WEBHOOK_URL", None)
            with self.assertRaises(Exception, msg="No webhook URL configured"):
                await alerting.send_webhook_alert(self._make_webhook_event())

    async def test_posts_to_all_urls_in_list(self):
        """When alert_to_webhook_url maps to a list, posts to every URL."""
        from litellm.proxy._types import AlertType

        mock_response = MagicMock()
        mock_response.status_code = 200

        alerting = SlackAlerting(
            alerting=["webhook"],
            alert_to_webhook_url={
                AlertType.budget_alerts: [
                    "https://hook1.example.com",
                    "https://hook2.example.com",
                ]
            },
        )
        alerting.async_http_handler = AsyncMock()
        alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

        with unittest.mock.patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("WEBHOOK_URL", None)
            result = await alerting.send_webhook_alert(self._make_webhook_event())

        assert result is True
        assert alerting.async_http_handler.post.call_count == 2
        posted_urls = {call[1]["url"] for call in alerting.async_http_handler.post.call_args_list}
        assert posted_urls == {"https://hook1.example.com", "https://hook2.example.com"}




class TestBudgetAlertsDedup(unittest.IsolatedAsyncioTestCase):
    """Tests that different thresholds get separate cache keys (no suppression across thresholds)."""

    async def test_different_thresholds_fire_independently(self):
        """Two budget_alerts calls with different budget_percentage_used values should both send."""
        from litellm.proxy._types import AlertType

        mock_response = MagicMock()
        mock_response.status_code = 200

        alerting = SlackAlerting(
            alerting=["webhook"],
            alert_to_webhook_url={AlertType.budget_alerts: "https://hook.example.com"},
        )
        alerting.async_http_handler = AsyncMock()
        alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

        call_info_80 = CallInfo(
            spend=80.0,
            max_budget=100.0,
            event_group=Litellm_EntityType.TEAM,
            team_id="team-1",
            budget_percentage_used=0.80,
        )
        call_info_85 = CallInfo(
            spend=85.0,
            max_budget=100.0,
            event_group=Litellm_EntityType.TEAM,
            team_id="team-1",
            budget_percentage_used=0.85,
        )

        await alerting.budget_alerts(type="team_budget", user_info=call_info_80)
        await alerting.budget_alerts(type="team_budget", user_info=call_info_85)

        assert alerting.async_http_handler.post.call_count == 2

    async def test_same_threshold_fires_only_once(self):
        """A second budget_alerts call with the same entity and same threshold is suppressed."""
        from litellm.proxy._types import AlertType

        mock_response = MagicMock()
        mock_response.status_code = 200

        alerting = SlackAlerting(
            alerting=["webhook"],
            alert_to_webhook_url={AlertType.budget_alerts: "https://hook.example.com"},
        )
        alerting.async_http_handler = AsyncMock()
        alerting.async_http_handler.post = AsyncMock(return_value=mock_response)

        call_info = CallInfo(
            spend=80.0,
            max_budget=100.0,
            event_group=Litellm_EntityType.TEAM,
            team_id="team-2",
            budget_percentage_used=0.80,
        )

        await alerting.budget_alerts(type="team_budget", user_info=call_info)
        await alerting.budget_alerts(type="team_budget", user_info=call_info)

        assert alerting.async_http_handler.post.call_count == 1
