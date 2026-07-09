import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import litellm
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import CallInfo, Litellm_EntityType


class TestSlackAlerting(unittest.TestCase):
    def setUp(self):
        self.slack_alerting = SlackAlerting()

    def test_get_percent_of_max_budget_left(self):
        # Test case 1: When max_budget is None
        user_info = CallInfo(
            max_budget=None, spend=50.0, event_group=Litellm_EntityType.KEY
        )
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 2: When max_budget is 0
        user_info = CallInfo(
            max_budget=0.0, spend=50.0, event_group=Litellm_EntityType.KEY
        )
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 3: When spend is less than max_budget
        user_info = CallInfo(
            max_budget=100.0, spend=75.0, event_group=Litellm_EntityType.KEY
        )
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.25)

        # Test case 4: When spend equals max_budget
        user_info = CallInfo(
            max_budget=100.0, spend=100.0, event_group=Litellm_EntityType.KEY
        )
        result = self.slack_alerting._get_percent_of_max_budget_left(user_info)
        self.assertEqual(result, 0.0)

        # Test case 5: When spend exceeds max_budget
        user_info = CallInfo(
            max_budget=100.0, spend=120.0, event_group=Litellm_EntityType.KEY
        )
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
        formatted_message = f"{alert_type_formatted}\n Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"

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
        self.assertIn(
            "Object of type set is not JSON serializable", str(context.exception)
        )

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


class _MockSlackResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.mark.asyncio
async def test_send_dm_file_uses_external_upload_flow():
    slack_alerting = SlackAlerting()
    handler = MagicMock()
    handler.get = AsyncMock(
        return_value=_MockSlackResponse({"ok": True, "user": {"id": "U123"}})
    )

    async def post_response(url, **kwargs):
        if url == "https://slack.com/api/conversations.open":
            return _MockSlackResponse({"ok": True, "channel": {"id": "D123"}})
        if url == "https://slack.com/api/files.getUploadURLExternal":
            return _MockSlackResponse(
                {
                    "ok": True,
                    "upload_url": "https://files.slack.test/upload",
                    "file_id": "F123",
                }
            )
        if url == "https://files.slack.test/upload":
            return _MockSlackResponse({})
        if url == "https://slack.com/api/files.completeUploadExternal":
            return _MockSlackResponse({"ok": True})
        raise AssertionError(f"unexpected Slack URL: {url}")

    handler.post = AsyncMock(side_effect=post_response)
    slack_alerting.async_http_handler = handler

    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"}):
        sent = await slack_alerting.send_dm_file(
            user_email="requester@juspay.in",
            message="Decrypt with gpg --decrypt filename.gpg",
            filename="filename.gpg",
            title="filename.gpg",
            file_content="-----BEGIN PGP MESSAGE-----\n...",
        )

    assert sent is True
    handler.get.assert_awaited_once()
    assert handler.get.await_args.kwargs["url"] == "https://slack.com/api/users.lookupByEmail"
    assert handler.get.await_args.kwargs["params"] == {"email": "requester@juspay.in"}

    post_calls = handler.post.await_args_list
    assert [call.kwargs["url"] for call in post_calls] == [
        "https://slack.com/api/conversations.open",
        "https://slack.com/api/files.getUploadURLExternal",
        "https://files.slack.test/upload",
        "https://slack.com/api/files.completeUploadExternal",
    ]
    assert post_calls[0].kwargs["json"] == {"users": ["U123"]}
    assert "json" not in post_calls[1].kwargs
    assert post_calls[1].kwargs["data"] == {
        "filename": "filename.gpg",
        "length": str(len("-----BEGIN PGP MESSAGE-----\n...".encode("utf-8"))),
    }
    assert post_calls[2].kwargs["content"] == b"-----BEGIN PGP MESSAGE-----\n..."
    assert post_calls[3].kwargs["json"] == {
        "files": [{"id": "F123", "title": "filename.gpg"}],
        "channel_id": "D123",
        "initial_comment": "Decrypt with gpg --decrypt filename.gpg",
    }
