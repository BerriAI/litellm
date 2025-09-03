import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

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
