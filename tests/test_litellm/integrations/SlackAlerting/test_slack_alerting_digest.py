"""
Tests for Slack Alert Digest Mode

Verifies that:
- Digest mode suppresses duplicate alerts within the interval
- Digest summary is emitted after the interval expires
- Non-digest alert types are unaffected
- Different (model, api_base) combos get separate digest entries
- The digest message format includes Start/End timestamps and Count
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType
from litellm.types.integrations.slack_alerting import AlertTypeConfig


class TestDigestMode(unittest.IsolatedAsyncioTestCase):
    """Test digest mode in SlackAlerting.send_alert()."""

    def setUp(self):
        os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/test"
        self.slack_alerting = SlackAlerting(
            alerting=["slack"],
            alert_type_config={
                "llm_requests_hanging": {"digest": True, "digest_interval": 60},
            },
        )
        # Prevent periodic flush from starting
        self.slack_alerting.periodic_started = True

    def tearDown(self):
        os.environ.pop("SLACK_WEBHOOK_URL", None)

    async def test_digest_suppresses_duplicate_alerts(self):
        """Sending the same alert type + model + api_base multiple times should NOT add to log_queue."""
        message = "`Requests are hanging`\nRequest Model: `gemini-2.5-flash`\nAPI Base: `None`"

        for _ in range(5):
            await self.slack_alerting.send_alert(
                message=message,
                level="Medium",
                alert_type=AlertType.llm_requests_hanging,
                alerting_metadata={},
                request_model="gemini-2.5-flash",
                api_base="None",
            )

        # No messages should be in the log queue - they're all in digest_buckets
        self.assertEqual(len(self.slack_alerting.log_queue), 0)
        # Should have exactly 1 digest bucket entry
        self.assertEqual(len(self.slack_alerting.digest_buckets), 1)
        # Count should be 5
        bucket = list(self.slack_alerting.digest_buckets.values())[0]
        self.assertEqual(bucket["count"], 5)

    async def test_different_models_get_separate_digests(self):
        """Different models should produce separate digest entries."""
        await self.slack_alerting.send_alert(
            message="`Requests are hanging`",
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata={},
            request_model="gemini-2.5-flash",
            api_base="None",
        )
        await self.slack_alerting.send_alert(
            message="`Requests are hanging`",
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata={},
            request_model="gpt-4",
            api_base="https://api.openai.com",
        )

        self.assertEqual(len(self.slack_alerting.digest_buckets), 2)

    async def test_non_digest_alert_goes_to_queue(self):
        """Alert types without digest enabled should go straight to the log queue."""
        message = "Budget exceeded"

        await self.slack_alerting.send_alert(
            message=message,
            level="High",
            alert_type=AlertType.budget_alerts,
            alerting_metadata={},
        )

        # Should be in log_queue, not digest_buckets
        self.assertGreater(len(self.slack_alerting.log_queue), 0)
        self.assertEqual(len(self.slack_alerting.digest_buckets), 0)

    async def test_flush_digest_buckets_emits_after_interval(self):
        """After the digest interval expires, _flush_digest_buckets should emit a summary."""
        message = "`Requests are hanging`\nRequest Model: `gemini-2.5-flash`\nAPI Base: `None`"

        # Send 3 alerts
        for _ in range(3):
            await self.slack_alerting.send_alert(
                message=message,
                level="Medium",
                alert_type=AlertType.llm_requests_hanging,
                alerting_metadata={},
                request_model="gemini-2.5-flash",
                api_base="None",
            )

        self.assertEqual(len(self.slack_alerting.log_queue), 0)
        self.assertEqual(len(self.slack_alerting.digest_buckets), 1)

        # Manually backdate the start_time to simulate interval expiration
        key = list(self.slack_alerting.digest_buckets.keys())[0]
        self.slack_alerting.digest_buckets[key]["start_time"] = datetime.now() - timedelta(seconds=120)

        # Flush digest buckets
        await self.slack_alerting._flush_digest_buckets()

        # Digest bucket should be cleared
        self.assertEqual(len(self.slack_alerting.digest_buckets), 0)
        # And a summary message should be in the log queue
        self.assertEqual(len(self.slack_alerting.log_queue), 1)
        payload_text = self.slack_alerting.log_queue[0]["payload"]["text"]
        self.assertIn("(Digest)", payload_text)
        self.assertIn("Count: `3`", payload_text)
        self.assertIn("Start:", payload_text)
        self.assertIn("End:", payload_text)

    async def test_flush_does_not_emit_before_interval(self):
        """Digest buckets should NOT be flushed before the interval expires."""
        message = "`Requests are hanging`"

        await self.slack_alerting.send_alert(
            message=message,
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata={},
            request_model="gemini-2.5-flash",
        )

        # Flush immediately (interval hasn't expired)
        await self.slack_alerting._flush_digest_buckets()

        # Bucket should still be there
        self.assertEqual(len(self.slack_alerting.digest_buckets), 1)
        self.assertEqual(len(self.slack_alerting.log_queue), 0)

    async def test_digest_message_format(self):
        """Verify the digest summary message format."""
        message = "`Requests are hanging - 600s+ request time`\nRequest Model: `gemini-2.5-flash`\nAPI Base: `None`"

        await self.slack_alerting.send_alert(
            message=message,
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata={},
            request_model="gemini-2.5-flash",
            api_base="None",
        )

        # Backdate and flush
        key = list(self.slack_alerting.digest_buckets.keys())[0]
        self.slack_alerting.digest_buckets[key]["start_time"] = datetime.now() - timedelta(seconds=120)

        await self.slack_alerting._flush_digest_buckets()

        payload_text = self.slack_alerting.log_queue[0]["payload"]["text"]
        self.assertIn("Alert type: `llm_requests_hanging` (Digest)", payload_text)
        self.assertIn("Level: `Medium`", payload_text)
        self.assertIn("Count: `1`", payload_text)
        self.assertIn("`Requests are hanging - 600s+ request time`", payload_text)

    async def test_digest_without_model_groups_by_alert_type_only(self):
        """When request_model is not provided, alerts group by alert type alone."""
        for _ in range(3):
            await self.slack_alerting.send_alert(
                message="Some hanging request",
                level="Medium",
                alert_type=AlertType.llm_requests_hanging,
                alerting_metadata={},
            )

        # All 3 should be in the same bucket (empty model and api_base)
        self.assertEqual(len(self.slack_alerting.digest_buckets), 1)
        bucket = list(self.slack_alerting.digest_buckets.values())[0]
        self.assertEqual(bucket["count"], 3)
        self.assertEqual(bucket["request_model"], "")
        self.assertEqual(bucket["api_base"], "")


class TestAlertTypeConfig(unittest.TestCase):
    """Test AlertTypeConfig model and initialization."""

    def test_default_values(self):
        config = AlertTypeConfig()
        self.assertFalse(config.digest)
        self.assertEqual(config.digest_interval, 86400)

    def test_custom_values(self):
        config = AlertTypeConfig(digest=True, digest_interval=3600)
        self.assertTrue(config.digest)
        self.assertEqual(config.digest_interval, 3600)

    def test_slack_alerting_init_with_config(self):
        sa = SlackAlerting(
            alerting=["slack"],
            alert_type_config={
                "llm_requests_hanging": {"digest": True, "digest_interval": 7200},
                "llm_too_slow": {"digest": True},
            },
        )
        self.assertIn("llm_requests_hanging", sa.alert_type_config)
        self.assertIn("llm_too_slow", sa.alert_type_config)
        self.assertTrue(sa.alert_type_config["llm_requests_hanging"].digest)
        self.assertEqual(sa.alert_type_config["llm_requests_hanging"].digest_interval, 7200)
        self.assertEqual(sa.alert_type_config["llm_too_slow"].digest_interval, 86400)

    def test_update_values_with_config(self):
        sa = SlackAlerting(alerting=["slack"])
        self.assertEqual(len(sa.alert_type_config), 0)

        sa.update_values(
            alert_type_config={"llm_exceptions": {"digest": True, "digest_interval": 1800}},
        )
        self.assertIn("llm_exceptions", sa.alert_type_config)
        self.assertTrue(sa.alert_type_config["llm_exceptions"].digest)


if __name__ == "__main__":
    unittest.main()
