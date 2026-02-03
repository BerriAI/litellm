"""Regression tests for Slack hanging-request alerting.

These tests live under `tests/litellm/` to satisfy LiteLLM's upstream PR
requirement of adding at least one test in that directory.

Primary regression: prevent false-positive "Requests are hanging" alerts by
gating on elapsed time since the request was registered for hanging checks.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
from litellm.types.integrations.slack_alerting import HangingRequestData


class TestHangingRequestCheckGate:
    @pytest.fixture
    def checker(self):
        mock_slack = MagicMock()
        mock_slack.alerting_threshold = 300
        mock_slack.send_alert = AsyncMock()

        mock_slack.internal_usage_cache = MagicMock()
        mock_slack.internal_usage_cache.async_get_cache = AsyncMock(return_value=None)

        return AlertingHangingRequestCheck(slack_alerting_object=mock_slack)

    @pytest.mark.asyncio
    async def test_does_not_alert_before_threshold(self, checker):
        request_id = "not_yet_hanging_request"
        hanging_data = HangingRequestData(
            request_id=request_id,
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        hanging_data.start_time = time.time()  # just registered

        await checker.hanging_request_cache.async_set_cache(
            key=request_id,
            value=hanging_data,
            ttl=checker.hanging_request_cache.default_ttl,
        )

        checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
            return_value=[request_id]
        )
        await checker.send_alerts_for_hanging_requests()

        checker.slack_alerting_object.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_alerts_once_after_threshold(self, checker):
        request_id = "hanging_request"
        hanging_data = HangingRequestData(
            request_id=request_id,
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        )
        hanging_data.start_time = time.time() - 301

        await checker.hanging_request_cache.async_set_cache(
            key=request_id,
            value=hanging_data,
            ttl=checker.hanging_request_cache.default_ttl,
        )

        checker.hanging_request_cache.async_get_oldest_n_keys = AsyncMock(
            return_value=[request_id]
        )

        await checker.send_alerts_for_hanging_requests()
        assert checker.slack_alerting_object.send_alert.await_count == 1

        # Second run: should not re-alert due to persisted `alert_sent`.
        await checker.send_alerts_for_hanging_requests()
        assert checker.slack_alerting_object.send_alert.await_count == 1
