import os
from datetime import datetime, timedelta, timezone
from typing import List

from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.pagerduty import (
    AlertingConfig,
    PagerDutyPayload,
    PagerDutyRequestBody,
)


class PagerDutyAlerting(CustomBatchLogger):
    def __init__(self, alerting_config: AlertingConfig, **kwargs):
        _api_key = os.getenv("PAGERDUTY_API_KEY")
        if not _api_key:
            raise ValueError("PAGERDUTY_API_KEY is not set")
        self.api_key: str = _api_key
        self.alerting_config: AlertingConfig = alerting_config

        # Track recent failures in-memory
        self._failure_events: List[datetime] = []

        super().__init__(**kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Record a failure event. Only send an alert to PagerDuty if the
        configured failure threshold is exceeded in the specified window.
        """
        now = datetime.now(timezone.utc)
        self._failure_events.append(now)

        # Prune events outside of the threshold window
        window_seconds = self.alerting_config.get(
            "failure_threshold_window_seconds", 60
        )
        threshold = self.alerting_config.get("failure_threshold", 1)
        cutoff = now - timedelta(seconds=window_seconds)

        self._failure_events = [t for t in self._failure_events if t > cutoff]
        # If the number of recent failures crosses the threshold, alert PagerDuty
        if len(self._failure_events) >= threshold:
            await self.send_alert_to_pagerduty(
                alert_message=(
                    f"High Failure Rate: {len(self._failure_events)} "
                    f"failures in the last {window_seconds} seconds."
                )
            )

    async def send_alert_to_pagerduty(self, alert_message: str):
        """
        Send [High] Alert to PagerDuty

        https://developer.pagerduty.com/api-reference/YXBpOjI3NDgyNjU-pager-duty-v2-events-api
        """
        async_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        payload: PagerDutyRequestBody = PagerDutyRequestBody(
            payload=PagerDutyPayload(
                summary=alert_message, severity="critical", source="LiteLLM Alert"
            ),
            routing_key=self.api_key,
            event_action="trigger",
        )

        response = await async_client.post(
            url="https://events.pagerduty.com/v2/enqueue",
            json=dict(payload),
            headers={"Content-Type": "application/json"},
        )
        return response
