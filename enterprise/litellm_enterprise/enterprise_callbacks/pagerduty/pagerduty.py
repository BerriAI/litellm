"""
PagerDuty Alerting Integration

Handles two types of alerts:
- High LLM API Failure Rate. Configure X fails in Y seconds to trigger an alert.
- High Number of Hanging LLM Requests. Configure X hangs in Y seconds to trigger an alert.

Note: This is a Free feature on the regular litellm docker image.

However, this is under the enterprise license
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Union

from litellm._logging import verbose_logger
from litellm.caching import DualCache
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.pagerduty import (
    AlertingConfig,
    PagerDutyInternalEvent,
    PagerDutyPayload,
    PagerDutyRequestBody,
)
from litellm.types.utils import (
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
)

PAGERDUTY_DEFAULT_FAILURE_THRESHOLD = 60
PAGERDUTY_DEFAULT_FAILURE_THRESHOLD_WINDOW_SECONDS = 60
PAGERDUTY_DEFAULT_HANGING_THRESHOLD_SECONDS = 60
PAGERDUTY_DEFAULT_HANGING_THRESHOLD_WINDOW_SECONDS = 600


class PagerDutyAlerting(SlackAlerting):
    """
    Tracks failed requests and hanging requests separately.
    If threshold is crossed for either type, triggers a PagerDuty alert.
    """

    def __init__(
        self, alerting_args: Optional[Union[AlertingConfig, dict]] = None, **kwargs
    ):
        super().__init__()
        _api_key = os.getenv("PAGERDUTY_API_KEY")
        if not _api_key:
            raise ValueError("PAGERDUTY_API_KEY is not set")

        self.api_key: str = _api_key
        alerting_args = alerting_args or {}
        self.pagerduty_alerting_args: AlertingConfig = AlertingConfig(
            failure_threshold=alerting_args.get(
                "failure_threshold", PAGERDUTY_DEFAULT_FAILURE_THRESHOLD
            ),
            failure_threshold_window_seconds=alerting_args.get(
                "failure_threshold_window_seconds",
                PAGERDUTY_DEFAULT_FAILURE_THRESHOLD_WINDOW_SECONDS,
            ),
            hanging_threshold_seconds=alerting_args.get(
                "hanging_threshold_seconds", PAGERDUTY_DEFAULT_HANGING_THRESHOLD_SECONDS
            ),
            hanging_threshold_window_seconds=alerting_args.get(
                "hanging_threshold_window_seconds",
                PAGERDUTY_DEFAULT_HANGING_THRESHOLD_WINDOW_SECONDS,
            ),
        )

        # Separate storage for failures vs. hangs
        self._failure_events: List[PagerDutyInternalEvent] = []
        self._hanging_events: List[PagerDutyInternalEvent] = []

    # ------------------ MAIN LOGIC ------------------ #

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Record a failure event. Only send an alert to PagerDuty if the
        configured *failure* threshold is exceeded in the specified window.
        """
        now = datetime.now(timezone.utc)
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if not standard_logging_payload:
            raise ValueError(
                "standard_logging_object is required for PagerDutyAlerting"
            )

        # Extract error details
        error_info: Optional[StandardLoggingPayloadErrorInformation] = (
            standard_logging_payload.get("error_information") or {}
        )
        _meta = standard_logging_payload.get("metadata") or {}

        self._failure_events.append(
            PagerDutyInternalEvent(
                failure_event_type="failed_response",
                timestamp=now,
                error_class=error_info.get("error_class"),
                error_code=error_info.get("error_code"),
                error_llm_provider=error_info.get("llm_provider"),
                user_api_key_hash=_meta.get("user_api_key_hash"),
                user_api_key_alias=_meta.get("user_api_key_alias"),
                user_api_key_org_id=_meta.get("user_api_key_org_id"),
                user_api_key_team_id=_meta.get("user_api_key_team_id"),
                user_api_key_user_id=_meta.get("user_api_key_user_id"),
                user_api_key_team_alias=_meta.get("user_api_key_team_alias"),
                user_api_key_end_user_id=_meta.get("user_api_key_end_user_id"),
                user_api_key_user_email=_meta.get("user_api_key_user_email"),
                user_api_key_request_route=_meta.get("user_api_key_request_route"),
            )
        )

        # Prune + Possibly alert
        window_seconds = self.pagerduty_alerting_args.get(
            "failure_threshold_window_seconds", 60
        )
        threshold = self.pagerduty_alerting_args.get("failure_threshold", 1)

        # If threshold is crossed, send PD alert for failures
        await self._send_alert_if_thresholds_crossed(
            events=self._failure_events,
            window_seconds=window_seconds,
            threshold=threshold,
            alert_prefix="High LLM API Failure Rate",
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Example of detecting hanging requests by waiting a given threshold.
        If the request didn't finish by then, we treat it as 'hanging'.
        """
        verbose_logger.info("Inside Proxy Logging Pre-call hook!")
        asyncio.create_task(
            self.hanging_response_handler(
                request_data=data, user_api_key_dict=user_api_key_dict
            )
        )
        return None

    async def hanging_response_handler(
        self, request_data: Optional[dict], user_api_key_dict: UserAPIKeyAuth
    ):
        """
        Checks if request completed by the time 'hanging_threshold_seconds' elapses.
        If not, we classify it as a hanging request.
        """
        verbose_logger.debug(
            f"Inside Hanging Response Handler!..sleeping for {self.pagerduty_alerting_args.get('hanging_threshold_seconds', PAGERDUTY_DEFAULT_HANGING_THRESHOLD_SECONDS)} seconds"
        )
        await asyncio.sleep(
            self.pagerduty_alerting_args.get(
                "hanging_threshold_seconds", PAGERDUTY_DEFAULT_HANGING_THRESHOLD_SECONDS
            )
        )

        if await self._request_is_completed(request_data=request_data):
            return  # It's not hanging if completed

        # Otherwise, record it as hanging
        self._hanging_events.append(
            PagerDutyInternalEvent(
                failure_event_type="hanging_response",
                timestamp=datetime.now(timezone.utc),
                error_class="HangingRequest",
                error_code="HangingRequest",
                error_llm_provider="HangingRequest",
                user_api_key_hash=user_api_key_dict.api_key,
                user_api_key_alias=user_api_key_dict.key_alias,
                user_api_key_org_id=user_api_key_dict.org_id,
                user_api_key_team_id=user_api_key_dict.team_id,
                user_api_key_user_id=user_api_key_dict.user_id,
                user_api_key_team_alias=user_api_key_dict.team_alias,
                user_api_key_end_user_id=user_api_key_dict.end_user_id,
                user_api_key_user_email=user_api_key_dict.user_email,
                user_api_key_request_route=user_api_key_dict.request_route,
            )
        )

        # Prune + Possibly alert
        window_seconds = self.pagerduty_alerting_args.get(
            "hanging_threshold_window_seconds",
            PAGERDUTY_DEFAULT_HANGING_THRESHOLD_WINDOW_SECONDS,
        )
        threshold: int = self.pagerduty_alerting_args.get(
            "hanging_threshold_fails", PAGERDUTY_DEFAULT_HANGING_THRESHOLD_SECONDS
        )

        # If threshold is crossed, send PD alert for hangs
        await self._send_alert_if_thresholds_crossed(
            events=self._hanging_events,
            window_seconds=window_seconds,
            threshold=threshold,
            alert_prefix="High Number of Hanging LLM Requests",
        )

    # ------------------ HELPERS ------------------ #

    async def _send_alert_if_thresholds_crossed(
        self,
        events: List[PagerDutyInternalEvent],
        window_seconds: int,
        threshold: int,
        alert_prefix: str,
    ):
        """
        1. Prune old events
        2. If threshold is reached, build alert, send to PagerDuty
        3. Clear those events
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        pruned = [e for e in events if e.get("timestamp", datetime.min) > cutoff]

        # Update the reference list
        events.clear()
        events.extend(pruned)

        # Check threshold
        verbose_logger.debug(
            f"Have {len(events)} events in the last {window_seconds} seconds. Threshold is {threshold}"
        )
        if len(events) >= threshold:
            # Build short summary of last N events
            error_summaries = self._build_error_summaries(events, max_errors=5)
            alert_message = (
                f"{alert_prefix}: {len(events)} in the last {window_seconds} seconds."
            )
            custom_details = {"recent_errors": error_summaries}

            await self.send_alert_to_pagerduty(
                alert_message=alert_message,
                custom_details=custom_details,
            )

            # Clear them after sending an alert, so we don't spam
            events.clear()

    def _build_error_summaries(
        self, events: List[PagerDutyInternalEvent], max_errors: int = 5
    ) -> List[PagerDutyInternalEvent]:
        """
        Build short text summaries for the last `max_errors`.
        Example: "ValueError (code: 500, provider: openai)"
        """
        recent = events[-max_errors:]
        summaries = []
        for fe in recent:
            # If any of these is None, show "N/A" to avoid messing up the summary string
            fe.pop("timestamp")
            summaries.append(fe)
        return summaries

    async def send_alert_to_pagerduty(self, alert_message: str, custom_details: dict):
        """
        Send [critical] Alert to PagerDuty

        https://developer.pagerduty.com/api-reference/YXBpOjI3NDgyNjU-pager-duty-v2-events-api
        """
        try:
            verbose_logger.debug(f"Sending alert to PagerDuty: {alert_message}")
            async_client: AsyncHTTPHandler = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )
            payload: PagerDutyRequestBody = PagerDutyRequestBody(
                payload=PagerDutyPayload(
                    summary=alert_message,
                    severity="critical",
                    source="LiteLLM Alert",
                    component="LiteLLM",
                    custom_details=custom_details,
                ),
                routing_key=self.api_key,
                event_action="trigger",
            )

            return await async_client.post(
                url="https://events.pagerduty.com/v2/enqueue",
                json=dict(payload),
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            verbose_logger.exception(f"Error sending alert to PagerDuty: {e}")
