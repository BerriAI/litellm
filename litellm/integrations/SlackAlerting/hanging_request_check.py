"""
Class to check for LLM API hanging requests


Notes:
- Do not create tasks that sleep, that can saturate the event loop
- Do not store large objects (eg. messages in memory) that can increase RAM usage
"""

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.types.integrations.slack_alerting import (
    HANGING_ALERT_BUFFER_TIME_SECONDS,
    MAX_OLDEST_HANGING_REQUESTS_TO_CHECK,
    HangingRequestData,
)

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
else:
    SlackAlerting = Any


class AlertingHangingRequestCheck:
    """
    Class to safely handle checking hanging requests alerts
    """

    def __init__(
        self,
        slack_alerting_object: SlackAlerting,
    ):
        self.slack_alerting_object = slack_alerting_object
        self.hanging_request_cache = InMemoryCache(
            default_ttl=self._hanging_request_cache_ttl_seconds(),
        )

    def _hanging_request_cache_ttl_seconds(self) -> int:
        """TTL for entries tracked by the hanging request checker.

        The background checker runs every `alerting_threshold / 2` seconds.
        With time-based gating (only alert once `alerting_threshold` has elapsed since
        registration), the *first* eligible check can occur up to ~1.5x the threshold
        after a request is registered (e.g. if it is registered just after a check).

        We keep entries long enough to ensure at least one post-threshold check occurs,
        plus a small buffer to avoid edge-case scheduling jitter.
        """

        alerting_threshold_seconds = float(
            self.slack_alerting_object.alerting_threshold
        )
        return int(
            math.ceil(
                alerting_threshold_seconds
                + (alerting_threshold_seconds / 2)
                + HANGING_ALERT_BUFFER_TIME_SECONDS
            )
        )

    async def add_request_to_hanging_request_check(
        self,
        request_data: Optional[dict] = None,
    ):
        """
        Add a request to the hanging request cache. This is the list of request_ids that gets periodicall checked for hanging requests
        """
        if request_data is None:
            return

        request_metadata = get_litellm_metadata_from_kwargs(kwargs=request_data)
        request_id = request_data.get("litellm_call_id")
        if not request_id:
            return
        model = request_data.get("model", "")
        api_base: Optional[str] = None

        # Best-effort API base resolution.
        # In some proxy paths, `deployment` may be absent and `litellm_params` contains
        # the routing params used to compute api_base.
        if request_data.get("deployment", None) is not None and isinstance(
            request_data["deployment"], dict
        ):
            api_base = litellm.get_api_base(
                model=model,
                optional_params=request_data["deployment"].get("litellm_params", {}),
            )
        elif isinstance(request_data.get("litellm_params"), dict):
            api_base = litellm.get_api_base(
                model=model,
                optional_params=request_data.get("litellm_params", {}),
            )
        elif isinstance(request_data.get("api_base"), str):
            api_base = request_data.get("api_base")

        alerting_metadata = request_metadata.get("alerting_metadata")
        if not isinstance(alerting_metadata, dict):
            alerting_metadata = None

        model_info = request_metadata.get("model_info")
        deployment_id: Optional[str] = None
        if isinstance(model_info, dict) and isinstance(model_info.get("id"), str):
            deployment_id = model_info.get("id")

        hanging_request_data = HangingRequestData(
            request_id=request_id,
            model=model,
            api_base=api_base,
            organization_id=request_metadata.get("user_api_key_org_id"),
            team_id=request_metadata.get("user_api_key_team_id"),
            deployment_id=deployment_id,
            key_alias=request_metadata.get("user_api_key_alias", ""),
            team_alias=request_metadata.get("user_api_key_team_alias", ""),
            alerting_metadata=alerting_metadata,
        )

        await self.hanging_request_cache.async_set_cache(
            key=request_id,
            value=hanging_request_data,
            ttl=self._hanging_request_cache_ttl_seconds(),
        )
        return

    async def send_alerts_for_hanging_requests(self):
        """
        Send alerts for hanging requests
        """
        #########################################################
        # Find all requests that have been hanging for more than the alerting threshold
        # Get the last N oldest items in the cache and check if they have completed
        #########################################################

        # Prefer the cache carried by the SlackAlerting instance.
        # This avoids importing `litellm.proxy.proxy_server` (proxy extras are
        # optional in many environments and importing the proxy server can have
        # heavyweight dependencies).
        internal_usage_cache = getattr(
            self.slack_alerting_object, "internal_usage_cache", None
        )
        if internal_usage_cache is None:
            return

        hanging_requests = await self.hanging_request_cache.async_get_oldest_n_keys(
            n=MAX_OLDEST_HANGING_REQUESTS_TO_CHECK,
        )

        hanging_request_ttl_seconds = self._hanging_request_cache_ttl_seconds()
        alerting_threshold_seconds = float(
            self.slack_alerting_object.alerting_threshold
        )

        for request_id in hanging_requests:
            hanging_request_data: Optional[
                HangingRequestData
            ] = await self.hanging_request_cache.async_get_cache(
                key=request_id,
            )

            if hanging_request_data is None:
                continue

            request_status = await self._get_request_status_from_internal_cache(
                internal_usage_cache=internal_usage_cache,
                request_id=hanging_request_data.request_id,
            )
            # this means the request status was either success or fail
            # and is not hanging
            if self._request_is_completed(request_status):
                # clear this request from hanging request cache since the request was either success or failed
                self.hanging_request_cache._remove_key(
                    key=request_id,
                )
                continue

            # Prevent errant alerts before alerting_threshold has elapsed.
            # NOTE: This class historically only checked for a missing request_status marker,
            # which could send "Requests are hanging" alerts for in-flight requests that
            # were still within the configured threshold window.
            time_since_start = time.time() - hanging_request_data.start_time
            if time_since_start < alerting_threshold_seconds:
                continue

            # Avoid repeated alerts for the same request.
            if hanging_request_data.alert_sent:
                continue

            # Reduce completion-race false positives: request may complete between the
            # first status check above and the Slack send below.
            request_status = await self._get_request_status_from_internal_cache(
                internal_usage_cache=internal_usage_cache,
                request_id=hanging_request_data.request_id,
            )
            if self._request_is_completed(request_status):
                self.hanging_request_cache._remove_key(
                    key=request_id,
                )
                continue

            ################
            # Send the Alert on Slack
            ################
            await self.send_hanging_request_alert(
                hanging_request_data=hanging_request_data,
                elapsed_seconds=time_since_start,
                threshold_seconds=alerting_threshold_seconds,
            )

            # Mark as alerted to avoid repeated messages for the same request_id.
            hanging_request_data.alert_sent = True

            # Persist updated object back into cache (do not rely on object reference semantics).
            await self.hanging_request_cache.async_set_cache(
                key=request_id,
                value=hanging_request_data,
                ttl=hanging_request_ttl_seconds,
            )

        return

    async def _get_request_status_from_internal_cache(
        self,
        internal_usage_cache: Any,
        request_id: str,
    ) -> Any:
        """Read request status from either DualCache or InternalUsageCache.

        In the proxy server, `SlackAlerting.internal_usage_cache` is usually a
        `DualCache` (parameter name: `parent_otel_span`).

        Historically, hanging alerts used `ProxyLogging.internal_usage_cache`
        (parameter name: `litellm_parent_otel_span`).

        We support both without importing the proxy server.
        """

        cache_key = "request_status:{}".format(request_id)
        try:
            return await internal_usage_cache.async_get_cache(
                key=cache_key,
                parent_otel_span=None,
                local_only=True,
            )
        except TypeError:
            return await internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )

    def _request_is_completed(self, request_status: Any) -> bool:
        """Return True if the cached request_status indicates completion.

        In the proxy, status is set as a string ('success' | 'fail').
        Some deployments may have historically stored a dict with a 'status' field.
        """

        if request_status in ("success", "fail"):
            return True
        if isinstance(request_status, dict):
            status_value = request_status.get("status")
            return status_value in ("success", "fail")
        return False

    async def check_for_hanging_requests(
        self,
    ):
        """
        Background task that checks all request ids in self.hanging_request_cache to check if they have completed

        Runs every alerting_threshold/2 seconds to check for hanging requests
        """
        while True:
            verbose_proxy_logger.debug("Checking for hanging requests....")
            await self.send_alerts_for_hanging_requests()
            await asyncio.sleep(self.slack_alerting_object.alerting_threshold / 2)

    async def send_hanging_request_alert(
        self,
        hanging_request_data: HangingRequestData,
        elapsed_seconds: Optional[float] = None,
        threshold_seconds: Optional[float] = None,
    ):
        """
        Send a hanging request alert
        """
        from litellm.integrations.SlackAlerting.slack_alerting import AlertType

        ################
        # Send the Alert on Slack
        ################
        elapsed_str = (
            f"{round(elapsed_seconds, 2)}s"
            if isinstance(elapsed_seconds, (int, float))
            else "unknown"
        )
        threshold_str = (
            f"{round(threshold_seconds, 2)}s"
            if isinstance(threshold_seconds, (int, float))
            else "unknown"
        )

        request_info = f"""Request ID: `{hanging_request_data.request_id}`
Request Model: `{hanging_request_data.model}`
Elapsed: `{elapsed_str}` (threshold: `{threshold_str}`)
API Base: `{hanging_request_data.api_base}`
Deployment ID: `{hanging_request_data.deployment_id}`
Organization ID: `{hanging_request_data.organization_id}`
Team ID: `{hanging_request_data.team_id}`
Key Alias: `{hanging_request_data.key_alias}`
Team Alias: `{hanging_request_data.team_alias}`"""

        alerting_message = f"`Requests are hanging - {self.slack_alerting_object.alerting_threshold}s+ request time`"
        await self.slack_alerting_object.send_alert(
            message=alerting_message + "\n" + request_info,
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata=hanging_request_data.alerting_metadata or {},
        )
