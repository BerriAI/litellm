"""
Class to check for LLM API hanging requests


Notes:
- Do not create tasks that sleep, that can saturate the event loop
- Do not store large objects (eg. messages in memory) that can increase RAM usage
"""

import asyncio
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
            default_ttl=int(
                self.slack_alerting_object.alerting_threshold
                + HANGING_ALERT_BUFFER_TIME_SECONDS
            ),
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
        model = request_data.get("model", "")
        api_base: Optional[str] = None

        if request_data.get("deployment", None) is not None and isinstance(
            request_data["deployment"], dict
        ):
            api_base = litellm.get_api_base(
                model=model,
                optional_params=request_data["deployment"].get("litellm_params", {}),
            )

        hanging_request_data = HangingRequestData(
            request_id=request_data.get("litellm_call_id", ""),
            model=model,
            api_base=api_base,
            key_alias=request_metadata.get("user_api_key_alias", ""),
            team_alias=request_metadata.get("user_api_key_team_alias", ""),
        )

        await self.hanging_request_cache.async_set_cache(
            key=hanging_request_data.request_id,
            value=hanging_request_data,
            ttl=int(
                self.slack_alerting_object.alerting_threshold
                + HANGING_ALERT_BUFFER_TIME_SECONDS
            ),
        )
        return

    async def send_alerts_for_hanging_requests(self):
        """
        Send alerts for hanging requests
        """
        from litellm.proxy.proxy_server import proxy_logging_obj

        #########################################################
        # Find all requests that have been hanging for more than the alerting threshold
        # Get the last 50 oldest items in the cache and check if they have completed
        #########################################################
        # check if request_id is in internal usage cache
        if proxy_logging_obj.internal_usage_cache is None:
            return

        hanging_requests = await self.hanging_request_cache.async_get_oldest_n_keys(
            n=MAX_OLDEST_HANGING_REQUESTS_TO_CHECK,
        )

        for request_id in hanging_requests:
            hanging_request_data: Optional[HangingRequestData] = (
                await self.hanging_request_cache.async_get_cache(
                    key=request_id,
                )
            )

            if hanging_request_data is None:
                continue

            request_status = (
                await proxy_logging_obj.internal_usage_cache.async_get_cache(
                    key="request_status:{}".format(hanging_request_data.request_id),
                    litellm_parent_otel_span=None,
                    local_only=True,
                )
            )
            # this means the request status was either success or fail
            # and is not hanging
            if request_status is not None:
                # clear this request from hanging request cache since the request was either success or failed
                self.hanging_request_cache._remove_key(
                    key=request_id,
                )
                continue

            ################
            # Send the Alert on Slack
            ################
            await self.send_hanging_request_alert(
                hanging_request_data=hanging_request_data
            )

        return

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
    ):
        """
        Send a hanging request alert
        """
        from litellm.integrations.SlackAlerting.slack_alerting import AlertType

        ################
        # Send the Alert on Slack
        ################
        request_info = f"""Request Model: `{hanging_request_data.model}`
API Base: `{hanging_request_data.api_base}`
Key Alias: `{hanging_request_data.key_alias}`
Team Alias: `{hanging_request_data.team_alias}`"""

        alerting_message = f"`Requests are hanging - {self.slack_alerting_object.alerting_threshold}s+ request time`"
        await self.slack_alerting_object.send_alert(
            message=alerting_message + "\n" + request_info,
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata=hanging_request_data.alerting_metadata or {},
        )
