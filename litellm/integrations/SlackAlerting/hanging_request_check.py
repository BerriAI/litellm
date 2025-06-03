"""
Class to check for LLM API hanging requests


Notes:
- Do not create tasks that sleep, that can saturate the event loop
- Do not store large objects (eg. messages in memory) that can increase RAM usage
"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.utils import ProxyLogging

if TYPE_CHECKING:
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
else:
    SlackAlerting = Any


class HangingRequestData(BaseModel):
    request_id: str
    model: str
    api_base: Optional[str] = None
    key_alias: Optional[str] = None
    team_alias: Optional[str] = None
    alerting_metadata: Optional[dict] = None


HANGING_ALERT_BUFFER_TIME_SECONDS = 60


class AlertingHangingRequestCheck:
    """
    Class to safely handle checking hanging requests alerts
    """

    def __init__(
        self,
        alerting_threshold: float,
        slack_alerting_object: SlackAlerting,
    ):
        self.alerting_threshold = alerting_threshold
        self.hanging_request_cache = InMemoryCache(
            default_ttl=int(alerting_threshold + HANGING_ALERT_BUFFER_TIME_SECONDS),
        )
        self.slack_alerting_object = slack_alerting_object

    async def add_request_to_hanging_request_check(
        self,
        request_data: Optional[dict] = None,
    ):
        """
        Add a request to the hanging request cache. This is the list of request_ids that gets periodicall checked for hanging requests
        """
        if request_data is None:
            return

        request_metadata = request_data.get("metadata", {})
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
            ttl=int(self.alerting_threshold + HANGING_ALERT_BUFFER_TIME_SECONDS),
        )
        return

    async def check_for_hanging_requests(
        self,
        proxy_logging_object: ProxyLogging,
    ):
        """
        Background task that checks all request ids in self.hanging_request_cache to check if they have completed
        """

        #########################################################
        # Find all requests that have been hanging for more than the alerting threshold
        # Get the last 50 oldest items in the cache and check if they have completed
        #########################################################
        # check if request_id is in internal usage cache
        if proxy_logging_object.internal_usage_cache is None:
            return

        hanging_requests = await self.hanging_request_cache.async_get_oldest_n_keys(
            n=100,
        )

        for request_id in hanging_requests:
            request_data: Optional[HangingRequestData] = (
                await self.hanging_request_cache.async_get_cache(
                    key=request_id,
                )
            )

            if request_data is None:
                continue

            request_status = (
                await proxy_logging_object.internal_usage_cache.async_get_cache(
                    key="request_status:{}".format(request_data.request_id),
                    litellm_parent_otel_span=None,
                )
            )

            # this means the request status was either success or fail
            # and is not hanging
            if request_status is not None:
                continue
        pass

    async def send_hanging_request_alert(
        self,
        request_data: HangingRequestData,
    ):
        """
        Send a hanging request alert
        """
        from litellm.integrations.SlackAlerting.slack_alerting import AlertType

        ################
        # Send the Alert on Slack
        ################
        request_info = f"""
        Request Model: `{request_data.model}`\n 
        API Base: `{request_data.api_base}`\n
        Key Alias: `{request_data.key_alias}`\n
        Team Alias: `{request_data.team_alias}`\n
        """
        alerting_message = (
            f"`Requests are hanging - {self.alerting_threshold}s+ request time`"
        )
        await self.slack_alerting_object.send_alert(
            message=alerting_message + request_info,
            level="Medium",
            alert_type=AlertType.llm_requests_hanging,
            alerting_metadata=request_data.alerting_metadata or {},
        )
        pass
