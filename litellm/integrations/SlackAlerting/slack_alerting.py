#### What this does ####
#    Class for sending Slack Alerts #
import asyncio
import datetime
import os
import random
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from openai import APIError

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
import litellm.types
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.constants import HOURS_IN_A_DAY
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.SlackAlerting.budget_alert_types import get_budget_alert_type
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.exception_mapping_utils import (
    _add_key_name_and_team_to_alert,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import (
    AlertType,
    CallInfo,
    Litellm_EntityType,
    VirtualKeyEvent,
    WebhookEvent,
)
from litellm.types.integrations.slack_alerting import *

from ..email_templates.templates import *
from .batching_handler import send_to_webhook, squash_payloads
from .utils import _add_langfuse_trace_id_to_alert, process_slack_alerting_variables

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    Router = _Router
else:
    Router = Any


class SlackAlerting(CustomBatchLogger):
    """
    Class for sending Slack Alerts
    """

    # Class variables or attributes
    def __init__(
        self,
        internal_usage_cache: Optional[DualCache] = None,
        alerting_threshold: Optional[
            float
        ] = None,  # threshold for slow / hanging llm responses (in seconds)
        alerting: Optional[List] = [],
        alert_types: List[AlertType] = DEFAULT_ALERT_TYPES,
        alert_to_webhook_url: Optional[
            Dict[AlertType, Union[List[str], str]]
        ] = None,  # if user wants to separate alerts to diff channels
        alerting_args={},
        default_webhook_url: Optional[str] = None,
        **kwargs,
    ):
        if alerting_threshold is None:
            alerting_threshold = 300
        self.alerting_threshold = alerting_threshold
        self.alerting = alerting
        self.alert_types = alert_types
        self.internal_usage_cache = internal_usage_cache or DualCache()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.alert_to_webhook_url = process_slack_alerting_variables(
            alert_to_webhook_url=alert_to_webhook_url
        )
        self.is_running = False
        self.alerting_args = SlackAlertingArgs(**alerting_args)
        self.default_webhook_url = default_webhook_url
        self.flush_lock = asyncio.Lock()
        self.periodic_started = False
        super().__init__(**kwargs, flush_lock=self.flush_lock)

    def update_values(
        self,
        alerting: Optional[List] = None,
        alerting_threshold: Optional[float] = None,
        alert_types: Optional[List[AlertType]] = None,
        alert_to_webhook_url: Optional[Dict[AlertType, Union[List[str], str]]] = None,
        alerting_args: Optional[Dict] = None,
        llm_router: Optional[Router] = None,
    ):
        if alerting is not None:
            self.alerting = alerting
            asyncio.create_task(self.periodic_flush())
            self.periodic_started = True
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
        if alert_types is not None:
            self.alert_types = alert_types
        if alerting_args is not None:
            self.alerting_args = SlackAlertingArgs(**alerting_args)
            if not self.periodic_started: 
                asyncio.create_task(self.periodic_flush())
                self.periodic_started = True
                
        if alert_to_webhook_url is not None:
            # update the dict
            if self.alert_to_webhook_url is None:
                self.alert_to_webhook_url = process_slack_alerting_variables(
                    alert_to_webhook_url=alert_to_webhook_url
                )
            else:
                _new_values = (
                    process_slack_alerting_variables(
                        alert_to_webhook_url=alert_to_webhook_url
                    )
                    or {}
                )
                self.alert_to_webhook_url.update(_new_values)
        if llm_router is not None:
            self.llm_router = llm_router

    async def deployment_in_cooldown(self):
        pass

    async def deployment_removed_from_cooldown(self):
        pass

    def _all_possible_alert_types(self):
        # used by the UI to show all supported alert types
        # Note: This is not the alerts the user has configured, instead it's all possible alert types a user can select
        # return list of all values AlertType enum
        return list(AlertType)

    def _response_taking_too_long_callback_helper(
        self,
        kwargs,  # kwargs to completion
        start_time,
        end_time,  # start/end time
    ):
        try:
            time_difference = end_time - start_time
            # Convert the timedelta to float (in seconds)
            time_difference_float = time_difference.total_seconds()
            litellm_params = kwargs.get("litellm_params", {})
            model = kwargs.get("model", "")
            api_base = litellm.get_api_base(model=model, optional_params=litellm_params)
            messages = kwargs.get("messages", None)
            # if messages does not exist fallback to "input"
            if messages is None:
                messages = kwargs.get("input", None)

            # only use first 100 chars for alerting
            _messages = str(messages)[:100]

            return time_difference_float, model, api_base, _messages
        except Exception as e:
            raise e

    def _get_deployment_latencies_to_alert(self, metadata=None):
        if metadata is None:
            return None

        if "_latency_per_deployment" in metadata:
            # Translate model_id to -> api_base
            # _latency_per_deployment is a dictionary that looks like this:
            """
            _latency_per_deployment: {
                api_base: 0.01336697916666667
            }
            """
            _message_to_send = ""
            _deployment_latencies = metadata["_latency_per_deployment"]
            if len(_deployment_latencies) == 0:
                return None
            _deployment_latency_map: Optional[dict] = None
            try:
                # try sorting deployments by latency
                _deployment_latencies = sorted(
                    _deployment_latencies.items(), key=lambda x: x[1]
                )
                _deployment_latency_map = dict(_deployment_latencies)
            except Exception:
                pass

            if _deployment_latency_map is None:
                return

            for api_base, latency in _deployment_latency_map.items():
                _message_to_send += f"\n{api_base}: {round(latency,2)}s"
            _message_to_send = "```" + _message_to_send + "```"
            return _message_to_send

    async def response_taking_too_long_callback(
        self,
        kwargs,  # kwargs to completion
        completion_response,  # response from completion
        start_time,
        end_time,  # start/end time
    ):
        if self.alerting is None or self.alert_types is None:
            return

        (
            time_difference_float,
            model,
            api_base,
            messages,
        ) = self._response_taking_too_long_callback_helper(
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
        )
        if litellm.turn_off_message_logging or litellm.redact_messages_in_exceptions:
            messages = "Message not logged. litellm.redact_messages_in_exceptions=True"
        request_info = f"\nRequest Model: `{model}`\nAPI Base: `{api_base}`\nMessages: `{messages}`"
        slow_message = f"`Responses are slow - {round(time_difference_float,2)}s response time > Alerting threshold: {self.alerting_threshold}s`"
        alerting_metadata: dict = {}
        if time_difference_float > self.alerting_threshold:
            # add deployment latencies to alert
            if (
                kwargs is not None
                and "litellm_params" in kwargs
                and "metadata" in kwargs["litellm_params"]
            ):
                _metadata: dict = kwargs["litellm_params"]["metadata"]
                request_info = _add_key_name_and_team_to_alert(
                    request_info=request_info, metadata=_metadata
                )

                _deployment_latency_map = self._get_deployment_latencies_to_alert(
                    metadata=_metadata
                )
                if _deployment_latency_map is not None:
                    request_info += (
                        f"\nAvailable Deployment Latencies\n{_deployment_latency_map}"
                    )

                if "alerting_metadata" in _metadata:
                    alerting_metadata = _metadata["alerting_metadata"]
            await self.send_alert(
                message=slow_message + request_info,
                level="Low",
                alert_type=AlertType.llm_too_slow,
                alerting_metadata=alerting_metadata,
            )

    async def async_update_daily_reports(
        self, deployment_metrics: DeploymentMetrics
    ) -> int:
        """
        Store the perf by deployment in cache
        - Number of failed requests per deployment
        - Latency / output tokens per deployment

        'deployment_id:daily_metrics:failed_requests'
        'deployment_id:daily_metrics:latency_per_output_token'

        Returns
            int - count of metrics set (1 - if just latency, 2 - if failed + latency)
        """

        return_val = 0
        try:
            ## FAILED REQUESTS ##
            if deployment_metrics.failed_request:
                await self.internal_usage_cache.async_increment_cache(
                    key="{}:{}".format(
                        deployment_metrics.id,
                        SlackAlertingCacheKeys.failed_requests_key.value,
                    ),
                    value=1,
                    parent_otel_span=None,  # no attached request, this is a background operation
                )

                return_val += 1

            ## LATENCY ##
            if deployment_metrics.latency_per_output_token is not None:
                await self.internal_usage_cache.async_increment_cache(
                    key="{}:{}".format(
                        deployment_metrics.id, SlackAlertingCacheKeys.latency_key.value
                    ),
                    value=deployment_metrics.latency_per_output_token,
                    parent_otel_span=None,  # no attached request, this is a background operation
                )

                return_val += 1

            return return_val
        except Exception:
            return 0

    async def send_daily_reports(self, router) -> bool:  # noqa: PLR0915
        """
        Send a daily report on:
        - Top 5 deployments with most failed requests
        - Top 5 slowest deployments (normalized by latency/output tokens)

        Get the value from redis cache (if available) or in-memory and send it

        Cleanup:
        - reset values in cache -> prevent memory leak

        Returns:
            True -> if successfuly sent
            False -> if not sent
        """

        ids = router.get_model_ids()

        # get keys
        failed_request_keys = [
            "{}:{}".format(id, SlackAlertingCacheKeys.failed_requests_key.value)
            for id in ids
        ]
        latency_keys = [
            "{}:{}".format(id, SlackAlertingCacheKeys.latency_key.value) for id in ids
        ]

        combined_metrics_keys = failed_request_keys + latency_keys  # reduce cache calls

        combined_metrics_values = await self.internal_usage_cache.async_batch_get_cache(
            keys=combined_metrics_keys
        )  # [1, 2, None, ..]

        if combined_metrics_values is None:
            return False

        all_none = True
        for val in combined_metrics_values:
            if val is not None and val > 0:
                all_none = False
                break

        if all_none:
            return False

        failed_request_values = combined_metrics_values[
            : len(failed_request_keys)
        ]  # # [1, 2, None, ..]
        latency_values = combined_metrics_values[len(failed_request_keys) :]

        # find top 5 failed
        ## Replace None values with a placeholder value (-1 in this case)
        placeholder_value = 0
        replaced_failed_values = [
            value if value is not None else placeholder_value
            for value in failed_request_values
        ]

        ## Get the indices of top 5 keys with the highest numerical values (ignoring None and 0 values)
        top_5_failed = sorted(
            range(len(replaced_failed_values)),
            key=lambda i: replaced_failed_values[i],
            reverse=True,
        )[:5]
        top_5_failed = [
            index for index in top_5_failed if replaced_failed_values[index] > 0
        ]

        # find top 5 slowest
        # Replace None values with a placeholder value (-1 in this case)
        placeholder_value = 0
        replaced_slowest_values = [
            value if value is not None else placeholder_value
            for value in latency_values
        ]

        # Get the indices of top 5 values with the highest numerical values (ignoring None and 0 values)
        top_5_slowest = sorted(
            range(len(replaced_slowest_values)),
            key=lambda i: replaced_slowest_values[i],
            reverse=True,
        )[:5]
        top_5_slowest = [
            index for index in top_5_slowest if replaced_slowest_values[index] > 0
        ]

        # format alert -> return the litellm model name + api base
        message = f"\n\nTime: `{time.time()}`s\nHere are today's key metrics 📈: \n\n"

        message += "\n\n*❗️ Top Deployments with Most Failed Requests:*\n\n"
        if not top_5_failed:
            message += "\tNone\n"
        for i in range(len(top_5_failed)):
            key = failed_request_keys[top_5_failed[i]].split(":")[0]
            _deployment = router.get_model_info(key)
            if isinstance(_deployment, dict):
                deployment_name = _deployment["litellm_params"].get("model", "")
            else:
                return False

            api_base = litellm.get_api_base(
                model=deployment_name,
                optional_params=(
                    _deployment["litellm_params"] if _deployment is not None else {}
                ),
            )
            if api_base is None:
                api_base = ""
            value = replaced_failed_values[top_5_failed[i]]
            message += f"\t{i+1}. Deployment: `{deployment_name}`, Failed Requests: `{value}`,  API Base: `{api_base}`\n"

        message += "\n\n*😅 Top Slowest Deployments:*\n\n"
        if not top_5_slowest:
            message += "\tNone\n"
        for i in range(len(top_5_slowest)):
            key = latency_keys[top_5_slowest[i]].split(":")[0]
            _deployment = router.get_model_info(key)
            if _deployment is not None:
                deployment_name = _deployment["litellm_params"].get("model", "")
            else:
                deployment_name = ""
            api_base = litellm.get_api_base(
                model=deployment_name,
                optional_params=(
                    _deployment["litellm_params"] if _deployment is not None else {}
                ),
            )
            value = round(replaced_slowest_values[top_5_slowest[i]], 3)
            message += f"\t{i+1}. Deployment: `{deployment_name}`, Latency per output token: `{value}s/token`,  API Base: `{api_base}`\n\n"

        # cache cleanup -> reset values to 0
        latency_cache_keys = [(key, 0) for key in latency_keys]
        failed_request_cache_keys = [(key, 0) for key in failed_request_keys]
        combined_metrics_cache_keys = latency_cache_keys + failed_request_cache_keys
        await self.internal_usage_cache.async_set_cache_pipeline(
            cache_list=combined_metrics_cache_keys
        )

        message += f"\n\nNext Run is at: `{time.time() + self.alerting_args.daily_report_frequency}`s"

        # send alert
        await self.send_alert(
            message=message,
            level="Low",
            alert_type=AlertType.daily_reports,
            alerting_metadata={},
        )

        return True

    async def response_taking_too_long(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        type: Literal["hanging_request", "slow_response"] = "hanging_request",
        request_data: Optional[dict] = None,
    ):
        if self.alerting is None or self.alert_types is None:
            return
        model: str = ""
        if request_data is not None:
            model = request_data.get("model", "")
            messages = request_data.get("messages", None)
            if messages is None:
                # if messages does not exist fallback to "input"
                messages = request_data.get("input", None)

            # try casting messages to str and get the first 100 characters, else mark as None
            try:
                messages = str(messages)
                messages = messages[:100]
            except Exception:
                messages = ""

            if (
                litellm.turn_off_message_logging
                or litellm.redact_messages_in_exceptions
            ):
                messages = (
                    "Message not logged. litellm.redact_messages_in_exceptions=True"
                )
            request_info = f"\nRequest Model: `{model}`\nMessages: `{messages}`"
        else:
            request_info = ""

        if type == "hanging_request":
            await asyncio.sleep(
                self.alerting_threshold
            )  # Set it to 5 minutes - i'd imagine this might be different for streaming, non-streaming, non-completion (embedding + img) requests
            alerting_metadata: dict = {}
            if await self._request_is_completed(request_data=request_data) is True:
                return

            if request_data is not None:
                if request_data.get("deployment", None) is not None and isinstance(
                    request_data["deployment"], dict
                ):
                    _api_base = litellm.get_api_base(
                        model=model,
                        optional_params=request_data["deployment"].get(
                            "litellm_params", {}
                        ),
                    )

                    if _api_base is None:
                        _api_base = ""

                    request_info += f"\nAPI Base: {_api_base}"
                elif request_data.get("metadata", None) is not None and isinstance(
                    request_data["metadata"], dict
                ):
                    # In hanging requests sometime it has not made it to the point where the deployment is passed to the `request_data``
                    # in that case we fallback to the api base set in the request metadata
                    _metadata: dict = request_data["metadata"]
                    _api_base = _metadata.get("api_base", "")

                    request_info = _add_key_name_and_team_to_alert(
                        request_info=request_info, metadata=_metadata
                    )

                    if _api_base is None:
                        _api_base = ""

                    if "alerting_metadata" in _metadata:
                        alerting_metadata = _metadata["alerting_metadata"]
                    request_info += f"\nAPI Base: `{_api_base}`"
                # only alert hanging responses if they have not been marked as success
                alerting_message = (
                    f"`Requests are hanging - {self.alerting_threshold}s+ request time`"
                )

                if "langfuse" in litellm.success_callback:
                    langfuse_url = await _add_langfuse_trace_id_to_alert(
                        request_data=request_data,
                    )

                    if langfuse_url is not None:
                        request_info += "\n🪢 Langfuse Trace: {}".format(langfuse_url)

                # add deployment latencies to alert
                _deployment_latency_map = self._get_deployment_latencies_to_alert(
                    metadata=request_data.get("metadata", {})
                )
                if _deployment_latency_map is not None:
                    request_info += f"\nDeployment Latencies\n{_deployment_latency_map}"

                await self.send_alert(
                    message=alerting_message + request_info,
                    level="Medium",
                    alert_type=AlertType.llm_requests_hanging,
                    alerting_metadata=alerting_metadata,
                )

    async def failed_tracking_alert(self, error_message: str, failing_model: str):
        """
        Raise alert when tracking failed for specific model

        Args:
            error_message (str): Error message
            failing_model (str): Model that failed tracking
        """
        if self.alerting is None or self.alert_types is None:
            # do nothing if alerting is not switched on
            return
        if "failed_tracking_spend" not in self.alert_types:
            return

        _cache: DualCache = self.internal_usage_cache
        message = "Failed Tracking Cost for " + error_message
        _cache_key = "budget_alerts:failed_tracking:{}".format(failing_model)
        result = await _cache.async_get_cache(key=_cache_key)
        if result is None:
            await self.send_alert(
                message=message,
                level="High",
                alert_type=AlertType.failed_tracking_spend,
                alerting_metadata={},
            )
            await _cache.async_set_cache(
                key=_cache_key,
                value="SENT",
                ttl=self.alerting_args.budget_alert_ttl,
            )

    async def budget_alerts(
        self,
        type: Literal[
            "token_budget",
            "soft_budget",
            "user_budget",
            "team_budget",
            "proxy_budget",
            "projected_limit_exceeded",
        ],
        user_info: CallInfo,
    ):
        """
        Send a budget alert on slack or webhook

        Args:
            type: The type of budget alert to send
            user_info: The user info to send the alert for
        """
        ## PREVENTITIVE ALERTING ## - https://github.com/BerriAI/litellm/issues/2727
        # - Alert once within 24hr period
        # - Cache this information
        # - Don't re-alert, if alert already sent
        _cache: DualCache = self.internal_usage_cache

        if self.alerting is None or self.alert_types is None:
            # do nothing if alerting is not switched on
            return
        if "budget_alerts" not in self.alert_types:
            return

        # Get the appropriate budget alert type handler
        budget_alert_class = get_budget_alert_type(type)
        _id = budget_alert_class.get_id(user_info)
        user_info_json = user_info.model_dump(exclude_none=True)
        user_info_str = self._get_user_info_str(user_info)
        event_message = budget_alert_class.get_event_message()

        # Set default event unless we're in projected_limit_exceeded
        event: Optional[
            Literal[
                "budget_crossed",
                "threshold_crossed",
                "projected_limit_exceeded",
                "soft_budget_crossed",
            ]
        ] = (
            "projected_limit_exceeded" if type == "projected_limit_exceeded" else None
        )

        webhook_event: Optional[WebhookEvent] = None

        # percent of max_budget left to spend
        if user_info.max_budget is None and user_info.soft_budget is None:
            return

        # check if crossed budget
        event, event_message = self._get_event_and_event_message(
            event=event,
            user_info=user_info,
            event_message=event_message,
        )

        # send alert
        if event is not None and user_info.event_group is not None:
            _cache_key = "budget_alerts:{}:{}".format(event, _id)
            result = await _cache.async_get_cache(key=_cache_key)
            if result is None:
                webhook_event = WebhookEvent(
                    event=event,
                    event_message=event_message,
                    **user_info_json,
                )
                await self.send_alert(
                    message=event_message + "\n\n" + user_info_str,
                    level="High",
                    alert_type=AlertType.budget_alerts,
                    user_info=webhook_event,
                    alerting_metadata={},
                )
                await _cache.async_set_cache(
                    key=_cache_key,
                    value="SENT",
                    ttl=self.alerting_args.budget_alert_ttl,
                )

            return
        return

    def _get_event_and_event_message(
        self,
        user_info: CallInfo,
        event: Optional[
            Literal[
                "budget_crossed",
                "threshold_crossed",
                "soft_budget_crossed",
                "projected_limit_exceeded",
            ]
        ],
        event_message: str,
    ) -> Tuple[
        Optional[
            Literal[
                "budget_crossed",
                "threshold_crossed",
                "soft_budget_crossed",
                "projected_limit_exceeded",
            ]
        ],
        str,
    ]:
        """
        Get the event and event message for a budget alert

        This will append any new information to the event_message

        Handles Max Budget and Soft Budget Alerts
        """
        percent_left: float = self._get_percent_of_max_budget_left(user_info=user_info)

        #####################################################################
        # SOFT BUDGET CHECK
        # Check if the key/team/user has a soft budget set and they have crossed it
        #####################################################################
        if user_info.soft_budget is not None:
            if user_info.spend >= user_info.soft_budget:
                event = "soft_budget_crossed"
                event_message += f"Total Soft Budget:`{user_info.soft_budget}`"

        #####################################################################
        # MAX BUDGET CHECK
        # Check if the key/team/user has a max budget set and they have either
        ## a. Crossed their max budget
        ## b. Either 5% or 15% of their max budget is left
        #####################################################################
        if user_info.max_budget is not None:
            if user_info.spend >= user_info.max_budget:
                event = "budget_crossed"
                event_message += (
                    f"Budget Crossed\n Total Budget:`{user_info.max_budget}`"
                )
            elif percent_left <= SLACK_ALERTING_THRESHOLD_5_PERCENT:
                event = "threshold_crossed"
                event_message += "5% Threshold Crossed "
            elif percent_left <= SLACK_ALERTING_THRESHOLD_15_PERCENT:
                event = "threshold_crossed"
                event_message += "15% Threshold Crossed"

        return event, event_message

    def _get_percent_of_max_budget_left(self, user_info: CallInfo) -> float:
        """
        Get the percent of the max budget that is left
        """
        percent_left: float = 0.0
        current_spend: float = user_info.spend
        max_budget: Optional[float] = user_info.max_budget
        if max_budget is None:
            return percent_left
        if max_budget <= 0:
            return percent_left
        percent_left = (max_budget - current_spend) / max_budget
        return percent_left

    def _get_user_info_str(self, user_info: CallInfo) -> str:
        """
        Create a standard message for a budget alert
        """
        _all_fields_as_dict = user_info.model_dump(exclude_none=True)
        _all_fields_as_dict.pop("token")
        msg = ""
        for k, v in _all_fields_as_dict.items():
            if isinstance(v, Litellm_EntityType):
                v = v.value
            msg += f"*{k}:* `{v}`\n"

        return msg

    async def customer_spend_alert(
        self,
        token: Optional[str],
        key_alias: Optional[str],
        end_user_id: Optional[str],
        response_cost: Optional[float],
        max_budget: Optional[float],
    ):
        if (
            self.alerting is not None
            and "webhook" in self.alerting
            and end_user_id is not None
            and token is not None
            and response_cost is not None
        ):
            # log customer spend
            event = WebhookEvent(
                spend=response_cost,
                max_budget=max_budget,
                token=token,
                customer_id=end_user_id,
                user_id=None,
                team_id=None,
                user_email=None,
                key_alias=key_alias,
                projected_exceeded_date=None,
                projected_spend=None,
                event="spend_tracked",
                event_group=Litellm_EntityType.END_USER,
                event_message="Customer spend tracked. Customer={}, spend={}".format(
                    end_user_id, response_cost
                ),
            )

            await self.send_webhook_alert(webhook_event=event)

    def _count_outage_alerts(self, alerts: List[int]) -> str:
        """
        Parameters:
        - alerts: List[int] -> list of error codes (either 408 or 500+)

        Returns:
        - str -> formatted string. This is an alert message, giving a human-friendly description of the errors.
        """
        error_breakdown = {"Timeout Errors": 0, "API Errors": 0, "Unknown Errors": 0}
        for alert in alerts:
            if alert == 408:
                error_breakdown["Timeout Errors"] += 1
            elif alert >= 500:
                error_breakdown["API Errors"] += 1
            else:
                error_breakdown["Unknown Errors"] += 1

        error_msg = ""
        for key, value in error_breakdown.items():
            if value > 0:
                error_msg += "\n{}: {}\n".format(key, value)

        return error_msg

    def _outage_alert_msg_factory(
        self,
        alert_type: Literal["Major", "Minor"],
        key: Literal["Model", "Region"],
        key_val: str,
        provider: str,
        api_base: Optional[str],
        outage_value: BaseOutageModel,
    ) -> str:
        """Format an alert message for slack"""
        headers = {f"{key} Name": key_val, "Provider": provider}
        if api_base is not None:
            headers["API Base"] = api_base  # type: ignore

        headers_str = "\n"
        for k, v in headers.items():
            headers_str += f"*{k}:* `{v}`\n"
        return f"""\n\n
*⚠️ {alert_type} Service Outage*

{headers_str}

*Errors:*
{self._count_outage_alerts(alerts=outage_value["alerts"])}

*Last Check:* `{round(time.time() - outage_value["last_updated_at"], 4)}s ago`\n\n
"""

    async def region_outage_alerts(
        self,
        exception: APIError,
        deployment_id: str,
    ) -> None:
        """
        Send slack alert if specific provider region is having an outage.

        Track for 408 (Timeout) and >=500 Error codes
        """
        ## CREATE (PROVIDER+REGION) ID ##
        if self.llm_router is None:
            return

        deployment = self.llm_router.get_deployment(model_id=deployment_id)

        if deployment is None:
            return

        model = deployment.litellm_params.model
        ### GET PROVIDER ###
        provider = deployment.litellm_params.custom_llm_provider
        if provider is None:
            model, provider, _, _ = litellm.get_llm_provider(model=model)

        ### GET REGION ###
        region_name = deployment.litellm_params.region_name
        if region_name is None:
            region_name = litellm.utils._get_model_region(
                custom_llm_provider=provider, litellm_params=deployment.litellm_params
            )

        if region_name is None:
            return

        ### UNIQUE CACHE KEY ###
        cache_key = provider + region_name

        outage_value: Optional[ProviderRegionOutageModel] = (
            await self.internal_usage_cache.async_get_cache(key=cache_key)
        )

        if (
            getattr(exception, "status_code", None) is None
            or (
                exception.status_code != 408  # type: ignore
                and exception.status_code < 500  # type: ignore
            )
            or self.llm_router is None
        ):
            return

        if outage_value is None:
            _deployment_set = set()
            _deployment_set.add(deployment_id)
            outage_value = ProviderRegionOutageModel(
                provider_region_id=cache_key,
                alerts=[exception.status_code],  # type: ignore
                minor_alert_sent=False,
                major_alert_sent=False,
                last_updated_at=time.time(),
                deployment_ids=_deployment_set,
            )

            ## add to cache ##
            await self.internal_usage_cache.async_set_cache(
                key=cache_key,
                value=outage_value,
                ttl=self.alerting_args.region_outage_alert_ttl,
            )
            return

        if len(outage_value["alerts"]) < self.alerting_args.max_outage_alert_list_size:
            outage_value["alerts"].append(exception.status_code)  # type: ignore
        else:  # prevent memory leaks
            pass
        _deployment_set = outage_value["deployment_ids"]
        _deployment_set.add(deployment_id)
        outage_value["deployment_ids"] = _deployment_set
        outage_value["last_updated_at"] = time.time()

        ## MINOR OUTAGE ALERT SENT ##
        if (
            outage_value["minor_alert_sent"] is False
            and len(outage_value["alerts"])
            >= self.alerting_args.minor_outage_alert_threshold
            and len(_deployment_set) > 1  # make sure it's not just 1 bad deployment
        ):
            msg = self._outage_alert_msg_factory(
                alert_type="Minor",
                key="Region",
                key_val=region_name,
                api_base=None,
                outage_value=outage_value,
                provider=provider,
            )
            # send minor alert
            await self.send_alert(
                message=msg,
                level="Medium",
                alert_type=AlertType.outage_alerts,
                alerting_metadata={},
            )
            # set to true
            outage_value["minor_alert_sent"] = True

        ## MAJOR OUTAGE ALERT SENT ##
        elif (
            outage_value["major_alert_sent"] is False
            and len(outage_value["alerts"])
            >= self.alerting_args.major_outage_alert_threshold
            and len(_deployment_set) > 1  # make sure it's not just 1 bad deployment
        ):
            msg = self._outage_alert_msg_factory(
                alert_type="Major",
                key="Region",
                key_val=region_name,
                api_base=None,
                outage_value=outage_value,
                provider=provider,
            )

            # send minor alert
            await self.send_alert(
                message=msg,
                level="High",
                alert_type=AlertType.outage_alerts,
                alerting_metadata={},
            )
            # set to true
            outage_value["major_alert_sent"] = True

        ## update cache ##
        await self.internal_usage_cache.async_set_cache(
            key=cache_key, value=outage_value
        )

    async def outage_alerts(
        self,
        exception: APIError,
        deployment_id: str,
    ) -> None:
        """
        Send slack alert if model is badly configured / having an outage (408, 401, 429, >=500).

        key = model_id

        value = {
        - model_id
        - threshold
        - alerts []
        }

        ttl = 1hr
        max_alerts_size = 10
        """
        try:
            outage_value: Optional[OutageModel] = await self.internal_usage_cache.async_get_cache(key=deployment_id)  # type: ignore
            if (
                getattr(exception, "status_code", None) is None
                or (
                    exception.status_code != 408  # type: ignore
                    and exception.status_code < 500  # type: ignore
                )
                or self.llm_router is None
            ):
                return

            ### EXTRACT MODEL DETAILS ###
            deployment = self.llm_router.get_deployment(model_id=deployment_id)
            if deployment is None:
                return

            model = deployment.litellm_params.model
            provider = deployment.litellm_params.custom_llm_provider
            if provider is None:
                try:
                    model, provider, _, _ = litellm.get_llm_provider(model=model)
                except Exception:
                    provider = ""
            api_base = litellm.get_api_base(
                model=model, optional_params=deployment.litellm_params
            )

            if outage_value is None:
                outage_value = OutageModel(
                    model_id=deployment_id,
                    alerts=[exception.status_code],  # type: ignore
                    minor_alert_sent=False,
                    major_alert_sent=False,
                    last_updated_at=time.time(),
                )

                ## add to cache ##
                await self.internal_usage_cache.async_set_cache(
                    key=deployment_id,
                    value=outage_value,
                    ttl=self.alerting_args.outage_alert_ttl,
                )
                return

            if (
                len(outage_value["alerts"])
                < self.alerting_args.max_outage_alert_list_size
            ):
                outage_value["alerts"].append(exception.status_code)  # type: ignore
            else:  # prevent memory leaks
                pass

            outage_value["last_updated_at"] = time.time()

            ## MINOR OUTAGE ALERT SENT ##
            if (
                outage_value["minor_alert_sent"] is False
                and len(outage_value["alerts"])
                >= self.alerting_args.minor_outage_alert_threshold
            ):
                msg = self._outage_alert_msg_factory(
                    alert_type="Minor",
                    key="Model",
                    key_val=model,
                    api_base=api_base,
                    outage_value=outage_value,
                    provider=provider,
                )
                # send minor alert
                await self.send_alert(
                    message=msg,
                    level="Medium",
                    alert_type=AlertType.outage_alerts,
                    alerting_metadata={},
                )
                # set to true
                outage_value["minor_alert_sent"] = True
            elif (
                outage_value["major_alert_sent"] is False
                and len(outage_value["alerts"])
                >= self.alerting_args.major_outage_alert_threshold
            ):
                msg = self._outage_alert_msg_factory(
                    alert_type="Major",
                    key="Model",
                    key_val=model,
                    api_base=api_base,
                    outage_value=outage_value,
                    provider=provider,
                )
                # send minor alert
                await self.send_alert(
                    message=msg,
                    level="High",
                    alert_type=AlertType.outage_alerts,
                    alerting_metadata={},
                )
                # set to true
                outage_value["major_alert_sent"] = True

            ## update cache ##
            await self.internal_usage_cache.async_set_cache(
                key=deployment_id, value=outage_value
            )
        except Exception:
            pass

    async def model_added_alert(
        self, model_name: str, litellm_model_name: str, passed_model_info: Any
    ):
        base_model_from_user = getattr(passed_model_info, "base_model", None)
        model_info = {}
        base_model = ""
        if base_model_from_user is not None:
            model_info = litellm.model_cost.get(base_model_from_user, {})
            base_model = f"Base Model: `{base_model_from_user}`\n"
        else:
            model_info = litellm.model_cost.get(litellm_model_name, {})
        model_info_str = ""
        for k, v in model_info.items():
            if k == "input_cost_per_token" or k == "output_cost_per_token":
                # when converting to string it should not be 1.63e-06
                v = "{:.8f}".format(v)

            model_info_str += f"{k}: {v}\n"

        message = f"""
*🚅 New Model Added*
Model Name: `{model_name}`
{base_model}

Usage OpenAI Python SDK:
```
import openai
client = openai.OpenAI(
    api_key="your_api_key",
    base_url={os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")}
)

response = client.chat.completions.create(
    model="{model_name}", # model to send to the proxy
    messages = [
        {{
            "role": "user",
            "content": "this is a test request, write a short poem"
        }}
    ]
)
```

Model Info: 
```
{model_info_str}
```
"""

        alert_val = self.send_alert(
            message=message,
            level="Low",
            alert_type=AlertType.new_model_added,
            alerting_metadata={},
        )

        if alert_val is not None and asyncio.iscoroutine(alert_val):
            await alert_val

    async def model_removed_alert(self, model_name: str):
        pass

    async def send_webhook_alert(self, webhook_event: WebhookEvent) -> bool:
        """
        Sends structured alert to webhook, if set.

        Currently only implemented for budget alerts

        Returns -> True if sent, False if not.

        Raises Exception
            - if WEBHOOK_URL is not set
        """

        webhook_url = os.getenv("WEBHOOK_URL", None)
        if webhook_url is None:
            raise Exception("Missing webhook_url from environment")

        payload = webhook_event.model_dump_json()
        headers = {"Content-type": "application/json"}

        response = await self.async_http_handler.post(
            url=webhook_url,
            headers=headers,
            data=payload,
        )
        if response.status_code == 200:
            return True
        else:
            print("Error sending webhook alert. Error=", response.text)  # noqa

        return False

    async def _check_if_using_premium_email_feature(
        self,
        premium_user: bool,
        email_logo_url: Optional[str] = None,
        email_support_contact: Optional[str] = None,
    ):
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            if email_logo_url is not None or email_support_contact is not None:
                raise ValueError(
                    f"Trying to Customize Email Alerting\n {CommonProxyErrors.not_premium_user.value}"
                )
        return

    async def send_key_created_or_user_invited_email(
        self, webhook_event: WebhookEvent
    ) -> bool:
        try:
            from litellm.proxy.utils import send_email

            if self.alerting is None or "email" not in self.alerting:
                # do nothing if user does not want email alerts
                verbose_proxy_logger.error(
                    "Error sending email alert - 'email' not in self.alerting %s",
                    self.alerting,
                )
                return False
            from litellm.proxy.proxy_server import premium_user, prisma_client

            email_logo_url = os.getenv(
                "SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None)
            )
            email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)
            await self._check_if_using_premium_email_feature(
                premium_user, email_logo_url, email_support_contact
            )
            if email_logo_url is None:
                email_logo_url = LITELLM_LOGO_URL
            if email_support_contact is None:
                email_support_contact = LITELLM_SUPPORT_CONTACT

            event_name = webhook_event.event_message
            recipient_email = webhook_event.user_email
            recipient_user_id = webhook_event.user_id
            if (
                recipient_email is None
                and recipient_user_id is not None
                and prisma_client is not None
            ):
                user_row = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": recipient_user_id}
                )

                if user_row is not None:
                    recipient_email = user_row.user_email

            key_token = webhook_event.token
            key_budget = webhook_event.max_budget
            base_url = os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")

            email_html_content = "Alert from LiteLLM Server"
            if recipient_email is None:
                verbose_proxy_logger.error(
                    "Trying to send email alert to no recipient",
                    extra=webhook_event.dict(),
                )

            if webhook_event.event == "key_created":
                email_html_content = KEY_CREATED_EMAIL_TEMPLATE.format(
                    email_logo_url=email_logo_url,
                    recipient_email=recipient_email,
                    key_budget=key_budget,
                    key_token=key_token,
                    base_url=base_url,
                    email_support_contact=email_support_contact,
                )
            elif webhook_event.event == "internal_user_created":
                # GET TEAM NAME
                team_id = webhook_event.team_id
                team_name = "Default Team"
                if team_id is not None and prisma_client is not None:
                    team_row = await prisma_client.db.litellm_teamtable.find_unique(
                        where={"team_id": team_id}
                    )
                    if team_row is not None:
                        team_name = team_row.team_alias or "-"
                email_html_content = USER_INVITED_EMAIL_TEMPLATE.format(
                    email_logo_url=email_logo_url,
                    recipient_email=recipient_email,
                    team_name=team_name,
                    base_url=base_url,
                    email_support_contact=email_support_contact,
                )
            else:
                verbose_proxy_logger.error(
                    "Trying to send email alert on unknown webhook event",
                    extra=webhook_event.model_dump(),
                )

            webhook_event.model_dump_json()
            email_event = {
                "to": recipient_email,
                "subject": f"LiteLLM: {event_name}",
                "html": email_html_content,
            }

            await send_email(
                receiver_email=email_event["to"],
                subject=email_event["subject"],
                html=email_event["html"],
            )

            return True

        except Exception as e:
            verbose_proxy_logger.error("Error sending email alert %s", str(e))
            return False

    async def send_email_alert_using_smtp(
        self, webhook_event: WebhookEvent, alert_type: str
    ) -> bool:
        """
        Sends structured Email alert to an SMTP server

        Currently only implemented for budget alerts

        Returns -> True if sent, False if not.
        """
        from litellm.proxy.proxy_server import premium_user
        from litellm.proxy.utils import send_email

        email_logo_url = os.getenv(
            "SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None)
        )
        email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)
        await self._check_if_using_premium_email_feature(
            premium_user, email_logo_url, email_support_contact
        )

        if email_logo_url is None:
            email_logo_url = LITELLM_LOGO_URL
        if email_support_contact is None:
            email_support_contact = LITELLM_SUPPORT_CONTACT

        event_name = webhook_event.event_message
        recipient_email = webhook_event.user_email
        user_name = webhook_event.user_id
        max_budget = webhook_event.max_budget
        email_html_content = "Alert from LiteLLM Server"
        if recipient_email is None:
            verbose_proxy_logger.error(
                "Trying to send email alert to no recipient", extra=webhook_event.dict()
            )

        if webhook_event.event == "budget_crossed":
            email_html_content = f"""
            <img src="{email_logo_url}" alt="LiteLLM Logo" width="150" height="50" />

            <p> Hi {user_name}, <br/>

            Your LLM API usage this month has reached your account's <b> monthly budget of ${max_budget} </b> <br /> <br />

            API requests will be rejected until either (a) you increase your monthly budget or (b) your monthly usage resets at the beginning of the next calendar month. <br /> <br />

            If you have any questions, please send an email to {email_support_contact} <br /> <br />

            Best, <br />
            The LiteLLM team <br />
            """

        webhook_event.model_dump_json()
        email_event = {
            "to": recipient_email,
            "subject": f"LiteLLM: {event_name}",
            "html": email_html_content,
        }

        await send_email(
            receiver_email=email_event["to"],
            subject=email_event["subject"],
            html=email_event["html"],
        )
        if webhook_event.event_group == "team":
            from litellm.integrations.email_alerting import send_team_budget_alert

            await send_team_budget_alert(webhook_event=webhook_event)

        return False

    async def send_alert(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: AlertType,
        alerting_metadata: dict,
        user_info: Optional[WebhookEvent] = None,
        **kwargs,
    ):
        """
        Alerting based on thresholds: - https://github.com/BerriAI/litellm/issues/1298

        - Responses taking too long
        - Requests are hanging
        - Calls are failing
        - DB Read/Writes are failing
        - Proxy Close to max budget
        - Key Close to max budget

        Parameters:
            level: str - Low|Medium|High - if calls might fail (Medium) or are failing (High); Currently, no alerts would be 'Low'.
            message: str - what is the alert about
        """
        if self.alerting is None:
            return

        if (
            "webhook" in self.alerting
            and alert_type == "budget_alerts"
            and user_info is not None
        ):
            await self.send_webhook_alert(webhook_event=user_info)

        if (
            "email" in self.alerting
            and alert_type == "budget_alerts"
            and user_info is not None
        ):
            # only send budget alerts over Email
            await self.send_email_alert_using_smtp(
                webhook_event=user_info, alert_type=alert_type
            )

        if "slack" not in self.alerting:
            return
        if alert_type not in self.alert_types:
            return

        from datetime import datetime

        # Get the current timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
        if alert_type == "daily_reports" or alert_type == "new_model_added":
            formatted_message = message
        else:
            formatted_message = (
                f"Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
            )

        if kwargs:
            for key, value in kwargs.items():
                formatted_message += f"\n\n{key}: `{value}`\n\n"
        if alerting_metadata:
            for key, value in alerting_metadata.items():
                formatted_message += f"\n\n*Alerting Metadata*: \n{key}: `{value}`\n\n"
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        # check if we find the slack webhook url in self.alert_to_webhook_url
        if (
            self.alert_to_webhook_url is not None
            and alert_type in self.alert_to_webhook_url
        ):
            slack_webhook_url: Optional[Union[str, List[str]]] = (
                self.alert_to_webhook_url[alert_type]
            )
        elif self.default_webhook_url is not None:
            slack_webhook_url = self.default_webhook_url
        else:
            slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", None)

        if slack_webhook_url is None:
            raise ValueError("Missing SLACK_WEBHOOK_URL from environment")
        payload = {"text": formatted_message}
        headers = {"Content-type": "application/json"}

        if isinstance(slack_webhook_url, list):
            for url in slack_webhook_url:
                self.log_queue.append(
                    {
                        "url": url,
                        "headers": headers,
                        "payload": payload,
                        "alert_type": alert_type,
                    }
                )
        else:
            self.log_queue.append(
                {
                    "url": slack_webhook_url,
                    "headers": headers,
                    "payload": payload,
                    "alert_type": alert_type,
                }
            )

        if len(self.log_queue) >= self.batch_size:
            await self.flush_queue()

    async def async_send_batch(self):
        if not self.log_queue:
            return

        squashed_queue = squash_payloads(self.log_queue)
        tasks = [
            send_to_webhook(
                slackAlertingInstance=self, item=item["item"], count=item["count"]
            )
            for item in squashed_queue.values()
        ]
        await asyncio.gather(*tasks)
        self.log_queue.clear()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log deployment latency"""
        try:
            if "daily_reports" in self.alert_types:
                litellm_params = kwargs.get("litellm_params", {}) or {}
                model_info = litellm_params.get("model_info", {}) or {}
                model_id = model_info.get("id", "") or ""
                response_s: timedelta = end_time - start_time

                final_value = response_s

                if isinstance(response_obj, litellm.ModelResponse) and (
                    hasattr(response_obj, "usage")
                    and response_obj.usage is not None  # type: ignore
                    and hasattr(response_obj.usage, "completion_tokens")  # type: ignore
                ):
                    completion_tokens = response_obj.usage.completion_tokens  # type: ignore
                    if completion_tokens is not None and completion_tokens > 0:
                        final_value = float(
                            response_s.total_seconds() / completion_tokens
                        )
                if isinstance(final_value, timedelta):
                    final_value = final_value.total_seconds()

                await self.async_update_daily_reports(
                    DeploymentMetrics(
                        id=model_id,
                        failed_request=False,
                        latency_per_output_token=final_value,
                        updated_at=litellm.utils.get_utc_datetime(),
                    )
                )
        except Exception as e:
            verbose_proxy_logger.error(
                f"[Non-Blocking Error] Slack Alerting: Got error in logging LLM deployment latency: {str(e)}"
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failure + deployment latency"""
        _litellm_params = kwargs.get("litellm_params", {})
        _model_info = _litellm_params.get("model_info", {}) or {}
        model_id = _model_info.get("id", "")
        try:
            if "daily_reports" in self.alert_types:
                try:
                    await self.async_update_daily_reports(
                        DeploymentMetrics(
                            id=model_id,
                            failed_request=True,
                            latency_per_output_token=None,
                            updated_at=litellm.utils.get_utc_datetime(),
                        )
                    )
                except Exception as e:
                    verbose_logger.debug(f"Exception raises -{str(e)}")

            if isinstance(kwargs.get("exception", ""), APIError):
                if "outage_alerts" in self.alert_types:
                    await self.outage_alerts(
                        exception=kwargs["exception"],
                        deployment_id=model_id,
                    )

                if "region_outage_alerts" in self.alert_types:
                    await self.region_outage_alerts(
                        exception=kwargs["exception"], deployment_id=model_id
                    )
        except Exception:
            pass

    async def _run_scheduler_helper(self, llm_router) -> bool:
        """
        Returns:
        - True -> report sent
        - False -> report not sent
        """
        report_sent_bool = False

        report_sent = await self.internal_usage_cache.async_get_cache(
            key=SlackAlertingCacheKeys.report_sent_key.value,
            parent_otel_span=None,
        )  # None | float

        current_time = time.time()

        if report_sent is None:
            await self.internal_usage_cache.async_set_cache(
                key=SlackAlertingCacheKeys.report_sent_key.value,
                value=current_time,
            )
        elif isinstance(report_sent, float):
            # Check if current time - interval >= time last sent
            interval_seconds = self.alerting_args.daily_report_frequency

            if current_time - report_sent >= interval_seconds:
                # Sneak in the reporting logic here
                await self.send_daily_reports(router=llm_router)
                # Also, don't forget to update the report_sent time after sending the report!
                await self.internal_usage_cache.async_set_cache(
                    key=SlackAlertingCacheKeys.report_sent_key.value,
                    value=current_time,
                )
                report_sent_bool = True

        return report_sent_bool

    async def _run_scheduled_daily_report(self, llm_router: Optional[Any] = None):
        """
        If 'daily_reports' enabled

        Ping redis cache every 5 minutes to check if we should send the report

        If yes -> call send_daily_report()
        """
        if llm_router is None or self.alert_types is None:
            return

        if "daily_reports" in self.alert_types:
            while True:
                await self._run_scheduler_helper(llm_router=llm_router)
                interval = random.randint(
                    self.alerting_args.report_check_interval - 3,
                    self.alerting_args.report_check_interval + 3,
                )  # shuffle to prevent collisions
                await asyncio.sleep(interval)
        return

    async def send_weekly_spend_report(
        self,
        time_range: str = "7d",
    ):
        """
        Send a spend report for a configurable time range.

        Args:
            time_range: A string specifying the time range for the report, e.g., "1d", "7d", "30d"
        """
        if self.alerting is None or "spend_reports" not in self.alert_types:
            return

        try:
            from litellm.proxy.spend_tracking.spend_management_endpoints import (
                _get_spend_report_for_time_range,
            )

            # Parse the time range
            days = int(time_range[:-1])
            if time_range[-1].lower() != "d":
                raise ValueError("Time range must be specified in days, e.g., '7d'")

            todays_date = datetime.datetime.now().date()
            start_date = todays_date - datetime.timedelta(days=days)

            _event_cache_key = f"weekly_spend_report_sent_{start_date.strftime('%Y-%m-%d')}_{todays_date.strftime('%Y-%m-%d')}"
            if await self.internal_usage_cache.async_get_cache(key=_event_cache_key):
                return

            _resp = await _get_spend_report_for_time_range(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=todays_date.strftime("%Y-%m-%d"),
            )
            if _resp is None or _resp == ([], []):
                return

            spend_per_team, spend_per_tag = _resp

            _spend_message = f"*💸 Spend Report for `{start_date.strftime('%m-%d-%Y')} - {todays_date.strftime('%m-%d-%Y')}` ({days} days)*\n"

            if spend_per_team is not None:
                _spend_message += "\n*Team Spend Report:*\n"
                for spend in spend_per_team:
                    _team_spend = round(float(spend["total_spend"]), 4)
                    _spend_message += (
                        f"Team: `{spend['team_alias']}` | Spend: `${_team_spend}`\n"
                    )

            if spend_per_tag is not None:
                _spend_message += "\n*Tag Spend Report:*\n"
                for spend in spend_per_tag:
                    _tag_spend = round(float(spend["total_spend"]), 4)
                    _spend_message += f"Tag: `{spend['individual_request_tag']}` | Spend: `${_tag_spend}`\n"

            await self.send_alert(
                message=_spend_message,
                level="Low",
                alert_type=AlertType.spend_reports,
                alerting_metadata={},
            )

            await self.internal_usage_cache.async_set_cache(
                key=_event_cache_key,
                value="SENT",
                ttl=duration_in_seconds(time_range),
            )

        except ValueError as ve:
            verbose_proxy_logger.error(f"Invalid time range format: {ve}")
        except Exception as e:
            verbose_proxy_logger.error(f"Error sending spend report: {e}")

    async def send_monthly_spend_report(self):
        """ """
        try:
            from calendar import monthrange

            from litellm.proxy.spend_tracking.spend_management_endpoints import (
                _get_spend_report_for_time_range,
            )

            todays_date = datetime.datetime.now().date()
            first_day_of_month = todays_date.replace(day=1)
            _, last_day_of_month = monthrange(todays_date.year, todays_date.month)
            last_day_of_month = first_day_of_month + datetime.timedelta(
                days=last_day_of_month - 1
            )

            _event_cache_key = f"monthly_spend_report_sent_{first_day_of_month.strftime('%Y-%m-%d')}_{last_day_of_month.strftime('%Y-%m-%d')}"
            if await self.internal_usage_cache.async_get_cache(key=_event_cache_key):
                return

            _resp = await _get_spend_report_for_time_range(
                start_date=first_day_of_month.strftime("%Y-%m-%d"),
                end_date=last_day_of_month.strftime("%Y-%m-%d"),
            )

            if _resp is None or _resp == ([], []):
                return

            monthly_spend_per_team, monthly_spend_per_tag = _resp

            _spend_message = f"*💸 Monthly Spend Report for `{first_day_of_month.strftime('%m-%d-%Y')} - {last_day_of_month.strftime('%m-%d-%Y')}` *\n"

            if monthly_spend_per_team is not None:
                _spend_message += "\n*Team Spend Report:*\n"
                for spend in monthly_spend_per_team:
                    _team_spend = spend["total_spend"]
                    _team_spend = float(_team_spend)
                    # round to 4 decimal places
                    _team_spend = round(_team_spend, 4)
                    _spend_message += (
                        f"Team: `{spend['team_alias']}` | Spend: `${_team_spend}`\n"
                    )

            if monthly_spend_per_tag is not None:
                _spend_message += "\n*Tag Spend Report:*\n"
                for spend in monthly_spend_per_tag:
                    _tag_spend = spend["total_spend"]
                    _tag_spend = float(_tag_spend)
                    # round to 4 decimal places
                    _tag_spend = round(_tag_spend, 4)
                    _spend_message += f"Tag: `{spend['individual_request_tag']}` | Spend: `${_tag_spend}`\n"

            await self.send_alert(
                message=_spend_message,
                level="Low",
                alert_type=AlertType.spend_reports,
                alerting_metadata={},
            )

            await self.internal_usage_cache.async_set_cache(
                key=_event_cache_key,
                value="SENT",
                ttl=(30 * HOURS_IN_A_DAY * 60 * 60),  # 1 month
            )

        except Exception as e:
            verbose_proxy_logger.exception("Error sending weekly spend report %s", e)

    async def send_fallback_stats_from_prometheus(self):
        """
        Helper to send fallback statistics from prometheus server -> to slack

        This runs once per day and sends an overview of all the fallback statistics
        """
        try:
            from litellm.integrations.prometheus_helpers.prometheus_api import (
                get_fallback_metric_from_prometheus,
            )

            # call prometheuslogger.
            falllback_success_info_prometheus = (
                await get_fallback_metric_from_prometheus()
            )

            fallback_message = (
                f"*Fallback Statistics:*\n{falllback_success_info_prometheus}"
            )

            await self.send_alert(
                message=fallback_message,
                level="Low",
                alert_type=AlertType.fallback_reports,
                alerting_metadata={},
            )

        except Exception as e:
            verbose_proxy_logger.error("Error sending weekly spend report %s", e)

        pass

    async def send_virtual_key_event_slack(
        self,
        key_event: VirtualKeyEvent,
        alert_type: AlertType,
        event_name: str,
    ):
        """
        Handles sending Virtual Key related alerts

        Example:
        - New Virtual Key Created
        - Internal User Updated
        - Team Created, Updated, Deleted
        """
        try:
            message = f"`{event_name}`\n"

            key_event_dict = key_event.model_dump()

            # Add Created by information first
            message += "*Action Done by:*\n"
            for key, value in key_event_dict.items():
                if "created_by" in key:
                    message += f"{key}: `{value}`\n"

            # Add args sent to function in the alert
            message += "\n*Arguments passed:*\n"
            request_kwargs = key_event.request_kwargs
            for key, value in request_kwargs.items():
                if key == "user_api_key_dict":
                    continue
                message += f"{key}: `{value}`\n"

            await self.send_alert(
                message=message,
                level="High",
                alert_type=alert_type,
                alerting_metadata={},
            )

        except Exception as e:
            verbose_proxy_logger.error(
                "Error sending send_virtual_key_event_slack %s", e
            )

        return

    async def _request_is_completed(self, request_data: Optional[dict]) -> bool:
        """
        Returns True if the request is completed - either as a success or failure
        """
        if request_data is None:
            return False

        if (
            request_data.get("litellm_status", "") != "success"
            and request_data.get("litellm_status", "") != "fail"
        ):
            ## CHECK IF CACHE IS UPDATED
            litellm_call_id = request_data.get("litellm_call_id", "")
            status: Optional[str] = await self.internal_usage_cache.async_get_cache(
                key="request_status:{}".format(litellm_call_id), local_only=True
            )
            if status is not None and (status == "success" or status == "fail"):
                return True
        return False
