#### What this does ####
#    Class for sending Slack Alerts #
import asyncio
import datetime
import os
import random
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

from openai import APIError
from pydantic import TypeAdapter

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
import litellm.types
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.constants import (
    HOURS_IN_A_DAY,
    SLACK_DAILY_REPORT_LOCK_ID,
    SLACK_MODEL_DEPRECATION_LOCK_ID,
)
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.SlackAlerting.budget_alert_types import get_budget_alert_type
from litellm.integrations.SlackAlerting.hanging_request_check import (
    AlertingHangingRequestCheck,
)
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
    InvitationModel,
    InvitationNew,
    Litellm_EntityType,
    UserAPIKeyAuth,
    VirtualKeyEvent,
    WebhookEvent,
)
from litellm.repositories.table_repositories import InvitationLinkRepository
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.integrations.slack_alerting import *
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    DEPRECATION_IDLE_POLL_SECONDS,
)

from ..email_templates.templates import *
from .batching_handler import send_to_webhook, squash_payloads
from .utils import process_slack_alerting_variables

if TYPE_CHECKING:
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.router import Router as _Router

    Router = _Router
else:
    Router = Any


def _proxy_llm_router() -> Router | None:
    from litellm.proxy.proxy_server import llm_router

    return llm_router


class SlackAlerting(CustomBatchLogger):
    """
    Class for sending Slack Alerts
    """

    # Class variables or attributes
    def __init__(
        self,
        internal_usage_cache: DualCache | None = None,
        alerting_threshold: float | None = None,  # threshold for slow / hanging llm responses (in seconds)
        alerting: list | None = [],
        alert_types: list[AlertType] = DEFAULT_ALERT_TYPES,
        alert_to_webhook_url: dict[AlertType, list[str] | str]
        | None = None,  # if user wants to separate alerts to diff channels
        alerting_args={},
        default_webhook_url: str | None = None,
        alert_type_config: dict[str, dict] | None = None,
        **kwargs,
    ):
        if alerting_threshold is None:
            alerting_threshold = 300
        self.alerting_threshold = alerting_threshold
        self.alerting = alerting
        self.alert_types = alert_types
        self.internal_usage_cache = internal_usage_cache or DualCache()
        self.async_http_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        self.alert_to_webhook_url = process_slack_alerting_variables(alert_to_webhook_url=alert_to_webhook_url)
        self.is_running = False
        self.alerting_args = SlackAlertingArgs(**alerting_args)
        self.default_webhook_url = default_webhook_url
        self.flush_lock = asyncio.Lock()
        self.periodic_started = False
        self.hanging_request_check = AlertingHangingRequestCheck(
            slack_alerting_object=self,
        )
        self.alert_type_config: dict[str, AlertTypeConfig] = {}
        if alert_type_config:
            for key, val in alert_type_config.items():
                self.alert_type_config[key] = AlertTypeConfig(**val) if isinstance(val, dict) else val
        self.digest_buckets: dict[str, DigestEntry] = {}
        self.digest_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)

    def update_values(
        self,
        alerting: list | None = None,
        alerting_threshold: float | None = None,
        alert_types: list[AlertType] | None = None,
        alert_to_webhook_url: dict[AlertType, list[str] | str] | None = None,
        alerting_args: dict | None = None,
        llm_router: Router | None = None,
        alert_type_config: dict[str, dict] | None = None,
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
        if alert_type_config is not None:
            for key, val in alert_type_config.items():
                self.alert_type_config[key] = AlertTypeConfig(**val) if isinstance(val, dict) else val

        if alert_to_webhook_url is not None:
            # update the dict
            if self.alert_to_webhook_url is None:
                self.alert_to_webhook_url = process_slack_alerting_variables(alert_to_webhook_url=alert_to_webhook_url)
            else:
                _new_values: Final = process_slack_alerting_variables(alert_to_webhook_url=alert_to_webhook_url) or {}
                self.alert_to_webhook_url.update(_new_values)
        if llm_router is not None:
            self.llm_router = llm_router

    def _prepare_outage_value_for_cache(self, outage_value: dict | ProviderRegionOutageModel | OutageModel) -> dict:
        """
        Helper method to prepare outage value for Redis caching.
        Converts set objects to lists for JSON serialization.
        """
        # Convert to dict for processing
        cache_value: Final = dict(outage_value)

        if "deployment_ids" in cache_value and isinstance(cache_value["deployment_ids"], set):
            cache_value["deployment_ids"] = list(cache_value["deployment_ids"])
        return cache_value

    def _restore_outage_value_from_cache(self, outage_value: dict | None) -> dict | None:
        """
        Helper method to restore outage value after retrieving from cache.
        Converts list objects back to sets for proper handling.
        """
        if outage_value and isinstance(outage_value.get("deployment_ids"), list):
            outage_value["deployment_ids"] = set(outage_value["deployment_ids"])
        return outage_value

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
            time_difference: Final = end_time - start_time
            # Convert the timedelta to float (in seconds)
            time_difference_float: Final = time_difference.total_seconds()
            litellm_params: Final = kwargs.get("litellm_params", {})
            model: Final = kwargs.get("model", "")
            api_base: Final = litellm.get_api_base(model=model, optional_params=litellm_params)
            messages = kwargs.get("messages", None)
            # if messages does not exist fallback to "input"
            if messages is None:
                messages = kwargs.get("input", None)

            # only use first 100 chars for alerting
            _messages: Final = str(messages)[:100]

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
            _deployment_latency_map: dict | None = None
            try:
                # try sorting deployments by latency
                _deployment_latencies = sorted(_deployment_latencies.items(), key=lambda x: x[1])
                _deployment_latency_map = dict(_deployment_latencies)
            except Exception:
                pass

            if _deployment_latency_map is None:
                return

            for api_base, latency in _deployment_latency_map.items():
                _message_to_send += f"\n{api_base}: {round(latency, 2)}s"
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
        slow_message = f"`Responses are slow - {round(time_difference_float, 2)}s response time > Alerting threshold: {self.alerting_threshold}s`"
        alerting_metadata: dict = {}
        if time_difference_float > self.alerting_threshold:
            # add deployment latencies to alert
            if kwargs is not None and "litellm_params" in kwargs and "metadata" in kwargs["litellm_params"]:
                _metadata: Final[dict] = kwargs["litellm_params"]["metadata"]
                request_info = _add_key_name_and_team_to_alert(request_info=request_info, metadata=_metadata)

                _deployment_latency_map: Final = self._get_deployment_latencies_to_alert(metadata=_metadata)
                if _deployment_latency_map is not None:
                    request_info += f"\nAvailable Deployment Latencies\n{_deployment_latency_map}"

                if "alerting_metadata" in _metadata:
                    alerting_metadata = _metadata["alerting_metadata"]
            await self.send_alert(
                message=slow_message + request_info,
                level="Low",
                alert_type=AlertType.llm_too_slow,
                alerting_metadata=alerting_metadata,
                request_model=model,
                api_base=api_base,
            )

    async def async_update_daily_reports(self, deployment_metrics: DeploymentMetrics) -> int:
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
                    key=f"{deployment_metrics.id}:{SlackAlertingCacheKeys.failed_requests_key.value}",
                    value=1,
                    parent_otel_span=None,  # no attached request, this is a background operation
                )

                return_val += 1

            ## LATENCY ##
            if deployment_metrics.latency_per_output_token is not None:
                await self.internal_usage_cache.async_increment_cache(
                    key=f"{deployment_metrics.id}:{SlackAlertingCacheKeys.latency_key.value}",
                    value=deployment_metrics.latency_per_output_token,
                    parent_otel_span=None,  # no attached request, this is a background operation
                )

                return_val += 1

            return return_val
        except Exception:
            return 0

    async def send_daily_reports(self, router) -> bool:
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

        ids: Final = router.get_model_ids()

        # get keys
        failed_request_keys: Final = [f"{id}:{SlackAlertingCacheKeys.failed_requests_key.value}" for id in ids]
        latency_keys: Final = [f"{id}:{SlackAlertingCacheKeys.latency_key.value}" for id in ids]

        combined_metrics_keys: Final = failed_request_keys + latency_keys  # reduce cache calls

        combined_metrics_values: Final = await self.internal_usage_cache.async_batch_get_cache(
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

        failed_request_values: Final = combined_metrics_values[: len(failed_request_keys)]  # # [1, 2, None, ..]
        latency_values: Final = combined_metrics_values[len(failed_request_keys) :]

        # find top 5 failed
        ## Replace None values with a placeholder value (-1 in this case)
        placeholder_value = 0
        replaced_failed_values = [value if value is not None else placeholder_value for value in failed_request_values]

        ## Get the indices of top 5 keys with the highest numerical values (ignoring None and 0 values)
        top_5_failed = sorted(
            range(len(replaced_failed_values)),
            key=lambda i: replaced_failed_values[i],
            reverse=True,
        )[:5]
        top_5_failed = [index for index in top_5_failed if replaced_failed_values[index] > 0]

        # find top 5 slowest
        # Replace None values with a placeholder value (-1 in this case)
        placeholder_value = 0
        replaced_slowest_values: Final = [value if value is not None else placeholder_value for value in latency_values]

        # Get the indices of top 5 values with the highest numerical values (ignoring None and 0 values)
        top_5_slowest = sorted(
            range(len(replaced_slowest_values)),
            key=lambda i: replaced_slowest_values[i],
            reverse=True,
        )[:5]
        top_5_slowest = [index for index in top_5_slowest if replaced_slowest_values[index] > 0]

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
                optional_params=(_deployment["litellm_params"] if _deployment is not None else {}),
            )
            if api_base is None:
                api_base = ""
            value = replaced_failed_values[top_5_failed[i]]
            message += (
                f"\t{i + 1}. Deployment: `{deployment_name}`, Failed Requests: `{value}`,  API Base: `{api_base}`\n"
            )

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
                optional_params=(_deployment["litellm_params"] if _deployment is not None else {}),
            )
            value = round(replaced_slowest_values[top_5_slowest[i]], 3)
            message += f"\t{i + 1}. Deployment: `{deployment_name}`, Latency per output token: `{value}s/token`,  API Base: `{api_base}`\n\n"

        # cache cleanup -> reset values to 0
        latency_cache_keys: Final = [(key, 0) for key in latency_keys]
        failed_request_cache_keys: Final = [(key, 0) for key in failed_request_keys]
        combined_metrics_cache_keys: Final = latency_cache_keys + failed_request_cache_keys
        await self.internal_usage_cache.async_set_cache_pipeline(cache_list=combined_metrics_cache_keys)

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
        request_data: dict | None = None,
    ):
        if self.alerting is None or self.alert_types is None:
            return

        if AlertType.llm_requests_hanging not in self.alert_types:
            return

        await self.hanging_request_check.add_request_to_hanging_request_check(request_data=request_data)

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

        _cache: Final[DualCache] = self.internal_usage_cache
        message: Final = "Failed Tracking Cost for " + error_message
        _cache_key: Final = f"budget_alerts:failed_tracking:{failing_model}"
        result: Final = await _cache.async_get_cache(key=_cache_key)
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
            "user_budget",
            "soft_budget",
            "max_budget_alert",
            "team_budget",
            "organization_budget",
            "proxy_budget",
            "projected_limit_exceeded",
            "project_budget",
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
        _cache: Final[DualCache] = self.internal_usage_cache

        if self.alerting is None or self.alert_types is None:
            # do nothing if alerting is not switched on
            return
        if "budget_alerts" not in self.alert_types:
            return

        # Get the appropriate budget alert type handler
        budget_alert_class: Final = get_budget_alert_type(type)
        _id: Final = budget_alert_class.get_id(user_info)
        user_info_json: Final = user_info.model_dump(exclude_none=True)
        user_info_str: Final = self._get_user_info_str(user_info)
        event_message = budget_alert_class.get_event_message()

        # Set default event unless we're in projected_limit_exceeded
        event: (
            Literal["budget_crossed", "threshold_crossed", "projected_limit_exceeded", "soft_budget_crossed"] | None
        ) = "projected_limit_exceeded" if type == "projected_limit_exceeded" else None

        webhook_event: WebhookEvent | None = None

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
            _cache_key: Final = f"budget_alerts:{event}:{_id}"
            result: Final = await _cache.async_get_cache(key=_cache_key)
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
        event: Literal["budget_crossed", "threshold_crossed", "soft_budget_crossed", "projected_limit_exceeded"] | None,
        event_message: str,
    ) -> tuple[
        Literal["budget_crossed", "threshold_crossed", "soft_budget_crossed", "projected_limit_exceeded"] | None,
        str,
    ]:
        """
        Get the event and event message for a budget alert

        This will append any new information to the event_message

        Handles Max Budget and Soft Budget Alerts
        """
        percent_left: Final[float] = self._get_percent_of_max_budget_left(user_info=user_info)

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
                event_message += f"Budget Crossed\n Total Budget:`{user_info.max_budget}`"
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
        current_spend: Final[float] = user_info.spend
        max_budget: Final[float | None] = user_info.max_budget
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
        _all_fields_as_dict: Final = user_info.model_dump(exclude_none=True)
        _all_fields_as_dict.pop("token")
        msg = ""
        for k, v in _all_fields_as_dict.items():
            if isinstance(v, Litellm_EntityType):
                v = v.value
            msg += f"*{k}:* `{v}`\n"

        return msg

    async def customer_spend_alert(
        self,
        token: str | None,
        key_alias: str | None,
        end_user_id: str | None,
        response_cost: float | None,
        max_budget: float | None,
    ):
        if (
            self.alerting is not None
            and "webhook" in self.alerting
            and end_user_id is not None
            and token is not None
            and response_cost is not None
        ):
            # log customer spend
            event: Final = WebhookEvent(
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
                event_message=f"Customer spend tracked. Customer={end_user_id}, spend={response_cost}",
            )

            await self.send_webhook_alert(webhook_event=event)

    def _count_outage_alerts(self, alerts: list[int]) -> str:
        """
        Parameters:
        - alerts: List[int] -> list of error codes (either 408 or 500+)

        Returns:
        - str -> formatted string. This is an alert message, giving a human-friendly description of the errors.
        """
        error_breakdown: Final = {"Timeout Errors": 0, "API Errors": 0, "Unknown Errors": 0}
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
                error_msg += f"\n{key}: {value}\n"

        return error_msg

    def _outage_alert_msg_factory(
        self,
        alert_type: Literal["Major", "Minor"],
        key: Literal["Model", "Region"],
        key_val: str,
        provider: str,
        api_base: str | None,
        outage_value: BaseOutageModel,
    ) -> str:
        """Format an alert message for slack"""
        headers: Final = {f"{key} Name": key_val, "Provider": provider}
        if api_base is not None:
            headers["API Base"] = api_base

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

        deployment: Final = self.llm_router.get_deployment(model_id=deployment_id)

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
        cache_key: Final = provider + region_name

        outage_value: ProviderRegionOutageModel | None = await self.internal_usage_cache.async_get_cache(key=cache_key)

        # Convert deployment_ids back to set if it was stored as a list
        if outage_value is not None:
            outage_value = self._restore_outage_value_from_cache(outage_value)

        if (
            getattr(exception, "status_code", None) is None
            or (exception.status_code != 408 and exception.status_code < 500)
            or self.llm_router is None
        ):
            return

        if outage_value is None:
            _deployment_set = set()
            _deployment_set.add(deployment_id)
            outage_value = ProviderRegionOutageModel(
                provider_region_id=cache_key,
                alerts=[exception.status_code],
                minor_alert_sent=False,
                major_alert_sent=False,
                last_updated_at=time.time(),
                deployment_ids=_deployment_set,
            )

            ## add to cache ##
            # Convert set to list for JSON serialization
            cache_value = self._prepare_outage_value_for_cache(outage_value)
            await self.internal_usage_cache.async_set_cache(
                key=cache_key,
                value=cache_value,
                ttl=self.alerting_args.region_outage_alert_ttl,
            )
            return

        if len(outage_value["alerts"]) < self.alerting_args.max_outage_alert_list_size:
            outage_value["alerts"].append(exception.status_code)
        else:  # prevent memory leaks
            pass
        _deployment_set = outage_value["deployment_ids"]
        _deployment_set.add(deployment_id)
        outage_value["deployment_ids"] = _deployment_set
        outage_value["last_updated_at"] = time.time()

        ## MINOR OUTAGE ALERT SENT ##
        if (
            outage_value["minor_alert_sent"] is False
            and len(outage_value["alerts"]) >= self.alerting_args.minor_outage_alert_threshold
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
            and len(outage_value["alerts"]) >= self.alerting_args.major_outage_alert_threshold
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
        # Convert set to list for JSON serialization
        cache_value = self._prepare_outage_value_for_cache(outage_value)
        await self.internal_usage_cache.async_set_cache(key=cache_key, value=cache_value)

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
            outage_value: OutageModel | None = await self.internal_usage_cache.async_get_cache(key=deployment_id)
            if (
                getattr(exception, "status_code", None) is None
                or (exception.status_code != 408 and exception.status_code < 500)
                or self.llm_router is None
            ):
                return

            ### EXTRACT MODEL DETAILS ###
            deployment: Final = self.llm_router.get_deployment(model_id=deployment_id)
            if deployment is None:
                return

            model = deployment.litellm_params.model
            provider = deployment.litellm_params.custom_llm_provider
            if provider is None:
                try:
                    model, provider, _, _ = litellm.get_llm_provider(model=model)
                except Exception:
                    provider = ""
            api_base: Final = litellm.get_api_base(model=model, optional_params=deployment.litellm_params)

            if outage_value is None:
                outage_value = OutageModel(
                    model_id=deployment_id,
                    alerts=[exception.status_code],
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

            if len(outage_value["alerts"]) < self.alerting_args.max_outage_alert_list_size:
                outage_value["alerts"].append(exception.status_code)
            else:  # prevent memory leaks
                pass

            outage_value["last_updated_at"] = time.time()

            ## MINOR OUTAGE ALERT SENT ##
            if (
                outage_value["minor_alert_sent"] is False
                and len(outage_value["alerts"]) >= self.alerting_args.minor_outage_alert_threshold
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
                and len(outage_value["alerts"]) >= self.alerting_args.major_outage_alert_threshold
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
            # Convert set to list for JSON serialization
            cache_value: Final = self._prepare_outage_value_for_cache(outage_value)
            await self.internal_usage_cache.async_set_cache(key=deployment_id, value=cache_value)
        except Exception:
            pass

    async def model_added_alert(self, model_name: str, litellm_model_name: str, passed_model_info: Any):
        base_model_from_user: Final = getattr(passed_model_info, "base_model", None)
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
                v = f"{v:.8f}"

            model_info_str += f"{k}: {v}\n"

        message: Final = f"""
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

        alert_val: Final = self.send_alert(
            message=message,
            level="Low",
            alert_type=AlertType.new_model_added,
            alerting_metadata={},
        )

        if alert_val is not None and asyncio.iscoroutine(alert_val):
            await alert_val

    async def model_removed_alert(self, model_name: str):
        pass

    def _deprecation_alerts_enabled(self) -> bool:
        return self.alerting is not None and AlertType.model_deprecation_warnings in self.alert_types

    async def send_model_deprecation_alert(
        self,
        llm_router: Router | None = None,
        pod_lock_manager: "PodLockManager | None" = None,
    ) -> bool:
        """Alert on the router's deprecated and imminent models, True when one was sent

        The daily lock is claimed only once there is something to say, so an empty pass never blocks a
        later real one, and a sent alert is stamped in the shared cache for a day so sibling pods stop asking
        """
        if not self._deprecation_alerts_enabled():
            return False

        from litellm.proxy.common_utils.model_deprecation import (
            collect_model_deprecations,
            format_deprecation_alert_message,
        )

        snapshot: Final = collect_model_deprecations(llm_router=llm_router)
        message: Final = format_deprecation_alert_message(snapshot)
        if message is None:
            return False
        if not await self._claimed_deprecation_alert_window(pod_lock_manager):
            return False

        level: Final[Literal["Low", "Medium", "High"]] = "High" if snapshot.deprecated else "Medium"

        await self.send_alert(
            message=message,
            level=level,
            alert_type=AlertType.model_deprecation_warnings,
            alerting_metadata={  # mutable-ok: send_alert takes a dict payload
                "deprecated_count": len(snapshot.deprecated),
                "imminent_count": len(snapshot.imminent),
                "upcoming_count": len(snapshot.upcoming),
            },
        )
        await self.internal_usage_cache.async_set_cache(
            key=SlackAlertingCacheKeys.deprecation_alert_sent_key.value,
            value=time.time(),
            ttl=DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
        )
        return True

    async def _claimed_deprecation_alert_window(self, pod_lock_manager: "PodLockManager | None") -> bool:
        """Without a redis backed lock there is no fleet to coordinate, so a lone pod always alerts"""
        if pod_lock_manager is None:
            return True
        return (
            await pod_lock_manager.acquire_lock(
                cronjob_id=SLACK_MODEL_DEPRECATION_LOCK_ID,
                ttl=DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
                allow_reentrant=False,
            )
        ) is not False

    async def _deprecation_alert_sent_within_a_day(self) -> bool:
        return (
            await self.internal_usage_cache.async_get_cache(key=SlackAlertingCacheKeys.deprecation_alert_sent_key.value)
        ) is not None

    async def _run_deprecation_alert_pass(
        self, llm_router: Router | None, pod_lock_manager: "PodLockManager | None"
    ) -> bool:
        if llm_router is None or not self._deprecation_alerts_enabled():
            return False
        if await self._deprecation_alert_sent_within_a_day():
            return False
        return await self.send_model_deprecation_alert(llm_router=llm_router, pod_lock_manager=pod_lock_manager)

    async def run_scheduled_deprecation_check(
        self,
        get_llm_router: Callable[[], Router | None] = _proxy_llm_router,
        pod_lock_manager: "PodLockManager | None" = None,
    ) -> None:
        """Poll every pass for a loaded router, the alert being on, and no alert in the last day, then alert

        A pass that could not alert (no router yet, alert type off, a sibling pod holds the daily lock, or a
        redis blip at claim time) is retried on the next poll instead of costing a day, while a pass that
        raised (a missing webhook, say) backs off a full day so a misconfiguration logs once, not every poll
        """
        while True:
            try:
                await self._run_deprecation_alert_pass(get_llm_router(), pod_lock_manager)
            except Exception as e:  # noqa: BLE001  # a failed alert must not kill the loop
                verbose_proxy_logger.exception("Error in model deprecation alert loop: %s", e)
                await asyncio.sleep(DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS)
                continue
            await asyncio.sleep(DEPRECATION_IDLE_POLL_SECONDS)

    async def send_webhook_alert(self, webhook_event: WebhookEvent) -> bool:
        """
        Sends structured alert to webhook, if set.

        Currently only implemented for budget alerts

        Returns -> True if sent, False if not.

        Raises Exception
            - if WEBHOOK_URL is not set
        """

        webhook_url: Final = os.getenv("WEBHOOK_URL", None)
        if webhook_url is None:
            raise Exception("Missing webhook_url from environment")

        payload: Final = webhook_event.model_dump_json()
        headers: Final = {"Content-type": "application/json"}

        response: Final = await self.async_http_handler.post(
            url=webhook_url,
            headers=headers,
            data=payload,
        )
        if response.status_code == 200:
            return True
        else:
            print("Error sending webhook alert. Error=", response.text)  # noqa: T201

        return False

    async def _check_if_using_premium_email_feature(
        self,
        premium_user: bool,
        email_logo_url: str | None = None,
        email_support_contact: str | None = None,
    ):
        from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

        if premium_user is not True:
            if email_logo_url is not None or email_support_contact is not None:
                raise ValueError(f"Trying to Customize Email Alerting\n {CommonProxyErrors.not_premium_user.value}")

    async def _construct_user_invitation_link(self, recipient_user_id: str | None, base_url: str) -> str:
        from litellm.proxy.management_helpers.user_invitation import (
            create_invitation_for_user,
        )
        from litellm.proxy.proxy_server import prisma_client

        if recipient_user_id is None or prisma_client is None:
            return base_url

        try:
            existing_invitations: Final = TypeAdapter(list[InvitationModel]).validate_python(
                await InvitationLinkRepository(prisma_client).table.find_many(  # pyright: ignore[reportAny]  # untyped prisma boundary (any-ok), result validated by TypeAdapter
                    where={"user_id": recipient_user_id},  # mutable-ok: prisma find_many requires a dict where filter
                    order={"created_at": "desc"},  # mutable-ok: prisma find_many requires a dict order arg
                ),
                from_attributes=True,
            )
            invitation: Final = (
                existing_invitations[0]
                if existing_invitations
                else TypeAdapter(InvitationModel).validate_python(
                    await create_invitation_for_user(
                        data=InvitationNew(user_id=recipient_user_id),
                        user_api_key_dict=UserAPIKeyAuth(user_id=recipient_user_id),
                    ),
                    from_attributes=True,
                )
            )
        except Exception as e:  # noqa: BLE001  # best-effort link build; any DB/creation failure falls back to base_url
            verbose_proxy_logger.error(
                "Error creating invitation link for user_id %s: %s",
                recipient_user_id,
                str(e),
            )
            return base_url

        return f"{base_url.rstrip('/')}/ui/onboarding?invitation_id={invitation.id}"

    async def send_key_created_or_user_invited_email(self, webhook_event: WebhookEvent) -> bool:
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

            email_logo_url = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None))
            email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)
            await self._check_if_using_premium_email_feature(premium_user, email_logo_url, email_support_contact)
            if email_logo_url is None:
                email_logo_url = LITELLM_LOGO_URL
            if email_support_contact is None:
                email_support_contact = LITELLM_SUPPORT_CONTACT

            event_name: Final = webhook_event.event_message
            recipient_email = webhook_event.user_email
            recipient_user_id: Final = webhook_event.user_id
            if recipient_email is None and recipient_user_id is not None and prisma_client is not None:
                user_row = await UserRepository(prisma_client).table.find_unique(where={"user_id": recipient_user_id})

                if user_row is not None:
                    recipient_email = user_row.user_email

            key_token: Final = webhook_event.token
            key_budget: Final = webhook_event.max_budget
            base_url: Final = os.getenv("PROXY_BASE_URL", "http://0.0.0.0:4000")

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
                team_id: Final = webhook_event.team_id
                team_name = "Default Team"
                if team_id is not None and prisma_client is not None:
                    team_row: Final = await TeamRepository(prisma_client).table.find_unique(where={"team_id": team_id})
                    if team_row is not None:
                        team_name = team_row.team_alias or "-"
                invitation_link: Final = await self._construct_user_invitation_link(
                    recipient_user_id=recipient_user_id, base_url=base_url
                )
                email_html_content = USER_INVITED_EMAIL_TEMPLATE.format(
                    email_logo_url=email_logo_url,
                    recipient_email=recipient_email,
                    team_name=team_name,
                    base_url=invitation_link,
                    email_support_contact=email_support_contact,
                )
            else:
                verbose_proxy_logger.error(
                    "Trying to send email alert on unknown webhook event",
                    extra=webhook_event.model_dump(),
                )

            webhook_event.model_dump_json()
            email_event: Final = {
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

    async def send_email_alert_using_smtp(self, webhook_event: WebhookEvent, alert_type: str) -> bool:
        """
        Sends structured Email alert to an SMTP server

        Currently only implemented for budget alerts

        Returns -> True if sent, False if not.
        """
        from litellm.proxy.proxy_server import premium_user
        from litellm.proxy.utils import send_email

        email_logo_url = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", None))
        email_support_contact = os.getenv("EMAIL_SUPPORT_CONTACT", None)
        await self._check_if_using_premium_email_feature(premium_user, email_logo_url, email_support_contact)

        if email_logo_url is None:
            email_logo_url = LITELLM_LOGO_URL
        if email_support_contact is None:
            email_support_contact = LITELLM_SUPPORT_CONTACT

        event_name: Final = webhook_event.event_message
        recipient_email: Final = webhook_event.user_email
        user_name: Final = webhook_event.user_id
        max_budget: Final = webhook_event.max_budget
        email_html_content = "Alert from LiteLLM Server"
        if recipient_email is None:
            verbose_proxy_logger.error("Trying to send email alert to no recipient", extra=webhook_event.dict())

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
        email_event: Final = {
            "to": recipient_email,
            "subject": f"LiteLLM: {event_name}",
            "html": email_html_content,
        }

        await send_email(
            receiver_email=email_event["to"],
            subject=email_event["subject"],
            html=email_event["html"],
        )
        if webhook_event.event_group == Litellm_EntityType.TEAM:
            from litellm.integrations.email_alerting import send_team_budget_alert

            await send_team_budget_alert(webhook_event=webhook_event)

        return False

    async def send_alert(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: AlertType,
        alerting_metadata: dict,
        user_info: WebhookEvent | None = None,
        request_model: str | None = None,
        api_base: str | None = None,
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
            request_model: Optional[str] - model name for digest grouping
            api_base: Optional[str] - api base for digest grouping
        """
        if self.alerting is None:
            return

        # Start periodic flush if not already started
        if not self.periodic_started and self.alerting is not None and len(self.alerting) > 0:
            asyncio.create_task(self.periodic_flush())
            self.periodic_started = True

        if "webhook" in self.alerting and alert_type == "budget_alerts" and user_info is not None:
            await self.send_webhook_alert(webhook_event=user_info)

        if "email" in self.alerting and alert_type == "budget_alerts" and user_info is not None:
            # only send budget alerts over Email
            await self.send_email_alert_using_smtp(webhook_event=user_info, alert_type=alert_type)

        if "slack" not in self.alerting:
            return
        if alert_type not in self.alert_types:
            return

        from datetime import datetime

        # Check if digest mode is enabled for this alert type
        alert_type_name_str: Final = getattr(alert_type, "value", str(alert_type))
        _atc: Final = self.alert_type_config.get(alert_type_name_str)
        if _atc is not None and _atc.digest:
            # Resolve webhook URL for this alert type (needed for digest entry)
            if self.alert_to_webhook_url is not None and alert_type in self.alert_to_webhook_url:
                _digest_webhook: str | list[str] | None = self.alert_to_webhook_url[alert_type]
            elif self.default_webhook_url is not None:
                _digest_webhook = self.default_webhook_url
            else:
                _digest_webhook = os.getenv("SLACK_WEBHOOK_URL", None)
            if _digest_webhook is None:
                raise ValueError("Missing SLACK_WEBHOOK_URL from environment")

            digest_key: Final = f"{alert_type_name_str}:{request_model or ''}:{api_base or ''}"

            async with self.digest_lock:
                now: Final = datetime.now()
                if digest_key in self.digest_buckets:
                    self.digest_buckets[digest_key]["count"] += 1
                    self.digest_buckets[digest_key]["last_time"] = now
                else:
                    self.digest_buckets[digest_key] = DigestEntry(
                        alert_type=alert_type_name_str,
                        request_model=request_model or "",
                        api_base=api_base or "",
                        first_message=message,
                        level=level,
                        count=1,
                        start_time=now,
                        last_time=now,
                        webhook_url=_digest_webhook,
                    )
            return  # Suppress immediate alert; will be emitted by _flush_digest_buckets

        # Get the current timestamp
        current_time: Final = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url: Final = os.getenv("PROXY_BASE_URL", None)
        # Use .name if it's an enum, otherwise use as is
        alert_type_name: Final = getattr(alert_type, "name", alert_type)
        alert_type_formatted: Final = f"Alert type: `{alert_type_name}`"
        if alert_type == "daily_reports" or alert_type == "new_model_added":
            formatted_message = alert_type_formatted + message
        else:
            formatted_message = (
                f"{alert_type_formatted}\nLevel: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
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
        if self.alert_to_webhook_url is not None and alert_type in self.alert_to_webhook_url:
            slack_webhook_url: str | list[str] | None = self.alert_to_webhook_url[alert_type]
        elif self.default_webhook_url is not None:
            slack_webhook_url = self.default_webhook_url
        else:
            slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", None)

        if slack_webhook_url is None:
            raise ValueError("Missing SLACK_WEBHOOK_URL from environment")
        payload: Final = {"text": formatted_message}
        headers: Final = {"Content-type": "application/json"}

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

        squashed_queue: Final = squash_payloads(self.log_queue)
        tasks: Final = [
            send_to_webhook(slackAlertingInstance=self, item=item["item"], count=item["count"])
            for item in squashed_queue.values()
        ]
        await asyncio.gather(*tasks)
        self.log_queue.clear()

    async def _flush_digest_buckets(self):
        """Flush any digest buckets whose interval has expired.

        For each expired bucket, formats a digest summary message and
        appends it to the log_queue for delivery via the normal batching path.
        """
        from datetime import datetime

        now: Final = datetime.now()
        flushed_keys: Final[list[str]] = []

        async with self.digest_lock:
            for key, entry in self.digest_buckets.items():
                alert_type_name = entry["alert_type"]
                _atc = self.alert_type_config.get(alert_type_name)
                if _atc is None:
                    continue
                elapsed = (now - entry["start_time"]).total_seconds()
                if elapsed < _atc.digest_interval:
                    continue

                # Build digest summary message
                start_ts = entry["start_time"].strftime("%H:%M:%S")
                end_ts = entry["last_time"].strftime("%H:%M:%S")
                start_date = entry["start_time"].strftime("%Y-%m-%d")
                end_date = entry["last_time"].strftime("%Y-%m-%d")
                formatted_message = (
                    f"Alert type: `{alert_type_name}` (Digest)\n"
                    f"Level: `{entry['level']}`\n"
                    f"Start: `{start_date} {start_ts}`\n"
                    f"End: `{end_date} {end_ts}`\n"
                    f"Count: `{entry['count']}`\n\n"
                    f"Message: {entry['first_message']}"
                )
                _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
                if _proxy_base_url is not None:
                    formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

                payload = {"text": formatted_message}
                headers = {"Content-type": "application/json"}
                webhook_url = entry["webhook_url"]

                if isinstance(webhook_url, list):
                    for url in webhook_url:
                        self.log_queue.append(
                            {
                                "url": url,
                                "headers": headers,
                                "payload": payload,
                                "alert_type": alert_type_name,
                            }
                        )
                else:
                    self.log_queue.append(
                        {
                            "url": webhook_url,
                            "headers": headers,
                            "payload": payload,
                            "alert_type": alert_type_name,
                        }
                    )
                flushed_keys.append(key)

            for key in flushed_keys:
                del self.digest_buckets[key]

    async def periodic_flush(self):
        """Override base periodic_flush to also flush digest buckets."""
        while True:
            await asyncio.sleep(self.flush_interval)
            try:
                await self._flush_digest_buckets()
            except Exception as e:
                verbose_proxy_logger.debug("Error flushing digest buckets: %s", e)
            await self.flush_queue()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log deployment latency"""
        try:
            if "daily_reports" in self.alert_types:
                litellm_params: Final = kwargs.get("litellm_params", {}) or {}
                model_info: Final = litellm_params.get("model_info", {}) or {}
                model_id: Final = model_info.get("id", "") or ""
                response_s: Final[timedelta] = end_time - start_time

                final_value = response_s

                if isinstance(response_obj, litellm.ModelResponse) and (
                    hasattr(response_obj, "usage")
                    and response_obj.usage is not None
                    and hasattr(response_obj.usage, "completion_tokens")
                ):
                    completion_tokens: Final = response_obj.usage.completion_tokens
                    if completion_tokens is not None and completion_tokens > 0:
                        final_value = float(response_s.total_seconds() / completion_tokens)
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
                "[Non-Blocking Error] Slack Alerting: Got error in logging LLM deployment latency: %s", e
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failure + deployment latency"""
        _litellm_params: Final = kwargs.get("litellm_params", {})
        _model_info: Final = _litellm_params.get("model_info", {}) or {}
        model_id: Final = _model_info.get("id", "")
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
                    verbose_logger.debug("Exception raises -%s", e)

            if isinstance(kwargs.get("exception", ""), APIError):
                if "outage_alerts" in self.alert_types:
                    await self.outage_alerts(
                        exception=kwargs["exception"],
                        deployment_id=model_id,
                    )

                if "region_outage_alerts" in self.alert_types:
                    await self.region_outage_alerts(exception=kwargs["exception"], deployment_id=model_id)
        except Exception:
            pass

    async def _run_scheduler_helper(
        self,
        llm_router,
        pod_lock_manager: "PodLockManager | None" = None,
    ) -> bool:
        """
        Returns:
        - True -> report sent
        - False -> report not sent
        """
        report_sent_bool = False

        report_sent: Final = await self.internal_usage_cache.async_get_cache(
            key=SlackAlertingCacheKeys.report_sent_key.value,
            parent_otel_span=None,
        )  # None | float

        current_time: Final = time.time()

        if report_sent is None:
            await self.internal_usage_cache.async_set_cache(
                key=SlackAlertingCacheKeys.report_sent_key.value,
                value=current_time,
            )
        elif isinstance(report_sent, float):
            # Check if current time - interval >= time last sent
            interval_seconds: Final = self.alerting_args.daily_report_frequency

            if current_time - report_sent >= interval_seconds:
                if (
                    pod_lock_manager is not None
                    and (
                        await pod_lock_manager.acquire_lock(
                            cronjob_id=SLACK_DAILY_REPORT_LOCK_ID, ttl=interval_seconds, allow_reentrant=False
                        )
                    )
                    is False
                ):
                    return False
                # Sneak in the reporting logic here
                await self.send_daily_reports(router=llm_router)
                # Also, don't forget to update the report_sent time after sending the report!
                await self.internal_usage_cache.async_set_cache(
                    key=SlackAlertingCacheKeys.report_sent_key.value,
                    value=current_time,
                )
                report_sent_bool = True

        return report_sent_bool

    async def _run_scheduled_daily_report(
        self,
        llm_router: Any | None = None,
        pod_lock_manager: "PodLockManager | None" = None,
    ):
        """
        If 'daily_reports' enabled

        Ping redis cache every 5 minutes to check if we should send the report

        If yes -> call send_daily_report()
        """
        if llm_router is None or self.alert_types is None:
            return

        if "daily_reports" in self.alert_types:
            while True:
                await self._run_scheduler_helper(llm_router=llm_router, pod_lock_manager=pod_lock_manager)
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
            days: Final = int(time_range[:-1])
            if time_range[-1].lower() != "d":
                raise ValueError("Time range must be specified in days, e.g., '7d'")

            todays_date: Final = datetime.datetime.now().date()
            start_date: Final = todays_date - datetime.timedelta(days=days)

            _event_cache_key: Final = (
                f"weekly_spend_report_sent_{start_date.strftime('%Y-%m-%d')}_{todays_date.strftime('%Y-%m-%d')}"
            )
            if await self.internal_usage_cache.async_get_cache(key=_event_cache_key):
                return

            _resp: Final = await _get_spend_report_for_time_range(
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
                    _spend_message += f"Team: `{spend['team_alias']}` | Spend: `${_team_spend}`\n"

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
            verbose_proxy_logger.error("Invalid time range format: %s", ve)
        except Exception as e:
            verbose_proxy_logger.error("Error sending spend report: %s", e)

    async def send_monthly_spend_report(self):
        """ """
        try:
            from calendar import monthrange

            from litellm.proxy.spend_tracking.spend_management_endpoints import (
                _get_spend_report_for_time_range,
            )

            todays_date: Final = datetime.datetime.now().date()
            first_day_of_month: Final = todays_date.replace(day=1)
            _, last_day_of_month = monthrange(todays_date.year, todays_date.month)
            last_day_of_month = first_day_of_month + datetime.timedelta(days=last_day_of_month - 1)

            _event_cache_key = f"monthly_spend_report_sent_{first_day_of_month.strftime('%Y-%m-%d')}_{last_day_of_month.strftime('%Y-%m-%d')}"
            if await self.internal_usage_cache.async_get_cache(key=_event_cache_key):
                return

            _resp: Final = await _get_spend_report_for_time_range(
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
                    _spend_message += f"Team: `{spend['team_alias']}` | Spend: `${_team_spend}`\n"

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
            falllback_success_info_prometheus: Final = await get_fallback_metric_from_prometheus()

            fallback_message: Final = f"*Fallback Statistics:*\n{falllback_success_info_prometheus}"

            await self.send_alert(
                message=fallback_message,
                level="Low",
                alert_type=AlertType.fallback_reports,
                alerting_metadata={},
            )

        except Exception as e:
            verbose_proxy_logger.error("Error sending weekly spend report %s", e)

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

            key_event_dict: Final = key_event.model_dump()

            # Add Created by information first
            message += "*Action Done by:*\n"
            for key, value in key_event_dict.items():
                if "created_by" in key:
                    message += f"{key}: `{value}`\n"

            # Add args sent to function in the alert
            message += "\n*Arguments passed:*\n"
            request_kwargs: Final = key_event.request_kwargs
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
            verbose_proxy_logger.error("Error sending send_virtual_key_event_slack %s", e)

    async def _request_is_completed(self, request_data: dict | None) -> bool:
        """
        Returns True if the request is completed - either as a success or failure
        """
        if request_data is None:
            return False

        if request_data.get("litellm_status", "") != "success" and request_data.get("litellm_status", "") != "fail":
            ## CHECK IF CACHE IS UPDATED
            litellm_call_id: Final = request_data.get("litellm_call_id", "")
            status: Final[str | None] = await self.internal_usage_cache.async_get_cache(
                key=f"request_status:{litellm_call_id}", local_only=True
            )
            if status is not None and (status == "success" or status == "fail"):
                return True
        return False
