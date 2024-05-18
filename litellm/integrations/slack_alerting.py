#### What this does ####
#    Class for sending Slack Alerts #
import dotenv, os
from litellm.proxy._types import UserAPIKeyAuth
from litellm._logging import verbose_logger, verbose_proxy_logger
import litellm, threading
from typing import List, Literal, Any, Union, Optional, Dict
from litellm.caching import DualCache
import asyncio
import aiohttp
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
import datetime
from pydantic import BaseModel
from enum import Enum
from datetime import datetime as dt, timedelta, timezone
from litellm.integrations.custom_logger import CustomLogger
import random


class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):
        try:
            return self.model_dump()  # noqa
        except:
            # if using pydantic v1
            return self.dict()


class SlackAlertingArgs(LiteLLMBase):
    default_daily_report_frequency: int = 12 * 60 * 60  # 12 hours
    daily_report_frequency: int = int(
        os.getenv("SLACK_DAILY_REPORT_FREQUENCY", default_daily_report_frequency)
    )
    report_check_interval: int = 5 * 60  # 5 minutes


class DeploymentMetrics(LiteLLMBase):
    """
    Metrics per deployment, stored in cache

    Used for daily reporting
    """

    id: str
    """id of deployment in router model list"""

    failed_request: bool
    """did it fail the request?"""

    latency_per_output_token: Optional[float]
    """latency/output token of deployment"""

    updated_at: dt
    """Current time of deployment being updated"""


class SlackAlertingCacheKeys(Enum):
    """
    Enum for deployment daily metrics keys - {deployment_id}:{enum}
    """

    failed_requests_key = "failed_requests_daily_metrics"
    latency_key = "latency_daily_metrics"
    report_sent_key = "daily_metrics_report_sent"


class SlackAlerting(CustomLogger):
    """
    Class for sending Slack Alerts
    """

    # Class variables or attributes
    def __init__(
        self,
        internal_usage_cache: Optional[DualCache] = None,
        alerting_threshold: float = 300,  # threshold for slow / hanging llm responses (in seconds)
        alerting: Optional[List] = [],
        alert_types: List[
            Literal[
                "llm_exceptions",
                "llm_too_slow",
                "llm_requests_hanging",
                "budget_alerts",
                "db_exceptions",
                "daily_reports",
                "spend_reports",
                "cooldown_deployment",
                "new_model_added",
            ]
        ] = [
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
            "daily_reports",
            "spend_reports",
            "cooldown_deployment",
            "new_model_added",
        ],
        alert_to_webhook_url: Optional[
            Dict
        ] = None,  # if user wants to separate alerts to diff channels
        alerting_args={},
        default_webhook_url: Optional[str] = None,
    ):
        self.alerting_threshold = alerting_threshold
        self.alerting = alerting
        self.alert_types = alert_types
        self.internal_usage_cache = internal_usage_cache or DualCache()
        self.async_http_handler = AsyncHTTPHandler()
        self.alert_to_webhook_url = alert_to_webhook_url
        self.is_running = False
        self.alerting_args = SlackAlertingArgs(**alerting_args)
        self.default_webhook_url = default_webhook_url

    def update_values(
        self,
        alerting: Optional[List] = None,
        alerting_threshold: Optional[float] = None,
        alert_types: Optional[List] = None,
        alert_to_webhook_url: Optional[Dict] = None,
        alerting_args: Optional[Dict] = None,
    ):
        if alerting is not None:
            self.alerting = alerting
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
        if alert_types is not None:
            self.alert_types = alert_types
        if alerting_args is not None:
            self.alerting_args = SlackAlertingArgs(**alerting_args)
        if alert_to_webhook_url is not None:
            # update the dict
            if self.alert_to_webhook_url is None:
                self.alert_to_webhook_url = alert_to_webhook_url
            else:
                self.alert_to_webhook_url.update(alert_to_webhook_url)

    async def deployment_in_cooldown(self):
        pass

    async def deployment_removed_from_cooldown(self):
        pass

    def _all_possible_alert_types(self):
        # used by the UI to show all supported alert types
        # Note: This is not the alerts the user has configured, instead it's all possible alert types a user can select
        return [
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
        ]

    def _add_langfuse_trace_id_to_alert(
        self,
        request_data: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Returns langfuse trace url

        - check:
        -> existing_trace_id
        -> trace_id
        -> litellm_call_id
        """
        # do nothing for now
        if request_data is not None:
            trace_id = None
            if (
                request_data.get("metadata", {}).get("existing_trace_id", None)
                is not None
            ):
                trace_id = request_data["metadata"]["existing_trace_id"]
            elif request_data.get("metadata", {}).get("trace_id", None) is not None:
                trace_id = request_data["metadata"]["trace_id"]
            elif request_data.get("litellm_logging_obj", None) is not None and hasattr(
                request_data["litellm_logging_obj"], "model_call_details"
            ):
                trace_id = request_data["litellm_logging_obj"].model_call_details[
                    "litellm_call_id"
                ]
            if litellm.utils.langFuseLogger is not None:
                base_url = litellm.utils.langFuseLogger.Langfuse.base_url
                return f"{base_url}/trace/{trace_id}"
        return None

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
            try:
                # try sorting deployments by latency
                _deployment_latencies = sorted(
                    _deployment_latencies.items(), key=lambda x: x[1]
                )
                _deployment_latencies = dict(_deployment_latencies)
            except:
                pass
            for api_base, latency in _deployment_latencies.items():
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

        time_difference_float, model, api_base, messages = (
            self._response_taking_too_long_callback_helper(
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
        )
        if litellm.turn_off_message_logging:
            messages = "Message not logged. `litellm.turn_off_message_logging=True`."
        request_info = f"\nRequest Model: `{model}`\nAPI Base: `{api_base}`\nMessages: `{messages}`"
        slow_message = f"`Responses are slow - {round(time_difference_float,2)}s response time > Alerting threshold: {self.alerting_threshold}s`"
        if time_difference_float > self.alerting_threshold:
            # add deployment latencies to alert
            if (
                kwargs is not None
                and "litellm_params" in kwargs
                and "metadata" in kwargs["litellm_params"]
            ):
                _metadata = kwargs["litellm_params"]["metadata"]
                request_info = litellm.utils._add_key_name_and_team_to_alert(
                    request_info=request_info, metadata=_metadata
                )

                _deployment_latency_map = self._get_deployment_latencies_to_alert(
                    metadata=_metadata
                )
                if _deployment_latency_map is not None:
                    request_info += (
                        f"\nAvailable Deployment Latencies\n{_deployment_latency_map}"
                    )
            await self.send_alert(
                message=slow_message + request_info,
                level="Low",
                alert_type="llm_too_slow",
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
                )

                return_val += 1

            ## LATENCY ##
            if deployment_metrics.latency_per_output_token is not None:
                await self.internal_usage_cache.async_increment_cache(
                    key="{}:{}".format(
                        deployment_metrics.id, SlackAlertingCacheKeys.latency_key.value
                    ),
                    value=deployment_metrics.latency_per_output_token,
                )

                return_val += 1

            return return_val
        except Exception as e:
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
        message = f"\n\nHere are today's key metrics 📈: \n\n"

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
        await self.internal_usage_cache.async_batch_set_cache(
            cache_list=combined_metrics_cache_keys
        )

        # send alert
        await self.send_alert(message=message, level="Low", alert_type="daily_reports")

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
            except:
                messages = ""

            if litellm.turn_off_message_logging:
                messages = (
                    "Message not logged. `litellm.turn_off_message_logging=True`."
                )
            request_info = f"\nRequest Model: `{model}`\nMessages: `{messages}`"
        else:
            request_info = ""

        if type == "hanging_request":
            await asyncio.sleep(
                self.alerting_threshold
            )  # Set it to 5 minutes - i'd imagine this might be different for streaming, non-streaming, non-completion (embedding + img) requests
            if (
                request_data is not None
                and request_data.get("litellm_status", "") != "success"
                and request_data.get("litellm_status", "") != "fail"
            ):
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
                    _metadata = request_data["metadata"]
                    _api_base = _metadata.get("api_base", "")

                    request_info = litellm.utils._add_key_name_and_team_to_alert(
                        request_info=request_info, metadata=_metadata
                    )

                    if _api_base is None:
                        _api_base = ""
                    request_info += f"\nAPI Base: `{_api_base}`"
                # only alert hanging responses if they have not been marked as success
                alerting_message = (
                    f"`Requests are hanging - {self.alerting_threshold}s+ request time`"
                )

                if "langfuse" in litellm.success_callback:
                    langfuse_url = self._add_langfuse_trace_id_to_alert(
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
                    alert_type="llm_requests_hanging",
                )

    async def budget_alerts(
        self,
        type: Literal[
            "token_budget",
            "user_budget",
            "user_and_proxy_budget",
            "failed_budgets",
            "failed_tracking",
            "projected_limit_exceeded",
        ],
        user_max_budget: float,
        user_current_spend: float,
        user_info=None,
        error_message="",
    ):
        if self.alerting is None or self.alert_types is None:
            # do nothing if alerting is not switched on
            return
        if "budget_alerts" not in self.alert_types:
            return
        _id: str = "default_id"  # used for caching
        if type == "user_and_proxy_budget":
            user_info = dict(user_info)
            user_id = user_info["user_id"]
            _id = user_id
            max_budget = user_info["max_budget"]
            spend = user_info["spend"]
            user_email = user_info["user_email"]
            user_info = f"""\nUser ID: {user_id}\nMax Budget: ${max_budget}\nSpend: ${spend}\nUser Email: {user_email}"""
        elif type == "token_budget":
            token_info = dict(user_info)
            token = token_info["token"]
            _id = token
            spend = token_info["spend"]
            max_budget = token_info["max_budget"]
            user_id = token_info["user_id"]
            user_info = f"""\nToken: {token}\nSpend: ${spend}\nMax Budget: ${max_budget}\nUser ID: {user_id}"""
        elif type == "failed_tracking":
            user_id = str(user_info)
            _id = user_id
            user_info = f"\nUser ID: {user_id}\n Error {error_message}"
            message = "Failed Tracking Cost for" + user_info
            await self.send_alert(
                message=message, level="High", alert_type="budget_alerts"
            )
            return
        elif type == "projected_limit_exceeded" and user_info is not None:
            """
            Input variables:
            user_info = {
                "key_alias": key_alias,
                "projected_spend": projected_spend,
                "projected_exceeded_date": projected_exceeded_date,
            }
            user_max_budget=soft_limit,
            user_current_spend=new_spend
            """
            message = f"""\n🚨 `ProjectedLimitExceededError` 💸\n\n`Key Alias:` {user_info["key_alias"]} \n`Expected Day of Error`: {user_info["projected_exceeded_date"]} \n`Current Spend`: {user_current_spend} \n`Projected Spend at end of month`: {user_info["projected_spend"]} \n`Soft Limit`: {user_max_budget}"""
            await self.send_alert(
                message=message, level="High", alert_type="budget_alerts"
            )
            return
        else:
            user_info = str(user_info)

        # percent of max_budget left to spend
        if user_max_budget > 0:
            percent_left = (user_max_budget - user_current_spend) / user_max_budget
        else:
            percent_left = 0
        verbose_proxy_logger.debug(
            f"Budget Alerts: Percent left: {percent_left} for {user_info}"
        )

        ## PREVENTITIVE ALERTING ## - https://github.com/BerriAI/litellm/issues/2727
        # - Alert once within 28d period
        # - Cache this information
        # - Don't re-alert, if alert already sent
        _cache: DualCache = self.internal_usage_cache

        # check if crossed budget
        if user_current_spend >= user_max_budget:
            verbose_proxy_logger.debug("Budget Crossed for %s", user_info)
            message = "Budget Crossed for" + user_info
            result = await _cache.async_get_cache(key=message)
            if result is None:
                await self.send_alert(
                    message=message, level="High", alert_type="budget_alerts"
                )
                await _cache.async_set_cache(key=message, value="SENT", ttl=2419200)
            return

        # check if 5% of max budget is left
        if percent_left <= 0.05:
            message = "5% budget left for" + user_info
            cache_key = "alerting:{}".format(_id)
            result = await _cache.async_get_cache(key=cache_key)
            if result is None:
                await self.send_alert(
                    message=message, level="Medium", alert_type="budget_alerts"
                )

                await _cache.async_set_cache(key=cache_key, value="SENT", ttl=2419200)

            return

        # check if 15% of max budget is left
        if percent_left <= 0.15:
            message = "15% budget left for" + user_info
            result = await _cache.async_get_cache(key=message)
            if result is None:
                await self.send_alert(
                    message=message, level="Low", alert_type="budget_alerts"
                )
                await _cache.async_set_cache(key=message, value="SENT", ttl=2419200)
            return
        return

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

        await self.send_alert(
            message=message, level="Low", alert_type="new_model_added"
        )
        pass

    async def model_removed_alert(self, model_name: str):
        pass

    async def send_alert(
        self,
        message: str,
        level: Literal["Low", "Medium", "High"],
        alert_type: Literal[
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
            "daily_reports",
            "spend_reports",
            "new_model_added",
            "cooldown_deployment",
        ],
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

        if alert_type not in self.alert_types:
            return

        from datetime import datetime
        import json

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
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        # check if we find the slack webhook url in self.alert_to_webhook_url
        if (
            self.alert_to_webhook_url is not None
            and alert_type in self.alert_to_webhook_url
        ):
            slack_webhook_url = self.alert_to_webhook_url[alert_type]
        elif self.default_webhook_url is not None:
            slack_webhook_url = self.default_webhook_url
        else:
            slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", None)

        if slack_webhook_url is None:
            raise Exception("Missing SLACK_WEBHOOK_URL from environment")
        payload = {"text": formatted_message}
        headers = {"Content-type": "application/json"}

        response = await self.async_http_handler.post(
            url=slack_webhook_url,
            headers=headers,
            data=json.dumps(payload),
        )
        if response.status_code == 200:
            pass
        else:
            print("Error sending slack alert. Error=", response.text)  # noqa

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log deployment latency"""
        if "daily_reports" in self.alert_types:
            model_id = (
                kwargs.get("litellm_params", {}).get("model_info", {}).get("id", "")
            )
            response_s: timedelta = end_time - start_time

            final_value = response_s
            total_tokens = 0

            if isinstance(response_obj, litellm.ModelResponse):
                completion_tokens = response_obj.usage.completion_tokens
                final_value = float(response_s.total_seconds() / completion_tokens)

            await self.async_update_daily_reports(
                DeploymentMetrics(
                    id=model_id,
                    failed_request=False,
                    latency_per_output_token=final_value,
                    updated_at=litellm.utils.get_utc_datetime(),
                )
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failure + deployment latency"""
        if "daily_reports" in self.alert_types:
            model_id = (
                kwargs.get("litellm_params", {}).get("model_info", {}).get("id", "")
            )
            await self.async_update_daily_reports(
                DeploymentMetrics(
                    id=model_id,
                    failed_request=True,
                    latency_per_output_token=None,
                    updated_at=litellm.utils.get_utc_datetime(),
                )
            )

    async def _run_scheduler_helper(self, llm_router) -> bool:
        """
        Returns:
        - True -> report sent
        - False -> report not sent
        """
        report_sent_bool = False

        report_sent = await self.internal_usage_cache.async_get_cache(
            key=SlackAlertingCacheKeys.report_sent_key.value
        )  # None | datetime

        current_time = litellm.utils.get_utc_datetime()

        if report_sent is None:
            _current_time = current_time.isoformat()
            await self.internal_usage_cache.async_set_cache(
                key=SlackAlertingCacheKeys.report_sent_key.value,
                value=_current_time,
            )
        else:
            # Check if current time - interval >= time last sent
            delta_naive = timedelta(seconds=self.alerting_args.daily_report_frequency)
            if isinstance(report_sent, str):
                report_sent = dt.fromisoformat(report_sent)

            # Ensure report_sent is an aware datetime object
            if report_sent.tzinfo is None:
                report_sent = report_sent.replace(tzinfo=timezone.utc)

            # Calculate delta as an aware datetime object with the same timezone as report_sent
            delta = report_sent - delta_naive

            current_time_utc = current_time.astimezone(timezone.utc)
            delta_utc = delta.astimezone(timezone.utc)

            if current_time_utc >= delta_utc:
                # Sneak in the reporting logic here
                await self.send_daily_reports(router=llm_router)
                # Also, don't forget to update the report_sent time after sending the report!
                _current_time = current_time.isoformat()
                await self.internal_usage_cache.async_set_cache(
                    key=SlackAlertingCacheKeys.report_sent_key.value,
                    value=_current_time,
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

    async def send_weekly_spend_report(self):
        """ """
        try:
            from litellm.proxy.proxy_server import _get_spend_report_for_time_range

            todays_date = datetime.datetime.now().date()
            week_before = todays_date - datetime.timedelta(days=7)

            weekly_spend_per_team, weekly_spend_per_tag = (
                await _get_spend_report_for_time_range(
                    start_date=week_before.strftime("%Y-%m-%d"),
                    end_date=todays_date.strftime("%Y-%m-%d"),
                )
            )

            _weekly_spend_message = f"*💸 Weekly Spend Report for `{week_before.strftime('%m-%d-%Y')} - {todays_date.strftime('%m-%d-%Y')}` *\n"

            if weekly_spend_per_team is not None:
                _weekly_spend_message += "\n*Team Spend Report:*\n"
                for spend in weekly_spend_per_team:
                    _team_spend = spend["total_spend"]
                    _team_spend = float(_team_spend)
                    # round to 4 decimal places
                    _team_spend = round(_team_spend, 4)
                    _weekly_spend_message += (
                        f"Team: `{spend['team_alias']}` | Spend: `${_team_spend}`\n"
                    )

            if weekly_spend_per_tag is not None:
                _weekly_spend_message += "\n*Tag Spend Report:*\n"
                for spend in weekly_spend_per_tag:
                    _tag_spend = spend["total_spend"]
                    _tag_spend = float(_tag_spend)
                    # round to 4 decimal places
                    _tag_spend = round(_tag_spend, 4)
                    _weekly_spend_message += f"Tag: `{spend['individual_request_tag']}` | Spend: `${_tag_spend}`\n"

            await self.send_alert(
                message=_weekly_spend_message,
                level="Low",
                alert_type="spend_reports",
            )
        except Exception as e:
            verbose_proxy_logger.error("Error sending weekly spend report", e)

    async def send_monthly_spend_report(self):
        """ """
        try:
            from calendar import monthrange

            from litellm.proxy.proxy_server import _get_spend_report_for_time_range

            todays_date = datetime.datetime.now().date()
            first_day_of_month = todays_date.replace(day=1)
            _, last_day_of_month = monthrange(todays_date.year, todays_date.month)
            last_day_of_month = first_day_of_month + datetime.timedelta(
                days=last_day_of_month - 1
            )

            monthly_spend_per_team, monthly_spend_per_tag = (
                await _get_spend_report_for_time_range(
                    start_date=first_day_of_month.strftime("%Y-%m-%d"),
                    end_date=last_day_of_month.strftime("%Y-%m-%d"),
                )
            )

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
                alert_type="spend_reports",
            )
        except Exception as e:
            verbose_proxy_logger.error("Error sending weekly spend report", e)
