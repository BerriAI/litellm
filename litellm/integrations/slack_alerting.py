#### What this does ####
#    Class for sending Slack Alerts #
import dotenv, os

dotenv.load_dotenv()  # Loading env variables using dotenv
import copy
import traceback
from litellm._logging import verbose_logger, verbose_proxy_logger
import litellm
from typing import List, Literal, Any, Union, Optional, Dict
from litellm.caching import DualCache
import asyncio
import aiohttp
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
import datetime


class SlackAlerting:
    # Class variables or attributes
    def __init__(
        self,
        alerting_threshold: float = 300,
        alerting: Optional[List] = [],
        alert_types: Optional[
            List[
                Literal[
                    "llm_exceptions",
                    "llm_too_slow",
                    "llm_requests_hanging",
                    "budget_alerts",
                    "db_exceptions",
                ]
            ]
        ] = [
            "llm_exceptions",
            "llm_too_slow",
            "llm_requests_hanging",
            "budget_alerts",
            "db_exceptions",
        ],
        alert_to_webhook_url: Optional[
            Dict
        ] = None,  # if user wants to separate alerts to diff channels
    ):
        self.alerting_threshold = alerting_threshold
        self.alerting = alerting
        self.alert_types = alert_types
        self.internal_usage_cache = DualCache()
        self.async_http_handler = AsyncHTTPHandler()
        self.alert_to_webhook_url = alert_to_webhook_url
        self.langfuse_logger = None

        try:
            from litellm.integrations.langfuse import LangFuseLogger

            self.langfuse_logger = LangFuseLogger(
                os.getenv("LANGFUSE_PUBLIC_KEY"),
                os.getenv("LANGFUSE_SECRET_KEY"),
                flush_interval=1,
            )
        except:
            pass

        pass

    def update_values(
        self,
        alerting: Optional[List] = None,
        alerting_threshold: Optional[float] = None,
        alert_types: Optional[List] = None,
        alert_to_webhook_url: Optional[Dict] = None,
    ):
        if alerting is not None:
            self.alerting = alerting
        if alerting_threshold is not None:
            self.alerting_threshold = alerting_threshold
        if alert_types is not None:
            self.alert_types = alert_types

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
        request_info: str,
        request_data: Optional[dict] = None,
        kwargs: Optional[dict] = None,
        type: Literal["hanging_request", "slow_response"] = "hanging_request",
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ):
        import uuid

        # For now: do nothing as we're debugging why this is not working as expected
        if request_data is not None:
            trace_id = request_data.get("metadata", {}).get(
                "trace_id", None
            )  # get langfuse trace id
            if trace_id is None:
                trace_id = "litellm-alert-trace-" + str(uuid.uuid4())
                request_data["metadata"]["trace_id"] = trace_id
        elif kwargs is not None:
            _litellm_params = kwargs.get("litellm_params", {})
            trace_id = _litellm_params.get("metadata", {}).get(
                "trace_id", None
            )  # get langfuse trace id
            if trace_id is None:
                trace_id = "litellm-alert-trace-" + str(uuid.uuid4())
                _litellm_params["metadata"]["trace_id"] = trace_id

        # Log hanging request as an error on langfuse
        if type == "hanging_request":
            if self.langfuse_logger is not None:
                _logging_kwargs = copy.deepcopy(request_data)
                if _logging_kwargs is None:
                    _logging_kwargs = {}
                _logging_kwargs["litellm_params"] = {}
                request_data = request_data or {}
                _logging_kwargs["litellm_params"]["metadata"] = request_data.get(
                    "metadata", {}
                )
                # log to langfuse in a separate thread
                import threading

                threading.Thread(
                    target=self.langfuse_logger.log_event,
                    args=(
                        _logging_kwargs,
                        None,
                        start_time,
                        end_time,
                        None,
                        print,
                        "ERROR",
                        "Requests is hanging",
                    ),
                ).start()

        _langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        _langfuse_project_id = os.environ.get("LANGFUSE_PROJECT_ID")

        # langfuse urls look like: https://us.cloud.langfuse.com/project/************/traces/litellm-alert-trace-ididi9dk-09292-************

        _langfuse_url = (
            f"{_langfuse_host}/project/{_langfuse_project_id}/traces/{trace_id}"
        )
        request_info += f"\n🪢 Langfuse Trace: {_langfuse_url}"
        return request_info

    def _response_taking_too_long_callback(
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
            self._response_taking_too_long_callback(
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
            )
        )
        request_info = f"\nRequest Model: `{model}`\nAPI Base: `{api_base}`\nMessages: `{messages}`"
        slow_message = f"`Responses are slow - {round(time_difference_float,2)}s response time > Alerting threshold: {self.alerting_threshold}s`"
        if time_difference_float > self.alerting_threshold:
            if "langfuse" in litellm.success_callback:
                request_info = self._add_langfuse_trace_id_to_alert(
                    request_info=request_info, kwargs=kwargs, type="slow_response"
                )
            # add deployment latencies to alert
            if (
                kwargs is not None
                and "litellm_params" in kwargs
                and "metadata" in kwargs["litellm_params"]
            ):
                _metadata = kwargs["litellm_params"]["metadata"]

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

    async def log_failure_event(self, original_exception: Exception):
        pass

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
                    if _api_base is None:
                        _api_base = ""
                    request_info += f"\nAPI Base: `{_api_base}`"
                # only alert hanging responses if they have not been marked as success
                alerting_message = (
                    f"`Requests are hanging - {self.alerting_threshold}s+ request time`"
                )

                if "langfuse" in litellm.success_callback:
                    request_info = self._add_langfuse_trace_id_to_alert(
                        request_info=request_info,
                        request_data=request_data,
                        type="hanging_request",
                        start_time=start_time,
                        end_time=end_time,
                    )

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
        ],
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

        from datetime import datetime
        import json

        # Get the current timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        _proxy_base_url = os.getenv("PROXY_BASE_URL", None)
        formatted_message = (
            f"Level: `{level}`\nTimestamp: `{current_time}`\n\nMessage: {message}"
        )
        if _proxy_base_url is not None:
            formatted_message += f"\n\nProxy URL: `{_proxy_base_url}`"

        # check if we find the slack webhook url in self.alert_to_webhook_url
        if (
            self.alert_to_webhook_url is not None
            and alert_type in self.alert_to_webhook_url
        ):
            slack_webhook_url = self.alert_to_webhook_url[alert_type]
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
