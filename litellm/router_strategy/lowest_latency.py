#### What this does ####
#   picks based on response time (for streaming, this is time to first token)
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

import litellm
from litellm import ModelResponse, token_counter, verbose_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()


class RoutingArgs(LiteLLMBase):
    ttl: float = 1 * 60 * 60  # 1 hour
    lowest_latency_buffer: float = 0
    max_latency_list_size: int = 10


class LowestLatencyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(
        self, router_cache: DualCache, model_list: list, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.model_list = model_list
        self.routing_args = RoutingArgs(**routing_args)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update latency usage on success
            """
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                # ------------
                # Setup values
                # ------------
                """
                {
                    {model_group}_map: {
                        id: {
                            "latency": [..]
                            f"{date:hour:minute}" : {"tpm": 34, "rpm": 3}
                        }
                    }
                }
                """
                latency_key = f"{model_group}_map"

                current_date = datetime.now().strftime("%Y-%m-%d")
                current_hour = datetime.now().strftime("%H")
                current_minute = datetime.now().strftime("%M")
                precise_minute = f"{current_date}-{current_hour}-{current_minute}"

                response_ms: timedelta = end_time - start_time
                time_to_first_token_response_time: Optional[timedelta] = None

                if kwargs.get("stream", None) is not None and kwargs["stream"] is True:
                    # only log ttft for streaming request
                    time_to_first_token_response_time = (
                        kwargs.get("completion_start_time", end_time) - start_time
                    )

                final_value: Union[float, timedelta] = response_ms
                time_to_first_token: Optional[float] = None
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    _usage = getattr(response_obj, "usage", None)
                    if _usage is not None:
                        completion_tokens = _usage.completion_tokens
                        total_tokens = _usage.total_tokens
                        final_value = float(
                            response_ms.total_seconds() / completion_tokens
                        )

                        if time_to_first_token_response_time is not None:
                            time_to_first_token = float(
                                time_to_first_token_response_time.total_seconds()
                                / completion_tokens
                            )

                # ------------
                # Update usage
                # ------------

                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

                if id not in request_count_dict:
                    request_count_dict[id] = {}

                ## Latency
                if (
                    len(request_count_dict[id].get("latency", []))
                    < self.routing_args.max_latency_list_size
                ):
                    request_count_dict[id].setdefault("latency", []).append(final_value)
                else:
                    request_count_dict[id]["latency"] = request_count_dict[id][
                        "latency"
                    ][: self.routing_args.max_latency_list_size - 1] + [final_value]

                ## Time to first token
                if time_to_first_token is not None:
                    if (
                        len(request_count_dict[id].get("time_to_first_token", []))
                        < self.routing_args.max_latency_list_size
                    ):
                        request_count_dict[id].setdefault(
                            "time_to_first_token", []
                        ).append(time_to_first_token)
                    else:
                        request_count_dict[id][
                            "time_to_first_token"
                        ] = request_count_dict[id]["time_to_first_token"][
                            : self.routing_args.max_latency_list_size - 1
                        ] + [
                            time_to_first_token
                        ]

                if precise_minute not in request_count_dict[id]:
                    request_count_dict[id][precise_minute] = {}

                ## TPM
                request_count_dict[id][precise_minute]["tpm"] = (
                    request_count_dict[id][precise_minute].get("tpm", 0) + total_tokens
                )

                ## RPM
                request_count_dict[id][precise_minute]["rpm"] = (
                    request_count_dict[id][precise_minute].get("rpm", 0) + 1
                )

                self.router_cache.set_cache(
                    key=latency_key, value=request_count_dict, ttl=self.routing_args.ttl
                )  # reset map within window

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Check if Timeout Error, if timeout set deployment latency -> 100
        """
        try:
            _exception = kwargs.get("exception", None)
            if isinstance(_exception, litellm.Timeout):
                if kwargs["litellm_params"].get("metadata") is None:
                    pass
                else:
                    model_group = kwargs["litellm_params"]["metadata"].get(
                        "model_group", None
                    )

                    id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                    if model_group is None or id is None:
                        return
                    elif isinstance(id, int):
                        id = str(id)

                    # ------------
                    # Setup values
                    # ------------
                    """
                    {
                        {model_group}_map: {
                            id: {
                                "latency": [..]
                                f"{date:hour:minute}" : {"tpm": 34, "rpm": 3}
                            }
                        }
                    }
                    """
                    latency_key = f"{model_group}_map"
                    request_count_dict = (
                        self.router_cache.get_cache(key=latency_key) or {}
                    )

                    if id not in request_count_dict:
                        request_count_dict[id] = {}

                    ## Latency - give 1000s penalty for failing
                    if (
                        len(request_count_dict[id].get("latency", []))
                        < self.routing_args.max_latency_list_size
                    ):
                        request_count_dict[id].setdefault("latency", []).append(1000.0)
                    else:
                        request_count_dict[id]["latency"] = request_count_dict[id][
                            "latency"
                        ][: self.routing_args.max_latency_list_size - 1] + [1000.0]

                    self.router_cache.set_cache(
                        key=latency_key,
                        value=request_count_dict,
                        ttl=self.routing_args.ttl,
                    )  # reset map within window
            else:
                # do nothing if it's not a timeout error
                return
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.prompt_injection_detection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update latency usage on success
            """
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                # ------------
                # Setup values
                # ------------
                """
                {
                    {model_group}_map: {
                        id: {
                            "latency": [..]
                            "time_to_first_token": [..]
                            f"{date:hour:minute}" : {"tpm": 34, "rpm": 3}
                        }
                    }
                }
                """
                latency_key = f"{model_group}_map"

                current_date = datetime.now().strftime("%Y-%m-%d")
                current_hour = datetime.now().strftime("%H")
                current_minute = datetime.now().strftime("%M")
                precise_minute = f"{current_date}-{current_hour}-{current_minute}"

                response_ms: timedelta = end_time - start_time
                time_to_first_token_response_time: Optional[timedelta] = None
                if kwargs.get("stream", None) is not None and kwargs["stream"] is True:
                    # only log ttft for streaming request
                    time_to_first_token_response_time = (
                        kwargs.get("completion_start_time", end_time) - start_time
                    )

                final_value: Union[float, timedelta] = response_ms
                total_tokens = 0
                time_to_first_token: Optional[float] = None

                if isinstance(response_obj, ModelResponse):
                    _usage = getattr(response_obj, "usage", None)
                    if _usage is not None:
                        completion_tokens = _usage.completion_tokens
                        total_tokens = _usage.total_tokens
                        final_value = float(
                            response_ms.total_seconds() / completion_tokens
                        )

                        if time_to_first_token_response_time is not None:
                            time_to_first_token = float(
                                time_to_first_token_response_time.total_seconds()
                                / completion_tokens
                            )
                # ------------
                # Update usage
                # ------------

                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

                if id not in request_count_dict:
                    request_count_dict[id] = {}

                ## Latency
                if (
                    len(request_count_dict[id].get("latency", []))
                    < self.routing_args.max_latency_list_size
                ):
                    request_count_dict[id].setdefault("latency", []).append(final_value)
                else:
                    request_count_dict[id]["latency"] = request_count_dict[id][
                        "latency"
                    ][: self.routing_args.max_latency_list_size - 1] + [final_value]

                ## Time to first token
                if time_to_first_token is not None:
                    if (
                        len(request_count_dict[id].get("time_to_first_token", []))
                        < self.routing_args.max_latency_list_size
                    ):
                        request_count_dict[id].setdefault(
                            "time_to_first_token", []
                        ).append(time_to_first_token)
                    else:
                        request_count_dict[id][
                            "time_to_first_token"
                        ] = request_count_dict[id]["time_to_first_token"][
                            : self.routing_args.max_latency_list_size - 1
                        ] + [
                            time_to_first_token
                        ]

                if precise_minute not in request_count_dict[id]:
                    request_count_dict[id][precise_minute] = {}

                ## TPM
                request_count_dict[id][precise_minute]["tpm"] = (
                    request_count_dict[id][precise_minute].get("tpm", 0) + total_tokens
                )

                ## RPM
                request_count_dict[id][precise_minute]["rpm"] = (
                    request_count_dict[id][precise_minute].get("rpm", 0) + 1
                )

                self.router_cache.set_cache(
                    key=latency_key, value=request_count_dict, ttl=self.routing_args.ttl
                )  # reset map within window

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.router_strategy.lowest_latency.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Returns a deployment with the lowest latency
        """
        # get list of potential deployments
        latency_key = f"{model_group}_map"
        _latency_per_deployment = {}

        request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_latency = float("inf")

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%H")
        current_minute = datetime.now().strftime("%M")
        precise_minute = f"{current_date}-{current_hour}-{current_minute}"

        deployment = None

        if request_count_dict is None:  # base case
            return

        all_deployments = request_count_dict
        for d in healthy_deployments:
            ## if healthy deployment not yet used
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = {
                    "latency": [0],
                    precise_minute: {"tpm": 0, "rpm": 0},
                }

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0

        # randomly sample from all_deployments, incase all deployments have latency=0.0
        _items = all_deployments.items()

        all_deployments = random.sample(list(_items), len(_items))
        all_deployments = dict(all_deployments)
        ### GET AVAILABLE DEPLOYMENTS ### filter out any deployments > tpm/rpm limits

        potential_deployments = []
        for item, item_map in all_deployments.items():
            ## get the item from model list
            _deployment = None
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m

            if _deployment is None:
                continue  # skip to next one

            _deployment_tpm = (
                _deployment.get("tpm", None)
                or _deployment.get("litellm_params", {}).get("tpm", None)
                or _deployment.get("model_info", {}).get("tpm", None)
                or float("inf")
            )

            _deployment_rpm = (
                _deployment.get("rpm", None)
                or _deployment.get("litellm_params", {}).get("rpm", None)
                or _deployment.get("model_info", {}).get("rpm", None)
                or float("inf")
            )
            item_latency = item_map.get("latency", [])
            item_ttft_latency = item_map.get("time_to_first_token", [])
            item_rpm = item_map.get(precise_minute, {}).get("rpm", 0)
            item_tpm = item_map.get(precise_minute, {}).get("tpm", 0)

            # get average latency or average ttft (depending on streaming/non-streaming)
            total: float = 0.0
            if (
                request_kwargs is not None
                and request_kwargs.get("stream", None) is not None
                and request_kwargs["stream"] is True
                and len(item_ttft_latency) > 0
            ):
                for _call_latency in item_ttft_latency:
                    if isinstance(_call_latency, float):
                        total += _call_latency
            else:
                for _call_latency in item_latency:
                    if isinstance(_call_latency, float):
                        total += _call_latency
            item_latency = total / len(item_latency)

            # -------------- #
            # Debugging Logic
            # -------------- #
            # We use _latency_per_deployment to log to langfuse, slack - this is not used to make a decision on routing
            # this helps a user to debug why the router picked a specfic deployment      #
            _deployment_api_base = _deployment.get("litellm_params", {}).get(
                "api_base", ""
            )
            if _deployment_api_base is not None:
                _latency_per_deployment[_deployment_api_base] = item_latency
            # -------------- #
            # End of Debugging Logic
            # -------------- #

            if (
                item_tpm + input_tokens > _deployment_tpm
                or item_rpm + 1 > _deployment_rpm
            ):  # if user passed in tpm / rpm in the model_list
                continue
            else:
                potential_deployments.append((_deployment, item_latency))

        if len(potential_deployments) == 0:
            return None

        # Sort potential deployments by latency
        sorted_deployments = sorted(potential_deployments, key=lambda x: x[1])

        # Find lowest latency deployment
        lowest_latency = sorted_deployments[0][1]

        # Find deployments within buffer of lowest latency
        buffer = self.routing_args.lowest_latency_buffer * lowest_latency

        valid_deployments = [
            x for x in sorted_deployments if x[1] <= lowest_latency + buffer
        ]

        # Pick a random deployment from valid deployments
        random_valid_deployment = random.choice(valid_deployments)
        deployment = random_valid_deployment[0]

        if request_kwargs is not None and "metadata" in request_kwargs:
            request_kwargs["metadata"][
                "_latency_per_deployment"
            ] = _latency_per_deployment
        return deployment
