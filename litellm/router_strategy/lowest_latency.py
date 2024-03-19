#### What this does ####
#   picks based on response time (for streaming, this is time to first token)
from pydantic import BaseModel, Extra, Field, root_validator
import dotenv, os, requests, random
from typing import Optional, Union, List, Dict
from datetime import datetime, timedelta

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm import ModelResponse
from litellm import token_counter


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


class RoutingArgs(LiteLLMBase):
    ttl: int = 1 * 60 * 60  # 1 hour


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

                final_value = response_ms
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    completion_tokens = response_obj.usage.completion_tokens
                    total_tokens = response_obj.usage.total_tokens
                    final_value = float(response_ms.total_seconds() / completion_tokens)

                # ------------
                # Update usage
                # ------------

                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

                if id not in request_count_dict:
                    request_count_dict[id] = {}

                ## Latency
                request_count_dict[id].setdefault("latency", []).append(final_value)

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
            traceback.print_exc()
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

                final_value = response_ms
                total_tokens = 0

                if isinstance(response_obj, ModelResponse):
                    completion_tokens = response_obj.usage.completion_tokens
                    total_tokens = response_obj.usage.total_tokens
                    final_value = float(response_ms.total_seconds() / completion_tokens)

                # ------------
                # Update usage
                # ------------

                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

                if id not in request_count_dict:
                    request_count_dict[id] = {}

                ## Latency
                request_count_dict[id].setdefault("latency", []).append(final_value)

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
            traceback.print_exc()
            pass

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Returns a deployment with the lowest latency
        """
        # get list of potential deployments
        latency_key = f"{model_group}_map"

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
        except:
            input_tokens = 0

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
            item_rpm = item_map.get(precise_minute, {}).get("rpm", 0)
            item_tpm = item_map.get(precise_minute, {}).get("tpm", 0)

            # get average latency
            total: float = 0.0
            for _call_latency in item_latency:
                if isinstance(_call_latency, float):
                    total += _call_latency
            item_latency = total / len(item_latency)
            if item_latency == 0:
                deployment = _deployment
                break
            elif (
                item_tpm + input_tokens > _deployment_tpm
                or item_rpm + 1 > _deployment_rpm
            ):  # if user passed in tpm / rpm in the model_list
                continue
            elif item_latency < lowest_latency:
                lowest_latency = item_latency
                deployment = _deployment
        if deployment is None:
            deployment = random.choice(healthy_deployments)
        return deployment
