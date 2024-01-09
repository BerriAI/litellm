#### What this does ####
#   picks based on response time (for streaming, this is time to first token)

import dotenv, os, requests, random
from typing import Optional
from datetime import datetime, timedelta

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LowestLatencyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(self, router_cache: DualCache, model_list: list):
        self.router_cache = router_cache
        self.model_list = model_list

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

                response_ms = end_time - start_time

                # ------------
                # Setup values
                # ------------
                latency_key = f"{model_group}_latency_map"

                # ------------
                # Update usage
                # ------------

                ## Latency
                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}
                if id in request_count_dict and isinstance(request_count_dict[id], list):
                    request_count_dict[id] = request_count_dict[id].append(response_ms)
                else:
                    request_count_dict[id] = [response_ms]

                self.router_cache.set_cache(key=latency_key, value=request_count_dict, ttl=self.default_cache_time_seconds) # reset map within window 

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

                response_ms = end_time - start_time

                # ------------
                # Setup values
                # ------------
                latency_key = f"{model_group}_latency_map"

                # ------------
                # Update usage
                # ------------

                ## Latency
                request_count_dict = self.router_cache.get_cache(key=latency_key) or {}
                if id in request_count_dict and isinstance(request_count_dict[id], list):
                    request_count_dict[id] = request_count_dict[id] + [response_ms]
                else:
                    request_count_dict[id] = [response_ms]

                self.router_cache.set_cache(key=latency_key, value=request_count_dict, ttl=self.default_cache_time_seconds) # reset map within window 

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            traceback.print_exc()
            pass

    def get_available_deployments(self, model_group: str, healthy_deployments: list):
        """
        Returns a deployment with the lowest latency
        """
        # get list of potential deployments
        latency_key = f"{model_group}_latency_map"

        request_count_dict = self.router_cache.get_cache(key=latency_key) or {}

        # -----------------------
        # Find lowest used model
        # ----------------------
        lowest_latency = float("inf")
        deployment = None

        if request_count_dict is None:  # base case
            return

        all_deployments = request_count_dict
        for d in healthy_deployments:
            ## if healthy deployment not yet used
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = [0]

        for item, item_latency in all_deployments.items():
            ## get the item from model list
            _deployment = None
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m

            if _deployment is None:
                continue  # skip to next one
            
            # get average latency 
            total = 0.0
            for _call_latency in item_latency:
                if isinstance(_call_latency, timedelta):
                    total += float(_call_latency.total_seconds())
                elif isinstance(_call_latency, float):
                    total += _call_latency
            item_latency = total/len(item_latency)
            if item_latency == 0:
                deployment = _deployment
                break
            elif item_latency < lowest_latency:
                lowest_latency = item_latency
                deployment = _deployment
        if deployment is None:
            deployment = random.choice(healthy_deployments)
        return deployment
