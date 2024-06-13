#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
#   - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
#   - use litellm.success + failure callbacks to log when a request completed
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import dotenv, os, requests, random  # type: ignore
from typing import Optional
import traceback
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache, model_list: list):
        self.router_cache = router_cache
        self.mapping_deployment_to_id: dict = {}
        self.model_list = model_list

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Log when a model is being used.

        Caching based on model group.
        """
        try:
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

                request_count_api_key = f"{model_group}_request_count"
                # update cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id, 0) + 1

                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )
        except Exception as e:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception as e:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception as e:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception as e:
            pass

    def get_available_deployments(self, model_group: str, healthy_deployments: list):
        request_count_api_key = f"{model_group}_request_count"
        deployments = self.router_cache.get_cache(key=request_count_api_key) or {}
        all_deployments = deployments
        for d in healthy_deployments:
            ## if healthy deployment not yet used
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = 0
        # map deployment to id
        # pick least busy deployment
        min_traffic = float("inf")
        min_deployment = None
        for k, v in all_deployments.items():
            if v < min_traffic:
                min_traffic = v
                min_deployment = k
        if min_deployment is not None:
            ## check if min deployment is a string, if so, cast it to int
            for m in healthy_deployments:
                if m["model_info"]["id"] == min_deployment:
                    return m
            min_deployment = random.choice(healthy_deployments)
        else:
            min_deployment = random.choice(healthy_deployments)
        return min_deployment
