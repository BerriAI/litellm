#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
#   - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
#   - use litellm.success + failure callbacks to log when a request completed
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import dotenv, os, requests
from typing import Optional

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LeastBusyLoggingHandler(CustomLogger):
    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache
        self.mapping_deployment_to_id: dict = {}

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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )
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

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id) - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )
        except Exception as e:
            pass

    def get_available_deployments(self, model_group: str):
        request_count_api_key = f"{model_group}_request_count"
        return_dict = self.router_cache.get_cache(key=request_count_api_key) or {}
        # map deployment to id
        return return_dict
