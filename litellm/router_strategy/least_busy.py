#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
#   - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
#   - use litellm.success + failure callbacks to log when a request completed
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import random
from typing import Final

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Log when a model is being used.

        Caching based on model group.
        """
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key: Final = f"{model_group}_request_count"
                # update cache
                request_count_dict: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
                request_count_dict[id] = request_count_dict.get(id, 0) + 1

                self.router_cache.set_cache(key=request_count_api_key, value=request_count_dict)
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key: Final = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(key=request_count_api_key, value=request_count_dict)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key: Final = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(key=request_count_api_key, value=request_count_dict)

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key: Final = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(key=request_count_api_key, value=request_count_dict)

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key: Final = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(key=request_count_api_key, value=request_count_dict)

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    def _get_available_deployments(
        self,
        healthy_deployments: list,
        all_deployments: dict,
    ):
        """
        Helper to get deployments using least busy strategy
        """
        if not healthy_deployments:
            return None

        min_traffic = float("inf")
        min_deployment = healthy_deployments[0]

        for deployment in healthy_deployments:
            dep_id = str(deployment.get("model_info", {}).get("id", ""))
            traffic = all_deployments.get(dep_id, 0)

            if traffic == 0:
                return deployment

            if traffic < min_traffic:
                min_traffic = traffic
                min_deployment = deployment

        return min_deployment

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ):
        """
        Sync helper to get deployments using least busy strategy
        """
        request_count_api_key: Final = f"{model_group}_request_count"
        all_deployments: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )

    async def async_get_available_deployments(self, model_group: str, healthy_deployments: list):
        """
        Async helper to get deployments using least busy strategy
        """
        request_count_api_key: Final = f"{model_group}_request_count"
        all_deployments: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )
