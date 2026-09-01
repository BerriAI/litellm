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
                request_count_dict: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(key=request_count_api_key, value=request_count_dict)

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
                request_count_dict: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(key=request_count_api_key, value=request_count_dict)

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
                request_count_dict: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(key=request_count_api_key, value=request_count_dict)

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
                request_count_dict: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
                request_count_value: Final[int | None] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(key=request_count_api_key, value=request_count_dict)

                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    def _get_available_deployments(
        self,
        healthy_deployments: list,
        all_deployments: dict,
    ):
        for d in healthy_deployments:
            if d["model_info"]["id"] not in all_deployments:
                all_deployments[d["model_info"]["id"]] = 0
        # counts drain back to zero between requests, so in light traffic every request is a tie
        traffic: Final = tuple(all_deployments[d["model_info"]["id"]] for d in healthy_deployments)
        min_traffic: Final = min(traffic)
        return random.choice(tuple(d for d, t in zip(healthy_deployments, traffic) if t == min_traffic))

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ):
        request_count_api_key: Final = f"{model_group}_request_count"
        all_deployments: Final = self.router_cache.get_cache(key=request_count_api_key) or {}
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )

    async def async_get_available_deployments(self, model_group: str, healthy_deployments: list):
        request_count_api_key: Final = f"{model_group}_request_count"
        all_deployments: Final = await self.router_cache.async_get_cache(key=request_count_api_key) or {}
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )
