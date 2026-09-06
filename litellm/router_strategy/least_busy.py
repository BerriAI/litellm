import random
from typing import Final

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


def _count_key(model_group: str, deployment_id: str) -> str:
    return f"{model_group}_request_count:{deployment_id}"


def _took_a_slot(kwargs: dict) -> bool:
    # pre_call stamps api_call_start_time right before taking the slot; cache hits and pre-call rejections never get there
    return kwargs.get("cache_hit") is not True and kwargs.get("api_call_start_time") is not None


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

                self.router_cache.increment_cache(key=_count_key(model_group, id), value=1)
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None or not _took_a_slot(kwargs):
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                self._release(_count_key(model_group, id))

                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None or not _took_a_slot(kwargs):
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                self._release(_count_key(model_group, id))

                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None or not _took_a_slot(kwargs):
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                await self._async_release(_count_key(model_group, id))

                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None or not _took_a_slot(kwargs):
                pass
            else:
                model_group: Final = kwargs["litellm_params"]["metadata"].get("model_group", None)
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                await self._async_release(_count_key(model_group, id))

                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    def _release(self, key: str) -> None:
        if self.router_cache.increment_cache(key=key, value=-1) < 0:
            self.router_cache.set_cache(key=key, value=0)

    async def _async_release(self, key: str) -> None:
        remaining: Final = await self.router_cache.async_increment_cache(key=key, value=-1)
        if remaining is not None and remaining < 0:
            await self.router_cache.async_set_cache(key=key, value=0)

    def _read_counts(self, keys: list) -> list:
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is None:
            return [count or 0 for count in self.router_cache.in_memory_cache.batch_get_cache(keys)]
        found: Final = redis_cache.batch_get_cache(key_list=keys) or {}
        return [found.get(key) or 0 for key in keys]

    async def _async_read_counts(self, keys: list) -> list:
        redis_cache: Final = self.router_cache.redis_cache
        if redis_cache is None:
            return [count or 0 for count in await self.router_cache.in_memory_cache.async_batch_get_cache(keys)]
        found: Final = await redis_cache.async_batch_get_cache(key_list=keys) or {}
        return [found.get(key) or 0 for key in keys]

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
        keys: Final = [_count_key(model_group, str(d["model_info"]["id"])) for d in healthy_deployments]
        counts: Final = self._read_counts(keys)
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=dict(zip([d["model_info"]["id"] for d in healthy_deployments], counts)),
        )

    async def async_get_available_deployments(self, model_group: str, healthy_deployments: list):
        keys: Final = [_count_key(model_group, str(d["model_info"]["id"])) for d in healthy_deployments]
        counts: Final = await self._async_read_counts(keys)
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=dict(zip([d["model_info"]["id"] for d in healthy_deployments], counts)),
        )
