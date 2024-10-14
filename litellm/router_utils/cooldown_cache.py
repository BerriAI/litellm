"""
Wrapper around router cache. Meant to handle model cooldown logic
"""

import json
import time
from typing import List, Optional, Tuple, TypedDict

from litellm import verbose_logger
from litellm.caching.caching import DualCache


class CooldownCacheValue(TypedDict):
    exception_received: str
    status_code: str
    timestamp: float
    cooldown_time: float


class CooldownCache:
    def __init__(self, cache: DualCache, default_cooldown_time: float):
        self.cache = cache
        self.default_cooldown_time = default_cooldown_time

    def _common_add_cooldown_logic(
        self, model_id: str, original_exception, exception_status, cooldown_time: float
    ) -> Tuple[str, CooldownCacheValue]:
        try:
            current_time = time.time()
            cooldown_key = f"deployment:{model_id}:cooldown"

            # Store the cooldown information for the deployment separately
            cooldown_data = CooldownCacheValue(
                exception_received=str(original_exception),
                status_code=str(exception_status),
                timestamp=current_time,
                cooldown_time=cooldown_time,
            )

            return cooldown_key, cooldown_data
        except Exception as e:
            verbose_logger.error(
                "CooldownCache::_common_add_cooldown_logic - Exception occurred - {}".format(
                    str(e)
                )
            )
            raise e

    def add_deployment_to_cooldown(
        self,
        model_id: str,
        original_exception: Exception,
        exception_status: int,
        cooldown_time: Optional[float],
    ):
        try:
            _cooldown_time = cooldown_time or self.default_cooldown_time
            cooldown_key, cooldown_data = self._common_add_cooldown_logic(
                model_id=model_id,
                original_exception=original_exception,
                exception_status=exception_status,
                cooldown_time=_cooldown_time,
            )

            # Set the cache with a TTL equal to the cooldown time
            self.cache.set_cache(
                value=cooldown_data,
                key=cooldown_key,
                ttl=_cooldown_time,
            )
        except Exception as e:
            verbose_logger.error(
                "CooldownCache::add_deployment_to_cooldown - Exception occurred - {}".format(
                    str(e)
                )
            )
            raise e

    async def async_get_active_cooldowns(
        self, model_ids: List[str]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # Generate the keys for the deployments
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]

        # Retrieve the values for the keys using mget
        results = await self.cache.async_batch_get_cache(keys=keys) or []

        active_cooldowns = []
        # Process the results
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_active_cooldowns(
        self, model_ids: List[str]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # Generate the keys for the deployments
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]

        # Retrieve the values for the keys using mget
        results = self.cache.batch_get_cache(keys=keys) or []

        active_cooldowns = []
        # Process the results
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_min_cooldown(self, model_ids: List[str]) -> float:
        """Return min cooldown time required for a group of model id's."""

        # Generate the keys for the deployments
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]

        # Retrieve the values for the keys using mget
        results = self.cache.batch_get_cache(keys=keys) or []

        min_cooldown_time: Optional[float] = None
        # Process the results
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                if min_cooldown_time is None:
                    min_cooldown_time = cooldown_cache_value["cooldown_time"]
                elif cooldown_cache_value["cooldown_time"] < min_cooldown_time:
                    min_cooldown_time = cooldown_cache_value["cooldown_time"]

        return min_cooldown_time or self.default_cooldown_time


# Usage example:
# cooldown_cache = CooldownCache(cache=your_cache_instance, cooldown_time=your_cooldown_time)
# cooldown_cache.add_deployment_to_cooldown(deployment, original_exception, exception_status)
# active_cooldowns = cooldown_cache.get_active_cooldowns()
