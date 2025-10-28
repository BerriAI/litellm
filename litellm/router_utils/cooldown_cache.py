"""
Wrapper around router cache. Meant to handle model cooldown logic
"""

import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from typing_extensions import TypedDict

from litellm import verbose_logger
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class CooldownCacheValue(TypedDict):
    exception_received: str
    status_code: str
    timestamp: float
    cooldown_time: float


class CooldownCache:
    def __init__(self, cache: DualCache, default_cooldown_time: float):
        self.cache = cache
        self.default_cooldown_time = default_cooldown_time
        self.in_memory_cache = InMemoryCache()
        # Initialize the masker with custom settings for exception strings
        self.exception_masker = SensitiveDataMasker(
            visible_prefix=50,  # Show first 50 characters
            visible_suffix=0,  # Show last 0 characters
            mask_char="*",  # Use * for masking
        )

    def _common_add_cooldown_logic(
        self, model_id: str, original_exception, exception_status, cooldown_time: float
    ) -> Tuple[str, CooldownCacheValue]:
        try:
            current_time = time.time()
            cooldown_key = f"deployment:{model_id}:cooldown"

            # Store the cooldown information for the deployment separately
            cooldown_data = CooldownCacheValue(
                exception_received=self.exception_masker._mask_value(
                    str(original_exception)
                ),
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
            #########################################################
            # get cooldown time
            # 1. If dynamic cooldown time is set for the model/deployment, use that
            # 2. If no dynamic cooldown time is set, use the default cooldown time set on CooldownCache
            _cooldown_time = cooldown_time
            if _cooldown_time is None:
                _cooldown_time = self.default_cooldown_time
            #########################################################

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

    @staticmethod
    def get_cooldown_cache_key(model_id: str) -> str:
        return f"deployment:{model_id}:cooldown"

    async def async_get_active_cooldowns(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # Generate the keys for the deployments
        keys = [
            CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids
        ]

        # Retrieve the values for the keys using mget
        ## more likely to be none if no models ratelimited. So just check redis every 1s
        ## each redis call adds ~100ms latency.

        ## check in memory cache first
        results = await self.cache.async_batch_get_cache(
            keys=keys, parent_otel_span=parent_otel_span
        )
        active_cooldowns: List[Tuple[str, CooldownCacheValue]] = []

        if results is None or all(v is None for v in results):
            return active_cooldowns
        
        # Process the results
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_active_cooldowns(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # Generate the keys for the deployments
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]
        # Retrieve the values for the keys using mget
        results = (
            self.cache.batch_get_cache(keys=keys, parent_otel_span=parent_otel_span)
            or []
        )

        active_cooldowns = []
        # Process the results
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_min_cooldown(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> float:
        """Return min cooldown time required for a group of model id's."""

        # Generate the keys for the deployments
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]

        # Retrieve the values for the keys using mget
        results = (
            self.cache.batch_get_cache(keys=keys, parent_otel_span=parent_otel_span)
            or []
        )

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
