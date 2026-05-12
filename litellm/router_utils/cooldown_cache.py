"""
路由器缓存的封装层，用于处理模型冷却（cooldown）相关逻辑。
"""

import functools
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
        # 使用自定义配置初始化脱敏器，用于处理异常字符串
        self.exception_masker = SensitiveDataMasker(
            visible_prefix=50,  # 保留前 50 个字符
            visible_suffix=0,  # 保留后 0 个字符
            mask_char="*",  # 使用 * 进行脱敏替换
        )

    def _common_add_cooldown_logic(
        self, model_id: str, original_exception, exception_status, cooldown_time: float
    ) -> Tuple[str, CooldownCacheValue]:
        try:
            current_time = time.time()
            cooldown_key = CooldownCache.get_cooldown_cache_key(model_id)

            # 单独存储该部署（deployment）的冷却信息
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
            # 获取冷却时间
            # 1. 如果该模型/部署设置了动态冷却时间，则使用该动态值
            # 2. 否则使用 CooldownCache 上设置的默认冷却时间
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

            # 将缓存的 TTL 设置为冷却时间
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
    @functools.lru_cache(maxsize=1024)
    def get_cooldown_cache_key(model_id: str) -> str:
        return "deployment:" + model_id + ":cooldown"

    async def async_get_active_cooldowns(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # 根据各部署生成对应的缓存 key
        keys = [
            CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids
        ]

        # 使用 mget 批量获取 key 对应的值
        ## 如果没有模型被限流，结果很可能全部为 None，因此 Redis 只需每秒检查一次即可
        ## 每次 Redis 调用大约会增加 100ms 的延迟

        ## 优先检查内存缓存
        results = await self.cache.async_batch_get_cache(
            keys=keys, parent_otel_span=parent_otel_span
        )
        active_cooldowns: List[Tuple[str, CooldownCacheValue]] = []

        if results is None or all(v is None for v in results):
            return active_cooldowns

        # 处理查询结果
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_active_cooldowns(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> List[Tuple[str, CooldownCacheValue]]:
        # 根据各部署生成对应的缓存 key
        keys = [
            CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids
        ]
        # 使用 mget 批量获取 key 对应的值
        results = (
            self.cache.batch_get_cache(keys=keys, parent_otel_span=parent_otel_span)
            or []
        )

        active_cooldowns = []
        # 处理查询结果
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                active_cooldowns.append((model_id, cooldown_cache_value))

        return active_cooldowns

    def get_min_cooldown(
        self, model_ids: List[str], parent_otel_span: Optional[Span]
    ) -> float:
        """返回一组模型 ID 所需的最小冷却时间。"""

        # 根据各部署生成对应的缓存 key
        keys = [f"deployment:{model_id}:cooldown" for model_id in model_ids]

        # 使用 mget 批量获取 key 对应的值
        results = (
            self.cache.batch_get_cache(keys=keys, parent_otel_span=parent_otel_span)
            or []
        )

        min_cooldown_time: Optional[float] = None
        # 处理查询结果
        for model_id, result in zip(model_ids, results):
            if result and isinstance(result, dict):
                cooldown_cache_value = CooldownCacheValue(**result)  # type: ignore
                if min_cooldown_time is None:
                    min_cooldown_time = cooldown_cache_value["cooldown_time"]
                elif cooldown_cache_value["cooldown_time"] < min_cooldown_time:
                    min_cooldown_time = cooldown_cache_value["cooldown_time"]

        return min_cooldown_time or self.default_cooldown_time


# 使用示例：
# cooldown_cache = CooldownCache(cache=your_cache_instance, cooldown_time=your_cooldown_time)
# cooldown_cache.add_deployment_to_cooldown(deployment, original_exception, exception_status)
# active_cooldowns = cooldown_cache.get_active_cooldowns()
