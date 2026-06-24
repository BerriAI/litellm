import random
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


@dataclass
class BalancedSmartRoutingArgs:
    max_queue_ttl_s: float = 1.0
    queue_poll_s: float = 0.01
    max_concurrent_requests: int = 10
    default_tokens_per_second: float = 50.0
    active_request_weight: float = 10.0
    tokens_per_second_weight: float = 1.0
    ttft_weight: float = 1.0
    failure_penalty: float = 5.0
    failure_cooldown_s: float = 5.0
    ewma_alpha: float = 0.2
    redis_key_prefix: str = "balanced_smart_router"
    redis_ttl_s: int = 3600


@dataclass
class DeploymentStats:
    active: int = 0
    selected: int = 0
    successes: int = 0
    failures: int = 0
    ewma_tps: float = 0.0
    ewma_ttft_s: float = 0.0
    last_selected_at: float = 0.0
    last_failure_at: float = 0.0


class BalancedSmartRoutingHandler(CustomLogger):
    """
    Router strategy that combines bounded concurrency, observed throughput,
    time-to-first-token, and recent failures.

    The strategy acquires capacity before returning a deployment. Success and
    failure callbacks release that capacity. When Redis is configured for the
    router cache, acquire/release is atomic and shared across LiteLLM pods.
    """

    _ACQUIRE_SCRIPT = """
local key = KEYS[1]
local max_active = tonumber(ARGV[1])
local now = ARGV[2]
local ttl = tonumber(ARGV[3])
local active = tonumber(redis.call('HGET', key, 'active') or '0')
if active >= max_active then
  return 0
end
redis.call('HINCRBY', key, 'active', 1)
redis.call('HINCRBY', key, 'selected', 1)
redis.call('HSET', key, 'last_selected_at', now)
redis.call('EXPIRE', key, ttl)
return 1
"""

    _RELEASE_SUCCESS_SCRIPT = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local active = tonumber(redis.call('HGET', key, 'active') or '0')
if active > 0 then
  redis.call('HINCRBY', key, 'active', -1)
else
  redis.call('HSET', key, 'active', 0)
end
redis.call('HINCRBY', key, 'successes', 1)
redis.call('EXPIRE', key, ttl)
return 1
"""

    _RELEASE_FAILURE_SCRIPT = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local now = ARGV[2]
local active = tonumber(redis.call('HGET', key, 'active') or '0')
if active > 0 then
  redis.call('HINCRBY', key, 'active', -1)
else
  redis.call('HSET', key, 'active', 0)
end
redis.call('HINCRBY', key, 'failures', 1)
redis.call('HSET', key, 'last_failure_at', now)
redis.call('EXPIRE', key, ttl)
return 1
"""

    def __init__(self, router_cache: DualCache, routing_args: Optional[dict] = None):
        self.router_cache = router_cache
        allowed_args = {field.name for field in fields(BalancedSmartRoutingArgs)}
        args = {
            key: value
            for key, value in dict(routing_args or {}).items()
            if key in allowed_args
        }
        self.routing_args = BalancedSmartRoutingArgs(**args)

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[dict],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[dict]:
        deadline = time.time() + self.routing_args.max_queue_ttl_s
        while True:
            acquired = self._try_acquire_one(model_group, healthy_deployments)
            if acquired is not None:
                return acquired
            if time.time() >= deadline:
                return None
            time.sleep(self.routing_args.queue_poll_s)

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[dict],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> Optional[dict]:
        import asyncio

        deadline = time.time() + self.routing_args.max_queue_ttl_s
        while True:
            acquired = await self._async_try_acquire_one(
                model_group, healthy_deployments
            )
            if acquired is not None:
                return acquired
            if time.time() >= deadline:
                return None
            await asyncio.sleep(self.routing_args.queue_poll_s)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._release_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_release_success(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._release_failure(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._async_release_failure(kwargs)

    def _try_acquire_one(
        self, model_group: str, healthy_deployments: List[dict]
    ) -> Optional[dict]:
        for deployment, deployment_id, _stats in self._rank_deployments(
            model_group, healthy_deployments
        ):
            if self._acquire(deployment_id):
                return deployment
        return None

    async def _async_try_acquire_one(
        self, model_group: str, healthy_deployments: List[dict]
    ) -> Optional[dict]:
        ranked = await self._async_rank_deployments(model_group, healthy_deployments)
        for deployment, deployment_id, _stats in ranked:
            if await self._async_acquire(deployment_id):
                return deployment
        return None

    def _rank_deployments(
        self, model_group: str, healthy_deployments: List[dict]
    ) -> List[Tuple[dict, str, DeploymentStats]]:
        rows: List[Tuple[dict, str, DeploymentStats]] = []
        for deployment in healthy_deployments:
            deployment_id = self._deployment_id(model_group, deployment)
            rows.append((deployment, deployment_id, self._get_stats(deployment_id)))
        return self._sort_ranked_rows(rows)

    async def _async_rank_deployments(
        self, model_group: str, healthy_deployments: List[dict]
    ) -> List[Tuple[dict, str, DeploymentStats]]:
        rows: List[Tuple[dict, str, DeploymentStats]] = []
        for deployment in healthy_deployments:
            deployment_id = self._deployment_id(model_group, deployment)
            rows.append(
                (deployment, deployment_id, await self._async_get_stats(deployment_id))
            )
        return self._sort_ranked_rows(rows)

    def _sort_ranked_rows(
        self, rows: List[Tuple[dict, str, DeploymentStats]]
    ) -> List[Tuple[dict, str, DeploymentStats]]:
        now = time.time()
        available = [
            row
            for row in rows
            if now - row[2].last_failure_at >= self.routing_args.failure_cooldown_s
        ]
        rows_to_rank = available or rows
        random.shuffle(rows_to_rank)
        return sorted(rows_to_rank, key=lambda row: self._score(row[2], now))

    def _score(self, stats: DeploymentStats, now: float) -> float:
        concurrency_pressure = (
            stats.active / max(1, self.routing_args.max_concurrent_requests)
        ) * self.routing_args.active_request_weight

        observed_tps = stats.ewma_tps or self.routing_args.default_tokens_per_second
        tps_pressure = (
            self.routing_args.default_tokens_per_second / max(0.001, observed_tps)
        ) * self.routing_args.tokens_per_second_weight

        ttft_pressure = stats.ewma_ttft_s * self.routing_args.ttft_weight
        total = max(1, stats.successes + stats.failures)
        failure_pressure = (
            stats.failures / total
        ) * self.routing_args.failure_penalty
        cooldown_pressure = 0.0
        if now - stats.last_failure_at < self.routing_args.failure_cooldown_s:
            cooldown_pressure = self.routing_args.failure_penalty
        return (
            concurrency_pressure
            + tps_pressure
            + ttft_pressure
            + failure_pressure
            + cooldown_pressure
        )

    def _acquire(self, deployment_id: str) -> bool:
        redis_client = self._redis_client()
        if redis_client is not None:
            result = redis_client.eval(
                self._ACQUIRE_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.max_concurrent_requests,
                time.time(),
                self.routing_args.redis_ttl_s,
            )
            return int(result) == 1

        stats = self._get_stats(deployment_id)
        if stats.active >= self.routing_args.max_concurrent_requests:
            return False
        stats.active += 1
        stats.selected += 1
        stats.last_selected_at = time.time()
        self._set_stats(deployment_id, stats)
        return True

    async def _async_acquire(self, deployment_id: str) -> bool:
        redis_client = await self._async_redis_client()
        if redis_client is not None:
            result = await redis_client.eval(
                self._ACQUIRE_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.max_concurrent_requests,
                time.time(),
                self.routing_args.redis_ttl_s,
            )
            return int(result) == 1

        stats = await self._async_get_stats(deployment_id)
        if stats.active >= self.routing_args.max_concurrent_requests:
            return False
        stats.active += 1
        stats.selected += 1
        stats.last_selected_at = time.time()
        await self._async_set_stats(deployment_id, stats)
        return True

    def _release_success(self, kwargs, response_obj, start_time, end_time) -> None:
        deployment_id = self._deployment_id_from_kwargs(kwargs)
        if deployment_id is None:
            return

        redis_client = self._redis_client()
        if redis_client is not None:
            redis_client.eval(
                self._RELEASE_SUCCESS_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.redis_ttl_s,
            )
            stats = self._get_stats(deployment_id)
        else:
            stats = self._get_stats(deployment_id)
            stats.active = max(0, stats.active - 1)
            stats.successes += 1

        self._update_success_metrics(stats, response_obj, start_time, end_time)
        self._set_stats(deployment_id, stats)

    async def _async_release_success(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        deployment_id = self._deployment_id_from_kwargs(kwargs)
        if deployment_id is None:
            return

        redis_client = await self._async_redis_client()
        if redis_client is not None:
            await redis_client.eval(
                self._RELEASE_SUCCESS_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.redis_ttl_s,
            )
            stats = await self._async_get_stats(deployment_id)
        else:
            stats = await self._async_get_stats(deployment_id)
            stats.active = max(0, stats.active - 1)
            stats.successes += 1

        self._update_success_metrics(stats, response_obj, start_time, end_time)
        await self._async_set_stats(deployment_id, stats)

    def _release_failure(self, kwargs) -> None:
        deployment_id = self._deployment_id_from_kwargs(kwargs)
        if deployment_id is None:
            return
        redis_client = self._redis_client()
        if redis_client is not None:
            redis_client.eval(
                self._RELEASE_FAILURE_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.redis_ttl_s,
                time.time(),
            )
            return

        stats = self._get_stats(deployment_id)
        stats.active = max(0, stats.active - 1)
        stats.failures += 1
        stats.last_failure_at = time.time()
        self._set_stats(deployment_id, stats)

    async def _async_release_failure(self, kwargs) -> None:
        deployment_id = self._deployment_id_from_kwargs(kwargs)
        if deployment_id is None:
            return
        redis_client = await self._async_redis_client()
        if redis_client is not None:
            await redis_client.eval(
                self._RELEASE_FAILURE_SCRIPT,
                1,
                self._redis_key(deployment_id),
                self.routing_args.redis_ttl_s,
                time.time(),
            )
            return

        stats = await self._async_get_stats(deployment_id)
        stats.active = max(0, stats.active - 1)
        stats.failures += 1
        stats.last_failure_at = time.time()
        await self._async_set_stats(deployment_id, stats)

    def _update_success_metrics(
        self, stats: DeploymentStats, response_obj, start_time, end_time
    ) -> None:
        duration_s = self._duration_seconds(start_time, end_time)
        output_tokens = self._output_tokens(response_obj)
        if output_tokens > 0 and duration_s > 0:
            observed_tps = output_tokens / duration_s
            stats.ewma_tps = self._ewma(stats.ewma_tps, observed_tps)
        if duration_s > 0:
            stats.ewma_ttft_s = self._ewma(stats.ewma_ttft_s, duration_s)

    def _ewma(self, old: float, new: float) -> float:
        if old <= 0:
            return new
        alpha = self.routing_args.ewma_alpha
        return (alpha * new) + ((1 - alpha) * old)

    def _get_stats(self, deployment_id: str) -> DeploymentStats:
        redis_client = self._redis_client()
        if redis_client is not None:
            return self._stats_from_mapping(
                redis_client.hgetall(self._redis_key(deployment_id)) or {}
            )
        raw = self.router_cache.get_cache(key=self._cache_key(deployment_id)) or {}
        return self._stats_from_mapping(raw)

    async def _async_get_stats(self, deployment_id: str) -> DeploymentStats:
        redis_client = await self._async_redis_client()
        if redis_client is not None:
            return self._stats_from_mapping(
                await redis_client.hgetall(self._redis_key(deployment_id)) or {}
            )
        raw = (
            await self.router_cache.async_get_cache(key=self._cache_key(deployment_id))
            or {}
        )
        return self._stats_from_mapping(raw)

    def _set_stats(self, deployment_id: str, stats: DeploymentStats) -> None:
        redis_client = self._redis_client()
        if redis_client is not None:
            redis_client.hset(self._redis_key(deployment_id), mapping=asdict(stats))
            redis_client.expire(
                self._redis_key(deployment_id), self.routing_args.redis_ttl_s
            )
            return
        self.router_cache.set_cache(
            key=self._cache_key(deployment_id), value=asdict(stats)
        )

    async def _async_set_stats(self, deployment_id: str, stats: DeploymentStats) -> None:
        redis_client = await self._async_redis_client()
        if redis_client is not None:
            await redis_client.hset(
                self._redis_key(deployment_id), mapping=asdict(stats)
            )
            await redis_client.expire(
                self._redis_key(deployment_id), self.routing_args.redis_ttl_s
            )
            return
        await self.router_cache.async_set_cache(
            key=self._cache_key(deployment_id), value=asdict(stats)
        )

    def _stats_from_mapping(self, raw: Dict[Any, Any]) -> DeploymentStats:
        normalized: Dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            normalized[str(key)] = value

        return DeploymentStats(
            active=max(0, int(float(normalized.get("active", 0) or 0))),
            selected=max(0, int(float(normalized.get("selected", 0) or 0))),
            successes=max(0, int(float(normalized.get("successes", 0) or 0))),
            failures=max(0, int(float(normalized.get("failures", 0) or 0))),
            ewma_tps=float(normalized.get("ewma_tps", 0) or 0),
            ewma_ttft_s=float(normalized.get("ewma_ttft_s", 0) or 0),
            last_selected_at=float(normalized.get("last_selected_at", 0) or 0),
            last_failure_at=float(normalized.get("last_failure_at", 0) or 0),
        )

    def _deployment_id(self, model_group: str, deployment: dict) -> str:
        model_info = deployment.get("model_info", {}) or {}
        raw_id = (
            model_info.get("balanced_smart_backend_id")
            or model_info.get("id")
            or deployment.get("model_name")
            or model_group
        )
        return f"{model_group}:{raw_id}"

    def _deployment_id_from_kwargs(self, kwargs) -> Optional[str]:
        litellm_params = kwargs.get("litellm_params", {}) or {}
        model_info = litellm_params.get("model_info", {}) or {}
        metadata = litellm_params.get("metadata", {}) or {}
        model_group = metadata.get("model_group") or litellm_params.get("model")
        raw_id = model_info.get("balanced_smart_backend_id") or model_info.get("id")
        if model_group is None or raw_id is None:
            return None
        return f"{model_group}:{raw_id}"

    def _cache_key(self, deployment_id: str) -> str:
        return f"{self.routing_args.redis_key_prefix}:{deployment_id}"

    def _redis_key(self, deployment_id: str) -> str:
        redis_cache = getattr(self.router_cache, "redis_cache", None)
        key = self._cache_key(deployment_id)
        if redis_cache is not None:
            return redis_cache.check_and_fix_namespace(key)
        return key

    def _redis_client(self):
        redis_cache = getattr(self.router_cache, "redis_cache", None)
        return getattr(redis_cache, "redis_client", None)

    async def _async_redis_client(self):
        redis_cache = getattr(self.router_cache, "redis_cache", None)
        if redis_cache is None:
            return None
        await redis_cache.init_async_client()
        return getattr(redis_cache, "async_redis_conn", None)

    def _duration_seconds(self, start_time, end_time) -> float:
        if start_time is None or end_time is None:
            return 0.0
        delta = end_time - start_time
        if hasattr(delta, "total_seconds"):
            return max(0.0, delta.total_seconds())
        return max(0.0, float(delta))

    def _output_tokens(self, response_obj) -> int:
        usage = getattr(response_obj, "usage", None)
        if usage is None and isinstance(response_obj, dict):
            usage = response_obj.get("usage")
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return int(
            getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
