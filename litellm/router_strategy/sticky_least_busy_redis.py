"""
Sticky-Least-Busy-Redis routing strategy.

Routes requests from the same conversation to the same deployment (for KV cache reuse
on vLLM/SGLang nodes), but rebalances to the least-busy deployment when the sticky
target is overloaded.

Unlike sticky-least-busy which uses a consistent hash ring to map conversations to
deployments deterministically, this strategy stores the mapping in Redis:

  1. Hash the conversation identity (first user message + user identifier) to compute
     a sticky key that is constant across all turns.
  2. Look up the sticky key in Redis to find the previously assigned deployment.
  3. Compute reference load using avg+min blend: (avg_load + min_load) / 2.
     This catches skewed distributions where avg alone is pulled up by outliers.
  4. If found and healthy and not overloaded (load < threshold * reference), route there.
  5. If not found, unhealthy, or overloaded, assign the least-busy deployment and
     store/update the mapping in Redis.
  6. Track in-flight requests via Redis (atomic increment/decrement) with dedup
     to avoid the streaming bug where log_pre_api_call fires per SSE chunk.

Advantages over consistent hash ring:
  - First request always goes to the actual least-busy node (not an arbitrary ring slot).
  - When rebalancing occurs, the new mapping persists for future turns.
  - Simpler code (no hash ring, no virtual nodes, no bisect).
  - Adaptive: the mapping evolves with traffic patterns.

Requires Redis for both sticky mapping and request counting. Without Redis, degrades
to pure least-busy routing with random tie-breaking (no stickiness).
"""

import hashlib
import random
from typing import Dict, List, Optional

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class StickyLeastBusyRedisLoggingHandler(CustomLogger):
    """
    Routing handler that combines conversation stickiness with load-aware rebalancing,
    using Redis for sticky key-to-deployment mapping.

    Uses a class-level singleton to survive Router re-creation. The LiteLLM proxy
    may create new Router instances per-request or on config syncs. Without a
    singleton, each new instance gets a fresh _seen_call_ids dict, breaking
    streaming dedup and causing in-flight counts to grow monotonically.
    """

    _instance: Optional["StickyLeastBusyRedisLoggingHandler"] = None

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __new__(
        cls,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        cache_ttl: int = 600,
        sticky_ttl: int = 900,
    ):
        """
        Singleton: return existing instance if one exists.
        Only update router_cache (which may change across Router instances).
        """
        if cls._instance is not None:
            cls._instance.router_cache = router_cache
            verbose_router_logger.info(
                f"[StickyLeastBusyRedis REUSE] Reusing existing handler "
                f"(seen_call_ids={len(cls._instance._seen_call_ids)})"
            )
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        cache_ttl: int = 600,
        sticky_ttl: int = 900,
    ):
        """
        Args:
            router_cache: DualCache instance for Redis + in-memory caching.
            imbalance_threshold: If sticky node load > threshold * avg_load, rebalance.
            cache_ttl: TTL in seconds for request count cache keys.
            sticky_ttl: TTL in seconds for sticky key-to-deployment mapping in Redis.
                        Refreshed on every request so active conversations never expire.
        """
        # Skip re-initialization if already initialized (singleton reuse)
        if hasattr(self, "_initialized") and self._initialized:
            self.router_cache = router_cache
            return

        self._initialized = True
        self.router_cache = router_cache
        self.imbalance_threshold = imbalance_threshold
        self.cache_ttl = cache_ttl
        self.sticky_ttl = sticky_ttl

        # Streaming dedup: track which litellm_call_ids we've already incremented.
        # log_pre_api_call fires for every SSE chunk in streaming - only increment once.
        self._seen_call_ids: Dict[str, bool] = {}
        self._seen_call_ids_max_size: int = 10000

        # Prometheus metrics (lazy init — no-op if prometheus_client not installed)
        try:
            from prometheus_client import Counter, Gauge

            self._routing_decisions = Counter(
                "litellm_sticky_routing_decisions_total",
                "Routing decisions made by sticky-least-busy strategy",
                ["model_group", "deployment_id", "decision", "strategy"],
            )
            self._routing_in_flight = Gauge(
                "litellm_sticky_routing_in_flight",
                "In-flight requests per deployment tracked by sticky routing",
                ["model_group", "deployment_id"],
            )
            self._routing_fallback = Counter(
                "litellm_sticky_routing_fallback_total",
                "Fallback events in sticky routing",
                ["model_group", "reason", "strategy"],
            )
        except ValueError:
            # Already registered by another handler instance — reuse from registry
            from prometheus_client import REGISTRY

            self._routing_decisions = REGISTRY._names_to_collectors.get(
                "litellm_sticky_routing_decisions_total"
            )
            self._routing_in_flight = REGISTRY._names_to_collectors.get(
                "litellm_sticky_routing_in_flight"
            )
            self._routing_fallback = REGISTRY._names_to_collectors.get(
                "litellm_sticky_routing_fallback_total"
            )
            # Guard against partial registration — if any lookup returned None,
            # fall back to NoOpMetric to avoid AttributeError on .labels().inc()
            if not all([self._routing_decisions, self._routing_in_flight, self._routing_fallback]):
                from litellm.types.integrations.prometheus import NoOpMetric

                self._routing_decisions = self._routing_decisions or NoOpMetric()
                self._routing_in_flight = self._routing_in_flight or NoOpMetric()
                self._routing_fallback = self._routing_fallback or NoOpMetric()
        except Exception:
            from litellm.types.integrations.prometheus import NoOpMetric

            self._routing_decisions = NoOpMetric()
            self._routing_in_flight = NoOpMetric()
            self._routing_fallback = NoOpMetric()

        verbose_router_logger.info(
            f"[StickyLeastBusyRedis INIT] Initialized with "
            f"imbalance_threshold={imbalance_threshold}, "
            f"cache_ttl={cache_ttl}s, "
            f"sticky_ttl={sticky_ttl}s"
        )

    # =========================================================================
    # Prefix Hashing
    # =========================================================================

    @staticmethod
    def compute_sticky_key(
        messages: Optional[List[Dict[str, str]]],
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Compute a deterministic hash that identifies the conversation per user.

        The key must be STABLE across all turns of the same conversation so that
        consecutive messages route to the same node (KV cache reuse). We achieve
        this by hashing the conversation's "identity" -- the first user message
        plus a user identifier -- which never changes as the conversation grows.

        Algorithm:
        - None/empty messages -> None (no stickiness, degrades to least-busy).
        - Extract the first user message content (O(1) scan, stops at first user msg).
        - Combine with user_id (API key or user ID) for per-user differentiation.
        - Hash with SHA-256.

        This ensures:
        - Same conversation always produces the same hash on every turn.
        - Different conversations (different first user question) get different hashes.
        - The hash is deterministic across pods (SHA-256).
        - Different users with the same system prompt AND same first question
          get different hashes (per-user stickiness, no hotspot).
        """
        if not messages:
            verbose_router_logger.debug(
                "[StickyLeastBusyRedis STICKY-KEY] No messages provided, sticky_key=None"
            )
            return None

        # Extract the first user message content.
        # O(1) scan -- stops at first user message, doesn't touch the rest.
        first_user_content: Optional[str] = None
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")
                # Handle multimodal content (list of parts) -- extract text parts
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    first_user_content = " ".join(text_parts) if text_parts else ""
                else:
                    first_user_content = str(content) if content is not None else ""
                break
            elif role in ("system", "developer"):
                continue
            else:
                break

        if first_user_content is None:
            verbose_router_logger.debug(
                "[StickyLeastBusyRedis STICKY-KEY] No user message found, sticky_key=None"
            )
            return None

        # Combine first user message + user identifier for per-user stickiness.
        hash_input = first_user_content
        if user_id:
            hash_input = f"{user_id}:{first_user_content}"

        sticky_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis STICKY-KEY] "
            f"total_messages={len(messages)}, "
            f"has_user_id={user_id is not None}, "
            f"sticky_key={sticky_key[:16]}..."
        )
        return sticky_key

    # =========================================================================
    # Sticky Route Cache Keys (key -> deployment mapping)
    # =========================================================================

    def _get_sticky_route_cache_key(
        self, model_group: str, sticky_key: str
    ) -> str:
        return f"sticky_route:{model_group}:{sticky_key}"

    def _get_sticky_deployment(
        self, model_group: str, sticky_key: str
    ) -> Optional[str]:
        """Sync: look up sticky key-to-deployment mapping from Redis."""
        cache_key = self._get_sticky_route_cache_key(model_group, sticky_key)
        try:
            value = self.router_cache.get_cache(key=cache_key, redis_only=True)
            if value is not None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis REDIS-GET] "
                    f"sticky_key={sticky_key[:16]}... -> deployment_id={value}"
                )
                # Refresh TTL on read (active conversation keeps mapping alive)
                self._refresh_sticky_ttl(cache_key)
            else:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis REDIS-GET] "
                    f"sticky_key={sticky_key[:16]}... -> NOT FOUND"
                )
            return str(value) if value is not None else None
        except Exception as e:
            verbose_router_logger.warning(
                f"[StickyLeastBusyRedis WARNING] Redis GET failed: {e}"
            )
            return None

    async def _async_get_sticky_deployment(
        self, model_group: str, sticky_key: str
    ) -> Optional[str]:
        """Async: look up sticky key-to-deployment mapping from Redis."""
        cache_key = self._get_sticky_route_cache_key(model_group, sticky_key)
        try:
            value = await self.router_cache.async_get_cache(
                key=cache_key, redis_only=True
            )
            if value is not None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis REDIS-GET] "
                    f"sticky_key={sticky_key[:16]}... -> deployment_id={value}"
                )
                await self._async_refresh_sticky_ttl(cache_key)
            else:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis REDIS-GET] "
                    f"sticky_key={sticky_key[:16]}... -> NOT FOUND"
                )
            return str(value) if value is not None else None
        except Exception as e:
            verbose_router_logger.warning(
                f"[StickyLeastBusyRedis WARNING] Redis async GET failed: {e}"
            )
            return None

    def _set_sticky_deployment(
        self, model_group: str, sticky_key: str, deployment_id: str
    ) -> None:
        """Sync: store sticky key-to-deployment mapping in Redis."""
        cache_key = self._get_sticky_route_cache_key(model_group, sticky_key)
        try:
            self.router_cache.set_cache(
                key=cache_key, value=deployment_id, ttl=self.sticky_ttl
            )
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis REDIS-SET] "
                f"sticky_key={sticky_key[:16]}... -> deployment_id={deployment_id}"
            )
        except Exception as e:
            verbose_router_logger.warning(
                f"[StickyLeastBusyRedis WARNING] Redis SET failed: {e}"
            )

    async def _async_set_sticky_deployment(
        self, model_group: str, sticky_key: str, deployment_id: str
    ) -> None:
        """Async: store sticky key-to-deployment mapping in Redis."""
        cache_key = self._get_sticky_route_cache_key(model_group, sticky_key)
        try:
            await self.router_cache.async_set_cache(
                key=cache_key, value=deployment_id, ttl=self.sticky_ttl
            )
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis REDIS-SET] "
                f"sticky_key={sticky_key[:16]}... -> deployment_id={deployment_id}"
            )
        except Exception as e:
            verbose_router_logger.warning(
                f"[StickyLeastBusyRedis WARNING] Redis async SET failed: {e}"
            )

    # =========================================================================
    # Request Count Cache Keys
    # =========================================================================

    def _get_request_count_cache_key(
        self, model_group: str, deployment_id: str
    ) -> str:
        return f"sticky_lb:{model_group}:{deployment_id}:request_count"

    # =========================================================================
    # TTL Refresh
    # =========================================================================

    def _refresh_cache_ttl(self, cache_key: str) -> None:
        """
        Refresh Redis TTL on every increment/decrement for request count keys.

        The shared redis_cache.increment_cache only sets TTL on first key creation.
        By refreshing TTL on every access, the key only expires after cache_ttl
        seconds of ZERO activity to that deployment.
        """
        try:
            if (
                self.router_cache.redis_cache is not None
                and hasattr(self.router_cache.redis_cache, "redis_client")
                and self.router_cache.redis_cache.redis_client is not None
            ):
                self.router_cache.redis_cache.redis_client.expire(
                    cache_key, self.cache_ttl
                )
        except Exception:
            pass  # Best-effort

    async def _async_refresh_cache_ttl(self, cache_key: str) -> None:
        """Async variant: refresh Redis TTL for request count keys."""
        try:
            if self.router_cache.redis_cache is not None:
                _redis_client = self.router_cache.redis_cache.init_async_client()
                await _redis_client.expire(cache_key, self.cache_ttl)
        except Exception:
            pass

    def _refresh_sticky_ttl(self, cache_key: str) -> None:
        """Refresh Redis TTL for sticky route mapping keys."""
        try:
            if (
                self.router_cache.redis_cache is not None
                and hasattr(self.router_cache.redis_cache, "redis_client")
                and self.router_cache.redis_cache.redis_client is not None
            ):
                self.router_cache.redis_cache.redis_client.expire(
                    cache_key, self.sticky_ttl
                )
        except Exception:
            pass  # Best-effort

    async def _async_refresh_sticky_ttl(self, cache_key: str) -> None:
        """Async variant: refresh Redis TTL for sticky route mapping keys."""
        try:
            if self.router_cache.redis_cache is not None:
                _redis_client = self.router_cache.redis_cache.init_async_client()
                await _redis_client.expire(cache_key, self.sticky_ttl)
        except Exception:
            pass

    # =========================================================================
    # Streaming Dedup
    # =========================================================================

    def _should_increment(self, litellm_call_id: str) -> bool:
        """
        Returns True only for the FIRST call with this litellm_call_id.
        Subsequent calls (SSE streaming chunks) return False.
        """
        if litellm_call_id in self._seen_call_ids:
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DEDUP] Skipping duplicate increment "
                f"for call_id={litellm_call_id[:16]}... "
                f"(streaming chunk dedup)"
            )
            return False

        if len(self._seen_call_ids) >= self._seen_call_ids_max_size:
            evict_count = self._seen_call_ids_max_size // 10
            keys_to_remove = list(self._seen_call_ids.keys())[:evict_count]
            for key in keys_to_remove:
                self._seen_call_ids.pop(key, None)
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DEDUP] Evicted {evict_count} old call_ids "
                f"(was at capacity {self._seen_call_ids_max_size})"
            )

        self._seen_call_ids[litellm_call_id] = True
        return True

    def _cleanup_call_id(self, litellm_call_id: str) -> None:
        self._seen_call_ids.pop(litellm_call_id, None)

    # =========================================================================
    # CustomLogger Callbacks - Request Tracking
    # =========================================================================

    def log_pre_api_call(self, model, messages, kwargs):
        """Increment in-flight count. Deduped by litellm_call_id for streaming."""
        if kwargs is None:
            verbose_router_logger.debug(
                "[StickyLeastBusyRedis INCREMENT] Skipping: kwargs is None"
            )
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                verbose_router_logger.debug(
                    "[StickyLeastBusyRedis INCREMENT] Skipping: "
                    "missing litellm_params or metadata"
                )
                return

            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis INCREMENT] Skipping: "
                    f"model_group={model_group}, dep_id={dep_id}"
                )
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id and not self._should_increment(litellm_call_id):
                return

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = self.router_cache.increment_cache(
                key=cache_key, value=1, ttl=self.cache_ttl
            )
            self._refresh_cache_ttl(cache_key)
            self._routing_in_flight.labels(model_group, dep_id).inc()
            stream = kwargs.get("stream", False)
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis INCREMENT] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}, "
                f"stream={stream}, "
                f"call_id={litellm_call_id[:16] if litellm_call_id else 'None'}..."
            )
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusyRedis log_pre_api_call error: {e}"
            )

    def _decrement_request_count(self, kwargs, callback_type: str) -> None:
        if kwargs is None:
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                f"Skipping: kwargs is None"
            )
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                    f"Skipping: missing litellm_params or metadata"
                )
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                    f"Skipping: model_group={model_group}, dep_id={dep_id}"
                )
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = self.router_cache.increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            self._refresh_cache_ttl(cache_key)
            self._routing_in_flight.labels(model_group, dep_id).dec()
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}"
            )
            if new_value < 0:
                verbose_router_logger.warning(
                    f"[StickyLeastBusyRedis WARNING] Negative count detected "
                    f"for deployment_id={dep_id}, resetting to 0"
                )
                self.router_cache.set_cache(
                    key=cache_key, value=0, ttl=self.cache_ttl
                )

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id:
                self._cleanup_call_id(litellm_call_id)
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusyRedis decrement error: {e}"
            )

    async def _async_decrement_request_count(
        self, kwargs, callback_type: str
    ) -> None:
        if kwargs is None:
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                f"Skipping: kwargs is None"
            )
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                    f"Skipping: missing litellm_params or metadata"
                )
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                    f"Skipping: model_group={model_group}, dep_id={dep_id}"
                )
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = await self.router_cache.async_increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            await self._async_refresh_cache_ttl(cache_key)
            self._routing_in_flight.labels(model_group, dep_id).dec()
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis DECREMENT {callback_type}] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}"
            )
            if new_value < 0:
                verbose_router_logger.warning(
                    f"[StickyLeastBusyRedis WARNING] Negative count detected "
                    f"for deployment_id={dep_id}, resetting to 0"
                )
                await self.router_cache.async_set_cache(
                    key=cache_key, value=0, ttl=self.cache_ttl
                )

            litellm_call_id = kwargs.get("litellm_call_id") or litellm_params.get(
                "litellm_call_id"
            )
            if litellm_call_id:
                self._cleanup_call_id(litellm_call_id)
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusyRedis async decrement error: {e}"
            )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs, callback_type="SYNC-SUCCESS")
        if self.test_flag:
            self.logged_success += 1

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._decrement_request_count(kwargs, callback_type="SYNC-FAILURE")
        if self.test_flag:
            self.logged_failure += 1

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        await self._async_decrement_request_count(
            kwargs, callback_type="ASYNC-SUCCESS"
        )
        if self.test_flag:
            self.logged_success += 1

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        await self._async_decrement_request_count(
            kwargs, callback_type="ASYNC-FAILURE"
        )
        if self.test_flag:
            self.logged_failure += 1

    # =========================================================================
    # Load Querying
    # =========================================================================

    def _get_request_counts(
        self, model_group: str, healthy_deployments: list
    ) -> Dict[str, int]:
        """Sync: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        none_count = 0
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            count = self.router_cache.get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            result[dep_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                "[StickyLeastBusyRedis WARNING] Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(
                model_group, "redis_unavailable", "sticky_redis"
            ).inc()
        return result

    async def _async_get_request_counts(
        self, model_group: str, healthy_deployments: list
    ) -> Dict[str, int]:
        """Async: get in-flight counts for all healthy deployments from Redis."""
        result = {}
        none_count = 0
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            count = await self.router_cache.async_get_cache(
                key=cache_key, redis_only=True
            )
            if count is None:
                none_count += 1
            result[dep_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            verbose_router_logger.warning(
                "[StickyLeastBusyRedis WARNING] Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(
                model_group, "redis_unavailable", "sticky_redis"
            ).inc()
        return result

    # =========================================================================
    # Deployment Selection Helpers
    # =========================================================================

    @staticmethod
    def _extract_user_id(request_kwargs: Optional[Dict]) -> Optional[str]:
        """
        Extract a user identifier from request kwargs for per-user sticky routing.

        Looks for (in order of preference):
        1. metadata.user_api_key - the API key used for the request
        2. metadata.user_api_key_user_id - the user ID associated with the API key
        3. user - top-level user field

        Returns None if no identifier is found (falls back to message-only hashing).
        """
        if not request_kwargs:
            return None
        metadata = request_kwargs.get("metadata") or {}
        return (
            metadata.get("user_api_key")
            or metadata.get("user_api_key_user_id")
            or request_kwargs.get("user")
        )

    def _get_deployment_info(self, deployment: dict) -> str:
        """Helper to extract key deployment info for logging."""
        try:
            dep_id = deployment.get("model_info", {}).get("id", "unknown")
            api_base = deployment.get("litellm_params", {}).get(
                "api_base", "unknown"
            )
            model = deployment.get("litellm_params", {}).get("model", "unknown")
            return f"[id={dep_id}, model={model}, api_base={api_base}]"
        except Exception as e:
            return f"[error extracting info: {e}]"

    @staticmethod
    def _find_least_busy(
        dep_ids: List[str],
        request_counts: Dict[str, int],
        dep_id_to_deployment: Dict[str, dict],
        healthy_deployments: list,
    ) -> dict:
        """
        Find the deployment with the fewest in-flight requests.
        Random tie-breaking when multiple deployments share the minimum load.
        """
        min_load = float("inf")
        for did in dep_ids:
            load = request_counts.get(did, 0)
            if load < min_load:
                min_load = load

        min_deployments = [
            dep_id_to_deployment[did]
            for did in dep_ids
            if request_counts.get(did, 0) == min_load
        ]

        return (
            random.choice(min_deployments)
            if min_deployments
            else random.choice(healthy_deployments)
        )

    # =========================================================================
    # Deployment Selection Core (Sync)
    # =========================================================================

    def _select_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        request_counts: Dict[str, int],
        sticky_key: Optional[str],
    ) -> dict:
        """
        Sync core selection logic:
        1. If sticky_key, look up Redis for stored deployment mapping.
        2. If found and healthy and not overloaded, route there (sticky hit).
        3. If not found, unhealthy, or overloaded, assign least-busy and store in Redis.
        4. If no sticky key, route to least-busy.
        """
        dep_id_to_deployment: Dict[str, dict] = {}
        dep_ids: List[str] = []
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            dep_ids.append(dep_id)
            dep_id_to_deployment[dep_id] = d

        total_load = sum(request_counts.get(did, 0) for did in dep_ids)
        avg_load = total_load / len(dep_ids) if dep_ids else 0
        min_load = min(
            (request_counts.get(did, 0) for did in dep_ids), default=0
        )

        # Avg+min blend: catches skewed distributions where avg alone is pulled
        # up by outliers. E.g. loads [50,25,20,7,5] → avg=21.4 but min=5,
        # reference=(21.4+5)/2=13.2, so a node at 25 correctly triggers rebalance.
        reference_load = (avg_load + min_load) / 2

        # --- Log node status overview ---
        node_summary = ", ".join(
            f"{did}={request_counts.get(did, 0)}" for did in dep_ids
        )
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis ROUTING] "
            f"healthy_nodes={len(dep_ids)}, "
            f"total_in_flight={total_load}, "
            f"avg_load={avg_load:.2f}, "
            f"min_load={min_load}, "
            f"reference_load={reference_load:.2f}, "
            f"threshold={self.imbalance_threshold}, "
            f"loads=[{node_summary}]"
        )

        # Try sticky routing
        if sticky_key:
            stored_dep_id = self._get_sticky_deployment(model_group, sticky_key)

            if stored_dep_id and stored_dep_id in dep_id_to_deployment:
                # Key exists in Redis AND deployment is healthy
                preferred_load = request_counts.get(stored_dep_id, 0)
                effective_reference = max(reference_load, 1.0)
                threshold_value = self.imbalance_threshold * effective_reference

                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis STICKY-CHECK] "
                    f"stored_node={stored_dep_id}, "
                    f"preferred_load={preferred_load}, "
                    f"threshold_value={threshold_value:.2f} "
                    f"(= {self.imbalance_threshold} * "
                    f"max(({avg_load:.2f}+{min_load})/2, 1.0))"
                )

                if preferred_load < threshold_value:
                    # STICKY HIT -- route to stored deployment
                    selected = dep_id_to_deployment[stored_dep_id]
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis DECISION] STICKY -> "
                        f"{self._get_deployment_info(selected)} "
                        f"(load={preferred_load} < threshold={threshold_value:.2f})"
                    )
                    self._routing_decisions.labels(
                        model_group, stored_dep_id, "sticky", "sticky_redis"
                    ).inc()
                    return selected
                else:
                    # IMBALANCED -- find least-busy, update Redis
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-OVERRIDE] "
                        f"Overriding stickiness! "
                        f"stored_node={stored_dep_id} is overloaded "
                        f"(load={preferred_load} >= threshold={threshold_value:.2f}), "
                        f"reassigning to least-busy"
                    )
                    least_busy = self._find_least_busy(
                        dep_ids, request_counts, dep_id_to_deployment,
                        healthy_deployments,
                    )
                    lb_dep_id = least_busy["model_info"]["id"]
                    if isinstance(lb_dep_id, int):
                        lb_dep_id = str(lb_dep_id)
                    self._set_sticky_deployment(
                        model_group, sticky_key, lb_dep_id
                    )
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis DECISION] REBALANCE -> "
                        f"{self._get_deployment_info(least_busy)} "
                        f"(updated Redis mapping)"
                    )
                    self._routing_decisions.labels(
                        model_group, lb_dep_id, "rebalance", "sticky_redis"
                    ).inc()
                    return least_busy
            else:
                # Key doesn't exist in Redis OR deployment is unhealthy
                if stored_dep_id:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-CHECK] "
                        f"stored_node={stored_dep_id} not in healthy deployments, "
                        f"reassigning to least-busy"
                    )
                else:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-CHECK] "
                        f"No mapping in Redis for sticky_key={sticky_key[:16]}..., "
                        f"assigning least-busy"
                    )
                least_busy = self._find_least_busy(
                    dep_ids, request_counts, dep_id_to_deployment,
                    healthy_deployments,
                )
                lb_dep_id = least_busy["model_info"]["id"]
                if isinstance(lb_dep_id, int):
                    lb_dep_id = str(lb_dep_id)
                self._set_sticky_deployment(
                    model_group, sticky_key, lb_dep_id
                )
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECISION] ASSIGN -> "
                    f"{self._get_deployment_info(least_busy)} "
                    f"(stored in Redis)"
                )
                self._routing_decisions.labels(
                    model_group, lb_dep_id, "assign", "sticky_redis"
                ).inc()
                return least_busy
        else:
            verbose_router_logger.debug(
                "[StickyLeastBusyRedis STICKY-CHECK] "
                "No sticky key (no messages or no user message), "
                "using least-busy"
            )

        # No sticky key -- pure least-busy
        least_busy = self._find_least_busy(
            dep_ids, request_counts, dep_id_to_deployment, healthy_deployments
        )
        lb_dep_id = least_busy["model_info"]["id"]
        if isinstance(lb_dep_id, int):
            lb_dep_id = str(lb_dep_id)
        min_dep_ids = [
            did for did in dep_ids
            if request_counts.get(did, 0) == min(
                request_counts.get(d, 0) for d in dep_ids
            )
        ]
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis DECISION] LEAST-BUSY -> "
            f"{self._get_deployment_info(least_busy)} "
            f"(no sticky key, candidates={min_dep_ids})"
        )
        self._routing_decisions.labels(
            model_group, lb_dep_id, "least_busy", "sticky_redis"
        ).inc()
        return least_busy

    # =========================================================================
    # Deployment Selection Core (Async)
    # =========================================================================

    async def _async_select_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        request_counts: Dict[str, int],
        sticky_key: Optional[str],
    ) -> dict:
        """
        Async core selection logic (same as sync but with async Redis operations).
        """
        dep_id_to_deployment: Dict[str, dict] = {}
        dep_ids: List[str] = []
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            dep_ids.append(dep_id)
            dep_id_to_deployment[dep_id] = d

        total_load = sum(request_counts.get(did, 0) for did in dep_ids)
        avg_load = total_load / len(dep_ids) if dep_ids else 0
        min_load = min(
            (request_counts.get(did, 0) for did in dep_ids), default=0
        )

        # Avg+min blend (same as sync version)
        reference_load = (avg_load + min_load) / 2

        node_summary = ", ".join(
            f"{did}={request_counts.get(did, 0)}" for did in dep_ids
        )
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis ROUTING] "
            f"healthy_nodes={len(dep_ids)}, "
            f"total_in_flight={total_load}, "
            f"avg_load={avg_load:.2f}, "
            f"min_load={min_load}, "
            f"reference_load={reference_load:.2f}, "
            f"threshold={self.imbalance_threshold}, "
            f"loads=[{node_summary}]"
        )

        if sticky_key:
            stored_dep_id = await self._async_get_sticky_deployment(
                model_group, sticky_key
            )

            if stored_dep_id and stored_dep_id in dep_id_to_deployment:
                preferred_load = request_counts.get(stored_dep_id, 0)
                effective_reference = max(reference_load, 1.0)
                threshold_value = self.imbalance_threshold * effective_reference

                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis STICKY-CHECK] "
                    f"stored_node={stored_dep_id}, "
                    f"preferred_load={preferred_load}, "
                    f"threshold_value={threshold_value:.2f} "
                    f"(= {self.imbalance_threshold} * "
                    f"max(({avg_load:.2f}+{min_load})/2, 1.0))"
                )

                if preferred_load < threshold_value:
                    selected = dep_id_to_deployment[stored_dep_id]
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis DECISION] STICKY -> "
                        f"{self._get_deployment_info(selected)} "
                        f"(load={preferred_load} < threshold={threshold_value:.2f})"
                    )
                    self._routing_decisions.labels(
                        model_group, stored_dep_id, "sticky", "sticky_redis"
                    ).inc()
                    return selected
                else:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-OVERRIDE] "
                        f"Overriding stickiness! "
                        f"stored_node={stored_dep_id} is overloaded "
                        f"(load={preferred_load} >= threshold={threshold_value:.2f}), "
                        f"reassigning to least-busy"
                    )
                    least_busy = self._find_least_busy(
                        dep_ids, request_counts, dep_id_to_deployment,
                        healthy_deployments,
                    )
                    lb_dep_id = least_busy["model_info"]["id"]
                    if isinstance(lb_dep_id, int):
                        lb_dep_id = str(lb_dep_id)
                    await self._async_set_sticky_deployment(
                        model_group, sticky_key, lb_dep_id
                    )
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis DECISION] REBALANCE -> "
                        f"{self._get_deployment_info(least_busy)} "
                        f"(updated Redis mapping)"
                    )
                    self._routing_decisions.labels(
                        model_group, lb_dep_id, "rebalance", "sticky_redis"
                    ).inc()
                    return least_busy
            else:
                if stored_dep_id:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-CHECK] "
                        f"stored_node={stored_dep_id} not in healthy deployments, "
                        f"reassigning to least-busy"
                    )
                else:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusyRedis STICKY-CHECK] "
                        f"No mapping in Redis for sticky_key={sticky_key[:16]}..., "
                        f"assigning least-busy"
                    )
                least_busy = self._find_least_busy(
                    dep_ids, request_counts, dep_id_to_deployment,
                    healthy_deployments,
                )
                lb_dep_id = least_busy["model_info"]["id"]
                if isinstance(lb_dep_id, int):
                    lb_dep_id = str(lb_dep_id)
                await self._async_set_sticky_deployment(
                    model_group, sticky_key, lb_dep_id
                )
                verbose_router_logger.debug(
                    f"[StickyLeastBusyRedis DECISION] ASSIGN -> "
                    f"{self._get_deployment_info(least_busy)} "
                    f"(stored in Redis)"
                )
                self._routing_decisions.labels(
                    model_group, lb_dep_id, "assign", "sticky_redis"
                ).inc()
                return least_busy
        else:
            verbose_router_logger.debug(
                "[StickyLeastBusyRedis STICKY-CHECK] "
                "No sticky key (no messages or no user message), "
                "using least-busy"
            )

        least_busy = self._find_least_busy(
            dep_ids, request_counts, dep_id_to_deployment, healthy_deployments
        )
        lb_dep_id = least_busy["model_info"]["id"]
        if isinstance(lb_dep_id, int):
            lb_dep_id = str(lb_dep_id)
        min_dep_ids = [
            did for did in dep_ids
            if request_counts.get(did, 0) == min(
                request_counts.get(d, 0) for d in dep_ids
            )
        ]
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis DECISION] LEAST-BUSY -> "
            f"{self._get_deployment_info(least_busy)} "
            f"(no sticky key, candidates={min_dep_ids})"
        )
        self._routing_decisions.labels(
            model_group, lb_dep_id, "least_busy", "sticky_redis"
        ).inc()
        return least_busy

    # =========================================================================
    # Public API - Called by Router
    # =========================================================================

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> dict:
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis] get_available_deployments called "
            f"(SYNC) for model_group={model_group}"
        )
        try:
            request_counts = self._get_request_counts(model_group, healthy_deployments)
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            selected = self._select_deployment(
                model_group, healthy_deployments, request_counts, sticky_key
            )
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis RESULT] "
                f"model_group={model_group}, "
                f"sticky_key={sticky_key[:16] + '...' if sticky_key else 'None'}, "
                f"user_id={'present' if user_id else 'None'}, "
                f"selected={self._get_deployment_info(selected)}"
            )
            return selected
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusyRedis ERROR] Routing failed, falling back to "
                f"random selection: {e}"
            )
            self._routing_fallback.labels(
                model_group, "error", "sticky_redis"
            ).inc()
            return random.choice(healthy_deployments)

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> dict:
        verbose_router_logger.debug(
            f"[StickyLeastBusyRedis] async_get_available_deployments called "
            f"(ASYNC) for model_group={model_group}"
        )
        try:
            request_counts = await self._async_get_request_counts(
                model_group, healthy_deployments
            )
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            selected = await self._async_select_deployment(
                model_group, healthy_deployments, request_counts, sticky_key
            )
            verbose_router_logger.debug(
                f"[StickyLeastBusyRedis RESULT] "
                f"model_group={model_group}, "
                f"sticky_key={sticky_key[:16] + '...' if sticky_key else 'None'}, "
                f"user_id={'present' if user_id else 'None'}, "
                f"selected={self._get_deployment_info(selected)}"
            )
            return selected
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusyRedis ERROR] Async routing failed, falling back to "
                f"random selection: {e}"
            )
            self._routing_fallback.labels(
                model_group, "error", "sticky_redis"
            ).inc()
            return random.choice(healthy_deployments)
