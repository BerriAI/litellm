"""
Sticky-Least-Busy routing strategy.

Routes requests from the same conversation to the same deployment (for KV cache reuse
on vLLM/SGLang nodes), but rebalances to the least-busy deployment when the sticky
target is overloaded.

How this works:
  1. Hash the conversation identity (first user message + user ID) to compute
     a sticky key that is constant across all turns.
  2. Map sticky key to a preferred deployment via consistent hashing.
  3. Compute a reference load using the avg+min blend: (avg_load + min_load) / 2.
     This catches skewed distributions where avg alone is pulled up by outliers.
  4. If preferred deployment's in-flight count < threshold * reference_load, use it (sticky).
  5. If overloaded, route to the deployment with the fewest in-flight requests (rebalance).
  6. Track in-flight requests via Redis (atomic increment/decrement) with dedup
     to avoid the streaming bug where log_pre_api_call fires per SSE chunk.
"""

import hashlib
import json
import random
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class StickyLeastBusyLoggingHandler(CustomLogger):
    """
    Routing handler that combines conversation stickiness with load-aware rebalancing.

    Uses a class-level singleton to survive Router re-creation. The LiteLLM proxy
    may create new Router instances per-request or on config syncs. Without a
    singleton, each new instance gets a fresh _seen_call_ids dict, breaking
    streaming dedup and causing in-flight counts to grow monotonically.
    """

    _instance: Optional["StickyLeastBusyLoggingHandler"] = None

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __new__(
        cls,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        virtual_nodes: int = 150,
        cache_ttl: int = 600,
    ):
        """
        Singleton: return existing instance if one exists.
        Only update router_cache (which may change across Router instances).
        """
        if cls._instance is not None:
            # Update router_cache to the latest Router's cache (may have new Redis connection)
            cls._instance.router_cache = router_cache
            verbose_router_logger.info(
                f"[StickyLeastBusy REUSE] Reusing existing handler "
                f"(seen_call_ids={len(cls._instance._seen_call_ids)}, "
                f"rings={len(cls._instance._rings)})"
            )
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        router_cache: DualCache,
        imbalance_threshold: float = 1.5,
        virtual_nodes: int = 150,
        cache_ttl: int = 600,
    ):
        """
        Args:
            router_cache: DualCache instance for Redis + in-memory caching.
            imbalance_threshold: If sticky node load > threshold * reference_load, rebalance.
                reference_load = (avg_load + min_load) / 2 to catch skewed distributions.
            virtual_nodes: Number of virtual nodes per deployment on the consistent hash ring.
            cache_ttl: TTL in seconds for request count cache keys.
        """
        # Skip re-initialization if already initialized (singleton reuse)
        if hasattr(self, "_initialized") and self._initialized:
            # Always update router_cache (may point to a new Router's cache)
            self.router_cache = router_cache
            return

        self._initialized = True
        self.router_cache = router_cache
        self.imbalance_threshold = imbalance_threshold
        self.virtual_nodes = virtual_nodes
        self.cache_ttl = cache_ttl

        # Streaming dedup: track which litellm_call_ids we've already incremented.
        # log_pre_api_call fires for every SSE chunk in streaming - only increment once.
        self._seen_call_ids: Dict[str, bool] = {}
        self._seen_call_ids_max_size: int = 10000

        # Per-model-group consistent hash rings.
        # Each model group (e.g., "llama-70b", "kimi-k2-5-dev") may have different
        # deployments, so each needs its own ring. Keyed by model_group name.
        # Value: (frozenset_of_deployment_ids, sorted_ring_list)
        self._rings: Dict[str, Tuple[frozenset, List[Tuple[int, str]]]] = {}

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
            self._routing_redis_count = Gauge(
                "litellm_sticky_routing_redis_count",
                "Redis in-flight count per deployment as seen by routing at decision time",
                ["model_group", "deployment_id"],
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
            self._routing_redis_count = REGISTRY._names_to_collectors.get(
                "litellm_sticky_routing_redis_count"
            )
            # Guard against partial registration — if any lookup returned None,
            # fall back to NoOpMetric to avoid AttributeError on .labels().inc()
            if not all([self._routing_decisions, self._routing_in_flight, self._routing_fallback, self._routing_redis_count]):
                from litellm.types.integrations.prometheus import NoOpMetric

                self._routing_decisions = self._routing_decisions or NoOpMetric()
                self._routing_in_flight = self._routing_in_flight or NoOpMetric()
                self._routing_fallback = self._routing_fallback or NoOpMetric()
                self._routing_redis_count = self._routing_redis_count or NoOpMetric()
        except Exception:
            from litellm.types.integrations.prometheus import NoOpMetric

            self._routing_decisions = NoOpMetric()
            self._routing_in_flight = NoOpMetric()
            self._routing_fallback = NoOpMetric()
            self._routing_redis_count = NoOpMetric()

        verbose_router_logger.info(
            f"[StickyLeastBusy INIT] Initialized with "
            f"imbalance_threshold={imbalance_threshold}, "
            f"virtual_nodes={virtual_nodes}, "
            f"cache_ttl={cache_ttl}s"
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
        this by hashing the conversation's "identity" — the first user message
        plus a user identifier — which never changes as the conversation grows.

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
                "[StickyLeastBusy STICKY-KEY] No messages provided, sticky_key=None"
            )
            return None

        # Extract the first user message content.
        # O(1) scan — stops at first user message, doesn't touch the rest.
        first_user_content: Optional[str] = None
        for msg in messages:
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content", "")
                # Handle multimodal content (list of parts) — extract text parts
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
                "[StickyLeastBusy STICKY-KEY] No user message found, sticky_key=None"
            )
            return None

        # Combine first user message + user identifier for per-user stickiness.
        # If user_id is not available, fall back to message-only hashing.
        hash_input = first_user_content
        if user_id:
            hash_input = f"{user_id}:{first_user_content}"

        sticky_key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        verbose_router_logger.debug(
            f"[StickyLeastBusy STICKY-KEY] "
            f"total_messages={len(messages)}, "
            f"has_user_id={user_id is not None}, "
            f"sticky_key={sticky_key[:16]}..."
        )
        return sticky_key

    # =========================================================================
    # Consistent Hashing
    # =========================================================================

    def _build_hash_ring(self, model_group: str, deployment_ids: List[str]) -> None:
        """
        Build a consistent hash ring from deployment IDs using virtual nodes.
        Rings are cached per model_group — only rebuilds if the set of IDs
        for that model group has changed.
        """
        new_ids = frozenset(deployment_ids)
        cached = self._rings.get(model_group)
        if cached and cached[0] == new_ids:
            return

        prev_count = len(cached[0]) if cached else 0
        verbose_router_logger.info(
            f"[StickyLeastBusy RING-BUILD] Rebuilding hash ring for "
            f"model_group={model_group}: "
            f"prev_deployments={prev_count}, "
            f"new_deployments={len(new_ids)}, "
            f"ids={list(new_ids)}"
        )

        ring: List[Tuple[int, str]] = []
        for dep_id in deployment_ids:
            for i in range(self.virtual_nodes):
                key = f"{dep_id}:{i}"
                h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
                ring.append((h, dep_id))

        ring.sort(key=lambda x: x[0])
        self._rings[model_group] = (new_ids, ring)

        verbose_router_logger.info(
            f"[StickyLeastBusy RING-BUILD] Ring for {model_group} built with "
            f"{len(ring)} virtual nodes "
            f"({self.virtual_nodes} per deployment)"
        )

    def _get_deployment_for_key(
        self, model_group: str, sticky_key: str
    ) -> Optional[str]:
        """Map a sticky key to a deployment ID via the consistent hash ring."""
        cached = self._rings.get(model_group)
        if not cached or not cached[1]:
            verbose_router_logger.debug(
                f"[StickyLeastBusy RING-LOOKUP] Hash ring for "
                f"{model_group} is empty, returning None"
            )
            return None

        ring = cached[1]
        h = int(hashlib.md5(sticky_key.encode("utf-8")).hexdigest(), 16)
        idx = bisect_right(ring, (h,))
        if idx >= len(ring):
            idx = 0

        result = ring[idx][1]
        verbose_router_logger.debug(
            f"[StickyLeastBusy RING-LOOKUP] "
            f"model_group={model_group}, "
            f"sticky_key={sticky_key[:16]}... -> deployment_id={result}"
        )
        return result

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
        Refresh Redis TTL on every increment/decrement.

        The shared redis_cache.increment_cache only sets TTL on first key creation
        (when current_ttl == -1). For sustained traffic lasting > cache_ttl seconds,
        the key would expire and in-flight decrements would hit a fresh key at 0,
        going negative. By refreshing TTL on every access, the key only expires
        after cache_ttl seconds of ZERO activity to that deployment.
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
            pass  # Best-effort — if Redis is down, we can't refresh TTL anyway

    async def _async_refresh_cache_ttl(self, cache_key: str) -> None:
        """Async variant: refresh Redis TTL on every access."""
        try:
            if self.router_cache.redis_cache is not None:
                _redis_client = self.router_cache.redis_cache.init_async_client()
                await _redis_client.expire(cache_key, self.cache_ttl)
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
                f"[StickyLeastBusy DEDUP] Skipping duplicate increment "
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
                f"[StickyLeastBusy DEDUP] Evicted {evict_count} old call_ids "
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
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return

            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
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
                f"[StickyLeastBusy INCREMENT] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}, "
                f"stream={stream}, "
                f"call_id={litellm_call_id[:16] if litellm_call_id else 'None'}..."
            )
        except Exception as e:
            verbose_router_logger.error(
                f"StickyLeastBusy log_pre_api_call error: {e}"
            )

    def _decrement_request_count(self, kwargs, callback_type: str) -> None:
        if kwargs is None:
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
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
                f"[StickyLeastBusy DECREMENT {callback_type}] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}"
            )
            if new_value < 0:
                verbose_router_logger.warning(
                    f"[StickyLeastBusy WARNING] Negative count detected "
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
                f"StickyLeastBusy decrement error: {e}"
            )

    async def _async_decrement_request_count(
        self, kwargs, callback_type: str
    ) -> None:
        if kwargs is None:
            return
        try:
            litellm_params = kwargs.get("litellm_params")
            if litellm_params is None or litellm_params.get("metadata") is None:
                return
            model_group = litellm_params["metadata"].get("model_group")
            dep_id = (litellm_params.get("model_info") or {}).get("id")
            if model_group is None or dep_id is None:
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
                f"[StickyLeastBusy DECREMENT {callback_type}] "
                f"deployment_id={dep_id}, "
                f"model_group={model_group}, "
                f"new_count={new_value}"
            )
            if new_value < 0:
                verbose_router_logger.warning(
                    f"[StickyLeastBusy WARNING] Negative count detected "
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
                f"StickyLeastBusy async decrement error: {e}"
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
                "[StickyLeastBusy WARNING] Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(
                model_group, "redis_unavailable", "consistent_hashing"
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
                "[StickyLeastBusy WARNING] Redis returned None for all deployments "
                "- Redis may be unavailable. Load data will default to 0."
            )
            self._routing_fallback.labels(
                model_group, "redis_unavailable", "consistent_hashing"
            ).inc()
        return result

    # =========================================================================
    # Deployment Selection Core
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

    def _select_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        request_counts: Dict[str, int],
        sticky_key: Optional[str],
    ) -> dict:
        """
        Core selection logic:
        1. Build/update consistent hash ring from healthy deployment IDs.
        2. If sticky_key available, find preferred deployment via consistent hashing.
        3. Check if preferred deployment is within load threshold.
        4. If overloaded or no sticky key, fall back to least-busy.
        """
        dep_id_to_deployment: Dict[str, dict] = {}
        dep_ids: List[str] = []
        for d in healthy_deployments:
            dep_id = d["model_info"]["id"]
            if isinstance(dep_id, int):
                dep_id = str(dep_id)
            dep_ids.append(dep_id)
            dep_id_to_deployment[dep_id] = d

        self._build_hash_ring(model_group, dep_ids)

        # Expose Redis counts to Prometheus so Grafana can show what routing sees
        for did in dep_ids:
            self._routing_redis_count.labels(model_group, did).set(
                request_counts.get(did, 0)
            )

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

        # Calculate imbalance ratio for each deployment
        imbalance_ratios = []
        for did in dep_ids:
            load = request_counts.get(did, 0)
            ref = max(reference_load, 1.0)
            ratio = load / ref if ref > 0 else 0
            imbalance_ratios.append(f"{did}={ratio:.2f}")

        verbose_router_logger.info(
            f"[StickyLeastBusy ROUTING] model_group={model_group}, "
            f"healthy_deployments={len(dep_ids)}, "
            f"deployment_ids={dep_ids}, "
            f"total_in_flight={total_load}, "
            f"avg_load={avg_load:.2f}, "
            f"min_load={min_load}, "
            f"reference_load={reference_load:.2f}, "
            f"imbalance_threshold={self.imbalance_threshold}, "
            f"loads_per_deployment=[{node_summary}], "
            f"imbalance_ratios=[{', '.join(imbalance_ratios)}]"
        )

        # Try sticky routing
        if sticky_key:
            preferred_id = self._get_deployment_for_key(model_group, sticky_key)
            if preferred_id and preferred_id in dep_id_to_deployment:
                preferred_load = request_counts.get(preferred_id, 0)
                effective_reference = max(reference_load, 1.0)
                threshold_value = self.imbalance_threshold * effective_reference
                current_ratio = preferred_load / effective_reference if effective_reference > 0 else 0

                verbose_router_logger.info(
                    f"[StickyLeastBusy STICKY-CHECK] model_group={model_group}, "
                    f"sticky_key={sticky_key[:16]}..., "
                    f"preferred_deployment={preferred_id}, "
                    f"preferred_load={preferred_load}, "
                    f"effective_reference={effective_reference:.2f}, "
                    f"threshold_value={threshold_value:.2f}, "
                    f"current_imbalance_ratio={current_ratio:.2f}x "
                    f"(threshold_ratio={self.imbalance_threshold}x)"
                )

                if preferred_load < threshold_value:
                    selected = dep_id_to_deployment[preferred_id]
                    verbose_router_logger.info(
                        f"[StickyLeastBusy DECISION] STICKY -> deployment_id={preferred_id}, "
                        f"api_base={selected.get('litellm_params', {}).get('api_base', 'unknown')}, "
                        f"model={selected.get('litellm_params', {}).get('model', 'unknown')}, "
                        f"reason=load_{preferred_load}_below_threshold_{threshold_value:.2f}, "
                        f"imbalance_ratio={current_ratio:.2f}x"
                    )
                    self._routing_decisions.labels(
                        model_group, preferred_id, "sticky", "consistent_hashing"
                    ).inc()
                    return selected
                else:
                    verbose_router_logger.info(
                        f"[StickyLeastBusy STICKY-OVERRIDE] model_group={model_group}, "
                        f"preferred_deployment={preferred_id} OVERLOADED, "
                        f"load={preferred_load} exceeds threshold={threshold_value:.2f}, "
                        f"imbalance_ratio={current_ratio:.2f}x > {self.imbalance_threshold}x, "
                        f"falling_back_to=least_busy"
                    )
                    self._routing_decisions.labels(
                        model_group, preferred_id, "override", "consistent_hashing"
                    ).inc()
            else:
                verbose_router_logger.info(
                    f"[StickyLeastBusy STICKY-CHECK] model_group={model_group}, "
                    f"sticky_key={sticky_key[:16]}..., "
                    f"preferred_deployment={preferred_id} "
                    f"not_in_healthy_deployments={list(dep_id_to_deployment.keys())}, "
                    f"falling_back_to=least_busy"
                )
        else:
            verbose_router_logger.info(
                f"[StickyLeastBusy STICKY-CHECK] model_group={model_group}, "
                f"reason=no_sticky_key, "
                f"using=least_busy"
            )

        # Least-busy fallback with random tie-breaking
        # (min_load already computed above for reference_load)
        min_deployments = [
            dep_id_to_deployment[did]
            for did in dep_ids
            if request_counts.get(did, 0) == min_load
        ]
        min_dep_ids = [
            d["model_info"]["id"] for d in min_deployments
        ]

        selected = (
            random.choice(min_deployments)
            if min_deployments
            else random.choice(healthy_deployments)
        )
        selected_dep_id = selected["model_info"]["id"]
        if isinstance(selected_dep_id, int):
            selected_dep_id = str(selected_dep_id)

        # Calculate how much less loaded the selected node is compared to average
        load_difference = avg_load - min_load if dep_ids else 0
        load_reduction_pct = (load_difference / avg_load * 100) if avg_load > 0 else 0

        verbose_router_logger.info(
            f"[StickyLeastBusy DECISION] LEAST-BUSY -> deployment_id={selected_dep_id}, "
            f"api_base={selected.get('litellm_params', {}).get('api_base', 'unknown')}, "
            f"model={selected.get('litellm_params', {}).get('model', 'unknown')}, "
            f"selected_load={min_load}, "
            f"avg_load={avg_load:.2f}, "
            f"load_difference_from_avg={load_difference:.2f} ({load_reduction_pct:.1f}% reduction), "
            f"candidates_with_min_load={len(min_deployments)}/{len(dep_ids)}, "
            f"candidate_ids={min_dep_ids}"
        )
        self._routing_decisions.labels(
            model_group, selected_dep_id, "least_busy", "consistent_hashing"
        ).inc()
        return selected

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
        # Log available healthy deployments at INFO level for production visibility
        healthy_ids = [
            str(d.get("model_info", {}).get("id", "unknown"))
            for d in healthy_deployments
        ]
        verbose_router_logger.info(
            f"[StickyLeastBusy ROUTING-START] (SYNC) model_group={model_group}, "
            f"healthy_deployments_count={len(healthy_deployments)}, "
            f"healthy_deployment_ids={healthy_ids}"
        )
        try:
            request_counts = self._get_request_counts(model_group, healthy_deployments)
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            return self._select_deployment(
                model_group, healthy_deployments, request_counts, sticky_key
            )
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusy ERROR] Routing failed, falling back to "
                f"random selection: {e}"
            )
            self._routing_fallback.labels(
                model_group, "error", "consistent_hashing"
            ).inc()
            return random.choice(healthy_deployments)

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        request_kwargs: Optional[Dict] = None,
    ) -> dict:
        # Log available healthy deployments at INFO level for production visibility
        healthy_ids = [
            str(d.get("model_info", {}).get("id", "unknown"))
            for d in healthy_deployments
        ]
        verbose_router_logger.info(
            f"[StickyLeastBusy ROUTING-START] (ASYNC) model_group={model_group}, "
            f"healthy_deployments_count={len(healthy_deployments)}, "
            f"healthy_deployment_ids={healthy_ids}"
        )
        try:
            request_counts = await self._async_get_request_counts(
                model_group, healthy_deployments
            )
            user_id = self._extract_user_id(request_kwargs)
            sticky_key = self.compute_sticky_key(messages, user_id=user_id)
            return self._select_deployment(
                model_group, healthy_deployments, request_counts, sticky_key
            )
        except Exception as e:
            verbose_router_logger.error(
                f"[StickyLeastBusy ERROR] Async routing failed, falling back to "
                f"random selection: {e}"
            )
            self._routing_fallback.labels(
                model_group, "error", "consistent_hashing"
            ).inc()
            return random.choice(healthy_deployments)
