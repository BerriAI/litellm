"""
Sticky-Least-Busy routing strategy.

Routes requests from the same conversation to the same deployment (for KV cache reuse
on vLLM/SGLang nodes), but rebalances to the least-busy deployment when the sticky
target is overloaded.

How this works:
  1. Hash the conversation identity (system prompt + first user message) to compute
     a sticky key that is constant across all turns.
  2. Map sticky key to a preferred deployment via consistent hashing.
  3. If preferred deployment's in-flight count < threshold * avg_load, use it (sticky).
  4. If overloaded, route to the deployment with the fewest in-flight requests (rebalance).
  5. Track in-flight requests via Redis (atomic increment/decrement) with dedup
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
                f"ring_nodes={len(cls._instance._hash_ring)})"
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
            imbalance_threshold: If sticky node load > threshold * avg_load, rebalance.
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

        # Consistent hash ring (rebuilt when deployments change)
        self._hash_ring: List[Tuple[int, str]] = []
        self._ring_deployment_ids: frozenset = frozenset()

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
    ) -> Optional[str]:
        """
        Compute a deterministic hash that identifies the conversation.

        The key must be STABLE across all turns of the same conversation so that
        consecutive messages route to the same node (KV cache reuse). We achieve
        this by hashing the conversation's "identity" — the system prompt (if any)
        plus the first user message — which never changes as the conversation grows.

        Algorithm:
        - None/empty messages -> None (no stickiness, degrades to least-busy).
        - Extract up to the first 2 messages: [system_prompt, first_user_msg].
          If no system message, just [first_user_msg].
        - Hash this fixed identity with SHA-256 of canonical JSON.

        This ensures:
        - Same conversation always produces the same hash on every turn.
        - Different conversations (different first user question) get different hashes.
        - The hash is deterministic across pods (canonical JSON + SHA-256).
        - Different users with the same system prompt but different first questions
          get different hashes (no hotspot).
        """
        if not messages:
            verbose_router_logger.debug(
                "[StickyLeastBusy STICKY-KEY] No messages provided, sticky_key=None"
            )
            return None

        # Extract the conversation identity: system prompt (if any) + first user message.
        # This is constant across all turns of the same conversation.
        identity: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "")
            identity.append(msg)
            if role == "user":
                # Found first user message — we have enough to identify the conversation
                break
            elif role in ("system", "developer"):
                # System/developer prompt — include it but keep looking for first user message
                continue
            else:
                # assistant or other role before first user message — stop here
                break

        if not identity:
            verbose_router_logger.debug(
                "[StickyLeastBusy STICKY-KEY] No identity messages found, sticky_key=None"
            )
            return None

        try:
            canonical = json.dumps(
                identity, sort_keys=True, ensure_ascii=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            canonical = str(identity)

        sticky_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        identity_roles = [m.get("role", "?") for m in identity]
        verbose_router_logger.debug(
            f"[StickyLeastBusy STICKY-KEY] "
            f"total_messages={len(messages)}, "
            f"identity_messages={len(identity)}, "
            f"identity_roles={identity_roles}, "
            f"sticky_key={sticky_key[:16]}..."
        )
        return sticky_key

    # =========================================================================
    # Consistent Hashing
    # =========================================================================

    def _build_hash_ring(self, deployment_ids: List[str]) -> None:
        """
        Build a consistent hash ring from deployment IDs using virtual nodes.
        Only rebuilds if the set of IDs has changed.
        """
        new_ids = frozenset(deployment_ids)
        if new_ids == self._ring_deployment_ids:
            return

        verbose_router_logger.info(
            f"[StickyLeastBusy RING-BUILD] Rebuilding hash ring: "
            f"prev_deployments={len(self._ring_deployment_ids)}, "
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
        self._hash_ring = ring
        self._ring_deployment_ids = new_ids

        verbose_router_logger.info(
            f"[StickyLeastBusy RING-BUILD] Ring built with "
            f"{len(ring)} virtual nodes "
            f"({self.virtual_nodes} per deployment)"
        )

    def _get_deployment_for_key(self, sticky_key: str) -> Optional[str]:
        """Map a sticky key to a deployment ID via the consistent hash ring."""
        if not self._hash_ring:
            verbose_router_logger.debug(
                "[StickyLeastBusy RING-LOOKUP] Hash ring is empty, returning None"
            )
            return None

        h = int(hashlib.md5(sticky_key.encode("utf-8")).hexdigest(), 16)
        idx = bisect_right(self._hash_ring, (h,))
        if idx >= len(self._hash_ring):
            idx = 0

        result = self._hash_ring[idx][1]
        verbose_router_logger.debug(
            f"[StickyLeastBusy RING-LOOKUP] "
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
            dep_id = litellm_params.get("model_info", {}).get("id")
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
            dep_id = litellm_params.get("model_info", {}).get("id")
            if model_group is None or dep_id is None:
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = self.router_cache.increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            self._refresh_cache_ttl(cache_key)
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
            dep_id = litellm_params.get("model_info", {}).get("id")
            if model_group is None or dep_id is None:
                return
            if isinstance(dep_id, int):
                dep_id = str(dep_id)

            cache_key = self._get_request_count_cache_key(model_group, dep_id)
            new_value = await self.router_cache.async_increment_cache(
                key=cache_key, value=-1, ttl=self.cache_ttl
            )
            await self._async_refresh_cache_ttl(cache_key)
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
        return result

    # =========================================================================
    # Deployment Selection Core
    # =========================================================================

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

        self._build_hash_ring(dep_ids)

        total_load = sum(request_counts.get(did, 0) for did in dep_ids)
        avg_load = total_load / len(dep_ids) if dep_ids else 0

        # --- Log node status overview ---
        node_summary = ", ".join(
            f"{did}={request_counts.get(did, 0)}" for did in dep_ids
        )
        verbose_router_logger.debug(
            f"[StickyLeastBusy ROUTING] "
            f"healthy_nodes={len(dep_ids)}, "
            f"total_in_flight={total_load}, "
            f"avg_load={avg_load:.2f}, "
            f"threshold={self.imbalance_threshold}, "
            f"loads=[{node_summary}]"
        )

        # Try sticky routing
        if sticky_key:
            preferred_id = self._get_deployment_for_key(sticky_key)
            if preferred_id and preferred_id in dep_id_to_deployment:
                preferred_load = request_counts.get(preferred_id, 0)
                effective_avg = max(avg_load, 1.0)
                threshold_value = self.imbalance_threshold * effective_avg

                verbose_router_logger.debug(
                    f"[StickyLeastBusy STICKY-CHECK] "
                    f"preferred_node={preferred_id}, "
                    f"preferred_load={preferred_load}, "
                    f"threshold_value={threshold_value:.2f} "
                    f"(= {self.imbalance_threshold} * max({avg_load:.2f}, 1.0))"
                )

                if preferred_load < threshold_value:
                    selected = dep_id_to_deployment[preferred_id]
                    verbose_router_logger.debug(
                        f"[StickyLeastBusy DECISION] STICKY -> "
                        f"{self._get_deployment_info(selected)} "
                        f"(load={preferred_load} < threshold={threshold_value:.2f})"
                    )
                    return selected
                else:
                    verbose_router_logger.debug(
                        f"[StickyLeastBusy STICKY-OVERRIDE] "
                        f"Overriding stickiness! "
                        f"preferred_node={preferred_id} is overloaded "
                        f"(load={preferred_load} >= threshold={threshold_value:.2f}), "
                        f"falling back to least-busy"
                    )
            else:
                verbose_router_logger.debug(
                    f"[StickyLeastBusy STICKY-CHECK] "
                    f"preferred_node={preferred_id} not found in healthy deployments, "
                    f"falling back to least-busy"
                )
        else:
            verbose_router_logger.debug(
                "[StickyLeastBusy STICKY-CHECK] "
                "No sticky key (no messages or single new conversation), "
                "using least-busy"
            )

        # Least-busy fallback with random tie-breaking
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
        min_dep_ids = [
            d["model_info"]["id"] for d in min_deployments
        ]

        selected = (
            random.choice(min_deployments)
            if min_deployments
            else random.choice(healthy_deployments)
        )
        verbose_router_logger.debug(
            f"[StickyLeastBusy DECISION] LEAST-BUSY -> "
            f"{self._get_deployment_info(selected)} "
            f"(min_load={min_load}, "
            f"candidates_with_min_load={len(min_deployments)}, "
            f"candidate_ids={min_dep_ids})"
        )
        return selected

    # =========================================================================
    # Public API - Called by Router
    # =========================================================================

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        verbose_router_logger.debug(
            f"[StickyLeastBusy] get_available_deployments called "
            f"(SYNC) for model_group={model_group}"
        )
        request_counts = self._get_request_counts(model_group, healthy_deployments)
        sticky_key = self.compute_sticky_key(messages)
        return self._select_deployment(
            healthy_deployments, request_counts, sticky_key
        )

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> dict:
        verbose_router_logger.debug(
            f"[StickyLeastBusy] async_get_available_deployments called "
            f"(ASYNC) for model_group={model_group}"
        )
        request_counts = await self._async_get_request_counts(
            model_group, healthy_deployments
        )
        sticky_key = self.compute_sticky_key(messages)
        return self._select_deployment(
            healthy_deployments, request_counts, sticky_key
        )
