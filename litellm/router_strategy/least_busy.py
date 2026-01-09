#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
#   - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
#   - use litellm.success + failure callbacks to log when a request completed
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import random
from typing import List, Optional

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache

    def _get_request_count_cache_key(self, model_group: str, deployment_id: str) -> str:
        """
        Get the cache key for a specific deployment's request count.
        Uses individual keys per deployment for atomic operations.
        """
        return f"deployment:{model_group}:{deployment_id}:request_count"

    def log_pre_api_call(self, model, messages, kwargs):
        """
        Log when a model is being used.
        Uses atomic increment to avoid race conditions.
        """
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                cache_key = self._get_request_count_cache_key(model_group, id)
                # Atomic increment - no race condition possible
                # Use 10-minute TTL to handle long-running LLM requests
                self.router_cache.increment_cache(key=cache_key, value=1, ttl=600)
        except Exception:
            pass

    def _sync_decrement_request_count(self, model_group: str, deployment_id: str):
        """
        Sync helper to atomically decrement request count, ensuring it never goes below 0.
        """
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        # Use atomic increment with -1 to decrement
        # Maintain 10-minute TTL to handle long-running requests
        new_value = self.router_cache.increment_cache(key=cache_key, value=-1, ttl=600)
        # If we went negative due to a race condition (e.g., decrement before increment was visible),
        # reset to 0 to avoid negative counts affecting routing
        if new_value < 0:
            self.router_cache.set_cache(key=cache_key, value=0, ttl=600)

    async def _async_decrement_request_count(self, model_group: str, deployment_id: str):
        """
        Async helper to atomically decrement request count, ensuring it never goes below 0.
        """
        cache_key = self._get_request_count_cache_key(model_group, deployment_id)
        # Use atomic increment with -1 to decrement
        # Maintain 10-minute TTL to handle long-running requests
        new_value = await self.router_cache.async_increment_cache(key=cache_key, value=-1, ttl=600)
        # If we went negative due to a race condition, reset to 0
        if new_value < 0:
            await self.router_cache.async_set_cache(key=cache_key, value=0, ttl=600)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                self._sync_decrement_request_count(model_group, id)

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
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                self._sync_decrement_request_count(model_group, id)

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
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                await self._async_decrement_request_count(model_group, id)

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
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                await self._async_decrement_request_count(model_group, id)

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    def _get_request_counts_for_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> dict:
        """
        Sync helper to get request counts for all healthy deployments.
        Returns a dict of {deployment_id: request_count}.

        Uses redis_only=True to bypass in-memory cache and always read from Redis.
        This is critical for distributed deployments where multiple pods need to see
        the global request count, not their local stale view.
        """
        result = {}
        none_count = 0
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            # Use redis_only=True to get global count across all pods
            count = self.router_cache.get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            # Default to 0 if not in cache, ensure non-negative
            result[deployment_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            print("[Least-Busy WARNING] Redis returned None for all deployments - Redis may be unavailable. Falling back to random routing.")
        return result

    async def _async_get_request_counts_for_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ) -> dict:
        """
        Async helper to get request counts for all healthy deployments.
        Returns a dict of {deployment_id: request_count}.

        Uses redis_only=True to bypass in-memory cache and always read from Redis.
        This is critical for distributed deployments where multiple pods need to see
        the global request count, not their local stale view.
        """
        result = {}
        none_count = 0
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            cache_key = self._get_request_count_cache_key(model_group, deployment_id)
            # Use redis_only=True to get global count across all pods
            count = await self.router_cache.async_get_cache(key=cache_key, redis_only=True)
            if count is None:
                none_count += 1
            # Default to 0 if not in cache, ensure non-negative
            result[deployment_id] = max(0, int(count)) if count is not None else 0

        if none_count == len(healthy_deployments) and none_count > 0:
            print("[Least-Busy WARNING] Redis returned None for all deployments - Redis may be unavailable. Falling back to random routing.")
        return result

    def _get_available_deployments(
        self,
        healthy_deployments: list,
        all_deployments: dict,
    ):
        """
        Helper to get deployments using least busy strategy.

        When multiple deployments have the same minimum traffic count,
        randomly select among them to ensure fair distribution.
        """
        # Extract healthy deployment IDs for logging
        healthy_ids = [d["model_info"]["id"] for d in healthy_deployments]

        print(f"[Least-Busy DEBUG] Cached all_deployments: {all_deployments}")
        print(f"[Least-Busy DEBUG] Healthy deployment IDs: {healthy_ids}")

        # First pass: find the minimum traffic count
        min_traffic = float("inf")
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            traffic = all_deployments.get(deployment_id, 0)
            if traffic < min_traffic:
                min_traffic = traffic

        # Second pass: collect all deployments with minimum traffic
        # This fixes the tie-breaking bias where the first deployment always won
        min_deployments = []
        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            if isinstance(deployment_id, int):
                deployment_id = str(deployment_id)
            traffic = all_deployments.get(deployment_id, 0)
            if traffic == min_traffic:
                min_deployments.append(d)

        # Randomly select among deployments with equal minimum traffic
        if min_deployments:
            selected = random.choice(min_deployments)
            print(f"[Least-Busy DEBUG] Selected deployment ID: {selected['model_info']['id']} with traffic={min_traffic} (from {len(min_deployments)} candidates)")
            return selected
        else:
            # Fallback: should not happen if healthy_deployments is non-empty
            print("[Least-Busy DEBUG] WARNING: No deployment found, falling back to RANDOM choice")
            return random.choice(healthy_deployments)

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ):
        """
        Sync helper to get deployments using least busy strategy
        """
        all_deployments = self._get_request_counts_for_deployments(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )

    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: list
    ):
        """
        Async helper to get deployments using least busy strategy
        """
        all_deployments = await self._async_get_request_counts_for_deployments(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
        )
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )
