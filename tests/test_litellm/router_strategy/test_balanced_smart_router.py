import os
import sys
import time
from datetime import datetime, timezone, timedelta

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.router_strategy.balanced_smart import BalancedSmartRoutingHandler


def _deployment(deployment_id: str, backend_id: str = None) -> dict:
    model_info = {"id": deployment_id}
    if backend_id is not None:
        model_info["balanced_smart_backend_id"] = backend_id
    return {
        "model_name": "balanced-model",
        "litellm_params": {
            "model": f"openai/{deployment_id}",
            "api_key": "sk-test",
            "api_base": "https://example.invalid",
        },
        "model_info": model_info,
    }


def _router(routing_strategy_args=None) -> Router:
    return Router(
        model_list=[_deployment("a"), _deployment("b")],
        routing_strategy="balanced-smart",
        routing_strategy_args=routing_strategy_args or {},
    )


def _callback_kwargs(deployment_id: str, backend_id: str = None) -> dict:
    model_info = {"id": deployment_id}
    if backend_id is not None:
        model_info["balanced_smart_backend_id"] = backend_id
    return {
        "litellm_params": {
            "metadata": {"model_group": "balanced-model"},
            "model_info": model_info,
        }
    }


def test_router_accepts_balanced_smart_strategy():
    router = _router()

    strategy, selector = router._get_routing_context("balanced-model")

    assert strategy == "balanced-smart"
    assert selector is router.balanced_smart_logger
    assert isinstance(selector, BalancedSmartRoutingHandler)


@pytest.mark.asyncio
async def test_async_router_dispatch_acquires_capacity():
    router = _router({"max_concurrent_requests": 1, "max_queue_ttl_s": 0.01})

    deployment = await router.async_get_available_deployment(
        model="balanced-model",
        request_kwargs={},
    )

    assert deployment is not None
    stats = router.balanced_smart_logger._get_stats(
        f"balanced-model:{deployment['model_info']['id']}"
    )
    assert stats.active == 1


def test_max_concurrent_requests_spills_to_next_deployment():
    router = _router({"max_concurrent_requests": 1, "max_queue_ttl_s": 0})
    selector = router.balanced_smart_logger
    assert selector is not None
    selector._acquire("balanced-model:a")

    deployment = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a"), _deployment("b")],
    )

    assert deployment is not None
    assert deployment["model_info"]["id"] == "b"


def test_queue_ttl_expires_when_all_deployments_are_full():
    router = _router(
        {
            "max_concurrent_requests": 1,
            "max_queue_ttl_s": 0.01,
            "queue_poll_s": 0.001,
        }
    )
    selector = router.balanced_smart_logger
    assert selector is not None
    selector._acquire("balanced-model:a")
    selector._acquire("balanced-model:b")

    start = time.time()
    deployment = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a"), _deployment("b")],
    )

    assert deployment is None
    assert time.time() - start < 0.5


def test_tokens_per_second_and_ttft_affect_ranking():
    router = _router(
        {
            "max_concurrent_requests": 1,
            "active_request_weight": 0,
            "tokens_per_second_weight": 1,
            "ttft_weight": 1,
        }
    )
    selector = router.balanced_smart_logger
    assert selector is not None
    slow = selector._get_stats("balanced-model:a")
    slow.ewma_tps = 10
    slow.ewma_ttft_s = 1
    selector._set_stats("balanced-model:a", slow)
    fast = selector._get_stats("balanced-model:b")
    fast.ewma_tps = 100
    fast.ewma_ttft_s = 0.1
    selector._set_stats("balanced-model:b", fast)

    deployment = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a"), _deployment("b")],
    )

    assert deployment is not None
    assert deployment["model_info"]["id"] == "b"


def test_failure_cooldown_routes_away_from_recent_failure():
    router = _router({"failure_cooldown_s": 60, "max_queue_ttl_s": 0})
    selector = router.balanced_smart_logger
    assert selector is not None
    failed = selector._get_stats("balanced-model:a")
    failed.last_failure_at = time.time()
    selector._set_stats("balanced-model:a", failed)

    deployment = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a"), _deployment("b")],
    )

    assert deployment is not None
    assert deployment["model_info"]["id"] == "b"


def test_success_callback_releases_capacity_and_updates_metrics():
    router = _router({"max_concurrent_requests": 1})
    selector = router.balanced_smart_logger
    assert selector is not None
    assert selector._acquire("balanced-model:a")

    start_time = datetime.now(timezone.utc)
    selector.log_success_event(
        _callback_kwargs("a"),
        {"usage": {"completion_tokens": 20}},
        start_time,
        start_time + timedelta(seconds=2),
    )

    stats = selector._get_stats("balanced-model:a")
    assert stats.active == 0
    assert stats.successes == 1
    assert stats.ewma_tps == 10
    assert stats.ewma_ttft_s == 2


def test_failure_callback_releases_capacity_and_sets_cooldown():
    router = _router({"max_concurrent_requests": 1})
    selector = router.balanced_smart_logger
    assert selector is not None
    assert selector._acquire("balanced-model:a")

    selector.log_failure_event(_callback_kwargs("a"), None, None, None)

    stats = selector._get_stats("balanced-model:a")
    assert stats.active == 0
    assert stats.failures == 1
    assert stats.last_failure_at > 0


def test_balanced_smart_backend_id_shares_capacity_across_aliases():
    router = Router(
        model_list=[
            _deployment("alias-a", backend_id="shared-backend"),
            _deployment("alias-b", backend_id="shared-backend"),
        ],
        routing_strategy="balanced-smart",
        routing_strategy_args={"max_concurrent_requests": 1, "max_queue_ttl_s": 0},
    )
    selector = router.balanced_smart_logger
    assert selector is not None

    first = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[
            _deployment("alias-a", backend_id="shared-backend"),
            _deployment("alias-b", backend_id="shared-backend"),
        ],
    )
    second = selector.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[
            _deployment("alias-a", backend_id="shared-backend"),
            _deployment("alias-b", backend_id="shared-backend"),
        ],
    )

    assert first is not None
    assert second is None


def test_redis_backed_capacity_is_shared_across_handler_instances():
    class MiniRedis:
        def __init__(self):
            self.data = {}

        def eval(self, script, numkeys, key, *args):
            if "return 0" in script and "last_selected_at" in script:
                max_active = int(args[0])
                active = int(self.data.setdefault(key, {}).get("active", 0))
                if active >= max_active:
                    return 0
                self.data[key]["active"] = active + 1
                self.data[key]["selected"] = int(self.data[key].get("selected", 0)) + 1
                self.data[key]["last_selected_at"] = args[1]
                return 1
            if "successes" in script:
                current = int(self.data.setdefault(key, {}).get("active", 0))
                self.data[key]["active"] = max(0, current - 1)
                self.data[key]["successes"] = (
                    int(self.data[key].get("successes", 0)) + 1
                )
                return 1
            if "last_failure_at" in script:
                current = int(self.data.setdefault(key, {}).get("active", 0))
                self.data[key]["active"] = max(0, current - 1)
                self.data[key]["failures"] = int(self.data[key].get("failures", 0)) + 1
                self.data[key]["last_failure_at"] = args[1]
                return 1
            raise AssertionError("unexpected redis script")

        def hgetall(self, key):
            return dict(self.data.get(key, {}))

        def hset(self, key, mapping):
            self.data.setdefault(key, {}).update(mapping)

        def expire(self, key, ttl):
            return True

    class FakeRedisCache:
        def __init__(self):
            self.redis_client = MiniRedis()

        def check_and_fix_namespace(self, key: str) -> str:
            return key

    redis_cache = FakeRedisCache()
    cache_a = DualCache()
    cache_b = DualCache()
    cache_a.redis_cache = redis_cache
    cache_b.redis_cache = redis_cache
    args = {"max_concurrent_requests": 1, "max_queue_ttl_s": 0}
    handler_a = BalancedSmartRoutingHandler(router_cache=cache_a, routing_args=args)
    handler_b = BalancedSmartRoutingHandler(router_cache=cache_b, routing_args=args)

    first = handler_a.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a")],
    )
    second = handler_b.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a")],
    )
    start_time = datetime.now(timezone.utc)
    handler_a.log_success_event(
        _callback_kwargs("a"),
        {"usage": {"completion_tokens": 1}},
        start_time,
        start_time + timedelta(seconds=1),
    )
    third = handler_b.get_available_deployments(
        model_group="balanced-model",
        healthy_deployments=[_deployment("a")],
    )

    assert first is not None
    assert second is None
    assert third is not None
