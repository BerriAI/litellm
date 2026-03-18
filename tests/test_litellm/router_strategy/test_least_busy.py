import os
import sys
from collections import Counter

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler


def _make_deployment(id_val):
    return {
        "model_name": "test-model",
        "litellm_params": {"model": "openai/gpt-4.1-mini"},
        "model_info": {"id": str(id_val)},
    }


class TestLeastBusyTieBreaking:
    """Tests that least-busy strategy distributes requests across tied deployments."""

    def test_should_randomly_distribute_when_all_counts_are_zero(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(i) for i in range(3)]

        selections = Counter()
        for _ in range(300):
            chosen = handler._get_available_deployments(
                healthy_deployments=deployments, all_deployments={}
            )
            selections[chosen["model_info"]["id"]] += 1

        assert len(selections) == 3, (
            f"Expected all 3 deployments to be selected at least once, got: {selections}"
        )
        for dep_id, count in selections.items():
            assert count > 30, (
                f"Deployment {dep_id} selected only {count}/300 times — distribution is too skewed"
            )

    def test_should_randomly_distribute_when_counts_are_tied(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(i) for i in range(2)]
        tied_counts = {"0": 5, "1": 5}

        selections = Counter()
        for _ in range(200):
            chosen = handler._get_available_deployments(
                healthy_deployments=deployments,
                all_deployments=dict(tied_counts),
            )
            selections[chosen["model_info"]["id"]] += 1

        assert len(selections) == 2, (
            f"Expected both deployments to be selected, got: {selections}"
        )
        for dep_id, count in selections.items():
            assert count > 40, (
                f"Deployment {dep_id} selected only {count}/200 times — distribution is too skewed"
            )

    def test_should_pick_unique_minimum(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(i) for i in range(3)]
        counts = {"0": 10, "1": 1, "2": 50}

        for _ in range(50):
            chosen = handler._get_available_deployments(
                healthy_deployments=deployments,
                all_deployments=dict(counts),
            )
            assert chosen["model_info"]["id"] == "1"

    def test_should_skip_unhealthy_deployments_in_cache(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        healthy = [_make_deployment(1), _make_deployment(2)]
        counts_with_stale = {"0": 0, "1": 5, "2": 10}

        for _ in range(50):
            chosen = handler._get_available_deployments(
                healthy_deployments=healthy,
                all_deployments=dict(counts_with_stale),
            )
            assert chosen["model_info"]["id"] == "1"


class TestLeastBusyGetAvailableDeployments:
    """Tests the sync/async wrappers for cache interaction."""

    def test_should_use_cache_for_selection(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(0), _make_deployment(1)]

        cache.set_cache(
            key="test-model_request_count", value={"0": 10, "1": 2}
        )
        chosen = handler.get_available_deployments(
            model_group="test-model", healthy_deployments=deployments
        )
        assert chosen["model_info"]["id"] == "1"

    @pytest.mark.asyncio
    async def test_should_use_cache_for_async_selection(self):
        cache = DualCache()
        handler = LeastBusyLoggingHandler(router_cache=cache)
        deployments = [_make_deployment(0), _make_deployment(1)]

        await cache.async_set_cache(
            key="test-model_request_count", value={"0": 10, "1": 2}
        )
        chosen = await handler.async_get_available_deployments(
            model_group="test-model", healthy_deployments=deployments
        )
        assert chosen["model_info"]["id"] == "1"
