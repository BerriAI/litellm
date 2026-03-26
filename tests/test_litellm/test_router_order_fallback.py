"""
Tests for order-based fallback routing.

When deployments have `order` set in litellm_params, lower order deployments
should be tried first, and higher order deployments should be used as fallbacks
when lower order deployments fail.
"""

from typing import Optional

import pytest

from litellm import Router
from litellm.utils import _get_order_filtered_deployments

# ---------------------------------------------------------------------------
# Unit tests for _get_order_filtered_deployments
# ---------------------------------------------------------------------------


class TestGetOrderFilteredDeployments:
    def _make_deployment(self, order: Optional[int], dep_id: str) -> dict:
        params: dict = {"model": "gpt-4o", "api_key": "key"}
        if order is not None:
            params["order"] = order
        return {
            "model_name": "test-model",
            "litellm_params": params,
            "model_info": {"id": dep_id},
        }

    def test_returns_min_order_group(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(1, "c"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2
        assert all(d["model_info"]["id"] in ("a", "c") for d in result)

    def test_target_order_filters_to_exact_level(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
            self._make_deployment(3, "c"),
        ]
        result = _get_order_filtered_deployments(deps, target_order=2)
        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "b"

    def test_target_order_no_match_returns_all(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(2, "b"),
        ]
        result = _get_order_filtered_deployments(deps, target_order=99)
        assert len(result) == 2

    def test_no_order_set_returns_all(self):
        deps = [
            self._make_deployment(None, "a"),
            self._make_deployment(None, "b"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2

    def test_empty_list(self):
        result = _get_order_filtered_deployments([])
        assert result == []

    def test_single_order_returns_all_with_that_order(self):
        deps = [
            self._make_deployment(1, "a"),
            self._make_deployment(1, "b"),
        ]
        result = _get_order_filtered_deployments(deps)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests for order-based fallback in Router
# ---------------------------------------------------------------------------


def test_router_order_without_pre_call_checks():
    """Order filtering should work even when enable_pre_call_checks=False (default)."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
        enable_pre_call_checks=False,
    )

    for _ in range(20):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


def test_router_order_no_fallback_when_healthy():
    """When order=1 is healthy, order=2 should never be used."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 1",
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    for _ in range(50):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response._hidden_params["model_id"] == "1"


@pytest.mark.asyncio
async def test_router_order_fallback_on_failure():
    """When order=1 fails, order=2 should be tried as fallback."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad-key",
                    "mock_response": Exception("connection error"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good-key",
                    "mock_response": "success from order 2",
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "2"


@pytest.mark.asyncio
async def test_router_order_fallback_three_levels():
    """When order=1 and order=2 both fail, order=3 should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from order 3",
                    "order": 3,
                },
                "model_info": {"id": "3"},
            },
        ],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "3"


@pytest.mark.asyncio
async def test_router_order_fallback_then_external_fallback():
    """When all order levels fail, external fallbacks should be tried."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "good",
                    "mock_response": "success from external fallback",
                },
                "model_info": {"id": "fallback"},
            },
        ],
        fallbacks=[{"test-model": ["fallback-model"]}],
        num_retries=0,
    )

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert response._hidden_params["model_id"] == "fallback"
