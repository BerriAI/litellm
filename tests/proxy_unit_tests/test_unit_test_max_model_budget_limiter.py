import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
from datetime import datetime as dt_object
import time
import pytest
import litellm

import json
from litellm.types.utils import BudgetConfig as GenericBudgetInfo
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch
import pytest
from litellm.caching.caching import DualCache
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.proxy._types import UserAPIKeyAuth
import litellm


# Test class setup
@pytest.fixture
def budget_limiter():
    dual_cache = DualCache()
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=dual_cache)


# Test _get_model_without_custom_llm_provider
def test_get_model_without_custom_llm_provider(budget_limiter):
    # Test with custom provider
    assert (
        budget_limiter._get_model_without_custom_llm_provider("openai/gpt-4") == "gpt-4"
    )

    # Test without custom provider
    assert budget_limiter._get_model_without_custom_llm_provider("gpt-4") == "gpt-4"


# Test _get_request_model_budget_config
def test_get_request_model_budget_config(budget_limiter):
    internal_budget = {
        "gpt-4": GenericBudgetInfo(budget_limit=100.0, time_period="1d"),
        "claude-3": GenericBudgetInfo(budget_limit=50.0, time_period="1d"),
    }

    # Test direct model match
    config = budget_limiter._get_request_model_budget_config(
        model="gpt-4", internal_model_max_budget=internal_budget
    )
    assert config.max_budget == 100.0

    # Test model with provider
    config = budget_limiter._get_request_model_budget_config(
        model="openai/gpt-4", internal_model_max_budget=internal_budget
    )
    assert config.max_budget == 100.0

    # Test non-existent model
    config = budget_limiter._get_request_model_budget_config(
        model="non-existent", internal_model_max_budget=internal_budget
    )
    assert config is None


# Test is_key_within_model_budget
@pytest.mark.asyncio
async def test_is_key_within_model_budget(budget_limiter):
    # Mock user API key dict
    user_api_key = UserAPIKeyAuth(
        token="test-key",
        key_alias="test-alias",
        model_max_budget={"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}},
    )

    # Test when model is within budget
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=50.0
    ):
        assert (
            await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")
            is True
        )

    # Test when model exceeds budget
    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", return_value=150.0
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-4")

    # Test model not in budget config
    assert (
        await budget_limiter.is_key_within_model_budget(user_api_key, "non-existent")
        is True
    )


# Test _get_virtual_key_spend_for_model
@pytest.mark.asyncio
async def test_get_virtual_key_spend_for_model(budget_limiter):
    budget_config = GenericBudgetInfo(budget_limit=100.0, time_period="1d")

    # Mock cache get
    with patch.object(budget_limiter.dual_cache, "async_get_cache", return_value=50.0):
        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash="test-key", model="gpt-4", key_budget_config=budget_config
        )
        assert spend == 50.0

        # Test with provider prefix
        spend = await budget_limiter._get_virtual_key_spend_for_model(
            user_api_key_hash="test-key",
            model="openai/gpt-4",
            key_budget_config=budget_config,
        )
        assert spend == 50.0


@pytest.mark.asyncio
async def test_virtual_key_budget_tracking_respects_duration(budget_limiter):
    user_api_key = UserAPIKeyAuth(
        token="virtual-key",
        key_alias="vk-alias",
        model_max_budget={
            "gpt-5": {"budget_limit": 1e-9, "time_period": "30d"}
        },
    )

    logging_kwargs = {
        "standard_logging_object": {
            "response_cost": 6e-10,
            "model": "gpt-5",
            "metadata": {"user_api_key_hash": user_api_key.token},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": user_api_key.model_max_budget
            }
        },
        "proxy_server_request": {"body": {"model": "gpt-5"}},
    }

    await budget_limiter.async_log_success_event(
        logging_kwargs, None, datetime.now(), datetime.now()
    )

    assert (
        await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-5")
        is True
    )

    await budget_limiter.async_log_success_event(
        logging_kwargs, None, datetime.now(), datetime.now()
    )

    with pytest.raises(litellm.BudgetExceededError):
        await budget_limiter.is_key_within_model_budget(user_api_key, "gpt-5")


@pytest.mark.asyncio
async def test_async_filter_deployments_filters_over_budget(budget_limiter):
    virtual_key_hash = "vk-over-budget"
    user_budget = {
        "gpt-5": {"budget_limit": 1e-9, "time_period": "30d"},
        "gpt-5-mini": {"budget_limit": 1e-9, "time_period": "30d"},
    }

    await budget_limiter.dual_cache.async_set_cache(
        key=f"virtual_key_spend:{virtual_key_hash}:gpt-5:30d",
        value=2e-9,
    )

    healthy_deployments = [
        {
            "model_name": "feedback",
            "litellm_params": {"model": "openai/gpt-5"},
        },
        {
            "model_name": "feedback-fallback",
            "litellm_params": {"model": "openai/gpt-5-mini"},
        },
    ]

    request_kwargs = {
        "model": "feedback",
        "metadata": {
            "user_api_key_model_max_budget": user_budget,
            "user_api_key_hash": virtual_key_hash,
            "user_api_key_alias": "vk-alias",
        },
    }

    filtered = await budget_limiter.async_filter_deployments(
        model="feedback",
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert len(filtered) == 1
    assert filtered[0]["model_name"] == "feedback-fallback"


@pytest.mark.asyncio
async def test_async_filter_deployments_raises_when_all_over_budget(budget_limiter):
    virtual_key_hash = "vk-over-budget-both"
    user_budget = {
        "gpt-5": {"budget_limit": 1e-9, "time_period": "30d"},
        "gpt-5-mini": {"budget_limit": 1e-9, "time_period": "30d"},
    }

    await budget_limiter.dual_cache.async_set_cache(
        key=f"virtual_key_spend:{virtual_key_hash}:gpt-5:30d",
        value=2e-9,
    )
    await budget_limiter.dual_cache.async_set_cache(
        key=f"virtual_key_spend:{virtual_key_hash}:gpt-5-mini:30d",
        value=2e-9,
    )

    healthy_deployments = [
        {
            "model_name": "feedback",
            "litellm_params": {"model": "openai/gpt-5"},
        },
        {
            "model_name": "feedback-fallback",
            "litellm_params": {"model": "openai/gpt-5-mini"},
        },
    ]

    request_kwargs = {
        "model": "feedback",
        "metadata": {
            "user_api_key_model_max_budget": user_budget,
            "user_api_key_hash": virtual_key_hash,
            "user_api_key_alias": "vk-alias",
        },
    }

    with pytest.raises(litellm.BudgetExceededError):
        await budget_limiter.async_filter_deployments(
            model="feedback",
            healthy_deployments=healthy_deployments,
            messages=None,
            request_kwargs=request_kwargs,
        )
