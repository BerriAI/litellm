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
