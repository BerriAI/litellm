from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.proxy import health_check as hc_module
from litellm.proxy.health_check import _update_litellm_params_for_health_check


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_default():
    """
    Test that max_tokens defaults to 1 for non-wildcard models.
    """
    model_info = {}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 1


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_custom():
    """
    Test that max_tokens respects health_check_max_tokens from model_info.
    """
    model_info = {"health_check_max_tokens": 5}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_wildcard():
    """
    Test that max_tokens does NOT default to 1 for wildcard models.
    """
    model_info = {}
    litellm_params = {"model": "openai/*"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    # Should not be set to 1
    assert "max_tokens" not in updated_params or updated_params["max_tokens"] != 1


@pytest.mark.asyncio
async def test_ahealth_check_wildcard_models_respects_max_tokens():
    """
    Test that ahealth_check_wildcard_models respects max_tokens if passed,
    otherwise defaults to 10.
    """
    with (
        patch(
            "litellm.litellm_core_utils.llm_request_utils.pick_cheapest_chat_models_from_llm_provider",
            return_value=["gpt-4o-mini"],
        ),
        patch("litellm.acompletion", new_callable=AsyncMock),
    ):
        # Test Case 1: No max_tokens passed, should default to 10
        model_params = {}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 10

        # Test Case 2: Custom health_check_max_tokens passed via model_params, should be respected
        model_params = {"max_tokens": 3}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 3


@pytest.mark.asyncio
async def test_background_health_check_max_tokens_env_var(monkeypatch):
    """
    Test that BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var is used as global default
    for explicit (non-wildcard) models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 10


@pytest.mark.asyncio
async def test_per_model_overrides_global_env_var(monkeypatch):
    """
    Test that per-model health_check_max_tokens takes priority over
    BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)

    model_info = {"health_check_max_tokens": 5}
    litellm_params = {"model": "azure/gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


@pytest.mark.asyncio
async def test_global_env_var_applies_to_wildcard_models(monkeypatch):
    """
    Test that BACKGROUND_HEALTH_CHECK_MAX_TOKENS env var also applies to wildcard models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 15)

    model_info = {}
    litellm_params = {"model": "openai/*"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 15
