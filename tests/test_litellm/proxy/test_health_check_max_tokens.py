from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from litellm.proxy import health_check as hc_module
from litellm.proxy.health_check import (
    _resolve_health_check_max_tokens,
    _update_litellm_params_for_health_check,
)


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_default(monkeypatch):
    """
    Test that max_tokens defaults to 5 for non-wildcard models.
    """
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


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


def test_resolve_health_check_max_tokens_reasoning_specific_model_info():
    model_info = {
        "health_check_max_tokens_reasoning": 64,
        "health_check_max_tokens_non_reasoning": 2,
    }
    litellm_params = {"model": "openai/gpt-4o"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 2

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 64


def test_explicit_health_check_max_tokens_beats_reasoning_specific():
    model_info = {
        "health_check_max_tokens": 9,
        "health_check_max_tokens_reasoning": 64,
        "health_check_max_tokens_non_reasoning": 2,
    }
    litellm_params = {"model": "openai/gpt-4o"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 9


def test_reasoning_specific_falls_through_when_wrong_branch_only(monkeypatch):
    """Only non-reasoning key set but model is reasoning → fall back to default 5."""
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {"health_check_max_tokens_non_reasoning": 3}
    litellm_params = {"model": "openai/o1"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        assert _resolve_health_check_max_tokens(model_info, litellm_params) == 5


@pytest.mark.asyncio
async def test_background_split_env_reasoning_vs_non_reasoning(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 50)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 5

    litellm_params2 = {"model": "openai/o1"}
    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        updated2 = _update_litellm_params_for_health_check(model_info, litellm_params2)
        assert updated2["max_tokens"] == 50


@pytest.mark.asyncio
async def test_reasoning_env_precedence_over_global(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 20)

    model_info = {}
    litellm_params = {"model": "openai/gpt-5.4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=True):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 20


@pytest.mark.asyncio
async def test_non_reasoning_uses_global_when_reasoning_env_set(monkeypatch):
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", 10)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", 20)

    model_info = {}
    litellm_params = {"model": "azure/gpt-4"}

    with patch.object(hc_module.litellm, "supports_reasoning", return_value=False):
        updated = _update_litellm_params_for_health_check(model_info, litellm_params)
        assert updated["max_tokens"] == 10


def test_wildcard_ignores_reasoning_split_model_info(monkeypatch):
    """Wildcard routes do not use reasoning/non-reasoning model_info split."""
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS", None)
    monkeypatch.setattr(hc_module, "BACKGROUND_HEALTH_CHECK_MAX_TOKENS_REASONING", None)
    model_info = {
        "health_check_max_tokens_reasoning": 99,
        "health_check_max_tokens_non_reasoning": 7,
    }
    litellm_params = {"model": "openai/*"}

    assert _resolve_health_check_max_tokens(model_info, litellm_params) is None


def test_update_litellm_params_health_check_reasoning_effort():
    """model_info.health_check_reasoning_effort sets reasoning_effort for chat-style health checks."""
    model_info = {"health_check_reasoning_effort": "low"}
    litellm_params = {"model": "openai/gpt-5", "api_key": "x"}
    out = _update_litellm_params_for_health_check(model_info, dict(litellm_params))
    assert out.get("reasoning_effort") == "low"

    model_info = {"mode": "chat", "health_check_reasoning_effort": "none"}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == "none"

    model_info = {"mode": "completion", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == "low"

    model_info = {
        "health_check_reasoning_effort": {"effort": "none", "summary": "auto"},
    }
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-5.1", "api_key": "x"}
    )
    assert out.get("reasoning_effort") == {"effort": "none", "summary": "auto"}

    model_info = {"mode": "embedding", "health_check_reasoning_effort": "low"}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "text-embedding-3-small", "api_key": "x"}
    )
    assert "reasoning_effort" not in out

    model_info = {}
    out = _update_litellm_params_for_health_check(
        model_info, {"model": "openai/gpt-4o", "api_key": "x"}
    )
    assert "reasoning_effort" not in out
