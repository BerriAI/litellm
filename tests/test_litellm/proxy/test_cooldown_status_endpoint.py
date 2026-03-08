"""Tests for the cooldown status API endpoint."""
import pytest
from unittest.mock import MagicMock, patch
import time

from litellm.router_utils.cooldown_cache import CooldownCacheValue


@pytest.mark.asyncio
async def test_get_deployment_cooldown_status():
    """Should return cooldown status for all deployments."""
    from litellm.proxy.proxy_server import get_deployment_cooldown_status

    mock_router = MagicMock()
    mock_router.get_model_list.return_value = [
        {
            "model_name": "gpt-4",
            "model_info": {"id": "dep-1"},
            "litellm_params": {"model": "gpt-4"},
        },
        {
            "model_name": "gpt-4",
            "model_info": {"id": "dep-2"},
            "litellm_params": {"model": "gpt-4"},
        },
    ]

    cooldown_data = CooldownCacheValue(
        exception_received="Rate limit exceeded",
        status_code="429",
        timestamp=time.time(),
        cooldown_time=4.0,
    )
    mock_router.cooldown_cache.get_active_cooldowns.return_value = [
        ("dep-1", cooldown_data)
    ]

    with patch("litellm.proxy.proxy_server.llm_router", mock_router):
        result = await get_deployment_cooldown_status()

    assert len(result["cooldowns"]) == 1
    assert result["cooldowns"][0]["model_id"] == "dep-1"
    assert result["cooldowns"][0]["status"] == "cooldown"
    assert len(result["healthy"]) == 1
    assert result["healthy"][0]["model_id"] == "dep-2"


@pytest.mark.asyncio
async def test_get_deployment_cooldown_status_no_router():
    """Should return empty when no router is configured."""
    from litellm.proxy.proxy_server import get_deployment_cooldown_status

    with patch("litellm.proxy.proxy_server.llm_router", None):
        result = await get_deployment_cooldown_status()

    assert result["cooldowns"] == []
    assert result["healthy"] == []
