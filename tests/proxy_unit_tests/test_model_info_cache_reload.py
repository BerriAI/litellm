"""
Tests for the /model/info cache-miss reload fix (MLP-6153).

When llm_router.get_deployment() returns None for a given model ID, the handler
must call proxy_config.add_deployment() to reload the router cache from the DB,
then retry the lookup before raising a 400. This covers the multi-replica
eventual-consistency race where the pod serving the Read-after-Apply hasn't yet
loaded the newly created model into its local router cache.
"""
import pytest
import litellm.proxy.proxy_server as proxy_server
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import model_info_v1


def _make_deployment(model_id: str = "test-model-id") -> MagicMock:
    """Return a minimal mock Deployment accepted by _get_proxy_model_info."""
    deployment = MagicMock()
    deployment.model_dump.return_value = {
        "model_name": "claude-sonnet-4-5",
        "litellm_params": {"model": "vertex_ai/claude-sonnet-4-5", "custom_llm_provider": "vertex_ai"},
        "model_info": {"id": model_id, "base_model": "claude-sonnet-4-5"},
    }
    return deployment


def _mock_proxy_config(add_deployment_side_effect=None) -> MagicMock:
    cfg = MagicMock()
    cfg.add_deployment = AsyncMock(side_effect=add_deployment_side_effect)
    return cfg


@pytest.mark.asyncio
async def test_model_info_returns_deployment_when_cache_is_warm():
    """Happy path: router cache already has the model, no reload needed."""
    deployment = _make_deployment()
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = deployment

    mock_proxy_cfg = _mock_proxy_config()

    setattr(proxy_server, "llm_router", mock_router)
    setattr(proxy_server, "llm_model_list", [{"model_name": "claude-sonnet-4-5"}])
    setattr(proxy_server, "prisma_client", MagicMock())
    setattr(proxy_server, "proxy_config", mock_proxy_cfg)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    resp = await model_info_v1(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
        litellm_model_id="test-model-id",
    )

    assert resp["data"][0]["model_name"] == "claude-sonnet-4-5"
    mock_proxy_cfg.add_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_model_info_reloads_cache_on_miss_and_returns_deployment():
    """
    Core regression test: router returns None on first call (cache miss),
    add_deployment() is triggered, second call succeeds.
    This is the exact race condition that caused the Terraform provider error.
    """
    deployment = _make_deployment()
    mock_router = MagicMock()
    mock_router.get_deployment.side_effect = [None, deployment]

    mock_proxy_cfg = _mock_proxy_config()

    setattr(proxy_server, "llm_router", mock_router)
    setattr(proxy_server, "llm_model_list", [{"model_name": "claude-sonnet-4-5"}])
    setattr(proxy_server, "prisma_client", MagicMock())
    setattr(proxy_server, "proxy_config", mock_proxy_cfg)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    resp = await model_info_v1(
        user_api_key_dict=UserAPIKeyAuth(models=[]),
        litellm_model_id="test-model-id",
    )

    assert resp["data"][0]["model_name"] == "claude-sonnet-4-5"
    mock_proxy_cfg.add_deployment.assert_called_once()
    assert mock_router.get_deployment.call_count == 2


@pytest.mark.asyncio
async def test_model_info_raises_400_when_model_absent_after_reload():
    """
    If the model is still absent after the cache reload, the endpoint must
    raise 400 — not silently return empty data (which would confuse Terraform).
    """
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None  # absent before and after reload

    mock_proxy_cfg = _mock_proxy_config()

    setattr(proxy_server, "llm_router", mock_router)
    setattr(proxy_server, "llm_model_list", [{"model_name": "claude-sonnet-4-5"}])
    setattr(proxy_server, "prisma_client", MagicMock())
    setattr(proxy_server, "proxy_config", mock_proxy_cfg)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        await model_info_v1(
            user_api_key_dict=UserAPIKeyAuth(models=[]),
            litellm_model_id="unknown-id",
        )

    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail["error"]
    # Reload was still attempted
    mock_proxy_cfg.add_deployment.assert_called_once()


@pytest.mark.asyncio
async def test_model_info_skips_reload_when_no_prisma_client():
    """
    If prisma_client is None (no DB configured), there is nothing to reload
    from. The endpoint should skip the reload and raise 400 directly.
    """
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    mock_proxy_cfg = _mock_proxy_config()

    setattr(proxy_server, "llm_router", mock_router)
    setattr(proxy_server, "llm_model_list", [{"model_name": "claude-sonnet-4-5"}])
    setattr(proxy_server, "prisma_client", None)
    setattr(proxy_server, "proxy_config", mock_proxy_cfg)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        await model_info_v1(
            user_api_key_dict=UserAPIKeyAuth(models=[]),
            litellm_model_id="test-model-id",
        )

    assert exc_info.value.status_code == 400
    mock_proxy_cfg.add_deployment.assert_not_called()


@pytest.mark.asyncio
async def test_model_info_returns_deployment_even_if_reload_raises():
    """
    If add_deployment() itself throws (e.g. DB is temporarily unavailable),
    the exception must be swallowed and the endpoint falls through to the
    404 path gracefully — it should not propagate a 500.
    """
    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    mock_proxy_cfg = _mock_proxy_config(
        add_deployment_side_effect=Exception("DB connection lost")
    )

    setattr(proxy_server, "llm_router", mock_router)
    setattr(proxy_server, "llm_model_list", [{"model_name": "claude-sonnet-4-5"}])
    setattr(proxy_server, "prisma_client", MagicMock())
    setattr(proxy_server, "proxy_config", mock_proxy_cfg)
    setattr(proxy_server, "proxy_logging_obj", MagicMock())

    with pytest.raises(HTTPException) as exc_info:
        await model_info_v1(
            user_api_key_dict=UserAPIKeyAuth(models=[]),
            litellm_model_id="test-model-id",
        )

    # Must be 400 (model not found), not 500 (unhandled DB error)
    assert exc_info.value.status_code == 400
