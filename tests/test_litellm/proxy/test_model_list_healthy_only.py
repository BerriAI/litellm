"""
Tests for the opt-in `healthy_only` filter on GET /v1/models (`model_list`).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth


@pytest.fixture
def patched_model_list(monkeypatch):
    """Stub router + utility helpers used by `model_list`."""
    from litellm.proxy import utils as proxy_utils

    router = MagicMock()
    router.get_fully_blocked_model_names = MagicMock(return_value=set())
    router.async_get_fully_unhealthy_model_names = AsyncMock(
        return_value={"claude-sonnet"}
    )

    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "user_model", None)

    async def _fake_get_available_models_for_user(**kwargs):
        return ["gpt-4", "claude-sonnet"]

    monkeypatch.setattr(
        proxy_utils,
        "get_available_models_for_user",
        _fake_get_available_models_for_user,
    )

    def _fake_create_model_info_response(model_id, provider="openai", **kwargs):
        return {"id": model_id, "object": "model", "created": 0, "owned_by": provider}

    monkeypatch.setattr(
        proxy_utils, "create_model_info_response", _fake_create_model_info_response
    )

    return router


@pytest.mark.asyncio
async def test_model_list_healthy_only_hides_fully_unhealthy_models(
    patched_model_list,
):
    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        healthy_only=True,
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_list_default_keeps_unhealthy_models(patched_model_list):
    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]
    patched_model_list.async_get_fully_unhealthy_model_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_list_healthy_only_applies_to_scope_expand(
    patched_model_list, monkeypatch
):
    from litellm.proxy.auth import model_checks
    from litellm.proxy.management_endpoints import common_utils

    async def _fake_admin(**kwargs):
        return True

    monkeypatch.setattr(common_utils, "_user_has_admin_privileges", _fake_admin)
    monkeypatch.setattr(
        model_checks,
        "get_complete_model_list",
        lambda **kwargs: ["gpt-4", "claude-sonnet"],
    )
    patched_model_list.get_model_names = MagicMock(
        return_value=["gpt-4", "claude-sonnet"]
    )
    patched_model_list.get_model_access_groups = MagicMock(return_value={})

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        scope="expand",
        healthy_only=True,
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4"]
