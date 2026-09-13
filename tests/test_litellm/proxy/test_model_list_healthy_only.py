"""
Tests for the opt-in health filter on the model listing endpoints: the
per-request `healthy_only` query parameter and the proxy-wide
`general_settings.model_list_healthy_only` setting, across GET /v1/models
(`model_list`), GET /v1/models/{id} (`model_info`) and GET /v1/model/info
(`model_info_v1`).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

HEALTHY_ONLY_SETTING = {"model_list_healthy_only": True}


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
    monkeypatch.setattr(proxy_server, "general_settings", {})

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


@pytest.fixture
def patched_model_info_v1(monkeypatch):
    """Stub router + globals used by the `/v1/model/info` list path."""
    healthy_row = {
        "model_name": "gpt-4",
        "litellm_params": {"model": "gpt-4"},
        "model_info": {"id": "healthy-id", "db_model": False},
    }
    unhealthy_row = {
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "anthropic/claude-sonnet"},
        "model_info": {"id": "unhealthy-id", "db_model": False},
    }
    router = MagicMock()
    router.model_list = [healthy_row, unhealthy_row]
    router.get_model_list_from_model_alias.return_value = []
    router.get_model_names.return_value = ["gpt-4", "claude-sonnet"]
    router.get_model_access_groups.return_value = {}
    router.async_get_fully_unhealthy_model_names = AsyncMock(return_value={"claude-sonnet"})

    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "_enrich_model_info_with_litellm_data", lambda model, **kw: model)
    return router


def _admin_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        team_models=[],
    )


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


@pytest.mark.asyncio
async def test_model_list_general_setting_hides_unhealthy_models(patched_model_list, monkeypatch):
    """`model_list_healthy_only: true` filters callers that pass no query param."""
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_list_general_setting_applies_to_scope_expand(patched_model_list, monkeypatch):
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
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)
    patched_model_list.get_model_names = MagicMock(return_value=["gpt-4", "claude-sonnet"])
    patched_model_list.get_model_access_groups = MagicMock(return_value={})

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        scope="expand",
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_list_general_setting_false_keeps_unhealthy_models(patched_model_list, monkeypatch):
    """Explicit `false` must behave exactly like the unset default."""
    monkeypatch.setattr(proxy_server, "general_settings", {"model_list_healthy_only": False})

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]
    patched_model_list.async_get_fully_unhealthy_model_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_list_non_boolean_general_setting_does_not_filter(patched_model_list, monkeypatch):
    """A quoted YAML value is not a bool; never filter on an ambiguous value."""
    monkeypatch.setattr(proxy_server, "general_settings", {"model_list_healthy_only": "true"})

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]
    patched_model_list.async_get_fully_unhealthy_model_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_list_blocked_models_hidden_without_health_filter(
    patched_model_list,
):
    """Blocked-model hiding is independent of the health filter."""
    patched_model_list.get_fully_blocked_model_names = MagicMock(return_value={"gpt-4"})

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["claude-sonnet"]


@pytest.mark.asyncio
async def test_model_list_no_router_does_not_filter(patched_model_list, monkeypatch):
    """No router means no health state; fail open rather than hiding everything."""
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]


@pytest.mark.asyncio
async def test_model_list_general_setting_no_health_state_keeps_all_models(patched_model_list, monkeypatch):
    """Setting on but no background health checks running: hide nothing."""
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)
    patched_model_list.async_get_fully_unhealthy_model_names = AsyncMock(return_value=set())

    response = await proxy_server.model_list(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert [m["id"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]


@pytest.mark.asyncio
async def test_retrieve_model_general_setting_hides_unhealthy_model(patched_model_list, monkeypatch):
    """GET /v1/models/{id} must not serve a model the listing hides."""
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)

    with pytest.raises(HTTPException) as exc_info:
        await proxy_server.model_info(
            model_id="claude-sonnet",
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_retrieve_model_default_serves_unhealthy_model(patched_model_list, monkeypatch):
    """Without the opt-in, retrieve keeps serving unhealthy models."""
    import litellm

    deployment = MagicMock()
    deployment.litellm_params.model = "anthropic/claude-sonnet"
    patched_model_list.get_deployment_by_model_group_name.return_value = deployment
    monkeypatch.setattr(litellm, "get_llm_provider", lambda model: (model, "anthropic", None, None))

    response = await proxy_server.model_info(
        model_id="claude-sonnet",
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
    )
    assert response["id"] == "claude-sonnet"
    patched_model_list.async_get_fully_unhealthy_model_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_info_v1_healthy_only_hides_unhealthy_deployments(
    patched_model_info_v1,
):
    response = await proxy_server.model_info_v1(
        user_api_key_dict=_admin_key(),
        litellm_model_id=None,
        healthy_only=True,
    )
    assert [m["model_name"] for m in response["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_info_v1_general_setting_hides_unhealthy_deployments(patched_model_info_v1, monkeypatch):
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)

    response = await proxy_server.model_info_v1(
        user_api_key_dict=_admin_key(),
        litellm_model_id=None,
    )
    assert [m["model_name"] for m in response["data"]] == ["gpt-4"]


@pytest.mark.asyncio
async def test_model_info_v1_default_keeps_unhealthy_deployments(
    patched_model_info_v1,
):
    response = await proxy_server.model_info_v1(
        user_api_key_dict=_admin_key(),
        litellm_model_id=None,
    )
    assert [m["model_name"] for m in response["data"]] == ["gpt-4", "claude-sonnet"]
    patched_model_info_v1.async_get_fully_unhealthy_model_names.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_info_v1_litellm_model_id_lookup_ignores_health_filter(patched_model_info_v1, monkeypatch):
    """The by-id lookup backs the dashboard's model detail view; turning the
    proxy-wide filter on must not make an unhealthy model unopenable there."""
    monkeypatch.setattr(proxy_server, "general_settings", HEALTHY_ONLY_SETTING)
    deployment = MagicMock()
    deployment.model_dump.return_value = {
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "anthropic/claude-sonnet"},
        "model_info": {"id": "unhealthy-id"},
    }
    patched_model_info_v1.get_deployment.return_value = deployment

    response = await proxy_server.model_info_v1(
        user_api_key_dict=_admin_key(),
        litellm_model_id="unhealthy-id",
    )
    assert [m["model_name"] for m in response["data"]] == ["claude-sonnet"]
