from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.proxy._types import (
    BlockModelRequest,
    LitellmUserRoles,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.types.router import RouterRateLimitError


def _setup_model_block_mocks(monkeypatch, *, updated_blocked: bool):
    model_id = "model-123"

    existing_row = MagicMock()
    existing_row.model_dump.return_value = {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o"},
        "model_info": {"id": model_id},
    }

    updated_row = MagicMock()
    updated_row.model_id = model_id
    updated_row.blocked = updated_blocked

    model_table = MagicMock()
    model_table.find_unique = AsyncMock(return_value=existing_row)
    model_table.update = AsyncMock(return_value=updated_row)

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_proxymodeltable = model_table

    mock_router = MagicMock()
    mock_router.get_deployment.return_value = None

    mock_clear_cache = AsyncMock(return_value=None)
    mock_audit_log = AsyncMock(return_value=None)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_router)
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.model_management_endpoints.clear_cache",
        mock_clear_cache,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.model_management_endpoints.create_object_audit_log",
        mock_audit_log,
    )

    return model_id, model_table, updated_row, mock_clear_cache, mock_audit_log


def _proxy_admin() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_id="admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-admin",
    )


@pytest.mark.asyncio
async def test_model_block_endpoint_sets_blocked_true(monkeypatch):
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        block_model,
    )

    model_id, model_table, updated_row, mock_clear_cache, mock_audit_log = (
        _setup_model_block_mocks(monkeypatch, updated_blocked=True)
    )

    result = await block_model(
        data=BlockModelRequest(model_id=model_id),
        http_request=MagicMock(),
        user_api_key_dict=_proxy_admin(),
        litellm_changed_by="operator@example.com",
    )

    assert result == updated_row
    model_table.update.assert_awaited_once()
    update_kwargs = model_table.update.await_args.kwargs
    assert update_kwargs["where"] == {"model_id": model_id}
    assert update_kwargs["data"]["blocked"] is True
    assert update_kwargs["data"]["updated_by"] == "admin"
    assert "updated_at" in update_kwargs["data"]
    mock_clear_cache.assert_awaited_once_with()
    assert mock_audit_log.call_args.kwargs["action"] == "blocked"
    assert (
        mock_audit_log.call_args.kwargs["litellm_changed_by"] == "operator@example.com"
    )


@pytest.mark.asyncio
async def test_model_unblock_endpoint_sets_blocked_false(monkeypatch):
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        unblock_model,
    )

    model_id, model_table, updated_row, mock_clear_cache, mock_audit_log = (
        _setup_model_block_mocks(monkeypatch, updated_blocked=False)
    )

    result = await unblock_model(
        data=BlockModelRequest(model_id=model_id),
        http_request=MagicMock(),
        user_api_key_dict=_proxy_admin(),
        litellm_changed_by=None,
    )

    assert result == updated_row
    model_table.update.assert_awaited_once()
    assert model_table.update.await_args.kwargs["data"]["blocked"] is False
    mock_clear_cache.assert_awaited_once_with()
    assert mock_audit_log.call_args.kwargs["action"] == "unblocked"


@pytest.mark.asyncio
async def test_model_block_endpoint_requires_proxy_admin(monkeypatch):
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        block_model,
    )

    model_id, model_table, _, _, _ = _setup_model_block_mocks(
        monkeypatch, updated_blocked=True
    )
    non_admin = UserAPIKeyAuth(
        user_id="internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-user",
    )

    with pytest.raises(ProxyException) as exc_info:
        await block_model(
            data=BlockModelRequest(model_id=model_id),
            http_request=MagicMock(),
            user_api_key_dict=non_admin,
            litellm_changed_by=None,
        )

    assert exc_info.value.code == "403"
    assert "Only proxy admins" in exc_info.value.message
    model_table.update.assert_not_awaited()


def test_router_returns_no_healthy_deployment_when_model_is_fully_blocked():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o-0"},
                "model_info": {"id": "dep-0", "blocked": True},
            },
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o-1"},
                "model_info": {"id": "dep-1", "blocked": True},
            },
        ]
    )

    with pytest.raises(RouterRateLimitError) as exc_info:
        router.get_available_deployment(model="gpt-4o", request_kwargs={})

    assert "No deployments available for selected model" in str(exc_info.value)
    assert "Passed model=gpt-4o" in str(exc_info.value)


@pytest.mark.asyncio
async def test_route_request_returns_403_when_model_is_fully_blocked(monkeypatch):
    from litellm.proxy.route_llm_request import route_request

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "dep-0", "blocked": True},
            }
        ]
    )
    monkeypatch.setattr(
        "litellm.proxy.route_llm_request.add_shared_session_to_data",
        AsyncMock(return_value=None),
    )

    with pytest.raises(litellm.PermissionDeniedError) as exc_info:
        await route_request(
            data={"model": "gpt-4o"},
            llm_router=router,
            user_model=None,
            route_type="acreate_eval",
        )

    assert exc_info.value.status_code == 403
    assert "Model is blocked" in exc_info.value.message
