from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import URL

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder


@pytest.fixture
def proxy_server_auth_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[UserAPIKeyAuth]:
    api_key = "sk-admin-key-budget-test"
    token = UserAPIKeyAuth(
        api_key=api_key,
        token=api_key,
        user_id="admin-user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        max_budget=1.0,
    )

    cache = AsyncMock()
    cache.async_get_cache = AsyncMock(return_value=None)
    cache.get_cache = AsyncMock(return_value=None)
    cache.delete_cache = MagicMock()

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.internal_usage_cache = MagicMock()
    proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    proxy_logging_obj.budget_alerts = AsyncMock(return_value=None)

    for attr, value in {
        "prisma_client": MagicMock(),
        "user_api_key_cache": cache,
        "proxy_logging_obj": proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }.items():
        monkeypatch.setattr(ps, attr, value)

    with patch(
        "litellm.proxy.auth.resolvers.store.IdentityStore._resolve_key",
        new_callable=AsyncMock,
        return_value=token,
    ):
        yield token


async def _authenticate(api_key: str) -> UserAPIKeyAuth:
    request = Request(scope={"type": "http"})
    request._url = URL(url="/chat/completions")
    return await user_api_key_auth_builder(
        request=request,
        api_key=f"Bearer {api_key}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        request_data={},
    )


@pytest.mark.asyncio
async def test_proxy_admin_key_over_budget_is_rejected(
    proxy_server_auth_state: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admin keys skipped the key max_budget check and stayed usable after spend exceeded it."""
    monkeypatch.setattr(ps, "get_current_spend", AsyncMock(return_value=2.0))

    with pytest.raises(ProxyException) as exc_info:
        await _authenticate(proxy_server_auth_state.api_key or "")

    assert "Budget has been exceeded" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_proxy_admin_key_under_budget_is_allowed(
    proxy_server_auth_state: UserAPIKeyAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: the same admin path still authenticates while under budget."""
    monkeypatch.setattr(ps, "get_current_spend", AsyncMock(return_value=0.5))

    result = await _authenticate(proxy_server_auth_state.api_key or "")

    assert result.user_role == LitellmUserRoles.PROXY_ADMIN
