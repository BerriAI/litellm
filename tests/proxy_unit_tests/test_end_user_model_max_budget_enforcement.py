import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import update_valid_token_with_end_user_params

MODEL = "google/gemini-2.5-flash-lite"
MODEL_BUDGET = {"max_budget": 1e-05, "budget_duration": "1d"}


def _proxy_server_attrs_for_master_key_auth():
    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.delete_cache = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    limiter = AsyncMock()
    limiter.is_end_user_within_model_budget = AsyncMock(return_value=None)

    return {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": limiter,
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }, limiter


def test_update_valid_token_applies_end_user_model_max_budget_from_params():
    valid_token = UserAPIKeyAuth(token="test-key")
    end_user_params = {
        "end_user_id": "customer-1",
        "end_user_model_max_budget": {MODEL: MODEL_BUDGET},
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    assert result.end_user_id == "customer-1"
    assert result.end_user_model_max_budget == end_user_params["end_user_model_max_budget"]


@pytest.mark.asyncio
async def test_enforce_end_user_model_max_budget_passes_when_within_budget():
    from litellm.proxy.auth.user_api_key_auth import _enforce_end_user_model_max_budget_checks

    valid_token = UserAPIKeyAuth(
        token="master-key",
        end_user_id="customer-1",
        end_user_model_max_budget={MODEL: MODEL_BUDGET},
    )
    request = MagicMock()
    request_data = {"model": MODEL}

    with patch(
        "litellm.proxy.auth.user_api_key_auth._get_model_from_request_context",
        return_value=MODEL,
    ):
        with patch(
            "litellm.proxy.proxy_server.model_max_budget_limiter.is_end_user_within_model_budget",
            new_callable=AsyncMock,
        ) as mock_check:
            await _enforce_end_user_model_max_budget_checks(
                valid_token=valid_token,
                request_data=request_data,
                route="/v1/chat/completions",
                request=request,
            )

            mock_check.assert_awaited_once_with(
                end_user_id="customer-1",
                end_user_model_max_budget=valid_token.end_user_model_max_budget,
                model=MODEL,
            )


@pytest.mark.asyncio
async def test_enforce_end_user_model_max_budget_raises_when_over_budget():
    from litellm.proxy.auth.user_api_key_auth import _enforce_end_user_model_max_budget_checks

    valid_token = UserAPIKeyAuth(
        token="master-key",
        end_user_id="customer-1",
        end_user_model_max_budget={MODEL: MODEL_BUDGET},
    )
    request = MagicMock()
    request_data = {"model": MODEL}

    with patch(
        "litellm.proxy.auth.user_api_key_auth._get_model_from_request_context",
        return_value=MODEL,
    ):
        with patch(
            "litellm.proxy.proxy_server.model_max_budget_limiter.is_end_user_within_model_budget",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.side_effect = litellm.BudgetExceededError(
                message="Exceeded budget", current_cost=0.0002, max_budget=1e-05
            )

            with pytest.raises(litellm.BudgetExceededError):
                await _enforce_end_user_model_max_budget_checks(
                    valid_token=valid_token,
                    request_data=request_data,
                    route="/v1/chat/completions",
                    request=request,
                )

            mock_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_master_key_auth_skips_end_user_model_budget_when_flag_disabled():
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    attrs, limiter = _proxy_server_attrs_for_master_key_auth()
    limiter.is_end_user_within_model_budget.side_effect = litellm.BudgetExceededError(
        message="Exceeded budget", current_cost=0.0002, max_budget=1e-05
    )
    end_user = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(model_max_budget={MODEL: MODEL_BUDGET}),
    )
    originals = {k: getattr(proxy_server, k, None) for k in attrs}
    flag_original = litellm.enforce_end_user_model_max_budget_on_master_key
    litellm.enforce_end_user_model_max_budget_on_master_key = False

    try:
        for k, v in attrs.items():
            setattr(proxy_server, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/v1/chat/completions")

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.resolve_and_validate_end_user_id",
                new_callable=AsyncMock,
                return_value="customer-1",
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=end_user,
            ),
        ):
            result = await _user_api_key_auth_builder(
                request=request,
                api_key=f"Bearer {attrs['master_key']}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={"user": "customer-1", "model": MODEL},
            )

        assert result.end_user_id == "customer-1"
        assert result.end_user_model_max_budget == {MODEL: MODEL_BUDGET}
        limiter.is_end_user_within_model_budget.assert_not_awaited()
    finally:
        litellm.enforce_end_user_model_max_budget_on_master_key = flag_original
        for k, v in originals.items():
            setattr(proxy_server, k, v)


@pytest.mark.asyncio
async def test_master_key_auth_enforces_end_user_model_budget_when_flag_enabled():
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    attrs, limiter = _proxy_server_attrs_for_master_key_auth()
    limiter.is_end_user_within_model_budget.side_effect = litellm.BudgetExceededError(
        message="Exceeded budget", current_cost=0.0002, max_budget=1e-05
    )
    end_user = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(model_max_budget={MODEL: MODEL_BUDGET}),
    )
    originals = {k: getattr(proxy_server, k, None) for k in attrs}
    flag_original = litellm.enforce_end_user_model_max_budget_on_master_key
    litellm.enforce_end_user_model_max_budget_on_master_key = True

    try:
        for k, v in attrs.items():
            setattr(proxy_server, k, v)

        request = Request(scope={"type": "http"})
        request._url = URL(url="/v1/chat/completions")

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.resolve_and_validate_end_user_id",
                new_callable=AsyncMock,
                return_value="customer-1",
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=end_user,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._get_model_from_request_context",
                return_value=MODEL,
            ),
        ):
            with pytest.raises(ProxyException) as exc_info:
                await _user_api_key_auth_builder(
                    request=request,
                    api_key=f"Bearer {attrs['master_key']}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={"user": "customer-1", "model": MODEL},
                )

            assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
            limiter.is_end_user_within_model_budget.assert_awaited()
    finally:
        litellm.enforce_end_user_model_max_budget_on_master_key = flag_original
        for k, v in originals.items():
            setattr(proxy_server, k, v)
