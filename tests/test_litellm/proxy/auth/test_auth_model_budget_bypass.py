"""
Tests for per-model budget enforcement on early-return auth paths.

The _user_api_key_auth_builder function has two early-return paths that
previously skipped per-model budget checks entirely:

1. Cached PROXY_ADMIN key path — returns valid_token before the main
   budget-check block.
2. Master key path — returns _user_api_key_obj before the main
   budget-check block.

These tests verify that _check_model_max_budget is called on both paths
and that BudgetExceededError propagates correctly.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
import litellm.proxy.proxy_server as _proxy_server_mod
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import (
    _check_model_max_budget,
    _user_api_key_auth_builder,
)
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.proxy.proxy_server import hash_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(route: str = "/chat/completions"):
    """Build a minimal mock Request with a URL."""
    from fastapi import Request
    from starlette.datastructures import URL

    request = Request(scope={"type": "http"})
    request._url = URL(url=route)
    return request


def _mock_proxy_logging_obj():
    """Build a mock proxy_logging_obj that won't blow up on fire-and-forget tasks."""
    obj = MagicMock()
    obj.internal_usage_cache = MagicMock()
    obj.internal_usage_cache.dual_cache = AsyncMock()
    obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()
    obj.post_call_failure_hook = AsyncMock(return_value=None)
    return obj


def _set_proxy_attrs(overrides: dict):
    """
    Set attributes on the proxy_server module (imported inside
    _user_api_key_auth_builder) and return originals for teardown.
    """
    defaults = {
        "prisma_client": MagicMock(),
        "user_api_key_cache": AsyncMock(spec=DualCache),
        "proxy_logging_obj": _mock_proxy_logging_obj(),
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(
            spec=_PROXY_VirtualKeyModelMaxBudgetLimiter
        ),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    defaults.update(overrides)
    originals = {k: getattr(_proxy_server_mod, k, None) for k in defaults}
    for k, v in defaults.items():
        setattr(_proxy_server_mod, k, v)
    return originals


def _restore_proxy_attrs(originals: dict):
    for k, v in originals.items():
        setattr(_proxy_server_mod, k, v)


# ---------------------------------------------------------------------------
# Unit tests for _check_model_max_budget helper
# ---------------------------------------------------------------------------


class TestCheckModelMaxBudget:
    """Direct unit tests for the _check_model_max_budget helper."""

    @pytest.mark.asyncio
    async def test_should_skip_non_llm_route(self):
        """Budget checks should be skipped for non-LLM routes like /key/info."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/key/info",
            model_max_budget_limiter=limiter,
        )
        limiter.is_key_within_model_budget.assert_not_called()
        limiter.is_end_user_within_model_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_skip_when_no_model_in_request(self):
        """Budget checks should be skipped when the request has no model."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
        )
        limiter.is_key_within_model_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_check_key_level_budget(self):
        """Should call is_key_within_model_budget when model_max_budget is set."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        limiter.is_key_within_model_budget = AsyncMock(return_value=True)
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
            prisma_client=MagicMock(),
        )
        limiter.is_key_within_model_budget.assert_awaited_once_with(
            user_api_key_dict=token,
            model="gpt-4",
        )

    @pytest.mark.asyncio
    async def test_should_check_end_user_budget(self):
        """Should call is_end_user_within_model_budget when end-user budget is set."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        limiter.is_end_user_within_model_budget = AsyncMock(return_value=True)
        eu_mmb = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}
        token = UserAPIKeyAuth(
            token="hashed-key",
            end_user_id="user-123",
            end_user_model_max_budget=eu_mmb,
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
        )
        limiter.is_end_user_within_model_budget.assert_awaited_once_with(
            end_user_id="user-123",
            end_user_model_max_budget=eu_mmb,
            model="gpt-4",
        )

    @pytest.mark.asyncio
    async def test_should_skip_key_budget_when_token_is_none(self):
        """Key-level check requires valid_token.token to be set."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        token = UserAPIKeyAuth(
            token=None,
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
        )
        limiter.is_key_within_model_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_skip_end_user_budget_when_no_end_user_id(self):
        """End-user check requires end_user_id to be set."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        eu_mmb = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}
        token = UserAPIKeyAuth(
            token="hashed-key",
            end_user_id=None,
            end_user_model_max_budget=eu_mmb,
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
        )
        limiter.is_end_user_within_model_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_propagate_budget_exceeded_error(self):
        """BudgetExceededError from the limiter must propagate."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        limiter.is_key_within_model_budget = AsyncMock(
            side_effect=litellm.BudgetExceededError(
                current_cost=15.0, max_budget=10.0
            )
        )
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        with pytest.raises(litellm.BudgetExceededError):
            await _check_model_max_budget(
                valid_token=token,
                request_data={"model": "gpt-4"},
                route="/chat/completions",
                model_max_budget_limiter=limiter,
                prisma_client=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_should_check_both_key_and_end_user_budgets(self):
        """When both budgets are configured, both should be checked."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        limiter.is_key_within_model_budget = AsyncMock(return_value=True)
        limiter.is_end_user_within_model_budget = AsyncMock(return_value=True)
        eu_mmb = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
            end_user_id="user-123",
            end_user_model_max_budget=eu_mmb,
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
            prisma_client=MagicMock(),
        )
        limiter.is_key_within_model_budget.assert_awaited_once()
        limiter.is_end_user_within_model_budget.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_skip_key_budget_when_prisma_client_is_none(self):
        """Key-level check requires prisma_client to be non-None."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={"gpt-4": {"budget_limit": 10.0, "time_period": "1d"}},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
            prisma_client=None,
        )
        limiter.is_key_within_model_budget.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_skip_when_model_max_budget_empty(self):
        """Empty model_max_budget dict should be treated as no budget."""
        limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        token = UserAPIKeyAuth(
            token="hashed-key",
            model_max_budget={},
        )
        await _check_model_max_budget(
            valid_token=token,
            request_data={"model": "gpt-4"},
            route="/chat/completions",
            model_max_budget_limiter=limiter,
        )
        limiter.is_key_within_model_budget.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests: cached PROXY_ADMIN path
# ---------------------------------------------------------------------------


class TestCachedProxyAdminModelBudget:
    """
    Tests that the cached PROXY_ADMIN early-return path enforces
    per-model budgets via _check_model_max_budget.
    """

    @pytest.mark.asyncio
    async def test_should_enforce_end_user_model_budget_on_cached_admin(self):
        """
        When a cached PROXY_ADMIN token has end_user_model_max_budget set
        (via the end user's budget table), the budget check must run and
        reject the request if the budget is exceeded.
        """
        api_key = "sk-test-admin-key"
        hashed_key = hash_token(api_key)

        eu_mmb = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}

        cached_token = UserAPIKeyAuth(
            api_key=api_key,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            token=hashed_key,
            end_user_model_max_budget=eu_mmb,
        )

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        mock_limiter.is_end_user_within_model_budget = AsyncMock(
            side_effect=litellm.BudgetExceededError(
                current_cost=10.0, max_budget=5.0
            )
        )
        mock_limiter.is_key_within_model_budget = AsyncMock(return_value=True)

        originals = _set_proxy_attrs(
            {"model_max_budget_limiter": mock_limiter}
        )
        try:
            # get_key_object returns the cached admin token
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=cached_token,
            ), patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(ProxyException) as exc_info:
                    await _user_api_key_auth_builder(
                        request=_make_request("/chat/completions"),
                        api_key=f"Bearer {api_key}",
                        azure_api_key_header="",
                        anthropic_api_key_header=None,
                        google_ai_studio_api_key_header=None,
                        azure_apim_header=None,
                        request_data={"model": "gpt-4", "user": "end-user-1"},
                    )
                assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
        finally:
            _restore_proxy_attrs(originals)

    @pytest.mark.asyncio
    async def test_should_enforce_key_model_budget_on_cached_admin(self):
        """
        When a cached PROXY_ADMIN token has model_max_budget set,
        the key-level budget check must run and reject the request
        if the budget is exceeded.
        """
        api_key = "sk-test-admin-key-2"
        hashed_key = hash_token(api_key)

        cached_token = UserAPIKeyAuth(
            api_key=api_key,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            token=hashed_key,
            model_max_budget={
                "gpt-4": {"budget_limit": 10.0, "time_period": "1d"}
            },
        )

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        mock_limiter.is_key_within_model_budget = AsyncMock(
            side_effect=litellm.BudgetExceededError(
                current_cost=15.0, max_budget=10.0
            )
        )

        originals = _set_proxy_attrs(
            {"model_max_budget_limiter": mock_limiter}
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=cached_token,
            ), patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ):
                with pytest.raises(ProxyException) as exc_info:
                    await _user_api_key_auth_builder(
                        request=_make_request("/chat/completions"),
                        api_key=f"Bearer {api_key}",
                        azure_api_key_header="",
                        anthropic_api_key_header=None,
                        google_ai_studio_api_key_header=None,
                        azure_apim_header=None,
                        request_data={"model": "gpt-4"},
                    )
                assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
        finally:
            _restore_proxy_attrs(originals)

    @pytest.mark.asyncio
    async def test_should_pass_cached_admin_when_within_budget(self):
        """
        When a cached PROXY_ADMIN token is within budget, auth should
        succeed and return the token normally.
        """
        api_key = "sk-test-admin-key-3"
        hashed_key = hash_token(api_key)

        cached_token = UserAPIKeyAuth(
            api_key=api_key,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            token=hashed_key,
            model_max_budget={
                "gpt-4": {"budget_limit": 100.0, "time_period": "1d"}
            },
        )

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        mock_limiter.is_key_within_model_budget = AsyncMock(return_value=True)

        originals = _set_proxy_attrs(
            {"model_max_budget_limiter": mock_limiter}
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=cached_token,
            ), patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await _user_api_key_auth_builder(
                    request=_make_request("/chat/completions"),
                    api_key=f"Bearer {api_key}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={"model": "gpt-4"},
                )
                assert result is not None
                assert result.user_role == LitellmUserRoles.PROXY_ADMIN
                mock_limiter.is_key_within_model_budget.assert_awaited_once()
        finally:
            _restore_proxy_attrs(originals)


# ---------------------------------------------------------------------------
# Integration tests: master key path
# ---------------------------------------------------------------------------


class TestMasterKeyModelBudget:
    """
    Tests that the master key early-return path enforces per-model
    budgets via _check_model_max_budget.
    """

    @pytest.mark.asyncio
    async def test_should_enforce_end_user_model_budget_on_master_key(self):
        """
        When a request uses the master key and the end user has a
        model_max_budget configured (via budget table), the budget
        check must run and reject the request if exceeded.
        """
        from litellm.proxy._types import (
            LiteLLM_BudgetTable,
            LiteLLM_EndUserTable,
        )

        master_key = "sk-master-key"
        eu_mmb = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}

        end_user_obj = MagicMock(spec=LiteLLM_EndUserTable)
        end_user_obj.allowed_model_region = None
        end_user_obj.object_permission = None
        budget_table = MagicMock(spec=LiteLLM_BudgetTable)
        budget_table.tpm_limit = None
        budget_table.rpm_limit = None
        budget_table.max_budget = None
        budget_table.model_max_budget = eu_mmb
        end_user_obj.litellm_budget_table = budget_table

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        mock_limiter.is_end_user_within_model_budget = AsyncMock(
            side_effect=litellm.BudgetExceededError(
                current_cost=10.0, max_budget=5.0
            )
        )
        mock_limiter.is_key_within_model_budget = AsyncMock(return_value=True)

        originals = _set_proxy_attrs(
            {
                "master_key": master_key,
                "model_max_budget_limiter": mock_limiter,
            }
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=end_user_obj,
            ):
                with pytest.raises(ProxyException) as exc_info:
                    await _user_api_key_auth_builder(
                        request=_make_request("/chat/completions"),
                        api_key=f"Bearer {master_key}",
                        azure_api_key_header="",
                        anthropic_api_key_header=None,
                        google_ai_studio_api_key_header=None,
                        azure_apim_header=None,
                        request_data={
                            "model": "gpt-4",
                            "user": "end-user-1",
                        },
                    )
                assert exc_info.value.type == ProxyErrorTypes.budget_exceeded
        finally:
            _restore_proxy_attrs(originals)

    @pytest.mark.asyncio
    async def test_should_pass_master_key_when_no_end_user_budget(self):
        """
        Master key requests with no end-user model budget should
        succeed without budget checks being triggered.
        """
        master_key = "sk-master-key"

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)

        originals = _set_proxy_attrs(
            {
                "master_key": master_key,
                "model_max_budget_limiter": mock_limiter,
            }
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await _user_api_key_auth_builder(
                    request=_make_request("/chat/completions"),
                    api_key=f"Bearer {master_key}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={"model": "gpt-4"},
                )
                assert result is not None
                assert result.user_role == LitellmUserRoles.PROXY_ADMIN
                mock_limiter.is_key_within_model_budget.assert_not_called()
                mock_limiter.is_end_user_within_model_budget.assert_not_called()
        finally:
            _restore_proxy_attrs(originals)

    @pytest.mark.asyncio
    async def test_should_pass_master_key_on_non_llm_route(self):
        """
        Master key requests on non-LLM routes (e.g. /key/generate)
        should skip budget checks entirely.
        """
        master_key = "sk-master-key"

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)

        originals = _set_proxy_attrs(
            {
                "master_key": master_key,
                "model_max_budget_limiter": mock_limiter,
            }
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await _user_api_key_auth_builder(
                    request=_make_request("/key/generate"),
                    api_key=f"Bearer {master_key}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={},
                )
                assert result is not None
                mock_limiter.is_key_within_model_budget.assert_not_called()
                mock_limiter.is_end_user_within_model_budget.assert_not_called()
        finally:
            _restore_proxy_attrs(originals)

    @pytest.mark.asyncio
    async def test_should_pass_master_key_when_end_user_within_budget(self):
        """
        Master key request with an end user within budget should succeed.
        """
        from litellm.proxy._types import (
            LiteLLM_BudgetTable,
            LiteLLM_EndUserTable,
        )

        master_key = "sk-master-key"
        eu_mmb = {"gpt-4": {"budget_limit": 100.0, "time_period": "1d"}}

        end_user_obj = MagicMock(spec=LiteLLM_EndUserTable)
        end_user_obj.allowed_model_region = None
        end_user_obj.object_permission = None
        budget_table = MagicMock(spec=LiteLLM_BudgetTable)
        budget_table.tpm_limit = None
        budget_table.rpm_limit = None
        budget_table.max_budget = None
        budget_table.model_max_budget = eu_mmb
        end_user_obj.litellm_budget_table = budget_table

        mock_limiter = MagicMock(spec=_PROXY_VirtualKeyModelMaxBudgetLimiter)
        mock_limiter.is_end_user_within_model_budget = AsyncMock(
            return_value=True
        )
        mock_limiter.is_key_within_model_budget = AsyncMock(return_value=True)

        originals = _set_proxy_attrs(
            {
                "master_key": master_key,
                "model_max_budget_limiter": mock_limiter,
            }
        )
        try:
            with patch(
                "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
                new_callable=AsyncMock,
                return_value=end_user_obj,
            ):
                result = await _user_api_key_auth_builder(
                    request=_make_request("/chat/completions"),
                    api_key=f"Bearer {master_key}",
                    azure_api_key_header="",
                    anthropic_api_key_header=None,
                    google_ai_studio_api_key_header=None,
                    azure_apim_header=None,
                    request_data={
                        "model": "gpt-4",
                        "user": "end-user-1",
                    },
                )
                assert result is not None
                assert result.user_role == LitellmUserRoles.PROXY_ADMIN
                mock_limiter.is_end_user_within_model_budget.assert_awaited_once()
        finally:
            _restore_proxy_attrs(originals)
