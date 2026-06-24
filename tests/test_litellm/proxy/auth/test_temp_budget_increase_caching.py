"""
Tests that temp_budget_increase is applied on the cache-hit auth path, not
just the DB-fetch path. Also verifies the helper function does not mutate
its input — otherwise repeated cache hits would compound the increase on the
in-memory cached object, and Redis-backed replicas would diverge from the
replica that originally hit the DB.

Regression test for https://github.com/BerriAI/litellm/issues/25760
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import (
    _update_key_budget_with_temp_budget_increase,
)


def _make_token_with_temp_increase(
    *, base_max_budget: float, temp_increase: float, expired: bool = False
) -> UserAPIKeyAuth:
    expiry = datetime.now() + timedelta(days=-1 if expired else 1)
    return UserAPIKeyAuth(
        api_key="sk-test-temp-budget",
        token="hashed-test-temp-budget",
        user_id="user-test-temp-budget",
        max_budget=base_max_budget,
        spend=0.0,
        metadata={
            "temp_budget_increase": temp_increase,
            "temp_budget_expiry": expiry.isoformat(),
        },
    )


class TestUpdateKeyBudgetIsPure:
    """
    The helper used to mutate `valid_token.max_budget` in place. The cache
    stores object references, so mutation would compound across requests
    on the same replica. Now it must return a model_copy and leave the
    input untouched.
    """

    def test_does_not_mutate_input(self):
        token = _make_token_with_temp_increase(base_max_budget=2.0, temp_increase=100.0)
        original_id = id(token)

        result = _update_key_budget_with_temp_budget_increase(token)

        assert token.max_budget == 2.0, "input token must not be mutated"
        assert result.max_budget == 102.0
        assert id(result) != original_id, "must return a new instance"

    def test_repeated_calls_do_not_compound(self):
        token = _make_token_with_temp_increase(base_max_budget=2.0, temp_increase=100.0)

        first = _update_key_budget_with_temp_budget_increase(token)
        second = _update_key_budget_with_temp_budget_increase(token)
        third = _update_key_budget_with_temp_budget_increase(token)

        assert first.max_budget == 102.0
        assert second.max_budget == 102.0
        assert third.max_budget == 102.0
        assert token.max_budget == 2.0

    def test_returns_same_instance_when_no_temp_increase(self):
        token = UserAPIKeyAuth(
            api_key="sk-test",
            token="hashed-test",
            max_budget=2.0,
            spend=0.0,
            metadata={},
        )

        result = _update_key_budget_with_temp_budget_increase(token)

        assert result is token

    def test_returns_same_instance_when_expiry_passed(self):
        token = _make_token_with_temp_increase(
            base_max_budget=2.0, temp_increase=100.0, expired=True
        )

        result = _update_key_budget_with_temp_budget_increase(token)

        assert result is token
        assert result.max_budget == 2.0

    def test_no_op_when_max_budget_is_none(self):
        token = UserAPIKeyAuth(
            api_key="sk-test",
            token="hashed-test",
            max_budget=None,
            spend=0.0,
            metadata={
                "temp_budget_increase": 100.0,
                "temp_budget_expiry": (datetime.now() + timedelta(days=1)).isoformat(),
            },
        )

        result = _update_key_budget_with_temp_budget_increase(token)

        assert result is token
        assert result.max_budget is None


def _proxy_server_attrs_for_cache_hit(*, cached_token: UserAPIKeyAuth) -> dict:
    """
    Minimal proxy_server module attributes for exercising the cache-hit path
    inside _user_api_key_auth_builder. The cache returns `cached_token`,
    causing the early `get_key_object(check_cache_only=True)` to set
    `valid_token` and skip the DB-fetch block entirely.
    """
    mock_cache = AsyncMock()
    mock_cache.async_get_cache = AsyncMock(return_value=cached_token)
    mock_cache.delete_cache = MagicMock()

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache = MagicMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    mock_proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = (
        AsyncMock()
    )
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    return {
        "prisma_client": MagicMock(),
        "user_api_key_cache": mock_cache,
        "proxy_logging_obj": mock_proxy_logging_obj,
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }


@pytest.mark.asyncio
async def test_cache_hit_path_applies_temp_budget_increase():
    """
    When a virtual key is retrieved from cache (cache hit), the auth flow
    must apply temp_budget_increase to the in-flight token. Before this fix,
    the helper was only invoked inside the DB-fetch block, so cache hits
    enforced the stale base max_budget.
    """
    from fastapi import Request
    from starlette.datastructures import URL

    import litellm.proxy.proxy_server as _proxy_server_mod
    from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

    cached_token = _make_token_with_temp_increase(
        base_max_budget=2.0, temp_increase=100.0
    )
    cached_token.user_role = LitellmUserRoles.INTERNAL_USER

    attrs = _proxy_server_attrs_for_cache_hit(cached_token=cached_token)
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}

    try:
        for attr, val in attrs.items():
            setattr(_proxy_server_mod, attr, val)

        with (
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_key_object",
                new_callable=AsyncMock,
                return_value=cached_token,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.get_user_object",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.common_checks",
                new_callable=AsyncMock,
                return_value=cached_token,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._lookup_end_user_and_apply_budget",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._virtual_key_max_budget_check",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth._virtual_key_soft_budget_check",
                new_callable=AsyncMock,
            ),
            patch(
                "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
                return_value=True,
            ),
        ):
            request = Request(scope={"type": "http"})
            request._url = URL(url="/chat/completions")

            result = await _user_api_key_auth_builder(
                request=request,
                api_key="Bearer sk-test-temp-budget",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={},
            )

            assert result.max_budget == 102.0, (
                "cache-hit path must apply temp_budget_increase: "
                f"expected 102.0, got {result.max_budget}"
            )

        assert cached_token.max_budget == 2.0, (
            "the cached token reference must remain at base max_budget; "
            "in-place mutation would compound across cache hits"
        )
    finally:
        for attr, val in originals.items():
            setattr(_proxy_server_mod, attr, val)
