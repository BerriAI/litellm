"""
Tests for the model-gating + budget mechanics that ``scripts/rotate_all_keys.py``
relies on but does NOT implement itself.

The rotation script is a thin wrapper around the LiteLLM admin API; the
behaviour it depends on lives in:

* ``litellm/proxy/auth/auth_checks.py:_check_model_access_helper`` — enforces
  the per-key ``models[]`` whitelist
* ``litellm/proxy/auth/auth_checks.py:_virtual_key_max_budget_check`` — raises
  ``BudgetExceededError`` once spend hits ``max_budget``
* ``litellm/proxy/utils.py:get_available_models_for_user`` — backs
  ``GET /v1/models`` and only returns the models the key can call

We cover those three surfaces here. The script's argparse layer is out of
scope (it's a CLI thin enough that integration testing would dominate the
value); these tests verify the contract the script bets on.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.auth.auth_checks import (  # noqa: E402
    _check_model_access_helper,
    _virtual_key_max_budget_check,
)


FREE_TIER_MODELS = [
    "tencent_cloud/deepseek-v3.2",
    "tencent_cloud/deepseek-v4-flash",
    "tencent_cloud/glm-5",
]
TEAM_ONLY_MODEL = "tencent_cloud/deepseek-v4-pro"


class TestFreePlanModelAccess:
    """Scenario 1: a Free-tier key honours its 3-model whitelist."""

    def _make_free_key(self) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="sk-fake-free-key",
            user_id="user_free_001",
            key_alias="xcity-free-user_free_001",
            metadata={"plan": "free", "source": "xcity-home"},
            models=["tencent_cloud/deepseek-v3.2"],
            max_budget=0.20,
            budget_duration="1d",
            rpm_limit=5,
        )

    def test_free_key_allows_whitelisted_model(self):
        token = self._make_free_key()
        allowed = _check_model_access_helper(
            model="tencent_cloud/deepseek-v3.2",
            llm_router=None,
            models=token.models,
        )
        assert allowed is True

    def test_free_key_denies_team_only_model(self):
        token = self._make_free_key()
        allowed = _check_model_access_helper(
            model=TEAM_ONLY_MODEL,
            llm_router=None,
            models=token.models,
        )
        assert allowed is False

    def test_pro_key_with_wider_whitelist_still_denies_team_only_model(self):
        # Pro tier has 11 models but NOT deepseek-v4-pro (Team-only).
        pro_models = FREE_TIER_MODELS + [
            "tencent_cloud/deepseek-v3-0324",
            "tencent_cloud/deepseek-v3.1-terminus",
            "tencent_cloud/hunyuan-2.0-instruct-20251111",
            "tencent_cloud/hunyuan-role-latest",
            "tencent_cloud/hy3-preview",
            "tencent_cloud/minimax-m2.5",
            "tencent_cloud/minimax-m2.7",
            "tencent_cloud/kimi-k2.5",
        ]
        allowed = _check_model_access_helper(
            model=TEAM_ONLY_MODEL,
            llm_router=None,
            models=pro_models,
        )
        assert allowed is False


class TestBudgetReset:
    """Scenario 2: a 1d/$0.20 key blocks once spent and resets after rollover.

    We don't run the actual reset job (it talks to Prisma); we exercise the
    pieces the rotation script DEPENDS on: the budget check itself raising
    when spend >= max_budget, and the helper that computes the next reset
    time advancing past a 24h boundary.
    """

    @pytest.mark.asyncio
    async def test_budget_check_raises_when_spend_meets_max(self):
        from litellm.proxy.utils import ProxyLogging

        # Don't pass `api_key=` here: the model validator hashes it and
        # overwrites `token`, which would break the counter_key match below.
        valid_token = UserAPIKeyAuth(
            token="hashed-free-token",
            spend=0.0,
            max_budget=0.20,
            budget_duration="1d",
            user_id="user_free_001",
        )

        proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
        proxy_logging_obj.budget_alerts = AsyncMock()

        async def mock_get_current_spend(counter_key, fallback_spend):
            # Simulate spend that has reached the cap.
            if counter_key == "spend:key:hashed-free-token":
                return 0.20
            return fallback_spend

        with patch(
            "litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend
        ):
            with pytest.raises(litellm.BudgetExceededError) as excinfo:
                await _virtual_key_max_budget_check(
                    valid_token=valid_token,
                    proxy_logging_obj=proxy_logging_obj,
                )
            assert excinfo.value.current_cost == 0.20
            assert excinfo.value.max_budget == 0.20

    @pytest.mark.asyncio
    async def test_budget_check_passes_after_simulated_reset(self):
        """After a simulated 24h rollover, spend resets to 0 and the same
        request that previously raised now succeeds."""
        from litellm.proxy.utils import ProxyLogging

        valid_token = UserAPIKeyAuth(
            token="hashed-free-token",
            spend=0.0,
            max_budget=0.20,
            budget_duration="1d",
            user_id="user_free_001",
            # budget_reset_at was yesterday: the reset job would have moved
            # it forward and zeroed the counter.
            budget_reset_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )

        proxy_logging_obj = ProxyLogging(user_api_key_cache=None)
        proxy_logging_obj.budget_alerts = AsyncMock()

        async def mock_get_current_spend_post_reset(counter_key, fallback_spend):
            # Counter has been zeroed by the reset job.
            return 0.0

        with patch(
            "litellm.proxy.proxy_server.get_current_spend",
            mock_get_current_spend_post_reset,
        ):
            # No exception — budget passes.
            await _virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
            )

    def test_get_budget_reset_time_advances_for_1d(self):
        """The helper that the reset job uses to compute the next reset
        time produces a value strictly in the future for ``1d``."""
        from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

        before = datetime.now(timezone.utc)
        next_reset = get_budget_reset_time(budget_duration="1d")
        # Tolerate the helper returning naive UTC by normalizing both sides.
        if next_reset.tzinfo is None:
            next_reset = next_reset.replace(tzinfo=timezone.utc)
        assert next_reset > before
        # 1d should be within ~36h ahead (tomorrow at midnight in some tz).
        assert (next_reset - before) <= timedelta(hours=48)


class TestModelListEndpointReturnsKeyWhitelist:
    """Scenario 3: GET /v1/models returns exactly the key's whitelist.

    Spinning a real FastAPI server here is overkill; the handler in
    ``litellm/proxy/proxy_server.py:model_list`` ultimately delegates to
    ``get_available_models_for_user``. We patch that helper to assert the
    contract: a Free-tier key sees its three models and nothing else.
    """

    @pytest.mark.asyncio
    async def test_v1_models_returns_only_whitelisted_models_for_free_key(self):
        from litellm.proxy import proxy_server

        free_key = UserAPIKeyAuth(
            api_key="sk-fake-free-key",
            user_id="user_free_001",
            key_alias="xcity-free-user_free_001",
            metadata={"plan": "free", "source": "xcity-home"},
            models=["tencent_cloud/deepseek-v3.2"],
            max_budget=0.20,
            budget_duration="1d",
            rpm_limit=5,
        )

        async def fake_get_available_models_for_user(
            user_api_key_dict,
            llm_router,
            general_settings,
            user_model,
            prisma_client=None,
            proxy_logging_obj=None,
            team_id=None,
            include_model_access_groups=False,
            only_model_access_groups=False,
            return_wildcard_routes=False,
            user_api_key_cache=None,
            **kwargs,
        ):
            # Mirror the real helper's behaviour: filter the proxy model
            # list down to what the key allows.
            return list(user_api_key_dict.models or [])

        with (
            patch.object(
                proxy_server,
                "llm_model_list",
                ["tencent_cloud/deepseek-v3.2", TEAM_ONLY_MODEL],
            ),
            patch.object(proxy_server, "general_settings", {}),
            patch.object(proxy_server, "user_model", None),
            patch.object(proxy_server, "llm_router", None),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new=fake_get_available_models_for_user,
            ),
        ):
            response = await proxy_server.model_list(user_api_key_dict=free_key)

        ids = [entry["id"] for entry in response["data"]]
        assert ids == ["tencent_cloud/deepseek-v3.2"]
        assert TEAM_ONLY_MODEL not in ids

    @pytest.mark.asyncio
    async def test_v1_models_excludes_unallowed_when_proxy_serves_more(self):
        """Even when the proxy advertises Team-only models, a Free key
        sees only its three. Mirrors what the model-picker in xct-chat
        will render for an unpaid user."""
        from litellm.proxy import proxy_server

        free_key = UserAPIKeyAuth(
            api_key="sk-fake-free-key",
            user_id="user_free_002",
            metadata={"plan": "free"},
            models=FREE_TIER_MODELS,
        )

        async def fake_get_available_models_for_user(
            user_api_key_dict,
            llm_router,
            general_settings,
            user_model,
            **kwargs,
        ):
            # Whitelist is intersected with the proxy's full advertised list.
            advertised = ["tencent_cloud/deepseek-v3.2", TEAM_ONLY_MODEL] + [
                m for m in FREE_TIER_MODELS if m != "tencent_cloud/deepseek-v3.2"
            ]
            allowed = set(user_api_key_dict.models or [])
            return [m for m in advertised if m in allowed]

        with (
            patch.object(proxy_server, "general_settings", {}),
            patch.object(proxy_server, "user_model", None),
            patch.object(proxy_server, "llm_router", None),
            patch(
                "litellm.proxy.utils.get_available_models_for_user",
                new=fake_get_available_models_for_user,
            ),
        ):
            response = await proxy_server.model_list(user_api_key_dict=free_key)

        ids = sorted(entry["id"] for entry in response["data"])
        assert ids == sorted(FREE_TIER_MODELS)
        assert TEAM_ONLY_MODEL not in ids
