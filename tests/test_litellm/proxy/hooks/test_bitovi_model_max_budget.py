import pytest
from unittest.mock import AsyncMock, patch

from litellm.caching.caching import DualCache
from litellm.proxy._types import UpdateTeamRequest
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    _to_internal_model_max_budget,
    build_effective_model_max_budget_usage,
    resolve_effective_model_max_budget,
)
from litellm.proxy.management_endpoints.key_management_endpoints import validate_model_max_budget


@pytest.fixture
def budget_limiter():
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())


@pytest.mark.asyncio
async def test_build_effective_model_max_budget_usage_uses_team_member_spend_for_user(budget_limiter):
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }

    with patch.object(
        budget_limiter, "_get_team_member_model_spend_for_model", new_callable=AsyncMock, return_value=15.0
    ):
        usage = await build_effective_model_max_budget_usage(
            budget_limiter,
            api_key_hash="key-hash",
            team_id="team-1",
            user_id="user-1",
            key_model_max_budget={},
            team_model_max_budget=team_budget,
            team_member_model_max_budget=None,
        )

    assert usage["claude-sonnet-4-6"]["current_spend"] == 15.0
    assert usage["claude-sonnet-4-6"]["percent_used"] == 75.0
    assert usage["claude-sonnet-4-6"]["scope"] == "team"


async def test_build_effective_model_max_budget_usage_uses_key_spend_for_service_account_team_default(
    budget_limiter,
):
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }

    with patch.object(
        budget_limiter, "_get_virtual_key_spend_for_model", new_callable=AsyncMock, return_value=8.0
    ) as mock_key_spend:
        usage = await build_effective_model_max_budget_usage(
            budget_limiter,
            api_key_hash="sa-key-hash",
            team_id="team-1",
            user_id=None,
            key_model_max_budget={},
            team_model_max_budget=team_budget,
            team_member_model_max_budget=None,
        )

    mock_key_spend.assert_awaited_once()
    assert usage["claude-sonnet-4-6"]["current_spend"] == 8.0
    assert usage["claude-sonnet-4-6"]["scope"] == "key"


def test_match_budget_model_key_maps_bedrock_deployment_to_alias(budget_limiter) -> None:
    internal = _to_internal_model_max_budget(
        {"claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"}}
    )

    assert (
        budget_limiter._match_budget_model_key(
            model="bedrock/us.anthropic.claude-sonnet-4-6",
            internal_model_max_budget=internal,
        )
        == "claude-sonnet-4-6"
    )
    assert (
        budget_limiter._match_budget_model_key(
            model="us.anthropic.claude-sonnet-4-6",
            internal_model_max_budget=internal,
        )
        == "claude-sonnet-4-6"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_team_default_increments_bedrock_model_alias(budget_limiter):
    from unittest.mock import AsyncMock, patch

    virtual_key = "sa-key-hash"
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.04,
            "model": "bedrock/us.anthropic.claude-sonnet-4-6",
            "model_group": None,
            "metadata": {"user_api_key_hash": virtual_key},
            "end_user": "",
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {},
                "user_api_key_team_model_max_budget": team_budget,
                "user_api_key_team_id": "team-systems",
            },
        },
    }

    with patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_increment.assert_awaited_once()
    assert (
        mock_increment.call_args.kwargs["spend_key"]
        == "virtual_key_spend:sa-key-hash:claude-sonnet-4-6:1d"
    )


def test_update_team_request_accepts_model_max_budget() -> None:
    request = UpdateTeamRequest(
        team_id="team-1",
        model_max_budget={
            "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
        },
    )

    dumped = request.model_dump(exclude_unset=True)
    assert dumped["model_max_budget"]["claude-sonnet-4-6"]["max_budget"] == 20.0
    assert dumped["model_max_budget"]["claude-sonnet-4-6"]["budget_duration"] == "1d"


def test_resolve_effective_model_max_budget_prefers_key_overrides() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
        "claude-haiku-4-5": {"budget_limit": 30.0, "time_period": "30d"},
    }
    key_budget = {
        "claude-sonnet-4-6": {"budget_limit": 5.0, "time_period": "7d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=key_budget,
        team_model_max_budget=team_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 5.0
    assert merged["claude-haiku-4-5"]["budget_limit"] == 30.0


def test_resolve_effective_model_max_budget_member_overrides_team() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    member_budget = {
        "claude-sonnet-4-6": {"budget_limit": 50.0, "time_period": "7d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=None,
        team_model_max_budget=team_budget,
        team_member_model_max_budget=member_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 50.0
    assert merged["claude-sonnet-4-6"]["time_period"] == "7d"


def test_resolve_effective_model_max_budget_three_layer_merge() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
        "claude-haiku-4-5": {"budget_limit": 10.0, "time_period": "1d"},
    }
    member_budget = {
        "claude-sonnet-4-6": {"budget_limit": 40.0, "time_period": "7d"},
    }
    key_budget = {
        "claude-sonnet-4-6": {"budget_limit": 100.0, "time_period": "30d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=key_budget,
        team_model_max_budget=team_budget,
        team_member_model_max_budget=member_budget,
    )

    assert merged is not None
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 100.0
    assert merged["claude-haiku-4-5"]["budget_limit"] == 10.0


def test_resolve_effective_model_max_budget_uses_team_when_key_empty() -> None:
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
    }

    merged = resolve_effective_model_max_budget(
        key_model_max_budget=None,
        team_model_max_budget=team_budget,
    )

    assert merged == team_budget


def test_validate_model_max_budget_without_enterprise_license() -> None:
    validate_model_max_budget(
        {
            "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "30d"},
        }
    )
