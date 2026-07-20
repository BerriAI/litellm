import pytest
from unittest.mock import AsyncMock, patch

from litellm.caching.caching import DualCache
from litellm.proxy._types import UpdateTeamRequest
from litellm.proxy.hooks.model_max_budget_limiter import (
    ResolvedModelBudgetMaps,
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
        "_increment_spend_in_current_window",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_increment.assert_awaited_once()
    assert mock_increment.call_args.kwargs["spend_key"].startswith(
        "virtual_key_spend:sa-key-hash:claude-sonnet-4-6:1d:w"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_reads_team_budget_from_litellm_metadata(budget_limiter):
    from unittest.mock import AsyncMock, patch

    virtual_key = "sa-key-hash"
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.0201767,
            "model": "bedrock/us.anthropic.claude-sonnet-4-6",
            "model_group": None,
            "metadata": {"user_api_key_hash": virtual_key},
            "end_user": "",
        },
        "litellm_params": {
            "litellm_metadata": {
                "user_api_key_model_max_budget": {},
                "user_api_key_team_model_max_budget": team_budget,
                "user_api_key_team_id": "team-systems",
            },
        },
    }

    with patch.object(
        budget_limiter,
        "_increment_spend_in_current_window",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_increment.assert_awaited_once()
    assert mock_increment.call_args.kwargs["spend_key"].startswith(
        "virtual_key_spend:sa-key-hash:claude-sonnet-4-6:1d:w"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_skips_when_budget_metadata_missing_from_both_bags(budget_limiter):
    from unittest.mock import AsyncMock, patch

    kwargs = {
        "standard_logging_object": {
            "response_cost": 0.02,
            "model": "bedrock/us.anthropic.claude-sonnet-4-6",
            "metadata": {"user_api_key_hash": "sa-key-hash"},
        },
        "litellm_params": {
            "litellm_metadata": {},
            "metadata": {},
        },
    }

    with patch.object(
        budget_limiter,
        "_increment_spend_in_current_window",
        new_callable=AsyncMock,
    ) as mock_increment, patch(
        "litellm.proxy.hooks.model_max_budget_limiter.resolve_budget_maps_for_increment",
        new_callable=AsyncMock,
        return_value=ResolvedModelBudgetMaps(None, None, None, "empty"),
    ):
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_increment.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_log_success_event_falls_back_to_user_api_key_auth_budgets(budget_limiter):
    from unittest.mock import AsyncMock, patch

    from litellm.proxy._types import UserAPIKeyAuth

    virtual_key = "auth-key-hash"
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    auth = UserAPIKeyAuth(
        token=virtual_key,
        auth_team_model_max_budget=team_budget,
    )
    kwargs = {
        "litellm_call_id": "call-auth-fallback",
        "standard_logging_object": {
            "response_cost": 0.03,
            "model": "bedrock/us.anthropic.claude-sonnet-4-6",
            "model_group": None,
            "metadata": {"user_api_key_hash": virtual_key},
            "end_user": "",
        },
        "litellm_params": {
            "litellm_metadata": {
                "user_api_key_auth": auth,
            },
        },
    }

    with patch(
        "litellm.proxy.hooks.model_max_budget_limiter.resolve_budget_maps_for_increment",
        new_callable=AsyncMock,
        return_value=ResolvedModelBudgetMaps(None, team_budget, None, "user_api_key_auth"),
    ), patch.object(
        budget_limiter,
        "_increment_spend_in_current_window",
        new_callable=AsyncMock,
    ) as mock_increment:
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    mock_increment.assert_awaited_once()
    assert mock_increment.call_args.kwargs["spend_key"].startswith(
        "virtual_key_spend:auth-key-hash:claude-sonnet-4-6:1d:w"
    )


@pytest.mark.asyncio
async def test_async_log_success_event_emits_callback_skipped_log_when_no_budget_maps(
    budget_limiter, caplog
):
    import logging
    from unittest.mock import AsyncMock, patch

    caplog.set_level(logging.INFO)
    kwargs = {
        "litellm_call_id": "call-skip-log",
        "standard_logging_object": {
            "response_cost": 0.02,
            "model": "claude-sonnet-4-6",
            "metadata": {"user_api_key_hash": "hash12345678"},
        },
        "litellm_params": {"metadata": {}},
    }

    with patch(
        "litellm.proxy.hooks.model_max_budget_limiter.resolve_budget_maps_for_increment",
        new_callable=AsyncMock,
        return_value=ResolvedModelBudgetMaps(None, None, None, "empty"),
    ), patch.object(
        budget_limiter,
        "_increment_spend_for_key",
        new_callable=AsyncMock,
    ):
        await budget_limiter.async_log_success_event(
            kwargs, response_obj=None, start_time=None, end_time=None
        )

    assert any(
        "model_max_budget_spend_trace event=callback_skipped" in record.message
        and "skip_reason=no_budget_maps" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_async_log_success_event_emits_increment_done_log(budget_limiter, caplog):
    import logging

    caplog.set_level(logging.INFO)
    virtual_key = "done-key-hash"
    team_budget = {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }
    kwargs = {
        "litellm_call_id": "call-increment-done",
        "standard_logging_object": {
            "response_cost": 0.04,
            "model": "claude-sonnet-4-6",
            "metadata": {"user_api_key_hash": virtual_key},
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_team_model_max_budget": team_budget,
            },
        },
    }

    await budget_limiter.async_log_success_event(
        kwargs, response_obj=None, start_time=None, end_time=None
    )

    assert any(
        "model_max_budget_spend_trace event=increment_done" in record.message
        for record in caplog.records
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


def _team_model_budget() -> dict:
    return {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }


def _budget_config_for_model(model: str = "claude-sonnet-4-6"):
    return _to_internal_model_max_budget(_team_model_budget())[model]


@pytest.mark.asyncio
async def test_human_user_single_key_team_budget_tracks_team_member_spend(budget_limiter):
    team_budget = _team_model_budget()
    budget_config = _budget_config_for_model()

    await budget_limiter._increment_model_budget_spend(
        response_cost=5.0,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="key-a",
        team_id="team-1",
        user_id="user-1",
        end_user_id=None,
        litellm_call_id=None,
    )

    usage = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="key-a",
        team_id="team-1",
        user_id="user-1",
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )

    assert usage["claude-sonnet-4-6"]["current_spend"] == 5.0
    assert usage["claude-sonnet-4-6"]["scope"] == "team"

    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key = UserAPIKeyAuth(
        token="key-a",
        team_id="team-1",
        user_id="user-1",
        model_max_budget={},
    )
    assert (
        await budget_limiter.is_key_within_model_budget(
            user_api_key,
            "claude-sonnet-4-6",
            team_model_max_budget=team_budget,
        )
        is True
    )


@pytest.mark.asyncio
async def test_human_user_two_keys_share_team_member_spend_pool(budget_limiter):
    import litellm

    team_budget = _team_model_budget()
    budget_config = _budget_config_for_model()
    key_a_spend = 7.0
    key_b_spend = 8.0
    key_a_final_increment = 6.0

    await budget_limiter._increment_model_budget_spend(
        response_cost=key_a_spend,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="key-a",
        team_id="team-1",
        user_id="user-1",
        end_user_id=None,
        litellm_call_id=None,
    )

    usage_on_key_a = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="key-a",
        team_id="team-1",
        user_id="user-1",
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    usage_on_key_b = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="key-b",
        team_id="team-1",
        user_id="user-1",
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    assert usage_on_key_a["claude-sonnet-4-6"]["current_spend"] == key_a_spend
    assert usage_on_key_b["claude-sonnet-4-6"]["current_spend"] == key_a_spend
    assert usage_on_key_a["claude-sonnet-4-6"]["scope"] == "team"
    assert usage_on_key_b["claude-sonnet-4-6"]["scope"] == "team"

    await budget_limiter._increment_model_budget_spend(
        response_cost=key_b_spend,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="key-b",
        team_id="team-1",
        user_id="user-1",
        end_user_id=None,
        litellm_call_id=None,
    )

    shared_spend_after_both = key_a_spend + key_b_spend
    usage_on_key_a_after_b = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="key-a",
        team_id="team-1",
        user_id="user-1",
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    assert usage_on_key_a_after_b["claude-sonnet-4-6"]["current_spend"] == shared_spend_after_both

    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_a = UserAPIKeyAuth(
        token="key-a",
        team_id="team-1",
        user_id="user-1",
        model_max_budget={},
    )
    assert (
        await budget_limiter.is_key_within_model_budget(
            user_api_key_a,
            "claude-sonnet-4-6",
            team_model_max_budget=team_budget,
        )
        is True
    )

    await budget_limiter._increment_model_budget_spend(
        response_cost=key_a_final_increment,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="key-a",
        team_id="team-1",
        user_id="user-1",
        end_user_id=None,
        litellm_call_id=None,
    )

    user_api_key_b = UserAPIKeyAuth(
        token="key-b",
        team_id="team-1",
        user_id="user-1",
        model_max_budget={},
    )
    with pytest.raises(litellm.BudgetExceededError):
        await budget_limiter.is_key_within_model_budget(
            user_api_key_b,
            "claude-sonnet-4-6",
            team_model_max_budget=team_budget,
        )


@pytest.mark.asyncio
async def test_service_account_two_keys_have_independent_spend_pools(budget_limiter):
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.model_max_budget_limiter import (
        TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX,
        VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
        _windowed_cache_key,
        current_model_budget_window,
    )

    team_budget = _team_model_budget()
    budget_config = _budget_config_for_model()
    sa_key_a_spend = 12.0
    sa_key_b_spend = 5.0

    await budget_limiter._increment_model_budget_spend(
        response_cost=sa_key_a_spend,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="sa-key-a",
        team_id="team-1",
        user_id=None,
        end_user_id=None,
        litellm_call_id=None,
    )

    usage_on_key_a = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="sa-key-a",
        team_id="team-1",
        user_id=None,
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    usage_on_key_b = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="sa-key-b",
        team_id="team-1",
        user_id=None,
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    assert usage_on_key_a["claude-sonnet-4-6"]["current_spend"] == sa_key_a_spend
    assert usage_on_key_b["claude-sonnet-4-6"]["current_spend"] == 0.0
    assert usage_on_key_a["claude-sonnet-4-6"]["scope"] == "key"
    assert usage_on_key_b["claude-sonnet-4-6"]["scope"] == "key"

    sa_key_b = UserAPIKeyAuth(
        token="sa-key-b",
        team_id="team-1",
        user_id=None,
        model_max_budget={},
    )
    assert (
        await budget_limiter.is_key_within_model_budget(
            sa_key_b,
            "claude-sonnet-4-6",
            team_model_max_budget=team_budget,
        )
        is True
    )

    await budget_limiter._increment_model_budget_spend(
        response_cost=sa_key_b_spend,
        model="claude-sonnet-4-6",
        budget_config=budget_config,
        scope="team",
        virtual_key="sa-key-b",
        team_id="team-1",
        user_id=None,
        end_user_id=None,
        litellm_call_id=None,
    )

    usage_on_key_a_after_b = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="sa-key-a",
        team_id="team-1",
        user_id=None,
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    usage_on_key_b_after_b = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="sa-key-b",
        team_id="team-1",
        user_id=None,
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )
    assert usage_on_key_a_after_b["claude-sonnet-4-6"]["current_spend"] == sa_key_a_spend
    assert usage_on_key_b_after_b["claude-sonnet-4-6"]["current_spend"] == sa_key_b_spend

    window_epoch = current_model_budget_window("1d").epoch
    team_member_key = _windowed_cache_key(
        f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:team-1:user-1:claude-sonnet-4-6:1d", window_epoch
    )
    sa_key_a_spend_key = _windowed_cache_key(
        f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:sa-key-a:claude-sonnet-4-6:1d", window_epoch
    )
    sa_key_b_spend_key = _windowed_cache_key(
        f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:sa-key-b:claude-sonnet-4-6:1d", window_epoch
    )
    assert await budget_limiter.dual_cache.async_get_cache(key=team_member_key) is None
    assert await budget_limiter.dual_cache.async_get_cache(key=sa_key_a_spend_key) == sa_key_a_spend
    assert await budget_limiter.dual_cache.async_get_cache(key=sa_key_b_spend_key) == sa_key_b_spend


def test_current_model_budget_window_is_calendar_aligned_utc() -> None:
    from datetime import datetime, timedelta, timezone

    from litellm.proxy.hooks.model_max_budget_limiter import current_model_budget_window

    window = current_model_budget_window("1d")

    assert window.reset_at.tzinfo is not None
    assert window.reset_at.utcoffset() == timedelta(0)
    assert (window.reset_at.hour, window.reset_at.minute, window.reset_at.second) == (0, 0, 0)
    assert window.reset_at - window.window_start == timedelta(days=1)
    assert window.epoch == int(window.window_start.timestamp())

    now = datetime.now(timezone.utc)
    assert window.window_start <= now < window.reset_at


@pytest.mark.asyncio
async def test_blocked_user_unblocks_when_calendar_window_rolls(budget_limiter):
    from datetime import datetime, timedelta, timezone

    import litellm
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks import model_max_budget_limiter as mod

    team_budget = _team_model_budget()
    budget_config = _budget_config_for_model()

    day1_start = datetime(2026, 7, 7, tzinfo=timezone.utc)
    day1 = mod.ModelBudgetWindow(
        window_start=day1_start,
        reset_at=day1_start + timedelta(days=1),
        epoch=int(day1_start.timestamp()),
    )
    day2_start = day1_start + timedelta(days=1)
    day2 = mod.ModelBudgetWindow(
        window_start=day2_start,
        reset_at=day2_start + timedelta(days=1),
        epoch=int(day2_start.timestamp()),
    )

    user_api_key = UserAPIKeyAuth(token="key-a", team_id="team-1", user_id="user-1", model_max_budget={})

    with patch.object(mod, "current_model_budget_window", return_value=day1):
        await budget_limiter._increment_model_budget_spend(
            response_cost=25.0,
            model="claude-sonnet-4-6",
            budget_config=budget_config,
            scope="team",
            virtual_key="key-a",
            team_id="team-1",
            user_id="user-1",
            end_user_id=None,
            litellm_call_id=None,
        )
        with pytest.raises(litellm.BudgetExceededError):
            await budget_limiter.is_key_within_model_budget(
                user_api_key,
                "claude-sonnet-4-6",
                team_model_max_budget=team_budget,
            )

    with patch.object(mod, "current_model_budget_window", return_value=day2):
        assert (
            await budget_limiter.is_key_within_model_budget(
                user_api_key,
                "claude-sonnet-4-6",
                team_model_max_budget=team_budget,
            )
            is True
        )


@pytest.mark.asyncio
async def test_cold_cache_reseeds_model_spend_from_spend_logs(budget_limiter):
    from litellm.proxy.hooks import model_max_budget_limiter as mod

    team_budget = _team_model_budget()

    reconcile = AsyncMock(return_value=18.0)
    with patch.object(mod, "_sum_model_window_spend_logs", reconcile):
        usage = await build_effective_model_max_budget_usage(
            budget_limiter,
            api_key_hash="key-a",
            team_id="team-1",
            user_id="user-1",
            key_model_max_budget={},
            team_model_max_budget=team_budget,
            team_member_model_max_budget=None,
        )
        assert usage["claude-sonnet-4-6"]["current_spend"] == 18.0

        usage_again = await build_effective_model_max_budget_usage(
            budget_limiter,
            api_key_hash="key-a",
            team_id="team-1",
            user_id="user-1",
            key_model_max_budget={},
            team_model_max_budget=team_budget,
            team_member_model_max_budget=None,
        )
        assert usage_again["claude-sonnet-4-6"]["current_spend"] == 18.0

    assert reconcile.await_count == 1


@pytest.mark.asyncio
async def test_usage_view_reports_calendar_window_boundaries(budget_limiter):
    from datetime import datetime, timedelta

    team_budget = _team_model_budget()

    usage = await build_effective_model_max_budget_usage(
        budget_limiter,
        api_key_hash="key-a",
        team_id="team-1",
        user_id="user-1",
        key_model_max_budget={},
        team_model_max_budget=team_budget,
        team_member_model_max_budget=None,
    )

    entry = usage["claude-sonnet-4-6"]
    reset_at = datetime.fromisoformat(str(entry["reset_at"]))
    window_start = datetime.fromisoformat(str(entry["window_start"]))
    assert (reset_at.hour, reset_at.minute, reset_at.second) == (0, 0, 0)
    assert reset_at - window_start == timedelta(days=1)
