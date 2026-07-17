import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_max_cost,
    get_budget_window_start,
    invalidate_budget_reservation_counters,
    release_budget_reservation,
    release_budget_reservation_on_cancel,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router


@pytest.fixture()
def spend_counter_state():
    import litellm.proxy.proxy_server as ps

    original_counter_cache = ps.spend_counter_cache
    original_key_cache = ps.user_api_key_cache
    original_prisma_client = ps.prisma_client

    counter_cache = DualCache()
    key_cache = DualCache()
    ps.spend_counter_cache = counter_cache
    ps.user_api_key_cache = key_cache
    ps.prisma_client = None

    try:
        yield counter_cache, key_cache
    finally:
        ps.spend_counter_cache = original_counter_cache
        ps.user_api_key_cache = original_key_cache
        ps.prisma_client = original_prisma_client


def _request_body() -> dict:
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }


async def _reserve(valid_token, cost, key_cache, proxy_logging_obj):
    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=cost,
    ):
        return await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_reservation_still_protects_under_budget_throttled_key(
    spend_counter_state, monkeypatch
):
    """An opted-in key that is still under budget keeps its reservation counter,
    so concurrent requests can't collectively overshoot max_budget."""
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-throttle-under",
        spend=0.0,
        max_budget=1.0,
        metadata={"throttle_on_budget_exceeded": True},
    )

    reservation = await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)

    assert reservation is not None
    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:key:key-throttle-under")
        == 0.6
    )


@pytest.mark.asyncio
async def test_reservation_does_not_block_over_budget_throttled_key(
    spend_counter_state, monkeypatch
):
    """Once an opted-in key is over budget the reservation path must not raise;
    the rate limiter throttles it instead."""
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-throttle-over",
        spend=0.0,
        max_budget=1.0,
        tpm_limit=1000,
        rpm_limit=100,
        metadata={"throttle_on_budget_exceeded": True},
    )

    # first reservation lands under budget (counter -> 0.6)
    await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)

    # any further request is over budget (0.6 + 0.6 > 1.0): the opted-in key is
    # released and allowed through (None), not blocked, and its over-budget
    # increment is released so the counter is not permanently inflated
    result = await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    assert result is None
    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:key:key-throttle-over")
        == 0.6
    )


@pytest.mark.asyncio
async def test_reservation_blocks_over_budget_non_throttled_key(
    spend_counter_state, monkeypatch
):
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.1)
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-no-optin-over",
        spend=0.0,
        max_budget=1.0,
    )

    await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)  # counter -> 1.0

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    assert exc_info.value.entity_type == "key"
    assert exc_info.value.entity_id == "key-no-optin-over"


@pytest.mark.asyncio
async def test_over_budget_window_counter_tags_clean_entity_id():
    from litellm.proxy.spend_tracking.budget_reservation import (
        _apply_over_budget_reservation_policy,
        _BudgetCounter,
    )

    counter = _BudgetCounter(
        counter_key="spend:key:test-token:window:1d",
        max_budget=1.0,
        fallback_spend=0.0,
        entity_type="Key",
        entity_id="test-token:1d",
        spend_log_entity_id="test-token",
    )

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _apply_over_budget_reservation_policy(
            counter=counter,
            valid_token=None,
            entry={"counter_key": counter.counter_key},
            applied_entries=[],
            reservation_cost=0.5,
            current_spend=2.0,
        )
    assert exc_info.value.entity_type == "key"
    assert exc_info.value.entity_id == "test-token"
    assert exc_info.value.max_budget == 1.0
    assert exc_info.value.current_cost == 2.0


def test_should_not_serialize_budget_reservation_on_user_api_key_auth():
    auth = UserAPIKeyAuth(
        token="key-budget-runtime-state",
        budget_reservation={
            "reserved_cost": 0.5,
            "entries": [{"counter_key": "spend:key:key-budget-runtime-state"}],
        },
    )

    assert "budget_reservation" not in auth.model_dump()
    assert "budget_reservation" not in auth.model_dump(exclude_none=True)
    assert "budget_reservation" not in auth.model_dump_json()


@pytest.mark.asyncio
async def test_should_shrink_second_key_reservation_to_remaining_budget(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-race",
        spend=0.0,
        max_budget=1.0,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert reservation is not None
        assert (
            counter_cache.in_memory_cache.get_cache(key="spend:key:key-budget-race")
            == 0.6
        )

        second_reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert second_reservation is not None
        assert second_reservation["reserved_cost"] == pytest.approx(0.4)
        assert counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-race"
        ) == pytest.approx(1.0)

        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=_request_body(),
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-race"
    ) == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-race"
    ) == pytest.approx(0.6)
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_shrink_second_end_user_reservation_to_remaining_budget(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-end-user",
        end_user_id="end-user-budget-race",
    )
    end_user_object = LiteLLM_EndUserTable(
        user_id="end-user-budget-race",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
            end_user_object=end_user_object,
        )
        assert reservation is not None
        assert counter_cache.in_memory_cache.get_cache(
            key="spend:end_user:end-user-budget-race"
        ) == pytest.approx(0.6)

        second_reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
            end_user_object=end_user_object,
        )
        assert second_reservation is not None
        assert second_reservation["reserved_cost"] == pytest.approx(0.4)
        assert counter_cache.in_memory_cache.get_cache(
            key="spend:end_user:end-user-budget-race"
        ) == pytest.approx(1.0)

        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=_request_body(),
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
                end_user_object=end_user_object,
            )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:end_user:end-user-budget-race"
    ) == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:end_user:end-user-budget-race"
    ) == pytest.approx(0.6)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        end_user_id="end-user-budget-race",
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:end_user:end-user-budget-race"
    ) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_should_shrink_second_tag_reservation_to_remaining_budget(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-budget-tag")
    request_body = _request_body()
    request_body["metadata"] = {
        "tags": ["tag-budget-race", "tag-without-budget", "tag-budget-race"]
    }
    await key_cache.async_set_cache(
        key="tag:tag-budget-race",
        value=LiteLLM_TagTable(
            tag_name="tag-budget-race",
            spend=0.0,
            budget_id="tag-budget-id",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="tag:tag-without-budget",
        value=LiteLLM_TagTable(
            tag_name="tag-without-budget",
            spend=0.0,
        ).model_dump(),
    )
    prisma_client = MagicMock()
    prisma_client.db.litellm_tagtable.find_many = AsyncMock(return_value=[])

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=prisma_client,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert reservation is not None
        assert reservation["entries"] == [
            {
                "counter_key": "spend:tag:tag-budget-race",
                "entity_type": "Tag",
                "entity_id": "tag-budget-race",
                "reserved_cost": 0.6,
                "applied_adjustment": 0.0,
            }
        ]
        assert counter_cache.in_memory_cache.get_cache(
            key="spend:tag:tag-budget-race"
        ) == pytest.approx(0.6)
        assert (
            counter_cache.in_memory_cache.get_cache(key="spend:tag:tag-without-budget")
            is None
        )

        second_reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=prisma_client,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert second_reservation is not None
        assert second_reservation["reserved_cost"] == pytest.approx(0.4)
        assert counter_cache.in_memory_cache.get_cache(
            key="spend:tag:tag-budget-race"
        ) == pytest.approx(1.0)

        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=request_body,
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=prisma_client,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:tag:tag-budget-race"
    ) == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:tag:tag-budget-race"
    ) == pytest.approx(0.6)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        tags=["tag-budget-race"],
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:tag:tag-budget-race"
    ) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_should_seed_and_update_end_user_and_tag_counters_without_reservation(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key="end_user_id:customer-1",
        value=LiteLLM_EndUserTable(
            user_id="customer-1",
            blocked=False,
            spend=4.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="tag:paid-tag",
        value=LiteLLM_TagTable(
            tag_name="paid-tag",
            spend=7.0,
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="tag:other-tag",
        value=LiteLLM_TagTable(
            tag_name="other-tag",
            spend=2.0,
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.50,
        end_user_id="customer-1",
        tags=["paid-tag", "paid-tag", "other-tag", ""],
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:end_user:customer-1"
    ) == pytest.approx(4.50)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:tag:paid-tag"
    ) == pytest.approx(7.50)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:tag:other-tag"
    ) == pytest.approx(2.50)


@pytest.mark.asyncio
async def test_should_reserve_team_member_and_org_budget_counters(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-shared",
        spend=0.0,
        max_budget=1.0,
        user_id="user-budget-shared",
        team_id="team-budget-shared",
        org_id="org-budget-shared",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-budget-shared",
        spend=0.0,
        max_budget=1.0,
    )
    user_object = LiteLLM_UserTable(
        user_id="user-budget-shared",
        spend=0.0,
    )
    await key_cache.async_set_cache(
        key="team_membership:user-budget-shared:team-budget-shared",
        value=LiteLLM_TeamMembership(
            user_id="user-budget-shared",
            team_id="team-budget-shared",
            spend=0.1,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="org_id:org-budget-shared:with_budget",
        value=LiteLLM_OrganizationTable(
            organization_id="org-budget-shared",
            organization_alias="shared-org",
            budget_id="org-budget-id",
            spend=0.1,
            models=[],
            created_by="test",
            updated_by="test",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
        ).model_dump(),
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.3,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=user_object,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team_member:user-budget-shared:team-budget-shared"
    ) == pytest.approx(0.4)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:org:org-budget-shared"
    ) == pytest.approx(0.4)

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reserve_user_budget_counter_for_team_key(spend_counter_state):
    """A user's personal budget must be reserved even when the key belongs to a team.

    Regression for GitHub issue #12905: previously the reservation path skipped the
    user spend counter whenever the key had a team, so a team key could overshoot the
    user's personal max_budget under concurrency.
    """
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-user-on-team",
        spend=0.0,
        user_id="user-on-team",
        team_id="team-no-budget",
    )
    team_object = LiteLLM_TeamTable(team_id="team-no-budget", spend=0.0, max_budget=None)
    user_object = LiteLLM_UserTable(user_id="user-on-team", spend=0.0, max_budget=5.0)

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.3,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=user_object,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert counter_cache.in_memory_cache.get_cache(key="spend:user:user-on-team") == pytest.approx(0.3)

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_skip_user_budget_counter_for_team_key_when_flag_set(spend_counter_state):
    """skip_user_budget_on_team_key=True restores the legacy behavior where a user's
    personal budget is not reserved for a team key."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-user-on-team-skip",
        spend=0.0,
        user_id="user-on-team-skip",
        team_id="team-no-budget-skip",
    )
    team_object = LiteLLM_TeamTable(team_id="team-no-budget-skip", spend=0.0, max_budget=None)
    user_object = LiteLLM_UserTable(user_id="user-on-team-skip", spend=0.0, max_budget=5.0)

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.3,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=user_object,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
            skip_user_budget_on_team_key=True,
        )

    assert counter_cache.in_memory_cache.get_cache(key="spend:user:user-on-team-skip") is None

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_with_budget_cache(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key="org_id:org-counter-with-budget:with_budget",
        value=LiteLLM_OrganizationTable(
            organization_id="org-counter-with-budget",
            organization_alias="shared-org",
            budget_id="org-budget-id",
            spend=2.0,
            models=[],
            created_by="test",
            updated_by="test",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        org_id="org-counter-with-budget",
        response_cost=0.25,
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:org:org-counter-with-budget"
    ) == pytest.approx(2.25)


@pytest.mark.asyncio
async def test_should_seed_org_counter_from_plain_org_cache(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    await key_cache.async_set_cache(
        key="org_id:org-counter-plain",
        value=LiteLLM_OrganizationTable(
            organization_id="org-counter-plain",
            organization_alias="shared-org",
            budget_id="org-budget-id",
            spend=2.0,
            models=[],
            created_by="test",
            updated_by="test",
        ).model_dump(),
    )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        org_id="org-counter-plain",
        response_cost=0.25,
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:org:org-counter-plain"
    ) == pytest.approx(2.25)


@pytest.mark.asyncio
async def test_should_cap_known_estimate_to_remaining_budget(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-known-estimate-cap",
        spend=0.9,
        max_budget=1.0,
    )
    counter_cache.in_memory_cache.set_cache(
        key="spend:key:key-budget-known-estimate-cap",
        value=0.9,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.1)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-known-estimate-cap"
    ) == pytest.approx(1.0)

    await release_budget_reservation(reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-known-estimate-cap"
    ) == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_should_clamp_reservation_to_default_when_output_cap_missing(
    spend_counter_state,
):
    """When max_tokens is not specified, _estimate_output_tokens falls back to
    DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK (16K), clamped by the model's
    max_output_tokens. Reservation must be a bounded per-request amount
    (mirroring parallel_request_limiter_v3's DEFAULT_MAX_TOKENS_ESTIMATE),
    not the entire remaining headroom."""
    from litellm.proxy.spend_tracking.budget_reservation import (
        DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK,
    )

    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-uncapped",
        spend=0.2,
        max_budget=10000.0,
    )
    await key_cache.async_set_cache(
        key="key-budget-uncapped",
        value=valid_token,
    )
    request_body = _request_body()
    request_body.pop("max_tokens")

    output_cost_per_token = 1e-5  # roughly Opus 4.5/4.7 output rate
    expected_cost = DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK * output_cost_per_token

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "input_cost_per_token": 0.0,
            "output_cost_per_token": output_cost_per_token,
            "max_output_tokens": 200000,  # well above the 16K fallback
        },
    ):
        estimated = estimate_request_max_cost(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
        )
        assert estimated == pytest.approx(expected_cost)

        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(expected_cost)
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reserve_tiered_pricing_cost(spend_counter_state):
    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    router = Router(
        model_list=[
            {
                "model_name": "dashscope/qwen3-max",
                "litellm_params": {
                    "model": "dashscope/qwen3-max",
                    "api_key": "sk-fake",
                },
                "model_info": {
                    "max_input_tokens": 258048,
                    "max_output_tokens": 65536,
                    "tiered_pricing": [
                        {
                            "input_cost_per_token": 1.2e-06,
                            "output_cost_per_token": 6e-06,
                            "range": [0, 32000],
                        },
                        {
                            "input_cost_per_token": 2.4e-06,
                            "output_cost_per_token": 1.2e-05,
                            "range": [32000, 128000],
                        },
                    ],
                },
            }
        ]
    )
    request_body = {
        "model": "dashscope/qwen3-max",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10,
    }

    estimated_cost = estimate_request_max_cost(
        request_body=request_body,
        route="/chat/completions",
        llm_router=router,
    )
    assert estimated_cost is not None
    assert estimated_cost > 0

    valid_token = UserAPIKeyAuth(
        token="key-tiered-pricing",
        spend=0.0,
        max_budget=estimated_cost,
    )
    reservation = await reserve_budget_for_request(
        request_body=request_body,
        route="/chat/completions",
        llm_router=router,
        valid_token=valid_token,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(estimated_cost)
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=router,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    await release_budget_reservation(reservation)


def test_tiered_reservation_is_all_or_nothing_with_output_tier_from_input_length():
    """Dashscope tiered pricing is all-or-nothing: the tier is chosen by the total
    input tokens and every token (input and output) is billed at that tier's rate.

    A long-context request with a large output allowance must reserve the output at
    the input-selected tier, not at the cheapest tier picked from the output volume.
    The earlier graduated calculation under-reserved such requests, letting a caller
    slip past a depleted budget."""
    tiered_pricing = [
        {"range": [0, 32000], "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
        {"range": [32000, 128000], "input_cost_per_token": 4e-06, "output_cost_per_token": 8e-06},
    ]
    input_tokens = 100000  # falls entirely in the second tier
    output_tokens = 1000

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={"tiered_pricing": tiered_pricing, "max_output_tokens": 200000},
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_input_tokens",
            return_value=input_tokens,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_output_tokens",
            return_value=output_tokens,
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
        )

    expected = (input_tokens * 4e-06) + (output_tokens * 8e-06)
    assert estimated == pytest.approx(expected)

    # What the old graduated math (with the output tier taken from output volume)
    # would have reserved. The all-or-nothing estimate must be strictly larger.
    graduated_under_reserve = (32000 * 1e-06) + (68000 * 4e-06) + (output_tokens * 2e-06)
    assert estimated > graduated_under_reserve


def test_tiered_reservation_uses_higher_reasoning_output_rate():
    """Some tiered models price reasoning output above standard output. The
    reasoning-token share is unknown before the request runs, so reservation must
    charge every output token at the higher of the two rates to avoid under-reserving
    reasoning-heavy requests."""
    tiered_pricing = [
        {
            "range": [0, 32000],
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 1.2e-06,
            "output_cost_per_reasoning_token": 4e-06,
        }
    ]
    input_tokens = 1000
    output_tokens = 500

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={"tiered_pricing": tiered_pricing, "max_output_tokens": 200000},
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_input_tokens",
            return_value=input_tokens,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_output_tokens",
            return_value=output_tokens,
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
        )

    expected = (input_tokens * 1e-06) + (output_tokens * 4e-06)
    assert estimated == pytest.approx(expected)

    # Reserving output at the plain rate would under-reserve reasoning-heavy calls.
    under_reserve = (input_tokens * 1e-06) + (output_tokens * 1.2e-06)
    assert estimated > under_reserve


def test_flat_reservation_uses_higher_reasoning_output_rate():
    """The same reasoning under-reservation gap exists for flat-rate models that
    declare output_cost_per_reasoning_token above output_cost_per_token."""
    input_tokens = 1000
    output_tokens = 500

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 1.2e-06,
                "output_cost_per_reasoning_token": 4e-06,
                "max_output_tokens": 200000,
            },
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_input_tokens",
            return_value=input_tokens,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_output_tokens",
            return_value=output_tokens,
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
        )

    expected = (input_tokens * 1e-06) + (output_tokens * 4e-06)
    assert estimated == pytest.approx(expected)
    under_reserve = (input_tokens * 1e-06) + (output_tokens * 1.2e-06)
    assert estimated > under_reserve


def test_reservation_uses_most_expensive_deployment_in_group():
    """When a model group mixes deployments with different tiered rates, reservation
    must estimate against the most expensive one. Reserving the cheaper sibling would
    let a caller repeatedly hit the alias and exceed the budget once routed to the
    costlier deployment."""
    cheap = [{"range": [0, 32000], "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}]
    expensive = [{"range": [0, 32000], "input_cost_per_token": 5e-06, "output_cost_per_token": 1e-05}]
    input_tokens = 1000
    output_tokens = 10

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
            return_value={"max_output_tokens": 200000},
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._get_deployment_tiered_pricing_tables",
            return_value=[cheap, expensive],
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_input_tokens",
            return_value=input_tokens,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation._estimate_output_tokens",
            return_value=output_tokens,
        ),
    ):
        estimated = estimate_request_max_cost(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
        )

    expected_expensive = (input_tokens * 5e-06) + (output_tokens * 1e-05)
    expected_cheap = (input_tokens * 1e-06) + (output_tokens * 2e-06)
    assert expected_expensive > expected_cheap
    assert estimated == pytest.approx(expected_expensive)


@pytest.mark.asyncio
async def test_should_clamp_reservation_to_model_ceiling_when_caller_overrequests(
    spend_counter_state,
):
    """An adversarial caller sending max_tokens=999_999_999 must not be able
    to inflate the per-request reservation up to the entire remaining team
    headroom. _estimate_output_tokens clamps the explicit value at the
    model's max_output_tokens — the model can only physically emit that
    many tokens anyway, so anything more is both wasteful and a DoS surface."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-overrequest",
        spend=0.0,
        max_budget=10000.0,
    )
    await key_cache.async_set_cache(
        key="key-budget-overrequest",
        value=valid_token,
    )

    request_body = _request_body()
    request_body["max_tokens"] = 999_999_999

    output_cost_per_token = 1e-5
    model_ceiling = 128_000
    expected_cost = model_ceiling * output_cost_per_token

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "input_cost_per_token": 0.0,
            "output_cost_per_token": output_cost_per_token,
            "max_output_tokens": model_ceiling,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(expected_cost)
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reserve_image_generation_cost_per_image(
    spend_counter_state,
):
    """Image-generation requests reserve `n × per-image cost` so concurrent
    requests against a depleted budget cannot all bypass the admission gate.
    The OpenAI ``dall-e-3`` entry exposes the per-image price as
    ``input_cost_per_image`` (a naming quirk), while other providers use
    ``output_cost_per_image`` — both must be honored."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-image-gen",
        spend=0.0,
        max_budget=10.0,
    )
    await key_cache.async_set_cache(key="key-image-gen", value=valid_token)

    request_body = {"model": "dall-e-3", "prompt": "a cat", "n": 3}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "image_generation",
            "input_cost_per_image": 0.04,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/v1/images/generations",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.12)  # 3 × $0.04
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reject_concurrent_image_request_against_depleted_budget(
    spend_counter_state,
):
    """Greptile P1 regression: with image-gen reservation in place, a second
    concurrent image request against a budget already pinned at the cap by
    the first reservation must raise BudgetExceededError instead of
    silently reaching the provider."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-image-deplete",
        spend=0.0,
        team_id="team-image-deplete",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-image-deplete",
        max_budget=0.04,
        spend=0.0,
    )
    await key_cache.async_set_cache(
        key=f"team_id:{team_object.team_id}",
        value=team_object,
    )

    request_body = {"model": "dall-e-3", "prompt": "a cat"}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "image_generation",
            "input_cost_per_image": 0.04,
        },
    ):
        first = await reserve_budget_for_request(
            request_body=request_body,
            route="/v1/images/generations",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert first is not None

        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=request_body,
                route="/v1/images/generations",
                llm_router=None,
                valid_token=valid_token,
                team_object=team_object,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    await release_budget_reservation(first)


@pytest.mark.asyncio
async def test_should_skip_reservation_for_per_pixel_image_model(
    spend_counter_state,
):
    """DALL-E 2-style per-pixel pricing depends on the requested ``size``,
    which we don't decode here. Fall through to read-time enforcement
    rather than guess."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-image-per-pixel",
        spend=0.0,
        max_budget=1.0,
    )
    await key_cache.async_set_cache(key="key-image-per-pixel", value=valid_token)

    request_body = {"model": "dall-e-2", "prompt": "a cat", "size": "256x256"}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "image_generation",
            "input_cost_per_pixel": 2.4414e-07,
            "output_cost_per_pixel": 0.0,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/v1/images/generations",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None


@pytest.mark.asyncio
async def test_should_use_token_pricing_for_chat_model_with_image_cost_field(
    spend_counter_state,
):
    """Several chat and embedding models carry ``input_cost_per_image`` /
    ``output_cost_per_image`` to price multimodal vision *input*, not image
    generation (e.g. gemini-3.1-pro-preview, azure/gpt-realtime-*,
    amazon.titan-embed-image-v1). _estimate_image_generation_cost must gate
    on ``mode`` so these models still go through the token-priced path —
    otherwise a long chat reserves a fraction of a cent instead of the true
    token cost."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-multimodal-chat",
        spend=0.0,
        max_budget=10.0,
    )
    await key_cache.async_set_cache(key="key-multimodal-chat", value=valid_token)

    # Roughly the gemini-3.1-pro-preview shape: chat-mode model that
    # carries an output_cost_per_image alongside token pricing.
    output_cost_per_token = 1.2e-5
    request_body = {
        "model": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1000,
    }
    expected_cost = 1000 * output_cost_per_token  # token-priced path, not 1 × $0.00012

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "chat",
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": output_cost_per_token,
            "output_cost_per_image": 0.00012,
            "max_output_tokens": 64000,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    # Token-priced path: reservation ≈ output_tokens × output_cost_per_token,
    # plus a small input-token contribution. Must NOT collapse to the
    # per-image price ($0.00012) which would indicate the image-gen branch
    # incorrectly fired for this chat model.
    assert reservation["reserved_cost"] == pytest.approx(expected_cost, rel=0.05)
    assert reservation["reserved_cost"] > 0.001  # well above per-image price
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reserve_image_edit_cost_per_image(
    spend_counter_state,
):
    """``image_edit`` models (Flux Kontext, Stability inpaint/outpaint, etc.)
    bill per generated image just like ``image_generation`` and must get
    the same atomic per-image reservation."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-image-edit",
        spend=0.0,
        max_budget=10.0,
    )
    await key_cache.async_set_cache(key="key-image-edit", value=valid_token)

    request_body = {"model": "stability/inpaint", "prompt": "a cat", "n": 2}

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "mode": "image_edit",
            "output_cost_per_image": 0.05,
        },
    ):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/v1/images/edits",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.10)  # 2 × $0.05
    await release_budget_reservation(reservation)


def test_should_start_window_without_reset_at_at_duration_boundary():
    before = datetime.now(timezone.utc) - timedelta(hours=1)

    window_start = get_budget_window_start({"budget_duration": "1h"})

    after = datetime.now(timezone.utc) - timedelta(hours=1)
    assert window_start is not None
    assert before <= window_start <= after


@pytest.mark.asyncio
async def test_should_skip_budget_window_with_unparseable_duration(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-malformed-window",
        spend=0.9,
        max_budget=10.0,
        budget_limits=[
            {
                "budget_duration": "not-a-duration",
                "max_budget": 1.0,
            }
        ],
    )
    counter_cache.in_memory_cache.set_cache(
        key="spend:key:key-budget-malformed-window",
        value=0.9,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.2,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert [entry["counter_key"] for entry in reservation["entries"]] == [
        "spend:key:key-budget-malformed-window"
    ]
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-malformed-window"
    ) == pytest.approx(1.1)
    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-malformed-window:window:not-a-duration"
        )
        is None
    )

    await release_budget_reservation(reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-malformed-window"
    ) == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_should_skip_window_reservation_when_db_baseline_unavailable(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-window-db-unavailable",
        budget_limits=[
            {
                "budget_duration": "1h",
                "max_budget": 1.0,
            }
        ],
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.5,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None
    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-window-db-unavailable:window:1h"
        )
        is None
    )


@pytest.mark.asyncio
async def test_should_skip_reservation_when_counter_increment_fails(
    spend_counter_state,
    monkeypatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-reserve-unavailable",
        spend=0.0,
        max_budget=1.0,
    )

    async def fail_increment_cache(*args, **kwargs):
        raise RuntimeError("counter unavailable")

    monkeypatch.setattr(counter_cache, "async_increment_cache", fail_increment_cache)

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.5,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.verbose_proxy_logger.warning"
        ) as mock_warning,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None
    assert mock_warning.call_count >= 1
    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-reserve-unavailable"
        )
        is None
    )


@pytest.mark.asyncio
async def test_should_skip_reservation_when_counter_initialization_fails(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-reserve-init-unavailable",
        spend=0.0,
        max_budget=1.0,
    )

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.5,
        ),
        patch(
            "litellm.proxy.proxy_server._ensure_spend_counter_initialized",
            side_effect=RuntimeError("redis unavailable"),
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.verbose_proxy_logger.warning"
        ) as mock_warning,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None
    assert mock_warning.call_count >= 1
    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-reserve-init-unavailable"
        )
        is None
    )


@pytest.mark.asyncio
async def test_should_release_tracked_entry_when_reservation_fails_after_increment(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-reserve-after-increment-failure",
        spend=0.0,
        max_budget=1.0,
    )

    import litellm.proxy.proxy_server as ps

    original_increment_counter = ps._increment_spend_counter_cache
    first_increment = True

    async def fail_after_increment(counter_key: str, increment: float):
        nonlocal first_increment
        if first_increment:
            first_increment = False
            await counter_cache.async_increment_cache(key=counter_key, value=increment)
            raise RuntimeError("lost increment response")
        return await original_increment_counter(
            counter_key=counter_key,
            increment=increment,
        )

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.5,
        ),
        patch(
            "litellm.proxy.proxy_server._increment_spend_counter_cache",
            side_effect=fail_after_increment,
        ),
        patch(
            "litellm.proxy.proxy_server._invalidate_spend_counter",
            side_effect=RuntimeError("invalidate unavailable"),
        ),
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is None
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-reserve-after-increment-failure"
    ) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_should_reconcile_reserved_counter_to_actual_spend(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-reconcile",
        spend=0.0,
        max_budget=1.0,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token="key-budget-reconcile",
        team_id="team-without-budget",
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
    )

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-reconcile"
    ) == pytest.approx(0.2)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-without-budget"
    ) == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_should_release_reservation_on_failure(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-release",
        spend=0.0,
        max_budget=1.0,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.4,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    await release_budget_reservation(reservation)
    await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-release"
    ) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_should_retry_partial_release_without_double_decrement(
    spend_counter_state,
    monkeypatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-partial-release",
        spend=0.0,
        max_budget=1.0,
        team_id="team-budget-partial-release",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-budget-partial-release",
        spend=0.0,
        max_budget=1.0,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.4,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    original_increment_cache = counter_cache.async_increment_cache
    fail_next_team_release = True

    async def flaky_increment_cache(key, value, *args, **kwargs):
        nonlocal fail_next_team_release
        if (
            key == "spend:team:team-budget-partial-release"
            and value < 0
            and fail_next_team_release
        ):
            fail_next_team_release = False
            raise RuntimeError("simulated counter failure")
        return await original_increment_cache(key=key, value=value, *args, **kwargs)

    monkeypatch.setattr(counter_cache, "async_increment_cache", flaky_increment_cache)

    with pytest.raises(RuntimeError):
        await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-partial-release"
    ) == pytest.approx(0.0)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-budget-partial-release"
    ) == pytest.approx(0.4)

    await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-partial-release"
    ) == pytest.approx(0.0)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-budget-partial-release"
    ) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_should_preserve_budget_error_and_continue_partial_cleanup(
    spend_counter_state,
    monkeypatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-cleanup-failure",
        spend=0.0,
        max_budget=1.0,
        team_id="team-budget-cleanup-failure",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-budget-cleanup-failure",
        spend=0.3,
        max_budget=0.3,
    )
    await key_cache.async_set_cache(
        key="team_id:team-budget-cleanup-failure",
        value=team_object,
    )

    original_increment_cache = counter_cache.async_increment_cache
    fail_key_cleanup = True

    async def flaky_increment_cache(key, value, *args, **kwargs):
        nonlocal fail_key_cleanup
        if key == "spend:key:key-budget-cleanup-failure" and value < 0:
            if fail_key_cleanup:
                fail_key_cleanup = False
                raise RuntimeError("simulated cleanup failure")
        return await original_increment_cache(key=key, value=value, *args, **kwargs)

    monkeypatch.setattr(counter_cache, "async_increment_cache", flaky_increment_cache)

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.4,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.verbose_proxy_logger.exception"
        ) as mock_log_exception,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=_request_body(),
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=team_object,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-cleanup-failure"
        )
        is None
    )
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-budget-cleanup-failure"
    ) == pytest.approx(0.3)
    mock_log_exception.assert_called()


@pytest.mark.asyncio
async def test_release_missing_counter_reseeds_from_db_instead_of_failing(
    spend_counter_state,
):
    """A reconcile/release that finds the counter missing must NOT delete it and
    raise (the old fail-open that left budgets unenforced after a Redis reload).
    It reseeds from the authoritative DB; with no DB it leaves the counter
    untouched and finalizes."""
    counter_cache, _ = spend_counter_state
    reservation = {
        "reserved_cost": 0.4,
        "entries": [
            {
                "counter_key": "spend:key:key-budget-missing-release",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    # must not raise
    await release_budget_reservation(reservation)

    # counter not driven negative / not corrupted; left absent (no DB to reseed)
    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-missing-release"
        )
        is None
    )
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_underflow_counter_reseeds_from_db(spend_counter_state):
    """When the release delta would drive the counter negative (counter was
    reset/reseeded mid-flight), reseed from the authoritative DB rather than
    deleting and failing open."""
    import litellm.proxy.proxy_server as ps

    counter_cache, _ = spend_counter_state
    await counter_cache.async_increment_cache(
        key="spend:key:key-budget-underflow-release",
        value=0.1,
    )
    reservation = {
        "reserved_cost": 0.4,
        "entries": [
            {
                "counter_key": "spend:key:key-budget-underflow-release",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    with patch.object(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=0.25)):
        await release_budget_reservation(reservation)

    # counter reseeded up to the authoritative DB value, not deleted or negated
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-underflow-release"
    ) == pytest.approx(0.25)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_non_numeric_counter_reseeds_from_db(spend_counter_state):
    """A non-numeric counter value (corrupt/stale) during release is recovered by
    reseeding from the DB, not by deleting the counter and raising."""
    import litellm.proxy.proxy_server as ps

    counter_cache, _ = spend_counter_state
    counter_cache.in_memory_cache.set_cache(
        key="spend:key:key-budget-nonnumeric-release",
        value="stale",
    )
    reservation = {
        "reserved_cost": 0.4,
        "entries": [
            {
                "counter_key": "spend:key:key-budget-nonnumeric-release",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    with patch.object(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=0.5)):
        await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-nonnumeric-release"
    ) == pytest.approx(0.5)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_should_invalidate_reserved_counters_after_persisted_spend_failure(
    spend_counter_state,
):
    counter_cache, _ = spend_counter_state
    await counter_cache.async_increment_cache(
        key="spend:key:key-budget-invalidate",
        value=0.4,
    )
    await counter_cache.async_increment_cache(
        key="spend:team:team-budget-invalidate",
        value=0.4,
    )

    await invalidate_budget_reservation_counters(
        {
            "reserved_cost": 0.4,
            "entries": [
                {"counter_key": "spend:key:key-budget-invalidate"},
                {"counter_key": "spend:team:team-budget-invalidate"},
            ],
        }
    )

    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:key:key-budget-invalidate")
        is None
    )
    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:team:team-budget-invalidate")
        is None
    )


@pytest.mark.asyncio
async def test_should_reserve_all_budgeted_counters(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-all",
        spend=0.0,
        max_budget=1.0,
        team_id="team-budget-all",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-budget-all",
        spend=0.0,
        max_budget=1.0,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.3,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:key:key-budget-all") == 0.3
    )
    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:team:team-budget-all") == 0.3
    )


@pytest.mark.asyncio
async def test_should_not_block_concurrent_team_request_when_first_request_lacks_max_tokens(
    spend_counter_state,
):
    """
    Regression test: a team-bound request with no max_tokens must not pin the
    team's spend counter at max_budget for the duration of the request.

    Repro of the integration-test team being falsely budget-blocked at the
    $2000 cap while DB spend is $0.144: the first request without max_tokens
    used to reserve the entire remaining headroom, leaving any subsequent
    request stuck behind a counter sitting at the cap until the success
    callback finished reconciling.
    """
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)

    valid_token = UserAPIKeyAuth(
        token="key-team-integration-tests",
        spend=0.0,
        team_id="team-integration-tests",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-integration-tests",
        max_budget=2000.0,
        spend=0.144,
    )
    await key_cache.async_set_cache(
        key=f"team_id:{team_object.team_id}",
        value=team_object,
    )

    request_body = _request_body()
    request_body.pop("max_tokens")

    # Realistic Opus 4.7 output pricing — the 16K fallback × $25/M ≈ $0.40
    # reservation per request, leaving ~5000 admittable concurrent requests
    # against a $2000 team budget.
    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 2.5e-5,
            "max_output_tokens": 128000,
        },
    ):
        first_reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

        # The team counter must not be pinned at max_budget while the first
        # request is in flight, otherwise concurrent requests false-positive.
        team_counter_after_first = (
            counter_cache.in_memory_cache.get_cache(
                key=f"spend:team:{team_object.team_id}"
            )
            or 0.0
        )
        assert team_counter_after_first < team_object.max_budget, (
            f"Team counter sat at {team_counter_after_first} after one uncapped "
            f"reservation against a {team_object.max_budget} budget — concurrent "
            "requests will be falsely blocked."
        )

        # Second request — same shape — must succeed without raising.
        second_reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=team_object,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        assert second_reservation is not None

    if first_reservation is not None:
        await release_budget_reservation(first_reservation)
    if second_reservation is not None:
        await release_budget_reservation(second_reservation)


@pytest.mark.asyncio
async def test_release_budget_reservation_on_cancel_gives_back_counter(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-cancel-give-back", spend=0.0, max_budget=10.0
    )

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=3.0,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_input_cost",
            return_value=0.5,
        ),
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    assert reservation is not None
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-give-back"
    ) == pytest.approx(3.0)

    await release_budget_reservation_on_cancel(reservation)

    # the provider already received the input, so the reservation is reconciled
    # to the input cost (0.5), not refunded to zero; the worst-case output
    # reservation (3.0 -> 0.5) is released
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-give-back"
    ) == pytest.approx(0.5)
    assert reservation["finalized"] is True

    # idempotent: a second cancel reconcile must not change the counter again
    await release_budget_reservation_on_cancel(reservation)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-give-back"
    ) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_release_budget_reservation_on_cancel_noop_when_finalized(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-cancel-finalized", spend=0.0, max_budget=10.0
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=3.0,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    assert reservation is not None
    reservation["finalized"] = True

    await release_budget_reservation_on_cancel(reservation)

    # already reconciled by the success/failure path -> must stay untouched
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-finalized"
    ) == pytest.approx(3.0)


async def _reserve_for_stream(counter_cache, key_cache, proxy_logging_obj, token: str):
    valid_token = UserAPIKeyAuth(token=token, spend=0.0, max_budget=10.0)
    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=2.0,
        ),
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_input_cost",
            return_value=0.5,
        ),
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    assert reservation is not None
    assert counter_cache.in_memory_cache.get_cache(
        key=f"spend:key:{token}"
    ) == pytest.approx(2.0)
    valid_token.budget_reservation = reservation
    return valid_token, reservation


def _drive_streaming_cancel(valid_token, iterator_hook):
    streaming_logging_obj = MagicMock()
    streaming_logging_obj.async_post_call_streaming_iterator_hook = iterator_hook
    return ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=MagicMock(),
        user_api_key_dict=valid_token,
        request_data=_request_body(),
        proxy_logging_obj=streaming_logging_obj,
        serialize_chunk=lambda chunk: chunk,
        serialize_error=lambda exc: str(exc),
    )


@pytest.mark.asyncio
async def test_streaming_cancel_before_any_chunk_reconciles_to_input_cost(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-cancel-no-chunk"
    )

    # Client disconnects before the upstream produced any output.
    async def cancel_before_chunk(user_api_key_dict, response, request_data):
        if False:
            yield ""  # make this an async generator
        raise asyncio.CancelledError()

    generator = _drive_streaming_cancel(valid_token, cancel_before_chunk)
    received = []
    with pytest.raises(asyncio.CancelledError):
        async for chunk in generator:
            received.append(chunk)

    assert received == []
    # no chunk delivered, but the provider already received the input, so the
    # reservation is reconciled to the input cost (0.5), not refunded to zero
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-no-chunk"
    ) == pytest.approx(0.5)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_streaming_cancel_after_chunk_keeps_reservation(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-cancel-after-chunk"
    )

    # Client consumes a chunk, then disconnects. Cancellation logs no cost, so
    # refunding here would let the caller read partial output for free.
    async def cancel_after_chunk(user_api_key_dict, response, request_data):
        yield "data: chunk\n\n"
        raise asyncio.CancelledError()

    generator = _drive_streaming_cancel(valid_token, cancel_after_chunk)
    received = []
    with pytest.raises(asyncio.CancelledError):
        async for chunk in generator:
            received.append(chunk)

    assert received == ["data: chunk\n\n"]
    # a consumed stream must NOT be refunded
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-after-chunk"
    ) == pytest.approx(2.0)
    assert reservation.get("finalized") is not True


@pytest.mark.asyncio
async def test_release_budget_reservation_on_cancel_swallows_release_errors():
    # If the release itself fails (e.g. Redis unavailable) it must not escape
    # the helper: doing so would replace the in-flight CancelledError /
    # GeneratorExit at the call site and disrupt the disconnect teardown.
    reservation = {
        "reserved_cost": 3.0,
        "entries": [{"counter_key": "spend:key:key-cancel-error"}],
        "finalized": False,
        "input_cost": 0.5,
    }
    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.reconcile_budget_reservation",
        new=AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        # must return without raising
        await release_budget_reservation_on_cancel(reservation)


@pytest.mark.asyncio
async def test_streaming_cancel_in_slow_path_before_yield_refunds(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-cancel-slowpath"
    )

    async def one_chunk(user_api_key_dict, response, request_data):
        yield "data: chunk\n\n"

    streaming_logging_obj = MagicMock()
    streaming_logging_obj.async_post_call_streaming_iterator_hook = one_chunk
    # On the slow path the per-chunk hook is awaited before the chunk is yielded
    # to the client; cancel there. Nothing has reached the client yet.
    streaming_logging_obj.async_post_call_streaming_hook = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=MagicMock(),
        user_api_key_dict=valid_token,
        request_data=_request_body(),
        proxy_logging_obj=streaming_logging_obj,
        serialize_chunk=lambda chunk: chunk,
        serialize_error=lambda exc: str(exc),
    )

    received = []
    # include_cost_in_streaming_usage forces fast_path off, so the hook above runs
    with patch.object(litellm, "include_cost_in_streaming_usage", True, create=True):
        with pytest.raises(asyncio.CancelledError):
            async for chunk in generator:
                received.append(chunk)

    assert received == []
    # cancellation happened before any chunk reached the client, but the
    # provider already received the input -> reconcile to the input cost (0.5)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-cancel-slowpath"
    ) == pytest.approx(0.5)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_streaming_disconnect_after_consuming_chunk_keeps_reservation(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-disconnect-after-chunk"
    )

    async def two_chunks(user_api_key_dict, response, request_data):
        yield "data: a\n\n"
        yield "data: b\n\n"

    generator = _drive_streaming_cancel(valid_token, two_chunks)

    # Client consumes one chunk, then disconnects. aclose() raises GeneratorExit
    # at the suspended yield, after the chunk already reached the client.
    first = await generator.__anext__()
    assert first == "data: a\n\n"
    await generator.aclose()

    # output was delivered, so the reservation must NOT be refunded
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-disconnect-after-chunk"
    ) == pytest.approx(2.0)
    assert reservation.get("finalized") is not True


@pytest.mark.asyncio
async def test_streaming_slow_path_processes_and_yields_chunk(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, _ = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-slowpath-ok"
    )

    async def one_chunk(user_api_key_dict, response, request_data):
        yield {"content": "hi"}

    streaming_logging_obj = MagicMock()
    streaming_logging_obj.async_post_call_streaming_iterator_hook = one_chunk
    streaming_logging_obj.async_post_call_streaming_hook = AsyncMock(
        side_effect=lambda **kwargs: kwargs["response"]
    )

    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=MagicMock(),
        user_api_key_dict=valid_token,
        request_data=_request_body(),
        proxy_logging_obj=streaming_logging_obj,
        serialize_chunk=lambda chunk: chunk,
        serialize_error=lambda exc: str(exc),
    )

    received = []
    # include_cost_in_streaming_usage forces the slow path so the per-chunk hook,
    # content accumulation, and cost-injection branch all run to a successful yield
    with patch.object(litellm, "include_cost_in_streaming_usage", True, create=True):
        async for chunk in generator:
            received.append(chunk)

    assert received == [{"content": "hi"}]
    streaming_logging_obj.async_post_call_streaming_hook.assert_awaited_once()
