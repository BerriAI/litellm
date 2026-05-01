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
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_max_cost,
    get_budget_window_start,
    invalidate_budget_reservation_counters,
    release_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging


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
async def test_should_prevent_second_key_reservation_over_budget(
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

    assert (
        counter_cache.in_memory_cache.get_cache(key="spend:key:key-budget-race") == 0.6
    )

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_prevent_second_end_user_reservation_over_budget(
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
async def test_should_prevent_second_tag_reservation_over_budget(
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
async def test_should_reserve_remaining_budget_when_output_cap_missing(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-uncapped",
        spend=0.2,
        max_budget=1.0,
    )
    await key_cache.async_set_cache(
        key="key-budget-uncapped",
        value=valid_token,
    )
    request_body = _request_body()
    request_body.pop("max_tokens")

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation._get_model_cost_info",
        return_value={
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 100.0,
            "max_output_tokens": 200000,
        },
    ):
        assert (
            estimate_request_max_cost(
                request_body=request_body,
                route="/chat/completions",
                llm_router=None,
            )
            is None
        )
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
    assert reservation["reserved_cost"] == pytest.approx(0.8)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-uncapped"
    ) == pytest.approx(1.0)

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_shrink_uncapped_reservation_when_counter_advances(
    spend_counter_state,
    monkeypatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-uncapped-race",
        spend=0.2,
        max_budget=1.0,
    )
    request_body = _request_body()
    request_body.pop("max_tokens")

    from litellm.proxy.spend_tracking import budget_reservation

    async def stale_counter_read(counter):
        await counter_cache.async_increment_cache(
            key=counter.counter_key,
            value=0.3,
        )
        return 0.2

    monkeypatch.setattr(
        budget_reservation,
        "_get_current_counter_value",
        stale_counter_read,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=None,
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
    assert reservation["reserved_cost"] == pytest.approx(0.7)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-uncapped-race"
    ) == pytest.approx(1.0)

    await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-uncapped-race"
    ) == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_should_shrink_uncapped_reservation_multiple_times(
    spend_counter_state,
    monkeypatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-double-resize",
        spend=0.2,
        max_budget=1.0,
        team_id="team-budget-double-resize",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-budget-double-resize",
        spend=0.2,
        max_budget=1.0,
    )
    request_body = _request_body()
    request_body.pop("max_tokens")

    from litellm.proxy.spend_tracking import budget_reservation

    stale_spend_by_counter_key = {
        "spend:key:key-budget-double-resize": 0.3,
        "spend:team:team-budget-double-resize": 0.4,
    }

    async def stale_counter_read(counter):
        await counter_cache.async_increment_cache(
            key=counter.counter_key,
            value=stale_spend_by_counter_key[counter.counter_key],
        )
        return 0.2

    monkeypatch.setattr(
        budget_reservation,
        "_get_current_counter_value",
        stale_counter_read,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=None,
    ):
        reservation = await reserve_budget_for_request(
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

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.6)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-double-resize"
    ) == pytest.approx(0.9)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-budget-double-resize"
    ) == pytest.approx(1.0)

    await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-double-resize"
    ) == pytest.approx(0.3)
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:team:team-budget-double-resize"
    ) == pytest.approx(0.4)


def test_should_start_window_without_reset_at_at_duration_boundary():
    before = datetime.now(timezone.utc) - timedelta(hours=1)

    window_start = get_budget_window_start({"budget_duration": "1h"})

    after = datetime.now(timezone.utc) - timedelta(hours=1)
    assert window_start is not None
    assert before <= window_start <= after


@pytest.mark.asyncio
async def test_should_seed_malformed_window_counter_from_parent_authoritative_spend(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-malformed-window",
        spend=0.0,
        budget_limits=[
            {
                "budget_duration": "not-a-duration",
                "max_budget": 1.0,
            }
        ],
    )

    import litellm.proxy.proxy_server as ps

    db_row = MagicMock()
    db_row.spend = 0.9
    prisma_client = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=db_row
    )
    ps.prisma_client = prisma_client

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.2,
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await reserve_budget_for_request(
                request_body=_request_body(),
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=prisma_client,
                user_api_key_cache=key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )

    prisma_client.db.litellm_verificationtoken.find_unique.assert_awaited_once_with(
        where={"token": "key-budget-malformed-window"}
    )
    assert counter_cache.in_memory_cache.get_cache(
        key="spend:key:key-budget-malformed-window:window:not-a-duration"
    ) == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_should_coalesce_malformed_window_counter_initialization(
    spend_counter_state,
):
    import asyncio as _asyncio

    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    token = "key-budget-malformed-window-concurrent"
    malformed_window_counter_key = f"spend:key:{token}:window:not-a-duration"
    valid_token = UserAPIKeyAuth(
        token=token,
        spend=0.0,
        budget_limits=[
            {
                "budget_duration": "not-a-duration",
                "max_budget": 1.0,
            }
        ],
    )

    import litellm.proxy.proxy_server as ps

    db_call_count = 0

    async def slow_find_unique(**kwargs):
        nonlocal db_call_count
        db_call_count += 1
        await _asyncio.sleep(0.05)
        db_row = MagicMock()
        db_row.spend = 0.35
        return db_row

    prisma_client = MagicMock()
    prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        side_effect=slow_find_unique
    )
    ps.prisma_client = prisma_client

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.2,
    ):
        results = await _asyncio.gather(
            *[
                reserve_budget_for_request(
                    request_body=_request_body(),
                    route="/chat/completions",
                    llm_router=None,
                    valid_token=valid_token,
                    team_object=None,
                    user_object=None,
                    prisma_client=prisma_client,
                    user_api_key_cache=key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                for _ in range(2)
            ],
            return_exceptions=True,
        )

    assert not any(isinstance(result, Exception) for result in results), results
    assert all(result is not None for result in results)
    assert db_call_count == 1
    assert counter_cache.in_memory_cache.get_cache(
        key=malformed_window_counter_key
    ) == pytest.approx(0.75)

    for reservation in results:
        await release_budget_reservation(reservation)

    assert counter_cache.in_memory_cache.get_cache(
        key=malformed_window_counter_key
    ) == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_should_not_re_read_uncapped_budget_after_reservation_fallback(
    spend_counter_state,
    monkeypatch,
):
    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-uncapped-read-once",
        spend=0.2,
        max_budget=1.0,
    )

    from litellm.proxy.spend_tracking import budget_reservation

    current_counter_reads = []

    async def mock_get_current_counter_value(counter):
        current_counter_reads.append(counter.counter_key)
        return counter.fallback_spend

    async def mock_reserve_counter(counter, reservation_cost):
        return None

    monkeypatch.setattr(
        budget_reservation,
        "_get_current_counter_value",
        mock_get_current_counter_value,
    )
    monkeypatch.setattr(
        budget_reservation,
        "_reserve_counter",
        mock_reserve_counter,
    )

    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=None,
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
    assert reservation["reserved_cost"] == pytest.approx(0.8)
    assert current_counter_reads == ["spend:key:key-budget-uncapped-read-once"]


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
        spend=0.0,
        max_budget=0.3,
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
    ) == pytest.approx(0.0)
    mock_log_exception.assert_called()


@pytest.mark.asyncio
async def test_should_not_create_negative_counter_when_release_counter_is_missing(
    spend_counter_state,
):
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

    with pytest.raises(RuntimeError, match="missing counter"):
        await release_budget_reservation(reservation)

    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-missing-release"
        )
        is None
    )
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_should_invalidate_counter_when_release_would_underflow(
    spend_counter_state,
):
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

    with pytest.raises(RuntimeError, match="negative"):
        await release_budget_reservation(reservation)

    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-underflow-release"
        )
        is None
    )
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_should_invalidate_non_numeric_counter_during_release(
    spend_counter_state,
):
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

    with pytest.raises(RuntimeError, match="non-numeric"):
        await release_budget_reservation(reservation)

    assert (
        counter_cache.in_memory_cache.get_cache(
            key="spend:key:key-budget-nonnumeric-release"
        )
        is None
    )
    assert reservation["finalized"] is False


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

    await release_budget_reservation(reservation)
