import asyncio
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import STREAM_SSE_KEEPALIVE_PING_BYTES
from litellm.llms.anthropic.messages.native.agentic_streaming_iterator import (
    AgenticAnthropicStreamingIterator,
)
from litellm.llms.anthropic.messages.native.streaming_iterator import (
    AnthropicMessagesStreamingResponse,
)
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_max_cost,
    invalidate_budget_reservation_counters,
    release_budget_reservation,
    release_budget_reservation_on_cancel,
)
from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    _approximate_input_size,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore, model_access_group_counter
from tests.unit.proxy.spend_tracking.native_budgets_replica import reserve_budget_for_request

pytestmark = pytest.mark.usefixtures("native_budgets_on")


@pytest.fixture()
def spend_counter_state():
    import litellm.proxy.proxy_server as ps

    original_counter_store = ps.spend_counter_store
    original_key_cache = ps.user_api_key_cache
    original_prisma_client = ps.prisma_client

    counter_cache = SpendCounterStore(connect(None).spend_counters())
    key_cache = UserApiKeyCache()
    ps.spend_counter_store = counter_cache
    ps.user_api_key_cache = key_cache
    ps.prisma_client = None

    try:
        yield counter_cache, key_cache
    finally:
        ps.spend_counter_store = original_counter_store
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
async def test_reservation_still_protects_under_budget_throttled_key(spend_counter_state, monkeypatch):
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
    assert counter_cache.get_local("spend:key:key-throttle-under") == 0.6


@pytest.mark.asyncio
async def test_reservation_does_not_block_over_budget_throttled_key(spend_counter_state, monkeypatch):
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
    assert counter_cache.get_local("spend:key:key-throttle-over") == 0.6


@pytest.mark.asyncio
async def test_reservation_blocks_over_budget_non_throttled_key(spend_counter_state, monkeypatch):
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
        assert counter_cache.get_local("spend:key:key-budget-race") == 0.6

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
        assert counter_cache.get_local("spend:key:key-budget-race") == pytest.approx(1.0)

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

    assert counter_cache.get_local("spend:key:key-budget-race") == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.get_local("spend:key:key-budget-race") == pytest.approx(0.6)
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
        assert counter_cache.get_local("spend:end_user:end-user-budget-race") == pytest.approx(0.6)

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
        assert counter_cache.get_local("spend:end_user:end-user-budget-race") == pytest.approx(1.0)

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

    assert counter_cache.get_local("spend:end_user:end-user-budget-race") == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.get_local("spend:end_user:end-user-budget-race") == pytest.approx(0.6)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        end_user_id="end-user-budget-race",
    )

    assert counter_cache.get_local("spend:end_user:end-user-budget-race") == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_should_shrink_second_tag_reservation_to_remaining_budget(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-budget-tag")
    request_body = _request_body()
    request_body["metadata"] = {"tags": ["tag-budget-race", "tag-without-budget", "tag-budget-race"]}
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
        assert counter_cache.get_local("spend:tag:tag-budget-race") == pytest.approx(0.6)
        assert counter_cache.get_local("spend:tag:tag-without-budget") is None

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
        assert counter_cache.get_local("spend:tag:tag-budget-race") == pytest.approx(1.0)

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

    assert counter_cache.get_local("spend:tag:tag-budget-race") == pytest.approx(1.0)

    await release_budget_reservation(second_reservation)
    assert counter_cache.get_local("spend:tag:tag-budget-race") == pytest.approx(0.6)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        tags=["tag-budget-race"],
    )

    assert counter_cache.get_local("spend:tag:tag-budget-race") == pytest.approx(0.2)


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

    assert counter_cache.get_local("spend:team_member:user-budget-shared:team-budget-shared") == pytest.approx(0.4)
    assert counter_cache.get_local("spend:org:org-budget-shared") == pytest.approx(0.4)

    await release_budget_reservation(reservation)


def _project_scoped_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token="key-project-scoped",
        spend=0.0,
        user_id="user-proj",
        team_id="team-proj",
        project_id="proj-1",
    )


async def _seed_project_scoped_budgets(
    key_cache: DualCache,
    team_member_spend: float,
    team_member_max_budget: float,
    project_spend: float,
    project_max_budget: float,
) -> None:
    await key_cache.async_set_cache(
        key="team_membership:user-proj:team-proj",
        value=LiteLLM_TeamMembership(
            user_id="user-proj",
            team_id="team-proj",
            spend=team_member_spend,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=team_member_max_budget),
        ).model_dump(),
    )
    await key_cache.async_set_cache(
        key="project_id:proj-1",
        value=LiteLLM_ProjectTableCachedObj(
            project_id="proj-1",
            team_id="team-proj",
            budget_id="project-budget-id",
            spend=project_spend,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=project_max_budget),
        ).model_dump(),
    )


@pytest.mark.asyncio
async def test_should_reserve_project_and_team_member_counters_for_project_scoped_key(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    await _seed_project_scoped_budgets(
        key_cache,
        team_member_spend=0.1,
        team_member_max_budget=1.0,
        project_spend=0.2,
        project_max_budget=1.0,
    )

    estimated = estimate_request_max_cost(request_body=_request_body(), route="/chat/completions", llm_router=None)
    assert estimated is not None and estimated > 0

    reservation = await reserve_budget_for_request(
        request_body=_request_body(),
        route="/chat/completions",
        llm_router=None,
        valid_token=_project_scoped_token(),
        team_object=LiteLLM_TeamTable(team_id="team-proj", spend=0.0, max_budget=None),
        user_object=LiteLLM_UserTable(user_id="user-proj", spend=0.0),
        prisma_client=None,
        user_api_key_cache=key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    assert reservation is not None
    assert counter_cache.get_local("spend:team_member:user-proj:team-proj") == pytest.approx(0.1 + estimated)
    assert counter_cache.get_local("spend:project:proj-1") == pytest.approx(0.2 + estimated)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token="key-project-scoped",
        team_id="team-proj",
        user_id="user-proj",
        response_cost=0.05,
        budget_reservation=reservation,
        project_id="proj-1",
    )

    assert counter_cache.get_local("spend:project:proj-1") == pytest.approx(0.25)
    assert counter_cache.get_local("spend:team_member:user-proj:team-proj") == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_exhausted_team_member_budget_still_blocks_project_scoped_key(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    await _seed_project_scoped_budgets(
        key_cache,
        team_member_spend=1.0,
        team_member_max_budget=1.0,
        project_spend=0.0,
        project_max_budget=100.0,
    )

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=_project_scoped_token(),
            team_object=LiteLLM_TeamTable(team_id="team-proj", spend=0.0, max_budget=None),
            user_object=LiteLLM_UserTable(user_id="user-proj", spend=0.0),
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert "TeamMember=user-proj:team-proj" in str(exc_info.value)
    assert counter_cache.get_local("spend:project:proj-1") in (None, pytest.approx(0.0))


@pytest.mark.asyncio
async def test_exhausted_project_budget_blocks_project_scoped_key(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    await _seed_project_scoped_budgets(
        key_cache,
        team_member_spend=0.0,
        team_member_max_budget=100.0,
        project_spend=5.0,
        project_max_budget=5.0,
    )

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=_project_scoped_token(),
            team_object=LiteLLM_TeamTable(team_id="team-proj", spend=0.0, max_budget=None),
            user_object=LiteLLM_UserTable(user_id="user-proj", spend=0.0),
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert "Project=proj-1" in str(exc_info.value)
    assert exc_info.value.entity_type == Litellm_EntityType.PROJECT.value
    assert counter_cache.get_local("spend:team_member:user-proj:team-proj") in (
        None,
        pytest.approx(0.0),
    )


@pytest.mark.asyncio
async def test_should_not_reserve_user_budget_counter_for_team_key(spend_counter_state):
    """The reservation path mirrors the read path: no personal user counter for a team key.

    A team-scoped key reserves against the key and team counters only, so the key
    owner's personal max_budget never gates a team request. Fails if the user
    counter is reserved for team keys again.
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

    assert counter_cache.get_local("spend:user:user-on-team") is None

    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_should_reserve_user_budget_counter_for_team_key_when_flag_enabled(spend_counter_state):
    """apply_user_budget_to_team_keys must widen the reservation path too.

    Read-time enforcement alone leaks budget under concurrency, so the opt-in has
    to reserve against the personal counter as well or a burst of team-key
    requests slips past the owner's max_budget.
    """
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-user-on-team-flagged",
        spend=0.0,
        user_id="user-on-team-flagged",
        team_id="team-no-budget",
    )
    team_object = LiteLLM_TeamTable(team_id="team-no-budget", spend=0.0, max_budget=None)
    user_object = LiteLLM_UserTable(user_id="user-on-team-flagged", spend=0.0, max_budget=5.0)

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
            apply_user_budget_to_team_keys=True,
        )

    assert counter_cache.get_local("spend:user:user-on-team-flagged") == pytest.approx(0.3)

    await release_budget_reservation(reservation)


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
    counter_cache.set_local("spend:key:key-budget-known-estimate-cap", 0.9)

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
    assert counter_cache.get_local("spend:key:key-budget-known-estimate-cap") == pytest.approx(1.0)

    await release_budget_reservation(reservation)
    assert counter_cache.get_local("spend:key:key-budget-known-estimate-cap") == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_fail_closed_rejects_known_estimate_exceeding_remaining_budget(
    spend_counter_state,
):
    """LIT-5922: with strict enforcement on, a request whose known estimate does
    not fit the remaining budget must be rejected before dispatch instead of
    having its reservation shrunk to the headroom and admitted, and the counter
    must be restored to the pre-request spend."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-known-estimate-fail-closed",
        spend=0.9,
        max_budget=1.0,
    )
    counter_cache.set_local("spend:key:key-budget-known-estimate-fail-closed", 0.9)

    with patch(  # test-quality-ok: reserve_budget_for_request takes no estimator, so pinning the estimate needs this attribute
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.6,
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
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
                fail_closed_budget_enforcement=True,
            )

    assert exc_info.value.current_cost == pytest.approx(0.9)
    assert exc_info.value.max_budget == pytest.approx(1.0)
    assert "Current cost: 0.9, Estimated request cost: 0.6, Max budget: 1.0" in str(exc_info.value)
    assert counter_cache.get_local("spend:key:key-budget-known-estimate-fail-closed") == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_fail_closed_tolerates_float_noise_when_estimate_exactly_fits(
    spend_counter_state,
):
    """0.1 + 0.2 lands a hair above 0.3 in floating point. Strict enforcement
    must treat that as fitting the budget, not reject it."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-fail-closed-float-noise",
        spend=0.1,
        max_budget=0.3,
    )
    counter_cache.set_local("spend:key:key-budget-fail-closed-float-noise", 0.1)

    with patch(  # test-quality-ok: reserve_budget_for_request takes no estimator, so pinning the estimate needs this attribute
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
            fail_closed_budget_enforcement=True,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.2)
    assert counter_cache.get_local("spend:key:key-budget-fail-closed-float-noise") == pytest.approx(0.3)


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
    counter_cache.set_local("spend:key:key-budget-malformed-window", 0.9)

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
    assert [entry["counter_key"] for entry in reservation["entries"]] == ["spend:key:key-budget-malformed-window"]
    assert counter_cache.get_local("spend:key:key-budget-malformed-window") == pytest.approx(1.1)
    assert counter_cache.get_local("spend:key:key-budget-malformed-window:window:not-a-duration") is None

    await release_budget_reservation(reservation)
    assert counter_cache.get_local("spend:key:key-budget-malformed-window") == pytest.approx(0.9)


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
    assert counter_cache.get_local("spend:key:key-budget-window-db-unavailable:window:1h") is None


@pytest.mark.asyncio
async def test_fail_closed_releases_earlier_counters_before_503(
    spend_counter_state,
):
    """#33923: when a later counter cannot be loaded in strict mode, the 503 is raised before any counter is
    reserved."""
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(
        token="key-budget-fail-closed-release",
        spend=0.0,
        max_budget=1.0,
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
        with pytest.raises(HTTPException) as exc_info:
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
                fail_closed_budget_enforcement=True,
            )

    assert exc_info.value.status_code == 503
    assert counter_cache.get_local("spend:key:key-budget-fail-closed-release") is None
    assert counter_cache.get_local("spend:key:key-budget-fail-closed-release:window:1h") is None


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
            "litellm.proxy.proxy_server.ensure_spend_counter_initialized",
            side_effect=RuntimeError("redis unavailable"),
        ),
        patch("litellm.proxy.spend_tracking.budget_reservation.verbose_proxy_logger.warning") as mock_warning,
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
    assert counter_cache.get_local("spend:key:key-budget-reserve-init-unavailable") is None


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

    assert counter_cache.get_local("spend:key:key-budget-reconcile") == pytest.approx(0.2)
    assert counter_cache.get_local("spend:team:team-without-budget") == pytest.approx(0.2)


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

    assert counter_cache.get_local("spend:key:key-budget-release") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_release_missing_counter_reseeds_from_db_instead_of_failing(
    spend_counter_state,
):
    """A reconcile/release that finds the counter missing must NOT delete it and
    raise (the old fail-open that left budgets unenforced after a Redis reload).
    It reseeds from the authoritative DB; with no DB it leaves the counter
    untouched and finalizes."""
    counter_cache, key_cache = spend_counter_state
    reservation = await _reserve(
        UserAPIKeyAuth(token="key-budget-missing-release", spend=0.0, max_budget=1.0),
        0.4,
        key_cache,
        ProxyLogging(user_api_key_cache=key_cache),
    )
    assert reservation is not None
    counter_cache.delete_local("spend:key:key-budget-missing-release")

    # must not raise
    await release_budget_reservation(reservation)

    # counter not driven negative / not corrupted; left absent (no DB to reseed)
    assert counter_cache.get_local("spend:key:key-budget-missing-release") is None
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_underflow_counter_reseeds_from_db(spend_counter_state):
    """When the release delta would drive the counter negative (counter was
    reset/reseeded mid-flight), reseed from the authoritative DB rather than
    deleting and failing open."""
    import litellm.proxy.proxy_server as ps

    counter_cache, key_cache = spend_counter_state
    reservation = await _reserve(
        UserAPIKeyAuth(token="key-budget-underflow-release", spend=0.0, max_budget=1.0),
        0.4,
        key_cache,
        ProxyLogging(user_api_key_cache=key_cache),
    )
    assert reservation is not None
    counter_cache.set_local("spend:key:key-budget-underflow-release", 0.1)

    with patch.object(ps.SpendCounterReseed, "from_db", AsyncMock(return_value=0.25)):
        await release_budget_reservation(reservation)

    # counter reseeded up to the authoritative DB value, not deleted or negated
    assert counter_cache.get_local("spend:key:key-budget-underflow-release") == pytest.approx(0.25)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_should_invalidate_reserved_counters_after_persisted_spend_failure(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    with patch(
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.4,
    ):
        reservation = await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=UserAPIKeyAuth(
                token="key-budget-invalidate", spend=0.0, max_budget=1.0, team_id="team-budget-invalidate"
            ),
            team_object=LiteLLM_TeamTable(team_id="team-budget-invalidate", spend=0.0, max_budget=1.0),
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=key_cache),
        )
    assert reservation is not None
    assert counter_cache.get_local("spend:key:key-budget-invalidate") == pytest.approx(0.4)
    assert counter_cache.get_local("spend:team:team-budget-invalidate") == pytest.approx(0.4)

    await invalidate_budget_reservation_counters(reservation)

    assert counter_cache.get_local("spend:key:key-budget-invalidate") is None
    assert counter_cache.get_local("spend:team:team-budget-invalidate") is None


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

    assert counter_cache.get_local("spend:key:key-budget-all") == 0.3
    assert counter_cache.get_local("spend:team:team-budget-all") == 0.3


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
        team_counter_after_first = counter_cache.get_local(f"spend:team:{team_object.team_id}") or 0.0
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
    valid_token = UserAPIKeyAuth(token="key-cancel-give-back", spend=0.0, max_budget=10.0)

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
    assert counter_cache.get_local("spend:key:key-cancel-give-back") == pytest.approx(3.0)

    await release_budget_reservation_on_cancel(reservation)

    # the provider already received the input, so the reservation is reconciled
    # to the input cost (0.5), not refunded to zero; the worst-case output
    # reservation (3.0 -> 0.5) is released
    assert counter_cache.get_local("spend:key:key-cancel-give-back") == pytest.approx(0.5)
    assert reservation["finalized"] is True

    # idempotent: a second cancel reconcile must not change the counter again
    await release_budget_reservation_on_cancel(reservation)
    assert counter_cache.get_local("spend:key:key-cancel-give-back") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_release_budget_reservation_on_cancel_noop_when_finalized(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-cancel-finalized", spend=0.0, max_budget=10.0)

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
    assert counter_cache.get_local("spend:key:key-cancel-finalized") == pytest.approx(3.0)


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
    assert counter_cache.get_local(f"spend:key:{token}") == pytest.approx(2.0)
    valid_token.budget_reservation = reservation
    return valid_token, reservation


async def _never_ending_stream():
    yield b'event: message_start\ndata: {"type": "message_start"}\n\n'
    await asyncio.sleep(30)


def _drive_streaming_cancel(valid_token, iterator_hook):
    streaming_logging_obj = MagicMock()
    streaming_logging_obj.async_post_call_streaming_iterator_hook = iterator_hook
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect = AsyncMock()
    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=MagicMock(),
        user_api_key_dict=valid_token,
        request_data=_request_body(),
        proxy_logging_obj=streaming_logging_obj,
        serialize_chunk=lambda chunk: chunk,
        serialize_error=lambda exc: str(exc),
    )
    return generator, streaming_logging_obj


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

    generator, streaming_logging_obj = _drive_streaming_cancel(valid_token, cancel_before_chunk)
    received = []

    async def _drain():
        async for chunk in generator:
            received.append(chunk)

    with pytest.raises(asyncio.CancelledError):
        await _drain()

    assert received == []
    # no chunk delivered, but the provider already received the input, so the
    # reservation is reconciled to the input cost (0.5), not refunded to zero
    assert counter_cache.get_local("spend:key:key-cancel-no-chunk") == pytest.approx(0.5)
    assert reservation["finalized"] is True
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect.assert_awaited_once()


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

    generator, streaming_logging_obj = _drive_streaming_cancel(valid_token, cancel_after_chunk)
    received = []

    async def _drain():
        async for chunk in generator:
            received.append(chunk)

    with pytest.raises(asyncio.CancelledError):
        await _drain()

    assert received == ["data: chunk\n\n"]
    # a consumed stream must NOT be refunded
    assert counter_cache.get_local("spend:key:key-cancel-after-chunk") == pytest.approx(2.0)
    assert reservation.get("finalized") is not True
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_cancel_after_only_keepalive_pings_reconciles_to_input_cost(
    spend_counter_state,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-cancel-after-ping"
    )

    async def cancel_after_ping(user_api_key_dict, response, request_data):
        yield STREAM_SSE_KEEPALIVE_PING_BYTES
        raise asyncio.CancelledError()

    generator, streaming_logging_obj = _drive_streaming_cancel(valid_token, cancel_after_ping)
    received = []

    async def _drain():
        async for chunk in generator:
            received.append(chunk)

    with pytest.raises(asyncio.CancelledError):
        await _drain()

    assert received == [STREAM_SSE_KEEPALIVE_PING_BYTES]
    assert counter_cache.get_local("spend:key:key-cancel-after-ping") == pytest.approx(0.5)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_streaming_cancel_while_holding_back_provider_output_keeps_reservation(
    spend_counter_state,
    monkeypatch: pytest.MonkeyPatch,
):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, reservation = await _reserve_for_stream(
        counter_cache, key_cache, proxy_logging_obj, "key-cancel-held-back"
    )

    held_back = AgenticAnthropicStreamingIterator(
        completion_stream=_never_ending_stream(),
        http_handler=MagicMock(),
        model="claude-haiku-4-5",
        messages=[],
        anthropic_messages_provider_config=MagicMock(),
        anthropic_messages_optional_request_params={},
        logging_obj=MagicMock(),
        custom_llm_provider="anthropic",
        kwargs={},
        hold_back=True,
        server_fulfilled_tool_names=frozenset({"headroom_retrieve"}),
        ping_interval_seconds=0.01,
    )
    router = Router(
        model_list=[
            {
                "model_name": "claude-haiku-4-5",
                "litellm_params": {"model": "anthropic/claude-haiku-4-5", "api_key": "sk-test"},
            }
        ]
    )
    provider_response = AnthropicMessagesStreamingResponse(
        completion_stream=held_back, hidden_params={"additional_headers": {}}
    )
    monkeypatch.setattr(  # test-quality-ok: the provider call the router streams from
        litellm, "anthropic_messages", AsyncMock(return_value=provider_response)
    )
    response = await router.aanthropic_messages(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
        stream=True,
    )

    async def ping_then_cancel(user_api_key_dict, response, request_data):
        yield await response.__anext__()
        while not response.has_buffered_provider_output:
            yield await response.__anext__()
        raise asyncio.CancelledError()

    streaming_logging_obj = MagicMock()
    streaming_logging_obj.async_post_call_streaming_iterator_hook = ping_then_cancel
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect = AsyncMock()
    generator = ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
        response=response,
        user_api_key_dict=valid_token,
        request_data=_request_body(),
        proxy_logging_obj=streaming_logging_obj,
        serialize_chunk=lambda chunk: chunk,
        serialize_error=lambda exc: str(exc),
    )

    received = []

    async def _drain():
        async for chunk in generator:
            received.append(chunk)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(_drain(), timeout=5)

    assert received and received == [STREAM_SSE_KEEPALIVE_PING_BYTES] * len(received)
    assert counter_cache.get_local("spend:key:key-cancel-held-back") == pytest.approx(2.0)
    assert reservation.get("finalized") is not True


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
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect = AsyncMock()
    # On the slow path the per-chunk hook is awaited before the chunk is yielded
    # to the client; cancel there. Nothing has reached the client yet.
    streaming_logging_obj.async_post_call_streaming_hook = AsyncMock(side_effect=asyncio.CancelledError())

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

        async def _drain():
            async for chunk in generator:
                received.append(chunk)

        with pytest.raises(asyncio.CancelledError):
            await _drain()

    assert received == []
    # cancellation happened before any chunk reached the client, but the
    # provider already received the input -> reconcile to the input cost (0.5)
    assert counter_cache.get_local("spend:key:key-cancel-slowpath") == pytest.approx(0.5)
    assert reservation["finalized"] is True
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect.assert_awaited_once()


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

    generator, streaming_logging_obj = _drive_streaming_cancel(valid_token, two_chunks)

    # Client consumes one chunk, then disconnects. aclose() raises GeneratorExit
    # at the suspended yield, after the chunk already reached the client.
    first = await generator.__anext__()
    assert first == "data: a\n\n"
    await generator.aclose()

    # output was delivered, so the reservation must NOT be refunded
    assert counter_cache.get_local("spend:key:key-disconnect-after-chunk") == pytest.approx(2.0)
    assert reservation.get("finalized") is not True
    streaming_logging_obj.arelease_max_parallel_requests_on_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_slow_path_processes_and_yields_chunk(spend_counter_state):
    counter_cache, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token, _ = await _reserve_for_stream(counter_cache, key_cache, proxy_logging_obj, "key-slowpath-ok")

    async def one_chunk(user_api_key_dict, response, request_data):
        yield {"content": "hi"}

    streaming_logging_obj: Final = MagicMock(spec=ProxyLogging)
    streaming_logging_obj.async_post_call_streaming_iterator_hook = one_chunk
    streaming_logging_obj.async_post_call_streaming_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["response"])

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


def _tiered_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "dashscope/qwen3-max",
                "litellm_params": {"model": "dashscope/qwen3-max", "api_key": "sk-fake"},
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


def _body_with_content_size(model: str, content_chars: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "token " * (content_chars // 6)}],
        "max_tokens": 10,
    }


@pytest.mark.asyncio
async def test_reservation_tokenizes_the_prompt_once(spend_counter_state):
    """Tokenizing is the reservation path's dominant CPU cost, so a request is
    tokenized once no matter how many cost estimates and pricing candidates it
    is priced against. The max-cost and input-cost estimates each used to
    re-tokenize the prompt, once per tiered-pricing candidate."""
    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-tokenize-once", spend=0.0, max_budget=100.0)
    request_body = _body_with_content_size("dashscope/qwen3-max", 600)
    real_token_counter = litellm.token_counter
    calls = []

    def counting_token_counter(**kwargs):
        calls.append(kwargs)
        return real_token_counter(**kwargs)

    with patch.object(litellm, "token_counter", counting_token_counter):
        reservation = await reserve_budget_for_request(
            request_body=request_body,
            route="/chat/completions",
            llm_router=_tiered_router(),
            valid_token=valid_token,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    assert reservation is not None
    assert reservation["reserved_cost"] > 0
    assert reservation["input_cost"] > 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_large_prompt_is_tokenized_off_the_event_loop(spend_counter_state):
    """Counting a large prompt inline blocks the event loop for the whole count,
    stalling every other request the worker is serving."""
    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-offloaded", spend=0.0, max_budget=100.0)
    request_body = _body_with_content_size("gpt-4o-mini", TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS + 6000)
    threads = []

    def recording_token_counter(**kwargs):
        threads.append(threading.current_thread())
        return 1000

    with patch.object(litellm, "token_counter", recording_token_counter):
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
    assert threads
    assert all(thread is not threading.main_thread() for thread in threads)


def _values_only_size(value: object) -> int:
    """The keys-ignoring walk the fixture below is sized to defeat"""
    if isinstance(value, Mapping):
        return sum(_values_only_size(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_values_only_size(item) for item in value)
    return len(value) if isinstance(value, str) else 0


_TOOL_PROPERTY_NAME_PREFIX = "service_metric_name_segment_" * 3


def _key_heavy_tool(index: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": f"lookup_service_metric_{index}",
            "parameters": {
                "type": "object",
                "properties": {
                    f"{_TOOL_PROPERTY_NAME_PREFIX}{index}_{field}": {"type": "string"} for field in range(24)
                },
            },
        },
    }


def _body_with_key_heavy_tool_schema(model: str) -> dict:
    """A tool schema whose bulk is property names rather than property values"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "which service is slow?"}],
        "tools": [_key_heavy_tool(index) for index in range(24)],
        "max_tokens": 10,
    }


@pytest.mark.asyncio
async def test_large_tool_schema_is_tokenized_off_the_event_loop(spend_counter_state):
    """Tool-schema property names are tokenized like any other text. Sizing a
    request by its values alone hides a large schema below the threshold, so it
    gets counted inline and stalls the loop the threshold exists to spare."""
    body = _body_with_key_heavy_tool_schema("gpt-4o-mini")
    assert _values_only_size(body["tools"]) < TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS
    assert _approximate_input_size(body) >= TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS

    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-tool-schema", spend=0.0, max_budget=100.0)
    threads = []

    def recording_token_counter(**kwargs):
        threads.append(threading.current_thread())
        return 1000

    with patch.object(litellm, "token_counter", recording_token_counter):
        reservation = await reserve_budget_for_request(
            request_body=body,
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
    assert threads
    assert all(thread is not threading.main_thread() for thread in threads)


@pytest.mark.asyncio
async def test_large_tool_choice_is_tokenized_off_the_event_loop(spend_counter_state):
    """tool_choice is handed to the tokenizer alongside the messages, so a
    request is only sized correctly if the heuristic covers it too."""
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "which service is slow?"}],
        "tool_choice": {
            "type": "function",
            "function": {"name": "lookup_" + "service_metric_" * 3000},
        },
        "max_tokens": 10,
    }
    assert _approximate_input_size(body) >= TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS

    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-tool-choice", spend=0.0, max_budget=100.0)
    threads = []

    def recording_token_counter(**kwargs):
        threads.append(threading.current_thread())
        return 1000

    with patch.object(litellm, "token_counter", recording_token_counter):
        reservation = await reserve_budget_for_request(
            request_body=body,
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
    assert threads
    assert all(thread is not threading.main_thread() for thread in threads)


@pytest.mark.asyncio
async def test_small_prompt_is_tokenized_inline(spend_counter_state):
    """A thread hand-off costs more than counting a small prompt"""
    _, key_cache = spend_counter_state
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-inline", spend=0.0, max_budget=100.0)
    threads = []

    def recording_token_counter(**kwargs):
        threads.append(threading.current_thread())
        return 10

    with patch.object(litellm, "token_counter", recording_token_counter):
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
    assert threads == [threading.main_thread()]


class _ModelAccessGroupBudgetPrisma:
    """Serves ``LiteLLM_ModelAccessGroupBudgetTable`` rows, recording what reached the database."""

    def __init__(self, **max_budget_by_group) -> None:
        self.rows = {
            group: SimpleNamespace(
                access_group_name=group,
                spend=7.0,
                litellm_budget_table=None
                if max_budget is None
                else SimpleNamespace(max_budget=max_budget, budget_limits=None),
            )
            for group, max_budget in max_budget_by_group.items()
        }
        self.batches = []
        self.db = SimpleNamespace(litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=self._find_many))

    async def _find_many(self, **kwargs):
        requested = list(kwargs["where"]["access_group_name"]["in"])
        self.batches.append(requested)
        return [self.rows[group] for group in requested if group in self.rows]


@pytest.mark.asyncio
async def test_model_access_group_counter_blocks_a_request_over_the_group_budget(spend_counter_state):
    """End to end through the reservation path, which is what runs when reservations are enabled."""
    counter_cache, key_cache = spend_counter_state
    prisma_client = _ModelAccessGroupBudgetPrisma(premium=1.0)
    valid_token = UserAPIKeyAuth(api_key="hashed", token="tok", matched_model_access_groups=["premium"])

    with patch(  # test-quality-ok: reserve_budget_for_request takes no estimator, so pinning the estimate needs this attribute
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=0.5,
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await reserve_budget_for_request(
                request_body=_request_body(),
                route="/chat/completions",
                llm_router=None,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=prisma_client,
                user_api_key_cache=key_cache,
                proxy_logging_obj=ProxyLogging(user_api_key_cache=key_cache),
            )

    assert exc_info.value.entity_id == "premium"
    assert exc_info.value.entity_type == Litellm_EntityType.MODEL_ACCESS_GROUP.value


async def _cache_model_access_group_budget(key_cache, group, spend, max_budget=None):
    await key_cache.async_set_cache(
        key=model_access_group_cache_key(group),
        value=ModelAccessGroupBudget(access_group_name=group, spend=spend, max_budget=max_budget),
        model_type=ModelAccessGroupBudget,
    )


async def _reserve_for_model_access_groups(key_cache, groups, estimate):
    """Reserve against the given groups, whose rows are already cached, so nothing hits the DB."""
    with patch(  # test-quality-ok: reserve_budget_for_request takes no estimator, so pinning the estimate needs this attribute
        "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
        return_value=estimate,
    ):
        return await reserve_budget_for_request(
            request_body=_request_body(),
            route="/chat/completions",
            llm_router=None,
            valid_token=UserAPIKeyAuth(
                api_key="hashed", token="tok-mag-counter", matched_model_access_groups=list(groups)
            ),
            team_object=None,
            user_object=None,
            prisma_client=_ModelAccessGroupBudgetPrisma(),
            user_api_key_cache=key_cache,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=key_cache),
        )


@pytest.mark.asyncio
async def test_reserved_model_access_group_is_not_charged_twice(spend_counter_state):
    """The reservation already wrote this counter, so the post-call pass has to skip it."""
    counter_cache, key_cache = spend_counter_state
    await _cache_model_access_group_budget(key_cache, "premium", spend=1.0, max_budget=25.0)

    reservation = await _reserve_for_model_access_groups(key_cache, ["premium"], estimate=0.6)
    counter_key = model_access_group_counter("premium")
    assert counter_cache.get_local(counter_key) == pytest.approx(1.6)

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        model_access_groups=["premium"],
    )

    # 1.0 recorded + the reservation reconciled down to the 0.2 actually spent. A second
    # increment would land at 1.4.
    assert counter_cache.get_local(counter_key) == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_unreserved_model_access_group_is_charged_alongside_a_reserved_one(spend_counter_state):
    """A budgetless group reserves nothing, so only the post-call pass can charge it.

    Both groups authorized the request and both get debited, each exactly once, whether or not
    the reservation path happened to hold a counter for them.
    """
    counter_cache, key_cache = spend_counter_state
    await _cache_model_access_group_budget(key_cache, "premium", spend=1.0, max_budget=25.0)
    await _cache_model_access_group_budget(key_cache, "starter", spend=4.0)

    reservation = await _reserve_for_model_access_groups(key_cache, ["premium", "starter"], estimate=0.6)
    assert [entry["entity_id"] for entry in reservation["entries"]] == ["premium"]

    from litellm.proxy.proxy_server import increment_spend_counters

    await increment_spend_counters(
        token=None,
        team_id=None,
        user_id=None,
        response_cost=0.2,
        budget_reservation=reservation,
        model_access_groups=["premium", "starter", "starter", "premium"],
    )

    assert counter_cache.get_local(model_access_group_counter("premium")) == pytest.approx(1.2)
    assert counter_cache.get_local(model_access_group_counter("starter")) == pytest.approx(4.2)
