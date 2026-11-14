import asyncio
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.types.caching import RedisPipelineIncrementOperation
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
from litellm.proxy.common_utils.reset_budget_job import _model_access_group_counter_key
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_model_access_group_budget_counters,
    estimate_request_max_cost,
    get_budget_window_start,
    invalidate_budget_reservation_counters,
    release_budget_reservation,
    release_budget_reservation_on_cancel,
    reserve_budget_for_request,
)
from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    _approximate_input_size,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from litellm.redis.stores.spend_counters import model_access_group_counter
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters, unavailable


@pytest.fixture()
def spend_counter_state():
    import litellm.proxy.proxy_server as ps

    original_counter_store = ps.spend_counter_store
    original_key_cache = ps.user_api_key_cache
    original_prisma_client = ps.prisma_client

    counter_cache = FakeSpendCounters(redis=None)
    key_cache = UserApiKeyCache()
    ps.spend_counter_store = counter_cache.store()
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

    assert counter_cache.get_local("spend:end_user:customer-1") == pytest.approx(4.50)
    assert counter_cache.get_local("spend:tag:paid-tag") == pytest.approx(7.50)
    assert counter_cache.get_local("spend:tag:other-tag") == pytest.approx(2.50)


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

    assert counter_cache.get_local("spend:org:org-counter-with-budget") == pytest.approx(2.25)


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

    assert counter_cache.get_local("spend:org:org-counter-plain") == pytest.approx(2.25)


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


def test_should_start_window_without_reset_at_at_duration_boundary():
    before = datetime.now(timezone.utc) - timedelta(hours=1)

    window_start = get_budget_window_start({"budget_duration": "1h"})

    after = datetime.now(timezone.utc) - timedelta(hours=1)
    assert window_start is not None
    assert before <= window_start <= after


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

    async def fail_after_increment(pending):
        for item in pending:
            counter_cache.increment_local(item.counter_key, item.increment)
        raise RuntimeError("lost increment response")

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.5,
        ),
        patch(
            "litellm.proxy.proxy_server.run_spend_counter_pipeline",
            side_effect=fail_after_increment,
        ),
        patch(
            "litellm.proxy.proxy_server.invalidate_spend_counter",
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
    assert counter_cache.get_local("spend:key:key-budget-reserve-after-increment-failure") == pytest.approx(0.0)


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

    original_increment_local = counter_cache.increment_local
    fail_next_team_release = True

    def flaky_increment_local(counter: str, amount: float) -> float:
        nonlocal fail_next_team_release
        if counter == "spend:team:team-budget-partial-release" and amount < 0 and fail_next_team_release:
            fail_next_team_release = False
            raise RuntimeError("simulated counter failure")
        return original_increment_local(counter, amount)

    monkeypatch.setattr(counter_cache, "increment_local", flaky_increment_local)

    with pytest.raises(RuntimeError):
        await release_budget_reservation(reservation)

    assert counter_cache.get_local("spend:key:key-budget-partial-release") == pytest.approx(0.0)
    assert counter_cache.get_local("spend:team:team-budget-partial-release") == pytest.approx(0.4)

    await release_budget_reservation(reservation)

    assert counter_cache.get_local("spend:key:key-budget-partial-release") == pytest.approx(0.0)
    assert counter_cache.get_local("spend:team:team-budget-partial-release") == pytest.approx(0.0)


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

    original_increment_local = counter_cache.increment_local
    fail_key_cleanup = True

    def flaky_increment_local(counter: str, amount: float) -> float:
        nonlocal fail_key_cleanup
        if counter == "spend:key:key-budget-cleanup-failure" and amount < 0 and fail_key_cleanup:
            fail_key_cleanup = False
            raise RuntimeError("simulated cleanup failure")
        return original_increment_local(counter, amount)

    monkeypatch.setattr(counter_cache, "increment_local", flaky_increment_local)

    with (
        patch(
            "litellm.proxy.spend_tracking.budget_reservation.estimate_request_max_cost",
            return_value=0.4,
        ),
        patch("litellm.proxy.spend_tracking.budget_reservation.verbose_proxy_logger.exception") as mock_log_exception,
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

    assert counter_cache.get_local("spend:key:key-budget-cleanup-failure") is None
    assert counter_cache.get_local("spend:team:team-budget-cleanup-failure") == pytest.approx(0.3)
    mock_log_exception.assert_called()


class _ExpiringRedis(dict[str, object]):
    """A Redis keyspace with real wall-clock key expiry; every write gives the key a fresh TTL."""

    def __init__(self, ttl: float) -> None:
        super().__init__()
        self.ttl = ttl
        self.expires_at: dict[str, float] = {}

    def _evict_expired(self, key: str) -> None:
        if self.expires_at.get(key, float("inf")) <= time.monotonic():
            super().pop(key, None)
            self.expires_at.pop(key, None)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            self._evict_expired(key)
        return super().__contains__(key)

    def __getitem__(self, key: str) -> object:
        self._evict_expired(key)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # dict.get overloads
        self._evict_expired(key)
        return super().get(key, default)

    def __setitem__(self, key: str, value: object) -> None:
        super().__setitem__(key, value)
        self.expires_at[key] = time.monotonic() + self.ttl

    def setdefault(self, key: str, default: object = None) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # dict.setdefault overloads
        if key not in self:
            self[key] = default
        return self[key]


class _ExpiringSpendCounters(FakeSpendCounters):
    """Spend counters whose Redis keys expire, counting the lease renewals that keep them alive."""

    def __init__(self, ttl: float = 60.0, fail_first_refresh: bool = False) -> None:
        super().__init__(redis=_ExpiringRedis(ttl), redis_ttl=ttl)
        self.refresh_attempts = 0
        self.refresh_count = 0
        self.fail_first_refresh = fail_first_refresh

    def refresh_ttl(self, counter: str) -> asyncio.Future[bool]:
        self.refresh_attempts += 1
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        redis = self.redis
        assert isinstance(redis, _ExpiringRedis)
        if self.fail_first_refresh and self.refresh_attempts == 1:
            future.set_exception(unavailable("breaker_open"))
            return future
        if counter not in redis:
            future.set_result(False)
            return future
        self.refresh_count += 1
        redis[counter] = redis[counter]
        future.set_result(True)
        return future


def _install(counters: FakeSpendCounters) -> FakeSpendCounters:
    import litellm.proxy.proxy_server as ps

    ps.spend_counter_store = counters.store()
    return counters


@pytest.mark.asyncio
async def test_reservation_survives_redis_counter_ttl_while_request_in_flight(
    spend_counter_state,
):
    """A request that runs longer than the counter TTL must keep its reservation in Redis
    (so a concurrent request on any worker still sees it), and renewal must stop once the
    reservation is reconciled so an idle counter still expires on its own."""
    counter_cache, key_cache = spend_counter_state
    redis_cache = _install(_ExpiringSpendCounters(ttl=0.2))
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-lease", spend=0.0, max_budget=1.0)
    counter_key = "spend:key:key-lease"

    reservation = await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    assert reservation is not None

    await asyncio.sleep(0.5)
    assert redis_cache.redis.get(counter_key) == pytest.approx(0.6)
    concurrent = await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    assert concurrent is not None
    assert concurrent["reserved_cost"] == pytest.approx(0.4)

    await release_budget_reservation(reservation)
    await release_budget_reservation(concurrent)
    await asyncio.sleep(0.15)
    refreshes_after_release = redis_cache.refresh_count
    await asyncio.sleep(0.35)
    assert redis_cache.refresh_count == refreshes_after_release
    assert redis_cache.redis.get(counter_key) is None


@pytest.mark.asyncio
async def test_reservation_lease_keeps_renewing_after_transient_redis_failure(
    spend_counter_state,
):
    """One failed EXPIRE (Redis blip, open circuit breaker) must not end renewal for the
    rest of the request."""
    counter_cache, key_cache = spend_counter_state
    redis_cache = _install(_ExpiringSpendCounters(ttl=0.2, fail_first_refresh=True))
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-lease-blip", spend=0.0, max_budget=1.0)

    reservation = await _reserve(valid_token, 0.6, key_cache, proxy_logging_obj)
    assert reservation is not None

    await asyncio.sleep(0.5)
    assert redis_cache.refresh_attempts >= 3
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_reservation_lease_stops_when_request_task_ends_without_reconciling(
    spend_counter_state,
):
    """A request whose task ends without reconciling (client disconnect path that skips the
    cost callbacks) must not keep renewing: the counter falls back to its plain TTL instead of
    pinning the reservation until the request timeout."""
    counter_cache, key_cache = spend_counter_state
    redis_cache = _install(_ExpiringSpendCounters(ttl=0.2))
    proxy_logging_obj = ProxyLogging(user_api_key_cache=key_cache)
    valid_token = UserAPIKeyAuth(token="key-lease-orphan", spend=0.0, max_budget=1.0)
    counter_key = "spend:key:key-lease-orphan"

    reservation = await asyncio.create_task(_reserve(valid_token, 0.6, key_cache, proxy_logging_obj))
    assert reservation is not None
    assert reservation["finalized"] is False

    await asyncio.sleep(0.5)
    assert redis_cache.refresh_count == 0
    assert redis_cache.redis.get(counter_key) is None


class _TeamMembershipFloorDb:
    """Stands in for `prisma_client.db`: only the team-membership row exists and its spend is the DB floor."""

    def __init__(self, spend: float) -> None:
        self.spend = spend

    def __getattr__(self, table_name: str) -> SimpleNamespace:
        row = SimpleNamespace(spend=self.spend) if table_name == "litellm_teammembership" else None
        return SimpleNamespace(find_unique=AsyncMock(return_value=row))


@pytest.mark.asyncio
async def test_reconcile_after_redis_counter_expiry_keeps_request_cost_enforced(
    spend_counter_state,
):
    """Redis key expired mid-stream while the pod's in-memory copy still holds the
    reserved value: reconcile must reseed from the DB floor plus the settled cost
    instead of applying ``actual - reserved`` to the empty key."""
    import litellm.proxy.proxy_server as ps

    counter_key = "spend:team_member:user-expiry:team-expiry"
    redis_cache = _install(_ExpiringSpendCounters())
    redis_cache.set_local(counter_key, 0.6)

    reservation = {
        "reserved_cost": 0.6,
        "entries": [
            {
                "counter_key": counter_key,
                "entity_type": "TeamMember",
                "entity_id": "user-expiry:team-expiry",
                "reserved_cost": 0.6,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    with (
        patch.object(  # test-quality-ok: the reseed reads the DB floor through a Prisma client the test has no seam for
            ps.SpendCounterReseed, "from_db", AsyncMock(return_value=0.3)
        )
    ):
        await ps.increment_spend_counters(
            token="key-expiry",
            team_id="team-expiry",
            user_id="user-expiry",
            response_cost=0.05,
            budget_reservation=reservation,
        )

    assert redis_cache.redis[counter_key] == pytest.approx(0.35)
    assert redis_cache.get_local(counter_key) == pytest.approx(0.35)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reconcile_before_db_update_does_not_double_count_when_flush_lands_between_passes(
    spend_counter_state,
):
    """The early reconcile (before the spend row is enqueued) reseeds from a DB
    floor that cannot yet include this request. When the periodic flush commits
    the row before increment_spend_counters runs its second reconcile, the
    applied_adjustment early-return must keep the counter from adding the cost
    a second time."""
    import litellm.proxy.proxy_server as ps
    from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation

    counter_cache, _ = spend_counter_state
    counter_key = "spend:team_member:user-flush:team-flush"
    redis_cache = _install(_ExpiringSpendCounters())
    redis_cache.set_local(counter_key, 0.6)
    db_floor = _TeamMembershipFloorDb(spend=0.3)
    ps.prisma_client = SimpleNamespace(db=db_floor)

    reservation = {
        "reserved_cost": 0.6,
        "entries": [
            {
                "counter_key": counter_key,
                "entity_type": "TeamMember",
                "entity_id": "user-flush:team-flush",
                "reserved_cost": 0.6,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    await reconcile_budget_reservation(budget_reservation=reservation, actual_cost=0.05, finalize=False)

    assert redis_cache.redis[counter_key] == pytest.approx(0.35)
    assert reservation["entries"][0]["applied_adjustment"] == pytest.approx(-0.55)
    assert reservation["finalized"] is False

    db_floor.spend = 0.35
    await ps.increment_spend_counters(
        token="key-flush",
        team_id="team-flush",
        user_id="user-flush",
        response_cost=0.05,
        budget_reservation=reservation,
    )

    assert redis_cache.redis[counter_key] == pytest.approx(0.35)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reserved_counter_deleted_during_spend_write_is_reseeded_instead_of_going_negative(
    spend_counter_state,
):
    import litellm.proxy.proxy_server as ps
    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters

    counter_cache, _ = spend_counter_state
    counter_key = "spend:key:key-deleted-mid-write"
    redis_cache = _install(_ExpiringSpendCounters())
    redis_cache.redis[counter_key] = 0.6
    redis_cache.set_local(counter_key, 0.6)

    async def _delete_counter_while_persisting(**kwargs: object) -> bool:
        redis_cache.redis.pop(counter_key, None)
        redis_cache.delete_local(counter_key)
        return True

    proxy_logging_obj = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(side_effect=_delete_counter_while_persisting)
    reservation = {
        "reserved_cost": 0.6,
        "entries": [
            {
                "counter_key": counter_key,
                "entity_type": "Key",
                "entity_id": "key-deleted-mid-write",
                "reserved_cost": 0.6,
                "applied_adjustment": 0.0,
            }
        ],
        "finalized": False,
    }

    with (
        patch.object(  # test-quality-ok: the reseed reads the DB floor through a Prisma client the test has no seam for
            ps.SpendCounterReseed, "from_db", AsyncMock(return_value=0.3)
        )
    ):
        charged = await _update_database_and_spend_counters(
            proxy_logging_obj=proxy_logging_obj,
            increment_spend_counters=ps.increment_spend_counters,
            user_api_key="key-deleted-mid-write",
            user_id=None,
            end_user_id=None,
            team_id=None,
            org_id=None,
            kwargs={},
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_cost=0.05,
            budget_reservation=reservation,
        )

    assert charged is True
    assert redis_cache.redis[counter_key] == pytest.approx(0.35), redis_cache.store
    assert reservation["finalized"] is True


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


async def _model_access_group_counters(matched, **max_budget_by_group):
    return await _get_model_access_group_budget_counters(
        valid_token=UserAPIKeyAuth(api_key="hashed", matched_model_access_groups=matched),
        prisma_client=_ModelAccessGroupBudgetPrisma(**max_budget_by_group),
        user_api_key_cache=UserApiKeyCache(),
    )


@pytest.mark.asyncio
async def test_model_access_group_with_a_budget_reserves_against_the_reset_jobs_counter_key():
    counters = await _model_access_group_counters(["premium"], premium=25.0)

    assert len(counters) == 1
    counter = counters[0]
    assert counter.counter_key == _model_access_group_counter_key(SimpleNamespace(access_group_name="premium"))
    assert counter.source_cache_key == model_access_group_cache_key("premium")
    assert counter.max_budget == 25.0
    assert counter.fallback_spend == 7.0
    assert counter.entity_type == "Model access group"
    assert counter.entity_id == "premium"


@pytest.mark.asyncio
async def test_model_access_group_without_a_budget_reserves_nothing():
    assert await _model_access_group_counters(["premium"], premium=None) == []


@pytest.mark.asyncio
async def test_model_access_group_with_a_zero_budget_reserves_nothing():
    """Zero is how a budget is cleared, not a ceiling that blocks every request."""
    assert await _model_access_group_counters(["premium"], premium=0.0) == []


@pytest.mark.asyncio
async def test_model_access_group_counters_come_from_the_auth_object():
    """Auth already resolved which granted groups serve the model; re-deriving it here would drift."""
    assert await _model_access_group_counters(None, premium=25.0) == []


@pytest.mark.asyncio
async def test_repeated_model_access_group_reserves_once():
    counters = await _model_access_group_counters(["premium", "premium"], premium=25.0)

    assert [counter.entity_id for counter in counters] == ["premium"]


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
async def test_model_access_group_counter_accumulates_across_calls_without_a_reservation(spend_counter_state):
    """With reservations disabled nothing writes the counter up front, so the cost callback must.

    Otherwise the read-time budget check enforces against the DB row's spend, which the cache
    holds for the full TTL, and a caller runs past the ceiling for that whole window.
    """
    counter_cache, key_cache = spend_counter_state
    await _cache_model_access_group_budget(key_cache, "premium", spend=1.0, max_budget=25.0)

    from litellm.proxy.proxy_server import increment_spend_counters

    counter_key = model_access_group_counter("premium")

    await increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.25, model_access_groups=["premium"]
    )
    assert counter_cache.get_local(counter_key) == pytest.approx(1.25)

    await increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.75, model_access_groups=["premium", "premium", ""]
    )
    assert counter_cache.get_local(counter_key) == pytest.approx(2.0)
    assert counter_cache.get_local(model_access_group_counter("")) is None
