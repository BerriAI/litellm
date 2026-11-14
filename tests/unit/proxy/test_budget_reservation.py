import asyncio
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

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
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters


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
