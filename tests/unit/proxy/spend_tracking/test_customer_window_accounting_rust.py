import asyncio
from collections import OrderedDict
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.models.budget import BudgetLimit
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.spend_tracking.budget_reservation import (
    customer_window_counters,
    estimate_request_max_cost,
    release_budget_reservation,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerScopeAuth,
)
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.proxy.spend_tracking.native_budgets_replica import reserve_budget_for_request

pytestmark = pytest.mark.usefixtures("native_budgets_on")

LIMITS: Final = (
    BudgetLimit(budget_duration="1mo", max_budget=10),
    BudgetLimit(budget_duration="12mo", max_budget=100),
)


@pytest.fixture
def accounting_state(monkeypatch):
    backend: Final = SpendCounterStore(connect(None).spend_counters())
    writer: Final = DBSpendUpdateWriter()
    prisma: Final = MagicMock()
    prisma.spend_log_transactions = []
    prisma._spend_log_transactions_lock = asyncio.Lock()
    prisma.tool_usage_transactions = []
    prisma._tool_usage_transactions_lock = asyncio.Lock()
    prisma.db.litellm_budgetwindowspend.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(ps, "spend_counter_store", backend)
    monkeypatch.setattr(ps, "proxy_logging_obj", SimpleNamespace(db_spend_update_writer=writer))
    monkeypatch.setattr(ps, "prisma_client", prisma)
    monkeypatch.setattr(SpendCounterReseed, "_locks", OrderedDict())
    return backend, writer, prisma


@pytest.mark.asyncio
@pytest.mark.parametrize("limiting_duration", ("1mo", "12mo"))
async def test_concurrent_requests_obey_both_caps_with_existing_reservation_engine(accounting_state, limiting_duration):
    backend, _, prisma = accounting_state
    request_body: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1000,
    }
    estimated: Final = estimate_request_max_cost(request_body, "/chat/completions", None)
    assert estimated is not None and estimated > 0
    limits: Final = tuple(
        BudgetLimit(budget_duration=duration, max_budget=estimated * (1.5 if duration == limiting_duration else 100))
        for duration in ("1mo", "12mo")
    )
    hierarchy: Final = CustomerHierarchyAuth(
        customer_org_id="org",
        customer_member_id="member",
        scopes=(CustomerScopeAuth(key="customer_member", entity_id="member", budget_limits=limits),),
    )
    auth: Final = UserAPIKeyAuth().model_copy(update={"customer_hierarchy": hierarchy})
    cache: Final = UserApiKeyCache()
    logging: Final = ProxyLogging(user_api_key_cache=cache)

    async def reserve():
        return await reserve_budget_for_request(
            request_body=dict(request_body),
            route="/chat/completions",
            llm_router=None,
            valid_token=auth.model_copy(),
            team_object=None,
            user_object=None,
            prisma_client=prisma,
            user_api_key_cache=cache,
            proxy_logging_obj=logging,
            fail_closed_budget_enforcement=True,
        )

    results: Final = await asyncio.gather(reserve(), reserve(), return_exceptions=True)
    accepted: Final = tuple(result for result in results if isinstance(result, dict))
    rejected: Final = tuple(result for result in results if isinstance(result, litellm.BudgetExceededError))
    assert len(accepted) == len(rejected) == 1
    counters: Final = customer_window_counters("customer_member", "member", limits)
    assert all(
        backend.get_local(counter.counter_key) == pytest.approx(accepted[0]["reserved_cost"]) for counter in counters
    )
    await release_budget_reservation(accepted[0])
    assert all(backend.get_local(counter.counter_key) == pytest.approx(0.0) for counter in counters)
