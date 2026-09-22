"""Live e2e: an over-budget key stays blocked while hundreds of other budget scopes take traffic (#40221).

The proxy keeps one in-memory spend counter per budget scope (key, team, end user, tag, ...)
in ``spend_counter_cache``. Every request that names a new ``user`` creates a new end-user
counter. When that cache is sized like the generic in-memory cache (200 entries), the 201st
active scope evicts a still-valid counter, and the next request on the evicted key reseeds it
from the DB. The DB row lags the counter until ``proxy_batch_write_at`` flushes, so a key the
proxy had just rejected for being over budget is admitted again.

The test blocks a tiny-budget key with a real call, then drives more than 200 distinct end users
on a second key, then re-probes the blocked key. The invariant (blocked stays blocked) must hold
on every stack. The eviction itself is only reachable on a proxy without Redis whose DB row still
lags (``proxy_batch_write_at`` above the churn time); with Redis the cross-pod counter answers
first, and a flushed row reseeds correctly, so those stacks re-prove the plain budget block.
The probe key's DB spend at re-probe time is reported in the failure message so a reader can
tell which path was exercised.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "claude-haiku-4-5"
TINY_BUDGET = 0.000001
GENERIC_CACHE_SIZE = 200
CHURN_SCOPES = 250
CHURN_WORKERS = 24


def _chat(client: BudgetClient, key: str, user: str) -> StreamingResponse:
    return client.chat(key, MODEL, f"scope churn {unique_marker()}", max_tokens=1, user=user)


class TestSpendCounterScopeChurn:
    @pytest.mark.covers("quota_management.budget.spend_counter.survives_scope_churn")
    def test_over_budget_key_stays_blocked_after_hundreds_of_other_scopes(
        self, client: BudgetClient, resources: ResourceManager
    ) -> None:
        run = unique_marker()
        probe_key = client.generate_key(max_budget=TINY_BUDGET, models=[MODEL])
        resources.defer(lambda: client.delete_key(probe_key))
        churn_key = client.generate_key(models=[MODEL])
        resources.defer(lambda: client.delete_key(churn_key))
        churn_users = [f"churn-{run}-{index}" for index in range(CHURN_SCOPES)]
        resources.defer(lambda: client.delete_customers(churn_users))

        first = _chat(client, probe_key, user=f"probe-{run}")
        assert first.ok, f"first call on a fresh key must be admitted: {first.status_code} {first.body[:300]}"
        blocked = _chat(client, probe_key, user=f"probe-{run}")
        assert is_budget_block(blocked), (
            f"precondition: the key must be over budget before the churn: {blocked.status_code} {blocked.body[:300]}"
        )

        def churn_call(user: str) -> StreamingResponse:
            return _chat(client, churn_key, user=user)

        with ThreadPoolExecutor(max_workers=CHURN_WORKERS) as pool:
            churn = list(pool.map(churn_call, churn_users))
        admitted = sum(1 for r in churn if r.ok)
        assert admitted > GENERIC_CACHE_SIZE, (
            f"only {admitted}/{CHURN_SCOPES} churn calls succeeded; cannot exceed {GENERIC_CACHE_SIZE} live scopes"
        )

        db_spend = client.proxy.key_info(probe_key).spend or 0.0
        after = _chat(client, probe_key, user=f"probe-{run}")
        assert is_budget_block(after), (
            f"over-budget key was admitted after {admitted} other scopes took traffic (DB spend {db_spend}): "
            f"{after.status_code} {after.body[:300]}; its spend counter was evicted and reseeded from lagging "
            "DB spend (#40221)"
        )
