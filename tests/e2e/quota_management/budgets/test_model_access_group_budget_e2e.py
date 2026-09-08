"""Live e2e: one shared budget across every key that can reach a model access group.

A model access group is a free-text label on a deployment (`model_info.access_groups`),
and a key is granted the group by name. The budget hangs off the group, not the key, so
the interesting behaviors are the ones a per-key budget cannot produce: a key that has
spent nothing of its own is refused once somebody else drained the pool, and draining one
group leaves a second group untouched, because a request is only charged to the groups
the caller was granted that also serve the model being called.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest

from budget_client import BudgetClient, is_budget_block
from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from models import KeyGenerateBody, LiteLLMParamsBody, ModelInfoBody, ModelNewBody

pytestmark = pytest.mark.e2e

BACKEND: Final = "openai/gpt-5.4-nano"
TINY_BUDGET: Final = 5e-6
MAX_TOKENS: Final = 16
DRAIN_TIMEOUT_SECONDS: Final = 180


@dataclass(frozen=True, slots=True)
class DrainedPool:
    """A model access group whose shared budget has been spent to exhaustion, the
    deployment inside it, the key that did the spending, and a second group holding
    its own deployment that was never given a budget at all."""

    access_group: str
    model: str
    spender_key: str
    free_access_group: str
    free_model: str


def _provider_key(env_var: str) -> str:
    return os.environ.get(env_var) or f"os.environ/{env_var}"


def _grouped_model(model_name: str, access_group: str) -> ModelNewBody:
    return ModelNewBody(
        model_name=model_name,
        litellm_params=LiteLLMParamsBody(model=BACKEND, api_key=_provider_key("OPENAI_API_KEY")),
        model_info=ModelInfoBody(access_groups=[access_group]),
    )


def _call(client: BudgetClient, key: str, model: str) -> StreamingResponse:
    return client.chat(key, model, f"hi {unique_marker()}", max_tokens=MAX_TOKENS)


def _drain(client: BudgetClient, key: str, model: str, access_group: str) -> None:
    """Spend the group's pool until the proxy refuses the next request. The first call
    lands under the cap and the block comes from the spend it recorded, so this needs at
    least one round trip through the spend writer, not just one request."""
    deadline: Final = time.monotonic() + DRAIN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = _call(client, key, model)
        if is_budget_block(result):
            return
        require_successful_call(result)
        time.sleep(1)
    pytest.fail(f"budget on model access group {access_group!r} never blocked a request")


@pytest.fixture(scope="module")
def drained(client: BudgetClient) -> Iterator[DrainedPool]:
    marker: Final = unique_marker()
    pool: Final = DrainedPool(
        access_group=f"e2e-mag-budget-{marker}",
        model=f"e2e-mag-budgeted-{marker}",
        spender_key=client.proxy.generate_key(KeyGenerateBody(models=[f"e2e-mag-budget-{marker}"])),
        free_access_group=f"e2e-mag-free-{marker}",
        free_model=f"e2e-mag-unbudgeted-{marker}",
    )
    created: Final = (
        client.proxy.register_model(_grouped_model(pool.model, pool.access_group)),
        client.proxy.register_model(_grouped_model(pool.free_model, pool.free_access_group)),
    )
    try:
        client.set_access_group_budget(pool.access_group, max_budget=TINY_BUDGET)
        _drain(client, pool.spender_key, pool.model, pool.access_group)
        yield pool
    finally:
        client.delete_access_group_budget(pool.access_group)
        client.proxy.delete_key(pool.spender_key)
        for model_id in created:
            client.proxy.delete_model(model_id)


class TestModelAccessGroupBudget:
    @pytest.mark.covers("quota_management.budget.model_access_group.blocks_over_limit")
    def test_the_key_that_drained_the_pool_stays_blocked(
        self, client: BudgetClient, drained: DrainedPool
    ) -> None:
        result = _call(client, drained.spender_key, drained.model)
        assert is_budget_block(result), (
            f"an exhausted pool served {drained.model!r} again: {result.status_code} {result.body[:300]}"
        )
        assert drained.access_group in result.body, (
            f"the block did not name the group that caused it: {result.body[:300]}"
        )

    @pytest.mark.covers("quota_management.budget.model_access_group.enforced_across_keys")
    def test_a_key_that_spent_nothing_is_blocked_by_the_shared_pool(
        self, client: BudgetClient, resources: ResourceManager, drained: DrainedPool
    ) -> None:
        newcomer = resources.key(models=[drained.access_group])

        result = _call(client, newcomer, drained.model)

        assert is_budget_block(result), (
            "a freshly minted key with no spend of its own was served by an exhausted "
            f"shared pool: {result.status_code} {result.body[:300]}"
        )

    @pytest.mark.covers("quota_management.budget.model_access_group.isolates_per_group")
    def test_a_drained_group_does_not_block_a_different_group(
        self, client: BudgetClient, resources: ResourceManager, drained: DrainedPool
    ) -> None:
        other = resources.key(models=[drained.free_access_group])

        result = _call(client, other, drained.free_model)

        assert not is_budget_block(result), (
            f"{drained.free_access_group!r} has no budget of its own but was blocked by "
            f"{drained.access_group!r}'s exhausted pool: {result.body[:300]}"
        )
        require_successful_call(result)

    @pytest.mark.covers("quota_management.budget.model_access_group.reports_spend")
    def test_the_budget_read_reports_the_spend_drawn_against_the_pool(
        self, client: BudgetClient, drained: DrainedPool
    ) -> None:
        """Enforcement runs off a live counter while the group's row is written by the
        batched spend writer, so the recorded spend an admin reads lands a beat after the
        block. Poll for it: what matters is that it arrives and matches the pool."""
        deadline = time.monotonic() + client.proxy.poll_timeout
        reported = client.access_group_budget(drained.access_group)
        while reported.spend < TINY_BUDGET and time.monotonic() < deadline:
            time.sleep(client.proxy.poll_interval)
            reported = client.access_group_budget(drained.access_group)

        assert reported.budget is not None, "the group lost the budget that just blocked it"
        assert reported.budget.max_budget == TINY_BUDGET
        assert reported.spend >= TINY_BUDGET, (
            f"the pool blocked at {TINY_BUDGET} but only {reported.spend} was ever recorded "
            f"against the group within {client.proxy.poll_timeout}s"
        )
