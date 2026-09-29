"""Cost-accounting helpers for the spend-tracking suite: the /spend/logs row shape
that carries the per-component cost breakdown, a poll that waits for it, and the
builders the cache-pricing tests share.

The shared SpendLogRow deliberately stays thin (most tests only read totals), so
the component-cost tests model the metadata they assert on here instead:
`metadata.cost_breakdown` (input/output/cache-read/cache-creation/reasoning costs
plus the service-tier pricing basis) and `metadata.additional_usage_values` (the
cache token counts the biller derived from the provider's usage).

Determinism strategy: every test registers its own deployment with explicit custom
rates for each component it asserts on (`register_priced_model`), so expected cost
is exactly tokens-on-the-row times configured rate, immune to provider price
changes. The rates are chosen ~100x above canonical and distinct from one another,
so a component billed at the wrong rate can never accidentally match.

OpenAI prompt caching is implicit and keyed on the exact token prefix, with a
1024-token minimum. `cacheable_prefix` builds a prefix whose first word is the
run's unique marker: unique marker = the whole prefix is novel (a fresh cache
write), same marker + different question = a cache read that still misses the
proxy's own response cache. How long the prefix has to be before the provider
actually reports a read varies by model, so callers pass `words` to suit theirs.

Two facts about the recorded bill that the assertions here encode, because the
two surfaces disagree on purpose. On the spend row, `input_cost` is gross: it
already contains the cache-read and cache-creation costs, so the row's total is
input + output + tool-usage and the fresh-token cost is input minus the two cache
components. In the response headers, `x-litellm-response-cost-input` is net of
cache, which is what makes the component headers sum to the total.
"""

import time
from collections.abc import Callable

from pydantic import BaseModel, RootModel

from e2e_config import unique_marker
from e2e_http import Success
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogsParams
from proxy_client import ProxyClient


class CostBreakdownRow(BaseModel):
    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_creation_cost: float | None = None
    reasoning_cost: float | None = None
    tool_usage_cost: float | None = None
    total_cost: float | None = None
    service_tier: str | None = None


class AdditionalUsageValues(BaseModel):
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class CostRowMetadata(BaseModel):
    cost_breakdown: CostBreakdownRow | None = None
    additional_usage_values: AdditionalUsageValues | None = None


class CostRow(BaseModel):
    request_id: str | None = None
    spend: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: CostRowMetadata | None = None

    @property
    def breakdown(self) -> CostBreakdownRow:
        assert self.metadata and self.metadata.cost_breakdown, (
            f"spend row {self.request_id} landed without a cost breakdown"
        )
        return self.metadata.cost_breakdown

    @property
    def cache_read_tokens(self) -> int:
        if self.metadata and self.metadata.additional_usage_values:
            return self.metadata.additional_usage_values.cache_read_input_tokens or 0
        return 0

    @property
    def cache_creation_tokens(self) -> int:
        if self.metadata and self.metadata.additional_usage_values:
            return self.metadata.additional_usage_values.cache_creation_input_tokens or 0
        return 0


class CostRows(RootModel[list[CostRow]]):
    pass


def approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def assert_total_is_sum_of_components(row: CostRow) -> None:
    """The row's total is input + output + tool usage. The cache components are
    already inside the gross input cost, so adding them again would double-bill."""
    breakdown = row.breakdown
    components = sum(
        cost or 0.0
        for cost in (breakdown.input_cost, breakdown.output_cost, breakdown.tool_usage_cost)
    )
    assert breakdown.total_cost is not None and approx_equal(breakdown.total_cost, components), (
        f"total_cost {breakdown.total_cost} != input + output + tool usage ({components}): {breakdown}"
    )
    assert row.spend is not None and approx_equal(row.spend, breakdown.total_cost), (
        f"row spend {row.spend} != breakdown total {breakdown.total_cost}"
    )


def assert_fresh_tokens_billed_at(row: CostRow, input_rate: float) -> None:
    """Strip the cache components out of the gross input cost and what is left must
    be the freshly-read tokens at the deployment's input rate."""
    breakdown = row.breakdown
    fresh_tokens = (row.prompt_tokens or 0) - row.cache_read_tokens - row.cache_creation_tokens
    fresh_cost = (
        (breakdown.input_cost or 0.0)
        - (breakdown.cache_read_cost or 0.0)
        - (breakdown.cache_creation_cost or 0.0)
    )
    assert breakdown.input_cost is not None and approx_equal(fresh_cost, fresh_tokens * input_rate), (
        f"input_cost {breakdown.input_cost} less cache read {breakdown.cache_read_cost} and "
        f"cache creation {breakdown.cache_creation_cost} leaves {fresh_cost}, not "
        f"{fresh_tokens} fresh tokens * {input_rate} (prompt {row.prompt_tokens}, "
        f"cache read {row.cache_read_tokens}, cache creation {row.cache_creation_tokens}); "
        "cached tokens are being billed at the input rate"
    )


def poll_cost_row(proxy: ProxyClient, request_id: str) -> CostRow | None:
    """Poll /spend/logs for the call's row until it lands with a cost breakdown
    (rows flush ~60s behind the call via proxy_batch_write_at); None on timeout."""
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        result = proxy.transport.get(
            "/spend/logs",
            headers=proxy.transport.master,
            params=SpendLogsParams(request_id=request_id),
            response_type=CostRows,
        )
        match result:
            case Success(data=data):
                rows = data.root
            case _:
                rows = []
        for row in rows:
            if row.metadata and row.metadata.cost_breakdown:
                return row
        time.sleep(proxy.poll_interval)
    return None


def poll_cost_row_where(
    proxy: ProxyClient, api_key: str, predicate: Callable[[CostRow], bool]
) -> CostRow | None:
    """Poll the key's own /spend/logs until one of its rows carries a cost breakdown
    the predicate accepts; None on timeout. For calls whose response id is not the
    id the bill is filed under, which is how a user finds the row in the UI anyway."""
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        result = proxy.transport.get(
            "/spend/logs",
            headers=proxy.transport.master,
            params=SpendLogsParams(api_key=api_key),
            response_type=CostRows,
        )
        match result:
            case Success(data=data):
                rows = data.root
            case _:
                rows = []
        for row in rows:
            if row.metadata and row.metadata.cost_breakdown and predicate(row):
                return row
        time.sleep(proxy.poll_interval)
    return None


def register_priced_model(
    proxy: ProxyClient,
    resources: ResourceManager,
    name_prefix: str,
    litellm_params: LiteLLMParamsBody,
) -> str:
    """Register a deployment with explicit custom rates (deleted on teardown) and
    return its unique model name."""
    model_name = f"{name_prefix}-{unique_marker()}"
    model_id = proxy.create_model(model_name, litellm_params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def cacheable_prefix(marker: str, *, words: int = 1200) -> str:
    """A prompt prefix above OpenAI's 1024-token caching minimum whose identity is
    fully determined by `marker` (it is the first word, and prefix caching matches
    from token zero). Raise `words` for models that only report a cache read on a
    substantially longer prefix."""
    return " ".join(marker if i == 0 else f"token{i:04d}" for i in range(words))
