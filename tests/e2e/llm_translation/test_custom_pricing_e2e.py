"""Live e2e: a deployment's custom per-token pricing is loaded, billed, and isolated.

Each test registers the deployment(s) it needs through /model/new (deleted on
teardown) instead of relying on a statically configured model, so the check is
self-contained and never inherits pricing another suite or a stale config left on
the shared proxy. custom-priced-flash sets input/output rates deliberately far
above the canonical gemini price; the isolation sibling shares the same
gemini/gemini-2.5-flash backend but sets no override. Three behaviors are checked
independently:

- billing: a real call's logged cost breakdown charges input and output tokens at
  the custom rates, each component checked separately (a base-rate bill lands
  ~100x lower; a swapped input/output rate passes a total-only check but not this)
- reporting: /model/info surfaces those rates for the deployment
- isolation: the sibling keeps its own price; an override that leaks into the
  shared backend cost map (LIT-3897) misprices it, making the sibling's rate match
  the custom one and failing the isolation check
"""

import time

import pytest
from pydantic import BaseModel, RootModel

from e2e_config import unique_marker
from proxy_client import ProxyClient
from e2e_http import Success, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    LiteLLMParamsBody,
    ModelInfoEntry,
    SpendLogsParams,
)

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-flash"
GEMINI_API_KEY = "os.environ/GEMINI_API_KEY"
# Deliberately ~100x above canonical gemini-2.5-flash (input 3e-7 / output 2.5e-6)
# so an override that is ignored or under-applied bills at the base rate and fails.
CUSTOM_INPUT_RATE = 5e-05
CUSTOM_OUTPUT_RATE = 1e-04


class _CostBreakdown(BaseModel):
    input_cost: float | None = None
    output_cost: float | None = None


class _RowMetadata(BaseModel):
    cost_breakdown: _CostBreakdown | None = None


class _SpendRow(BaseModel):
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: _RowMetadata | None = None


class _SpendRows(RootModel[list[_SpendRow]]):
    pass


def _approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def _provision(
    proxy: ProxyClient,
    resources: ResourceManager,
    prefix: str,
    *,
    input_cost_per_token: float | None,
    output_cost_per_token: float | None,
) -> str:
    """Register a fresh gemini/gemini-2.5-flash deployment (deleted on teardown) and
    return its model name. With the cost fields set the deployment carries a custom
    pricing override; with them None it is a plain sibling on the same backend. The
    marker keeps the name unique so concurrent runs on the shared proxy never
    collide."""
    model_name = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=BACKEND_MODEL,
            api_key=GEMINI_API_KEY,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def _provision_custom_priced(
    proxy: ProxyClient, resources: ResourceManager
) -> str:
    return _provision(
        proxy,
        resources,
        "custom-priced-flash",
        input_cost_per_token=CUSTOM_INPUT_RATE,
        output_cost_per_token=CUSTOM_OUTPUT_RATE,
    )


def _model_info_entry(entries: list[ModelInfoEntry], model_name: str) -> ModelInfoEntry:
    for entry in entries:
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info; the override did not load")


def _poll_breakdown_row(proxy: ProxyClient, key: str, response_id: str | None) -> _SpendRow:
    """Poll /spend/logs until the call's row lands with a cost breakdown (rows
    flush ~60s behind the call via proxy_batch_write_at)."""
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        result = proxy.transport.get(
            "/spend/logs",
            headers=proxy.transport.master,
            params=SpendLogsParams(api_key=key),
            response_type=_SpendRows,
        )
        match result:
            case Success(data=data):
                rows = data.root
            case _:
                rows = []
        priced = [
            row
            for row in rows
            if row.metadata
            and row.metadata.cost_breakdown
            and row.metadata.cost_breakdown.input_cost is not None
        ]
        for row in priced:
            if response_id and row.request_id == response_id:
                return row
        if priced and response_id is None:
            return priced[0]
        time.sleep(proxy.poll_interval)
    pytest.fail("no spend row with a cost breakdown landed before the deadline")


class TestCustomPricing:
    def test_custom_pricing_is_billed_at_configured_rate(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        scoped_key: str,
    ) -> None:
        model = _provision_custom_priced(proxy, resources)

        chat = unwrap(
            proxy.chat(
                scoped_key,
                ChatBody(
                    model=model,
                    messages=[
                        ChatMessage(
                            role="user", content=f"reply with one word {unique_marker()}"
                        )
                    ],
                    max_tokens=16,
                ),
            )
        )

        row = _poll_breakdown_row(proxy, scoped_key, chat.id)
        assert row.metadata and row.metadata.cost_breakdown  # guaranteed by the poll
        breakdown = row.metadata.cost_breakdown

        prompt = row.prompt_tokens or 0
        completion = row.completion_tokens or 0
        assert prompt > 0 and completion > 0, f"call tokens not logged on the row: {row}"

        input_cost = breakdown.input_cost
        output_cost = breakdown.output_cost
        assert input_cost is not None and output_cost is not None, (
            f"row cost breakdown missing input/output cost: {breakdown}"
        )
        assert _approx_equal(input_cost, prompt * CUSTOM_INPUT_RATE), (
            f"input_cost {input_cost} != {prompt} tokens * {CUSTOM_INPUT_RATE} "
            f"= {prompt * CUSTOM_INPUT_RATE}"
        )
        assert _approx_equal(output_cost, completion * CUSTOM_OUTPUT_RATE), (
            f"output_cost {output_cost} != {completion} tokens * {CUSTOM_OUTPUT_RATE} "
            f"= {completion * CUSTOM_OUTPUT_RATE}"
        )

    def test_model_info_reports_custom_pricing(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = _provision_custom_priced(proxy, resources)
        entry = _model_info_entry(proxy.model_info(), model)

        assert entry.litellm_params.input_cost_per_token == CUSTOM_INPUT_RATE, (
            f"/model/info litellm_params input rate "
            f"{entry.litellm_params.input_cost_per_token} != configured {CUSTOM_INPUT_RATE}"
        )
        assert entry.litellm_params.output_cost_per_token == CUSTOM_OUTPUT_RATE, (
            f"/model/info litellm_params output rate "
            f"{entry.litellm_params.output_cost_per_token} != configured {CUSTOM_OUTPUT_RATE}"
        )

    def test_custom_pricing_is_isolated_from_sibling_deployment(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        # Register the override first so its rate is in the backend cost map before
        # the sibling resolves; a leak (LIT-3897) would then poison the sibling.
        custom = _provision_custom_priced(proxy, resources)
        sibling = _provision(
            proxy,
            resources,
            "base-flash",
            input_cost_per_token=None,
            output_cost_per_token=None,
        )

        entries = {entry.model_name: entry for entry in proxy.model_info()}
        custom_entry = entries.get(custom)
        sibling_entry = entries.get(sibling)
        assert custom_entry is not None, f"{custom} absent from /model/info"
        assert sibling_entry is not None, f"{sibling} absent from /model/info"

        # custom-priced-flash overrides pricing; the sibling shares the same
        # gemini/gemini-2.5-flash backend but sets no override, so it must keep its
        # own price. Equal rates mean the override leaked into the shared cost map.
        assert (
            sibling_entry.model_info.input_cost_per_token
            != custom_entry.model_info.input_cost_per_token
        ), (
            f"{sibling} input rate {sibling_entry.model_info.input_cost_per_token} matches "
            f"{custom}'s override {custom_entry.model_info.input_cost_per_token}; "
            f"per-deployment custom pricing is not isolated"
        )
