"""Live e2e: a baseten deployment is priced from the registry cost map.

Registers a baseten/zai-org/GLM-5.3-Fast deployment through /model/new (deleted
on teardown) with no pricing override, so every rate the proxy reports and bills
comes from model_prices_and_context_window.json. Two behaviors are checked
independently:

- reporting: /model/info surfaces the registry input/output rates for the
  deployment (an unmapped model resolves no price and fails here)
- billing: a real call's logged cost breakdown charges input and output tokens
  at those registry rates, each component checked separately
"""

import os
import time
from typing import Final

import pytest
from pydantic import BaseModel, RootModel

from e2e_config import unique_marker
from e2e_http import Success
from lifecycle import ResourceManager
from models import (
    LiteLLMParamsBody,
    ModelInfoEntry,
    SpendLogsParams,
)
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

BACKEND_MODEL: Final = "baseten/zai-org/GLM-5.3-Fast"
BASETEN_API_KEY_ENV: Final = "BASETEN_API_KEY"


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
) -> str:
    """Register a fresh baseten GLM-5.3-Fast deployment (deleted on teardown) and
    return its model name. No pricing fields are set, so the deployment resolves
    its rates from the registry cost map. The marker keeps the name unique so
    concurrent runs on the shared proxy never collide."""
    assert os.environ.get(BASETEN_API_KEY_ENV), f"{BASETEN_API_KEY_ENV} must be set for this live e2e test"
    model_name = f"e2e-baseten-glm-5-3-fast-{unique_marker()}"
    model_id = proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=BACKEND_MODEL,
            api_key=f"os.environ/{BASETEN_API_KEY_ENV}",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name


def _model_info_entry(entries: list[ModelInfoEntry], model_name: str) -> ModelInfoEntry:
    for entry in entries:
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info; the deployment did not load")


def _poll_breakdown_row(proxy: ProxyClient, response_id: str) -> _SpendRow:
    """Poll /spend/logs until the call's row lands with a cost breakdown (rows
    flush ~60s behind the call via proxy_batch_write_at)."""
    deadline = time.monotonic() + proxy.poll_timeout
    while time.monotonic() < deadline:
        result = proxy.transport.get(
            "/spend/logs",
            headers=proxy.transport.master,
            params=SpendLogsParams(request_id=response_id),
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
            if row.metadata and row.metadata.cost_breakdown and row.metadata.cost_breakdown.input_cost is not None
        ]
        if priced:
            return priced[0]
        time.sleep(proxy.poll_interval)
    pytest.fail("no spend row with a cost breakdown landed before the deadline")


class TestBasetenGlm53FastPricing:
    def test_deployment_reports_registry_price(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model = _provision(proxy, resources)
        entry = _model_info_entry(proxy.model_info(), model)

        assert entry.model_info.input_cost_per_token is not None and (entry.model_info.input_cost_per_token > 0), (
            f"/model_info model_info input rate {entry.model_info.input_cost_per_token} "
            f"unset for {model}; the backend is not priced in the registry"
        )
        assert entry.model_info.output_cost_per_token is not None and (entry.model_info.output_cost_per_token > 0), (
            f"/model_info model_info output rate {entry.model_info.output_cost_per_token} "
            f"unset for {model}; the backend is not priced in the registry"
        )

    def test_completion_is_billed_at_registry_price(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        scoped_key: str,
        sdk: SdkClients,
    ) -> None:
        model = _provision(proxy, resources)
        entry = _model_info_entry(proxy.model_info(), model)
        input_rate = entry.model_info.input_cost_per_token
        output_rate = entry.model_info.output_cost_per_token
        assert input_rate is not None and input_rate > 0, f"registry input rate unset for {model}: {entry.model_info}"
        assert output_rate is not None and output_rate > 0, (
            f"registry output rate unset for {model}: {entry.model_info}"
        )

        completion = sdk.openai(scoped_key).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"reply with one word {unique_marker()}"}],
            max_tokens=16,
            extra_body=NO_PROXY_CACHE,
        )
        response_id = completion.id
        assert response_id, f"chat completion returned no id: {completion!r}"

        row = _poll_breakdown_row(proxy, response_id)
        assert row.metadata and row.metadata.cost_breakdown, f"poll returned a row without a breakdown: {row}"
        breakdown = row.metadata.cost_breakdown

        prompt = row.prompt_tokens or 0
        output = row.completion_tokens or 0
        assert prompt > 0 and output > 0, f"call tokens not logged on the row: {row}"

        input_cost = breakdown.input_cost
        output_cost = breakdown.output_cost
        assert input_cost is not None and output_cost is not None, (
            f"row cost breakdown missing input/output cost: {breakdown}"
        )
        assert _approx_equal(input_cost, prompt * input_rate), (
            f"input_cost {input_cost} != {prompt} tokens * {input_rate} = {prompt * input_rate}"
        )
        assert _approx_equal(output_cost, output * output_rate), (
            f"output_cost {output_cost} != {output} tokens * {output_rate} = {output * output_rate}"
        )
