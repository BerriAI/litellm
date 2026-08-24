"""Live e2e: a model's custom per-token pricing is loaded, billed, and isolated.

The gateway config declares ``custom-priced-flash`` (gemini-2.5-flash underneath)
with input/output rates deliberately far above the canonical gemini price, read
back here from the same config file. Three behaviors are checked independently:

- billing: a real call's logged cost breakdown charges input and output tokens at
  the custom rates, each component checked separately (a base-rate bill lands
  ~100x lower; a swapped input/output rate passes a total-only check but not this)
- reporting: /model/info surfaces those rates for the model
- isolation: gemini-2.5-flash shares the same underlying gemini/gemini-2.5-flash
  but sets no override, so it must keep its own price; an override that leaks into
  the shared cost map misprices it. A regression that reintroduces that leak makes
  the sibling's rate match the custom one and fails the isolation check.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, RootModel

from e2e_config import unique_marker
from e2e_http import Success, unwrap
from models import ChatBody, ChatMessage, CustomPricing, ModelInfoEntry, SpendLogsParams
from passthrough_client import PassthroughClient

pytestmark = pytest.mark.e2e

CUSTOM_MODEL = "custom-priced-flash"
BASE_MODEL = "gemini-2.5-flash"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "gateway" / "litellm-config.yml"


@dataclass(frozen=True, slots=True)
class _Rates:
    input_per_token: float
    output_per_token: float


class _ConfiguredModel(BaseModel):
    model_name: str
    litellm_params: CustomPricing


class _GatewayConfig(BaseModel):
    model_list: list[_ConfiguredModel]


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


def _configured_pricing(model_name: str) -> _Rates:
    """The custom rates declared for `model_name` in the gateway config the proxy
    runs with - the source of truth the billed and reported prices are checked
    against."""
    config = _GatewayConfig.model_validate(yaml.safe_load(CONFIG_PATH.read_text()))
    for entry in config.model_list:
        if entry.model_name == model_name:
            pricing = entry.litellm_params
            assert pricing.input_cost_per_token and pricing.output_cost_per_token, (
                f"{model_name} declares no custom per-token rates in {CONFIG_PATH.name}"
            )
            return _Rates(pricing.input_cost_per_token, pricing.output_cost_per_token)
    pytest.fail(f"{model_name} not found in {CONFIG_PATH.name}")


def _model_info_entry(
    entries: list[ModelInfoEntry], model_name: str
) -> ModelInfoEntry:
    for entry in entries:
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info; the override did not load")


def _poll_breakdown_row(
    client: PassthroughClient, key: str, response_id: str | None
) -> _SpendRow:
    """Poll /spend/logs until the call's row lands with a cost breakdown (rows
    flush ~60s behind the call via proxy_batch_write_at)."""
    deadline = time.monotonic() + client.gateway.poll_timeout
    while time.monotonic() < deadline:
        result = client.gateway.transport.get(
            "/spend/logs",
            headers=client.gateway.transport.master,
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
        time.sleep(client.gateway.poll_interval)
    pytest.fail("no spend row with a cost breakdown landed before the deadline")


def test_custom_pricing_is_billed_at_configured_rate(
    client: PassthroughClient, scoped_key: str
) -> None:
    rates = _configured_pricing(CUSTOM_MODEL)

    chat = unwrap(
        client.gateway.chat(
            scoped_key,
            ChatBody(
                model=CUSTOM_MODEL,
                messages=[
                    ChatMessage(
                        role="user", content=f"reply with one word {unique_marker()}"
                    )
                ],
                max_tokens=16,
            ),
        )
    )

    row = _poll_breakdown_row(client, scoped_key, chat.id)
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
    assert _approx_equal(input_cost, prompt * rates.input_per_token), (
        f"input_cost {input_cost} != {prompt} tokens * {rates.input_per_token} "
        f"= {prompt * rates.input_per_token}"
    )
    assert _approx_equal(output_cost, completion * rates.output_per_token), (
        f"output_cost {output_cost} != {completion} tokens * {rates.output_per_token} "
        f"= {completion * rates.output_per_token}"
    )


def test_model_info_reports_custom_pricing(client: PassthroughClient) -> None:
    rates = _configured_pricing(CUSTOM_MODEL)
    entry = _model_info_entry(client.gateway.model_info(), CUSTOM_MODEL)

    assert entry.litellm_params.input_cost_per_token == rates.input_per_token, (
        f"/model/info litellm_params input rate "
        f"{entry.litellm_params.input_cost_per_token} != configured "
        f"{rates.input_per_token}"
    )
    assert entry.litellm_params.output_cost_per_token == rates.output_per_token, (
        f"/model/info litellm_params output rate "
        f"{entry.litellm_params.output_cost_per_token} != configured "
        f"{rates.output_per_token}"
    )


def test_custom_pricing_is_isolated_from_sibling_deployment(
    client: PassthroughClient,
) -> None:
    entries = {entry.model_name: entry for entry in client.gateway.model_info()}
    custom = entries.get(CUSTOM_MODEL)
    base = entries.get(BASE_MODEL)
    assert custom is not None, f"{CUSTOM_MODEL} absent from /model/info"
    assert base is not None, f"{BASE_MODEL} absent from /model/info"

    # custom-priced-flash overrides pricing; gemini-2.5-flash shares the same
    # underlying gemini/gemini-2.5-flash but sets no override, so it must keep its
    # own price. Equal rates mean the override leaked into the shared cost map.
    assert (
        base.model_info.input_cost_per_token != custom.model_info.input_cost_per_token
    ), (
        f"{BASE_MODEL} input rate {base.model_info.input_cost_per_token} matches "
        f"{CUSTOM_MODEL}'s override {custom.model_info.input_cost_per_token}; "
        f"per-deployment custom pricing is not isolated"
    )
