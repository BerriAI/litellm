"""A streamed `/v1/messages` request served through the Responses bridge bills the same row as the plain request
when a cost margin or a cost discount is configured. The streamed terminal event carries litellm's own bill in
`usage.cost`, so the spend row must price the tokens the way the plain leg does instead of reading that bill back
as the provider's charge, which books input 0 / output total and applies the margin or discount a second time.

The stream_parity runner checks every leg against the suite's unadjusted expected row on the shared proxy, and a
margin or discount set there would reprice every other case running beside it, so each pricing boots its own proxy."""

from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.workers import cost_map_environment
from integration.cost_calculation.cases.openai import (
    GPT_6_LUNA_MESSAGES_BRIDGED_JSON,
    GPT_6_LUNA_MESSAGES_BRIDGED_STREAM,
)
from integration.cost_calculation.conftest import CostRow, poll_rows, register_scenario_deployment
from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, ExactExpected
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = [pytest.mark.timeout(240), pytest.mark.shares_worker_state]

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STOCK_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
_LEGS: Final[tuple[CostTrackingTestCase, ...]] = (GPT_6_LUNA_MESSAGES_BRIDGED_JSON, GPT_6_LUNA_MESSAGES_BRIDGED_STREAM)


@dataclass(frozen=True, slots=True)
class _Pricing:
    name: str
    litellm_settings: Mapping[str, JsonValue]
    spend_multiplier: float


PRICINGS: Final = (
    _Pricing("margin", MappingProxyType({"cost_margin_config": {"global": 0.5}}), 1.5),
    _Pricing("discount", MappingProxyType({"cost_discount_config": {"openai": 0.1}}), 0.9),
)


def _config(directory: Path, pricing: _Pricing) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(_STOCK_CONFIG.read_text()))
    settings: Final = _JSON_OBJECT.validate_python(stock["litellm_settings"])
    path: Final = directory / f"{pricing.name}-config.yaml"
    path.write_text(yaml.safe_dump({**stock, "litellm_settings": {**settings, **pricing.litellm_settings}}))
    return path


@pytest.fixture(scope="module")
def proxies(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Mapping[str, OwnedProxy]]:
    with gateway_from_environment() as gateway, ExitStack() as stack:
        overrides: Final = cost_map_environment(gateway.upstream_url)

        def boot(pricing: _Pricing) -> OwnedProxy:
            directory: Final = tmp_path_factory.mktemp(f"messages-bridged-stream-{pricing.name}-rig")
            return stack.enter_context(
                owned_proxy_process(
                    gateway,
                    directory,
                    overrides,
                    config=_config(directory, pricing),
                    remove_environment=("LITELLM_LOCAL_MODEL_COST_MAP",),
                )
            )

        yield MappingProxyType({pricing.name: boot(pricing) for pricing in PRICINGS})


def _bill_legs(gateway: Gateway) -> Mapping[str, CostRow]:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        deployments: Final = {
            leg.name: register_scenario_deployment(scenario, leg, sha256(leg.name.encode()).hexdigest()[:12], key)
            for leg in _LEGS
        }
        responses: Final = {
            leg.name: gateway.request(
                "POST", leg.endpoint, {**leg.request, "model": deployments[leg.name].model_name}, key=key
            )
            for leg in _LEGS
        }
        for leg in _LEGS:
            assert responses[leg.name].status_code == 200, (leg.name, responses[leg.name].text[:400])
        rows: Final = poll_rows(key, len(_LEGS))
        return MappingProxyType(
            {leg.name: next(row for row in rows if row.model_id == deployments[leg.name].identity) for leg in _LEGS}
        )


def _priced(row: CostRow) -> tuple[float | None, float | None, float | None, int | None, int | None]:
    breakdown: Final = row.breakdown
    assert breakdown is not None, row
    return (row.spend, breakdown.input_cost, breakdown.output_cost, row.prompt_tokens, row.completion_tokens)


@pytest.mark.parametrize("pricing", PRICINGS, ids=[pricing.name for pricing in PRICINGS])
def test_streamed_messages_bridged_row_matches_the_plain_row_under_a_margin_or_discount(
    proxies: Mapping[str, OwnedProxy], pricing: _Pricing
) -> None:
    billed: Final = _bill_legs(proxies[pricing.name].gateway)
    plain: Final = billed[GPT_6_LUNA_MESSAGES_BRIDGED_JSON.name]
    streamed: Final = billed[GPT_6_LUNA_MESSAGES_BRIDGED_STREAM.name]
    expected: Final = GPT_6_LUNA_MESSAGES_BRIDGED_JSON.expected
    assert isinstance(expected, ExactExpected), expected
    assert _priced(plain) == pytest.approx(
        (
            expected.spend * pricing.spend_multiplier,
            expected.input_cost,
            expected.output_cost,
            expected.prompt_tokens,
            expected.completion_tokens,
        )
    ), plain
    assert _priced(streamed) == pytest.approx(_priced(plain)), (plain, streamed)
