from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.unpriced_models import DeploymentPricing, PriceEntry, UnpricedModelFacts, unpriced_model
from litellm.proxy.auth.unpriced_model_adapter import price_entry
from tests.unit._auth.support.model_registry_reference import entry_has_priced_metric

_BASE: Final = UnpricedModelFacts("model", True)
_ROW: Final = DeploymentPricing((), True)


@pytest.mark.parametrize(
    ("entry", "expected"),
    (
        ({}, False),
        ({"input_cost_per_token": 0}, False),
        ({"input_cost_per_token": -1}, False),
        ({"cost_per_image": 0.1}, True),
        ({"cost_per_image": True}, False),
        ({"cost_per_image": "0.1"}, False),
        ({"cost_per_image": {"small": 0, "large": 0.2}}, True),
        ({"cost_per_image": {"large": True}}, False),
        ({"cost_per_image": {"large": {"nested": 1}}}, False),
        ({"tiered_pricing": None}, False),
        ({"tiered_pricing": ()}, True),
        ({"tiered_pricing": False}, True),
        ({"maximum_tokens": 1}, False),
        ({"input_cost_per_token": float("nan")}, False),
        ({"input_cost_per_token": float("inf")}, True),
    ),
)
def test_native_price_metrics_match_reference(entry: Mapping[str, object], expected: bool) -> None:
    facts: Final = replace(
        _BASE,
        group_found=True,
        deployments=(replace(_ROW, info_loaded=True, info=price_entry(entry)),),
        cost_deployments=(),
    )
    assert entry_has_priced_metric(entry) == expected
    assert unpriced_model(facts).unpriced == (not expected)


@pytest.mark.parametrize(
    ("facts", "kind", "index", "unpriced"),
    (
        (replace(_BASE, model=None), "Complete", 0, False),
        (replace(_BASE, model=""), "Complete", 0, False),
        (replace(_BASE, router_present=False), "Complete", 0, False),
        (_BASE, "ReadGroup", 0, False),
        (replace(_BASE, group_found=False), "Complete", 0, False),
        (replace(_BASE, group_found=True), "ReadDeployments", 0, False),
        (replace(_BASE, group_found=True, deployments=(_ROW,)), "ReadInfo", 0, False),
        (
            replace(_BASE, group_found=True, deployments=(replace(_ROW, parameter_keys=("input_cost_per_token",)),)),
            "Complete",
            0,
            False,
        ),
        (
            replace(_BASE, group_found=True, deployments=(replace(_ROW, parameter_keys=("tiered_pricing",)),)),
            "Complete",
            0,
            False,
        ),
        (replace(_BASE, group_found=True, deployments=()), "ReadCostDeployments", 0, False),
        (replace(_BASE, group_found=True, deployments=(), cost_deployments=(_ROW,)), "ReadCost", 0, False),
        (
            replace(
                _BASE,
                group_found=True,
                deployments=(),
                cost_deployments=(replace(_ROW, cost_keys=("input_cost_per_token",)),),
            ),
            "Complete",
            0,
            False,
        ),
        (
            replace(
                _BASE,
                group_found=True,
                deployments=(),
                cost_deployments=(replace(_ROW, cost_keys=("cost_per_second",)),),
            ),
            "Complete",
            0,
            True,
        ),
        (
            replace(
                _BASE, group_found=True, deployments=(replace(_ROW, info_loaded=True, info=PriceEntry(False, ())), _ROW)
            ),
            "ReadInfo",
            1,
            False,
        ),
    ),
)
def test_unpriced_reads_preserve_legacy_short_circuit_order(
    facts: UnpricedModelFacts, kind: str, index: int, unpriced: bool
) -> None:
    decision: Final = unpriced_model(facts)
    assert (decision.kind, decision.index, decision.unpriced) == (kind, index, unpriced)
