"""Freshness checks for the cost suite's data files; markerless, so it runs on
any pytest invocation of the folder without the stack. expected.json is the
oracle: these tests check its key set against the derived matrix, never its
values (the generator proposes, the file decides)."""

from __future__ import annotations

from typing import Final

import pytest
from cost_matrix import (
    CASES,
    CASES_FILE,
    COST_MAP,
    EXPECTED,
    FRONTIER_MODELS,
    CostMapEntry,
    cases_for,
    expected_key,
)


def test_expected_keys_match_derived_exact_cells() -> None:
    derived: Final = {
        expected_key(model, case)
        for model in FRONTIER_MODELS
        for case in cases_for(model)
        if case.exact_spend
    }
    golden: Final = set(EXPECTED)
    if derived != golden:
        missing: Final = sorted(derived - golden)
        stale: Final = sorted(golden - derived)
        pytest.fail(
            "expected.json is out of sync with the derived matrix; run "
            "uv run python tests/e2e/cost_calculation/generate_expected.py "
            f"(missing: {missing}; stale: {stale})"
        )


def test_deployments_reference_existing_map_keys() -> None:
    unknown: Final = sorted(
        spec.map_key for spec in CASES_FILE.deployments if spec.map_key not in COST_MAP
    )
    assert not unknown, f"deployments entries name map keys absent from cost_map.json: {unknown}"


def test_requires_rates_are_cost_map_fields() -> None:
    fields: Final = set(CostMapEntry.model_fields)
    unknown: Final = sorted(
        {field for case in CASES for field in case.requires_rates} - fields
    )
    assert not unknown, f"requires_rates names that are not CostMapEntry fields: {unknown}"


def test_no_two_entries_share_input_rate() -> None:
    rates: Final = tuple(entry.input_cost_per_token for entry in COST_MAP.values())
    assert len(rates) == len(set(rates)), (
        "two cost_map entries share input_cost_per_token; the suite relies on "
        "distinct rates so a wrong-model bill can never coincidentally match"
    )
