"""Runtime-priced Sail deployments for the completion-window (service_tier) suites.

Sail bills a request by the `metadata.completion_window` it ran under, and the
cost map carries one rate set per window: the base keys for `asap`, `*_balanced`
and `*_flex`. Every rate here is read off the proxy's own
/public/litellm_model_cost_map at run time, so the tests compare a bill against
the same numbers the gateway priced it from and never pin a vendor's price.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from models import CostMapEntry

SAIL_PROVIDER: Final = "sail"
SAIL_API_KEY: Final = "os.environ/SAIL_API_KEY"
COST_REL_TOLERANCE: Final = 1e-9


def within_rel(actual: float | None, expected: float) -> bool:
    """Spend math agrees to one part in 1e9, the tightest tolerance float sums hold; no value never agrees."""
    return actual is not None and abs(actual - expected) <= abs(expected) * COST_REL_TOLERANCE


def all_within_rel(actual: Sequence[float], expected: Sequence[float]) -> bool:
    """Every component of one bill agrees with the matching component of another."""
    return len(actual) == len(expected) and all(within_rel(a, e) for a, e in zip(actual, expected, strict=True))


@dataclass(frozen=True, slots=True)
class TierRates:
    """The three per-token rates one window bills at."""

    input: float
    output: float
    cache_read: float

    def bill(
        self, *, prompt_tokens: int, cached_tokens: int, completion_tokens: int, reasoning_tokens: int
    ) -> tuple[float, float, float, float, float]:
        """(input_cost, output_cost, cache_read_cost, reasoning_cost, total_cost) the
        gateway must file for this usage: cached prompt tokens at the cache-read rate,
        the rest of the prompt at the input rate, every completion token (reasoning
        included) at the output rate, reasoning also reported on its own line."""
        cache_read_cost: Final = cached_tokens * self.cache_read
        input_cost: Final = (prompt_tokens - cached_tokens) * self.input + cache_read_cost
        output_cost: Final = completion_tokens * self.output
        return (
            input_cost,
            output_cost,
            cache_read_cost,
            reasoning_tokens * self.output,
            input_cost + output_cost,
        )


@dataclass(frozen=True, slots=True)
class WindowPricedModel:
    """A Sail cost-map row priced for every completion window."""

    model: str
    asap: TierRates
    balanced: TierRates
    flex: TierRates


@dataclass(frozen=True, slots=True)
class FlexOnlyModel:
    """A Sail cost-map row priced for the flex window and nothing else."""

    model: str
    flex: TierRates


def _rates(input_rate: float | None, output_rate: float | None, cache_rate: float | None) -> TierRates | None:
    if input_rate is None or output_rate is None or cache_rate is None:
        return None
    return TierRates(input=input_rate, output=output_rate, cache_read=cache_rate)


def _base(entry: CostMapEntry) -> TierRates | None:
    return _rates(entry.input_cost_per_token, entry.output_cost_per_token, entry.cache_read_input_token_cost)


def _balanced(entry: CostMapEntry) -> TierRates | None:
    return _rates(
        entry.input_cost_per_token_balanced,
        entry.output_cost_per_token_balanced,
        entry.cache_read_input_token_cost_balanced,
    )


def _flex(entry: CostMapEntry) -> TierRates | None:
    return _rates(
        entry.input_cost_per_token_flex, entry.output_cost_per_token_flex, entry.cache_read_input_token_cost_flex
    )


def _sail_chat_rows(cost_map: Mapping[str, CostMapEntry]) -> tuple[tuple[str, CostMapEntry], ...]:
    return tuple(
        (model, entry)
        for model, entry in sorted(cost_map.items())
        if entry.litellm_provider == SAIL_PROVIDER and entry.mode == "chat" and entry.deprecation_date is None
    )


def _window_priced(model: str, entry: CostMapEntry) -> WindowPricedModel | None:
    asap: Final = _base(entry)
    balanced: Final = _balanced(entry)
    flex: Final = _flex(entry)
    if asap is None or balanced is None or flex is None:
        return None
    totals: Final = {asap.input + asap.output, balanced.input + balanced.output, flex.input + flex.output}
    return WindowPricedModel(model=model, asap=asap, balanced=balanced, flex=flex) if len(totals) == 3 else None


def cheapest_window_priced_model(cost_map: Mapping[str, CostMapEntry]) -> WindowPricedModel:
    """The Sail chat row with distinct asap, balanced and flex rate sets whose asap
    input rate is lowest, so the tier comparison runs on the cheapest eligible model."""
    priced: Final = tuple(
        priced_row
        for model, entry in _sail_chat_rows(cost_map)
        if (priced_row := _window_priced(model, entry)) is not None
    )
    assert priced, f"no sail chat row carries distinct asap, balanced and flex rates: {sorted(cost_map)}"
    return min(priced, key=lambda candidate: candidate.asap.input)


def cheapest_flex_only_model(cost_map: Mapping[str, CostMapEntry]) -> FlexOnlyModel:
    """The Sail chat row that carries flex rates and no balanced rates, cheapest first."""
    priced: Final = tuple(
        FlexOnlyModel(model=model, flex=flex)
        for model, entry in _sail_chat_rows(cost_map)
        if (flex := _flex(entry)) is not None and _balanced(entry) is None
    )
    assert priced, f"no sail chat row is priced for flex only: {sorted(cost_map)}"
    return min(priced, key=lambda candidate: candidate.flex.input)
