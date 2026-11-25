"""
Provider-neutral tiered pricing calculation.

Shared by provider cost calculators (e.g. Dashscope) and the proxy budget
reservation logic so neither has to depend on the other.
"""

from collections.abc import Mapping, Sequence
from typing import Final, SupportsFloat

from pydantic import TypeAdapter, ValidationError

_TIER_RANGE: Final = TypeAdapter(tuple[float, float])


def _coerce_cost_per_token(value: object) -> float:
    """
    Coerce a per-token cost into a float.

    Model cost values loaded from YAML config may arrive as strings (e.g.
    scientific notation like "4e-07"), which would break arithmetic.
    """
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    if isinstance(value, SupportsFloat):
        return float(value)
    return 0.0


def _tier_range(tier: Mapping[str, object]) -> tuple[float, float] | None:
    try:
        return _TIER_RANGE.validate_python(tier.get("range"))
    except ValidationError:
        return None


def select_tier_for_input(
    tiered_pricing: Sequence[Mapping[str, object]],
    input_tokens: int,
) -> Mapping[str, object] | None:
    """
    Select the pricing tier for a request based on its total input token count.

    Alibaba Model Studio (Dashscope) tiered pricing is all-or-nothing: the tier is
    chosen by the total input tokens of a single request and every token in the
    request (input and output) is billed at that one tier's rate, rather than
    graduated income-tax-style slicing. A tier matches when
    ``range_start < input_tokens <= range_end`` (so a request of exactly
    ``range_end`` tokens stays in the lower tier, matching the official
    ``0 < Token <= 32K`` phrasing). Requests above the highest declared range fall
    back to the last (most expensive) tier.
    """
    if not tiered_pricing or input_tokens <= 0:
        return None

    ranged_tiers: Final = sorted(
        ((bounds, tier) for tier in tiered_pricing if (bounds := _tier_range(tier)) is not None),
        key=lambda ranged: ranged[0][0],
    )
    if not ranged_tiers:
        return None

    matching: Final = tuple(tier for (start, end), tier in ranged_tiers if start < input_tokens <= end)
    if matching:
        return matching[0]
    return ranged_tiers[-1][1]


def tier_rate(
    tier: Mapping[str, object],
    cost_key: str,
    fallback_cost_key: str | None = None,
) -> float:
    """Read a per-token rate from a tier, coercing YAML string costs to float.

    A rate that is explicitly present wins over the fallback, an explicit zero
    included, so a tier can declare a token type free.
    """
    primary: Final = tier.get(cost_key)
    if primary is not None:
        return _coerce_cost_per_token(primary)
    if fallback_cost_key is None:
        return 0.0
    return _coerce_cost_per_token(tier.get(fallback_cost_key, 0))
