"""
Provider-neutral tiered pricing calculation.

Shared by provider cost calculators (e.g. Dashscope) and the proxy budget
reservation logic so neither has to depend on the other.
"""

from typing import Final


def _coerce_cost_per_token(value: float | str | None) -> float:
    """
    Coerce a per-token cost into a float.

    Model cost values loaded from YAML config may arrive as strings (e.g.
    scientific notation like "4e-07"), which would break arithmetic.
    """
    if value is None:
        return 0.0
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return float(value)


def select_tier_for_input(
    tiered_pricing: list[dict],
    input_tokens: int,
) -> dict | None:
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

    sorted_tiers: Final = sorted(tiered_pricing, key=lambda t: t.get("range", [0, 0])[0])
    valid_tiers: Final = [tier for tier in sorted_tiers if len(tier.get("range", [])) == 2]
    if not valid_tiers:
        return None

    matching: Final = [tier for tier in valid_tiers if tier["range"][0] < input_tokens <= tier["range"][1]]
    if matching:
        return matching[0]
    return valid_tiers[-1]


def tier_rate(
    tier: dict,
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
    return _coerce_cost_per_token(tier.get(fallback_cost_key, 0))
