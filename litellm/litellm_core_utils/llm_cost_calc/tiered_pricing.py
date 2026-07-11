"""
Provider-neutral graduated tiered pricing calculation.

Shared by provider cost calculators (e.g. Dashscope) and the proxy budget
reservation logic so neither has to depend on the other.
"""

from typing import List, Optional, Union


def _coerce_cost_per_token(value: Union[float, int, str, None]) -> float:
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


def calculate_tiered_cost(
    tokens: int,
    tiered_pricing: List[dict],
    cost_key: str,
    fallback_cost_key: Optional[str] = None,
) -> float:
    """
    Calculate cost for a given number of tokens based on a true tiered pricing structure.

    This function iterates through sorted pricing tiers, calculates the cost for the
    number of tokens that fall into each tier's range, and sums them up to get the total cost.

    Args:
        tokens (int): The total number of tokens to calculate the cost for.
        tiered_pricing (List[dict]): A list of dictionaries, where each dictionary
            represents a pricing tier.
        cost_key (str): The key in the tier dictionary that holds the per-token cost
            (e.g., 'input_cost_per_token').
        fallback_cost_key (Optional[str], optional): A fallback key to use if the
            primary `cost_key` is not found in a tier. Defaults to None.

    Returns:
        float: The total calculated cost for the given tokens.

    Example:
        >>> tiered_pricing = [
        ...     {"range": [0, 100000], "input_cost_per_token": 0.0001},
        ...     {"range": [100000, 500000], "input_cost_per_token": 0.00005},
        ... ]

        Calculating cost for 150,000 tokens:
        (100,000 * 0.0001) + (50,000 * 0.00005) = $12.5
    """
    if not tiered_pricing or tokens <= 0:
        return 0.0

    total_cost = 0.0
    tokens_processed = 0

    sorted_tiers = sorted(tiered_pricing, key=lambda x: x.get("range", [0, 0])[0])

    for tier in sorted_tiers:
        if tokens_processed >= tokens:
            break

        tier_range = tier.get("range", [])
        if len(tier_range) != 2:
            continue

        range_start, range_end = tier_range

        if tokens <= range_start:
            continue

        tier_start = max(range_start, tokens_processed)
        tier_end = min(range_end, tokens)

        if tier_end > tier_start:
            tokens_in_tier = tier_end - tier_start
            cost_per_token = tier.get(cost_key) or tier.get(fallback_cost_key, 0)
            total_cost += tokens_in_tier * _coerce_cost_per_token(cost_per_token)
            tokens_processed = tier_end

    # After loop, check if any tokens remain (i.e., tokens > highest tier's end range)
    # and charge them at the last tier's rate.
    if tokens_processed < tokens and sorted_tiers:
        last_tier = sorted_tiers[-1]
        remaining_tokens = tokens - tokens_processed
        cost_per_token = last_tier.get(cost_key) or last_tier.get(fallback_cost_key, 0)
        total_cost += remaining_tokens * _coerce_cost_per_token(cost_per_token)

    return total_cost
