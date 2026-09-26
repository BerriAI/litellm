"""Regression tests for calculate_cache_writing_cost undetailed remainder (#42663)."""

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import calculate_cache_writing_cost
from litellm.types.utils import CacheCreationTokenDetails


def test_calculate_cache_writing_cost_bills_undetailed_remainder_at_5m_rate():
    """
    Streamed Anthropic server-tool runs keep the message_start 5m/1h breakdown
    while cache_creation_input_tokens grows across iterations. Pricing must
    still bill the undetailed remainder at the 5m rate instead of dropping it.
    """
    rate_5m = 3.75e-6
    rate_1h = 6.0e-6
    details = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=31490,
        ephemeral_1h_input_tokens=0,
    )

    # Stale breakdown (message_start only) vs full streamed total.
    cost = calculate_cache_writing_cost(
        cache_creation_tokens=184457,
        cache_creation_token_details=details,
        cache_creation_cost_above_1hr=rate_1h,
        cache_creation_cost=rate_5m,
    )
    assert cost == pytest.approx(184457 * rate_5m)

    # Already-reconciled details (non-streaming aggregate path) must not double-bill.
    reconciled = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=184457,
        ephemeral_1h_input_tokens=0,
    )
    cost_reconciled = calculate_cache_writing_cost(
        cache_creation_tokens=184457,
        cache_creation_token_details=reconciled,
        cache_creation_cost_above_1hr=rate_1h,
        cache_creation_cost=rate_5m,
    )
    assert cost_reconciled == pytest.approx(184457 * rate_5m)

    # Mixed 5m + 1h with an undetailed remainder.
    mixed = CacheCreationTokenDetails(
        ephemeral_5m_input_tokens=1000,
        ephemeral_1h_input_tokens=2000,
    )
    cost_mixed = calculate_cache_writing_cost(
        cache_creation_tokens=5000,
        cache_creation_token_details=mixed,
        cache_creation_cost_above_1hr=rate_1h,
        cache_creation_cost=rate_5m,
    )
    assert cost_mixed == pytest.approx(1000 * rate_5m + 2000 * rate_1h + 2000 * rate_5m)


def test_calculate_cache_writing_cost_without_details_uses_total():
    rate_5m = 3.75e-6
    cost = calculate_cache_writing_cost(
        cache_creation_tokens=1000,
        cache_creation_token_details=None,
        cache_creation_cost_above_1hr=6.0e-6,
        cache_creation_cost=rate_5m,
    )
    assert cost == pytest.approx(1000 * rate_5m)
