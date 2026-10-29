from dataclasses import FrozenInstanceError
from typing import Final

import pytest

from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
from litellm.types.utils import CostBreakdown


@pytest.mark.parametrize(("amount", "discount", "expected"), [(1.0, 0.2, 0.8), (0.0, 0.0, 0.0), (0.1, 0.5, 0.0)])
def test_finalized_amount_keeps_nonnegative_cost(amount: float, discount: float, expected: float) -> None:
    finalized: Final = finalize_cost(amount, None, discount)
    assert finalized.amount == pytest.approx(expected)
    assert finalized.allowance_discount == min(amount, discount)


def test_finalized_breakdown_is_independent_of_input_and_callback_copies() -> None:
    original: Final[CostBreakdown] = {"total_cost": 1.0, "additional_costs": {"fee": 0.3}, "discount_amount": 0.1}
    finalized: Final = finalize_cost(1.0, original, 0.2)
    original["additional_costs"]["fee"] = 9.0
    copied: Final = finalized.logging_breakdown()
    assert copied is not None
    copied["additional_costs"]["fee"] = 7.0
    assert finalized.logging_breakdown() == {
        "total_cost": 0.8,
        "original_cost": 1.0,
        "discount_amount": pytest.approx(0.3),
        "additional_costs": {"fee": 0.3},
    }
    with pytest.raises(FrozenInstanceError):
        setattr(finalized, "amount", 9.0)


def test_no_adjustment_keeps_an_absent_breakdown() -> None:
    assert finalize_cost(0.0, None).logging_breakdown() is None


@pytest.mark.parametrize(("amount", "discount"), [(-0.5, 0.0), (0.5, -0.2)])
def test_finalization_preserves_existing_credit_and_rejects_negative_allowances(amount: float, discount: float) -> None:
    finalized: Final = finalize_cost(amount, None, discount)
    assert finalized.amount == amount
    assert finalized.allowance_discount == 0.0
