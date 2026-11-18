from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import adjusted_cost

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_a_discount_config_replaced_twice_prices_with_the_one_installed_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "cost_margin_config", {})

    def priced_with(fraction: float) -> float:
        monkeypatch.setattr(litellm, "cost_discount_config", {"anthropic": fraction})
        return adjusted_cost(10.0, "anthropic").cost

    costs: Final = tuple(priced_with(fraction) for fraction in (0.1, 0.5, 0.1))

    assert costs == pytest.approx((9.0, 5.0, 9.0))


@pytest.mark.parametrize(
    ("config", "edit", "expected"),
    [
        ("cost_discount_config", ("anthropic", 0.5), 5.0),
        ("cost_margin_config", ("global", 0.1), 11.0),
    ],
)
def test_a_cost_config_edited_in_place_applies_on_the_next_call(
    monkeypatch: pytest.MonkeyPatch, config: str, edit: tuple[str, float], expected: float
) -> None:
    monkeypatch.setattr(litellm, "cost_discount_config", {})
    monkeypatch.setattr(litellm, "cost_margin_config", {})
    unadjusted: Final = adjusted_cost(10.0, "anthropic").cost

    monkeypatch.setitem(getattr(litellm, config), *edit)
    edited: Final = adjusted_cost(10.0, "anthropic").cost

    assert unadjusted == pytest.approx(10.0)
    assert edited == pytest.approx(expected)
