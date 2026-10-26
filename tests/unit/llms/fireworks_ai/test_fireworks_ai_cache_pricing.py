from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm.exceptions import ModelNotMappedError
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token, get_billed_token_rates
from litellm.types.utils import Usage


@pytest.mark.parametrize(
    ("extra", "expected"),
    (
        ({}, 0.0001),
        ({"cache_read_input_token_cost": None}, 0.0001),
        ({"cache_read_input_token_cost": 0.0}, 0.0),
        ({"cache_read_input_token_cost": 0.3e-6}, 0.00003),
        ({"off_peak_pricing": {"hours_utc": "14:00-00:00", "input_cost_per_token": 1e-6}}, 0.00005),
        ({"off_peak_pricing": {"hours_utc": "14:00-00:00", "output_cost_per_token": 3e-6}}, 0.0001),
        ({"off_peak_pricing": {"hours_utc": "14:00-00:00", "input_cost_per_token": "1e-6"}}, 0.00005),
        (
            {
                "off_peak_pricing": {
                    "hours_utc": "14:00-00:00",
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": None,
                }
            },
            0.0001,
        ),
        ({"tiered_pricing": [{"range": [0, 200], "input_cost_per_token": 4e-6}]}, 0.0004),
    ),
)
def test_fireworks_default_cache_rate_matches_projected_rates_without_mutating_the_row(
    monkeypatch: pytest.MonkeyPatch,
    extra: Mapping[str, object],
    expected: float,
) -> None:
    model: Final = "fireworks_ai/cache-default-test"
    row: Final = {"input_cost_per_token": "2e-6", "litellm_provider": "fireworks_ai", "mode": "chat", **extra}
    original: Final = deepcopy(row)
    monkeypatch.setitem(litellm.model_cost, model, row)
    model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="fireworks_ai")
    usage: Final = Usage(prompt_tokens=100, completion_tokens=0, prompt_tokens_details={"cached_tokens": 100})
    when: Final = datetime(2026, 1, 1, 16, tzinfo=timezone.utc)
    assert generic_cost_per_token(
        model,
        usage,
        "fireworks_ai",
        model_info=model_info,
        current_time=when,
    ) == pytest.approx((expected, 0))
    assert get_billed_token_rates(
        model,
        "fireworks_ai",
        usage,
        model_info=model_info,
        current_time=when,
    ).cache_read_input_token_cost == pytest.approx(expected / 100)
    assert row == original


def test_fireworks_missing_input_rate_remains_unknown_without_a_cache_price(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "fireworks_ai/cache-unknown-test"
    row: Final = {"litellm_provider": "fireworks_ai", "mode": "chat", "output_cost_per_token": 3e-6}
    monkeypatch.setitem(litellm.model_cost, model, row)
    with pytest.raises(ModelNotMappedError):
        generic_cost_per_token(
            model,
            Usage(prompt_tokens=100, completion_tokens=0, prompt_tokens_details={"cached_tokens": 100}),
            "fireworks_ai",
        )
    assert row == {"litellm_provider": "fireworks_ai", "mode": "chat", "output_cost_per_token": 3e-6}
