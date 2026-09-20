from copy import deepcopy

import pytest

from litellm.constants import FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO
from litellm.llms.fireworks_ai.cache_pricing import with_default_cache_read_rate
from litellm.types.utils import ModelInfo


def test_explicit_cache_read_rate_and_missing_input_rate_keep_the_entry_untouched() -> None:
    explicit_info: ModelInfo = {"input_cost_per_token": 2e-6, "cache_read_input_token_cost": 1e-6}
    no_input_rate_info: ModelInfo = {"output_cost_per_token": 3e-6}

    assert with_default_cache_read_rate(explicit_info) is explicit_info
    assert with_default_cache_read_rate(no_input_rate_info) is no_input_rate_info


def test_missing_cache_read_rate_is_derived_for_standard_and_off_peak_without_mutating_the_entry() -> None:
    model_info: ModelInfo = {
        "input_cost_per_token": 2e-6,
        "off_peak_pricing": {
            "hours_utc": "14:00-00:00",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 3e-6,
        },
    }
    original: ModelInfo = deepcopy(model_info)

    derived = with_default_cache_read_rate(model_info)

    assert model_info == original
    assert derived is not model_info
    assert derived["cache_read_input_token_cost"] == pytest.approx(2e-6 * FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO)
    assert derived["off_peak_pricing"] == {
        "hours_utc": "14:00-00:00",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 3e-6,
        "cache_read_input_token_cost": 1e-6 * FIREWORKS_AI_DEFAULT_CACHE_READ_RATE_RATIO,
    }


def test_off_peak_window_without_its_own_input_rate_reuses_the_standard_derived_rate() -> None:
    model_info: ModelInfo = {
        "input_cost_per_token": 2e-6,
        "off_peak_pricing": {"hours_utc": "14:00-00:00", "output_cost_per_token": 3e-6},
    }

    derived = with_default_cache_read_rate(model_info)

    assert derived["off_peak_pricing"]["cache_read_input_token_cost"] == derived["cache_read_input_token_cost"]


def test_string_rates_from_config_are_coerced_before_the_discount_is_applied() -> None:
    model_info: ModelInfo = {
        "input_cost_per_token": "2e-6",
        "off_peak_pricing": {
            "hours_utc": "14:00-00:00",
            "input_cost_per_token": "1e-6",
        },
    }

    derived = with_default_cache_read_rate(model_info)

    assert derived["cache_read_input_token_cost"] == pytest.approx(1e-6)
    assert derived["off_peak_pricing"]["cache_read_input_token_cost"] == pytest.approx(5e-7)
