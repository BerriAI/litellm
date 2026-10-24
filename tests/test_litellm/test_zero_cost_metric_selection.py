"""Tests for cost-metric selection on models priced at zero.

`select_cost_metric_for_model` chose a metric by truthiness, so a rate of 0 --
a valid price declaring the model free -- was indistinguishable from an absent
one, and the speech cost path raised ValueError claiming fields the model had
explicitly set were missing.
"""

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    select_cost_metric_for_model,
)


@pytest.mark.parametrize(
    "model_info, expected",
    [
        ({"key": "tts-1", "input_cost_per_token": 1.5e-05}, "cost_per_token"),
        ({"key": "tts-1", "input_cost_per_character": 1.5e-05}, "cost_per_character"),
        # character pricing wins when both are present
        (
            {
                "key": "tts-1",
                "input_cost_per_character": 1.5e-05,
                "input_cost_per_token": 3.0e-05,
            },
            "cost_per_character",
        ),
    ],
)
def test_priced_models_select_their_metric(model_info, expected):
    assert select_cost_metric_for_model(model_info) == expected


@pytest.mark.parametrize(
    "model_info, expected",
    [
        ({"key": "free-tts", "input_cost_per_character": 0}, "cost_per_character"),
        ({"key": "free-tts", "input_cost_per_token": 0}, "cost_per_token"),
        ({"key": "free-tts", "input_cost_per_character": 0.0}, "cost_per_character"),
        (
            {"key": "free-tts", "input_cost_per_character": 0, "input_cost_per_token": 0},
            "cost_per_character",
        ),
    ],
)
def test_zero_is_a_price_not_an_absent_field(model_info, expected):
    # A zero rate declares the model free. Selecting on truthiness dropped
    # through to the ValueError below, so a free TTS model could not be costed.
    assert select_cost_metric_for_model(model_info) == expected


def test_missing_cost_fields_still_raise():
    with pytest.raises(ValueError, match="does not have"):
        select_cost_metric_for_model({"key": "unpriced-tts"})


def test_explicit_none_is_treated_as_absent():
    with pytest.raises(ValueError, match="does not have"):
        select_cost_metric_for_model(
            {"key": "unpriced-tts", "input_cost_per_character": None, "input_cost_per_token": None}
        )
