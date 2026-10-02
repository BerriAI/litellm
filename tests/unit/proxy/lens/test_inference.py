from typing import Final

import pytest

from litellm.proxy.lens.inference import Deployment, DeploymentParams, Prices, completion_charge, quote
from litellm.types.utils import ModelResponse


def test_missing_optional_price_tiers_use_base_rates() -> None:
    prices: Final = Prices.model_validate(
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "input_cost_per_token_above_200k_tokens": None,
            "output_cost_per_token_above_200k_tokens": None,
            "input_cost_per_token_above_128k_tokens": None,
            "output_cost_per_token_above_128k_tokens": None,
        }
    )
    assert prices == Prices(input_cost_per_token=0.001, output_cost_per_token=0.002)


def test_custom_priced_model_charges_reported_tokens() -> None:
    deployment: Final = Deployment(
        litellm_params=DeploymentParams(
            model="openai/lens-test", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
    )
    response: Final = ModelResponse(
        model="lens-test", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    )
    assert completion_charge((deployment,), response, 10) == pytest.approx(0.04)
    assert quote((deployment,), "hello") > 0.04
