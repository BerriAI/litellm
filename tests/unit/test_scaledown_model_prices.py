import pytest

import litellm
from litellm.types.utils import Usage

SCALEDOWN_MODELS = (
    "scaledown/classify",
    "scaledown/decisions",
    "scaledown/extract",
    "scaledown/summarize",
    "scaledown/compress",
)

INPUT_COST_PER_TOKEN = 5e-08


@pytest.mark.parametrize("model", SCALEDOWN_MODELS)
def test_scaledown_model_is_priced_on_input_tokens_only(model):
    entry = litellm.model_cost[model]

    assert entry["litellm_provider"] == "scaledown"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == INPUT_COST_PER_TOKEN
    assert entry["output_cost_per_token"] == 0.0


@pytest.mark.parametrize("model", SCALEDOWN_MODELS)
def test_scaledown_cost_ignores_output_tokens(model):
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        custom_llm_provider="scaledown",
        usage_object=Usage(prompt_tokens=1_000_000, completion_tokens=500_000),
    )

    assert prompt_cost == pytest.approx(0.05)
    assert completion_cost == 0.0


def test_every_scaledown_entry_in_the_cost_map_is_covered_here():
    in_map = {key for key in litellm.model_cost if key.startswith("scaledown/")}

    assert in_map == set(SCALEDOWN_MODELS)
