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

# Source: ScaleDown's standard rate of $0.05 per million input tokens with output unbilled,
# as shown in the ScaleDown usage dashboard (checked 2026-10-09) and its billing service.
INPUT_COST_PER_MILLION_TOKENS = 0.05


@pytest.mark.parametrize("model", SCALEDOWN_MODELS)
def test_scaledown_model_is_priced_on_input_tokens_only(model: str) -> None:
    entry = litellm.model_cost[model]

    assert entry["litellm_provider"] == "scaledown"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] * 1_000_000 == pytest.approx(INPUT_COST_PER_MILLION_TOKENS)
    assert entry["output_cost_per_token"] == 0.0


@pytest.mark.parametrize("model", SCALEDOWN_MODELS)
def test_scaledown_cost_ignores_output_tokens(model: str) -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        custom_llm_provider="scaledown",
        usage_object=Usage(prompt_tokens=1_000_000, completion_tokens=500_000),
    )

    assert prompt_cost == pytest.approx(INPUT_COST_PER_MILLION_TOKENS)
    assert completion_cost == 0.0


def test_every_scaledown_entry_in_the_cost_map_is_covered_here() -> None:
    in_map = {key for key in litellm.model_cost if key.startswith("scaledown/")}

    assert in_map == set(SCALEDOWN_MODELS)
