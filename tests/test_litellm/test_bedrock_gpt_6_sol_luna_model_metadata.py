import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# AWS has not published Bedrock prices for these yet. Bedrock US Geo and
# In-Region rates for OpenAI models are the OpenAI API rate plus a 10% fee,
# so these entries are the OpenAI rate for the same model times 1.1.
US_UPLIFT = 1.1
BEDROCK_TO_OPENAI = {
    "us.openai.gpt-6-sol": "gpt-6-sol",
    "bedrock_mantle/openai.gpt-6-sol": "gpt-6-sol",
    "us.openai.gpt-6-luna": "gpt-6-luna",
    "bedrock_mantle/openai.gpt-6-luna": "gpt-6-luna",
}
PRICE_KEYS = (
    "input_cost_per_token",
    "input_cost_per_token_above_272k_tokens",
    "cache_creation_input_token_cost",
    "cache_read_input_token_cost",
    "output_cost_per_token",
    "output_cost_per_token_above_272k_tokens",
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", BEDROCK_TO_OPENAI)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert model in main_cost, f"{model} missing from model_prices_and_context_window.json"
    assert model in backup_cost, f"{model} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[model] == main_cost[model], f"{model} differs between main and backup model cost maps"


@pytest.mark.parametrize("model,openai_model", BEDROCK_TO_OPENAI.items())
def test_prices_are_openai_rate_plus_us_uplift(model, openai_model):
    cost_map = _load(MAIN_PATH)
    entry = cost_map[model]
    openai_entry = cost_map[openai_model]

    for key in PRICE_KEYS:
        assert entry[key] == pytest.approx(openai_entry[key] * US_UPLIFT), key


@pytest.mark.parametrize("model", BEDROCK_TO_OPENAI)
def test_cost_per_token_is_not_zero(model, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    entry = _load(MAIN_PATH)[model]

    prompt_cost, completion_cost = litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=1000)

    assert prompt_cost == pytest.approx(1000 * entry["input_cost_per_token"])
    assert completion_cost == pytest.approx(1000 * entry["output_cost_per_token"])
    assert prompt_cost > 0 and completion_cost > 0
