import json
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# us-east-1 standard-tier rates from the AWS Pricing API (USD per token)
MANTLE_MODEL_PRICES = {
    "bedrock_mantle/deepseek.v3.1": (5.8e-07, 1.68e-06),
    "bedrock_mantle/moonshotai.kimi-k2-thinking": (6e-07, 2.5e-06),
    "bedrock_mantle/qwen.qwen3-32b": (1.5e-07, 6e-07),
    "bedrock_mantle/qwen.qwen3-235b-a22b-2507": (2.2e-07, 8.8e-07),
    "bedrock_mantle/qwen.qwen3-coder-30b-a3b-instruct": (1.5e-07, 6e-07),
    "bedrock_mantle/qwen.qwen3-coder-480b-a35b-instruct": (4.5e-07, 1.8e-06),
    "bedrock_mantle/qwen.qwen3-next-80b-a3b-instruct": (1.4e-07, 1.2e-06),
    "bedrock_mantle/qwen.qwen3-vl-235b-a22b-instruct": (5.3e-07, 2.66e-06),
}


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", MANTLE_MODEL_PRICES)
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert model in main_cost, f"{model} missing from model_prices_and_context_window.json"
    assert model in backup_cost, f"{model} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[model] == main_cost[model], f"{model} differs between main and backup model cost maps"


@pytest.mark.parametrize("model,prices", MANTLE_MODEL_PRICES.items())
def test_mantle_model_prices(model, prices):
    entry = _load(MAIN_PATH)[model]
    input_cost, output_cost = prices

    assert entry["litellm_provider"] == "bedrock_mantle"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == input_cost
    assert entry["output_cost_per_token"] == output_cost


@pytest.mark.parametrize("model,prices", MANTLE_MODEL_PRICES.items())
def test_mantle_model_cost_per_token(model, prices, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    input_cost, output_cost = prices

    prompt_cost, completion_cost = litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=1000)

    assert prompt_cost == pytest.approx(1000 * input_cost)
    assert completion_cost == pytest.approx(1000 * output_cost)
