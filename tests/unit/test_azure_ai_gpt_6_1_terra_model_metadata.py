import json
from pathlib import Path

MODEL = "azure_ai/gpt-6.1-terra"


def _load_cost_map(filename: str = "model_prices_and_context_window.json") -> dict:
    with open(Path(__file__).parents[2] / filename) as f:
        return json.load(f)


def test_azure_ai_gpt_6_1_terra_present_in_both_cost_maps():
    """Regression test for #48571: azure foundry GPT-6.1-terra had no price entry."""
    main_cost = _load_cost_map()
    backup_cost = _load_cost_map("litellm/model_prices_and_context_window_backup.json")

    assert MODEL in main_cost, f"{MODEL} missing from model_prices_and_context_window.json"
    assert MODEL in backup_cost, f"{MODEL} missing from model_prices_and_context_window_backup.json"
    assert backup_cost[MODEL] == main_cost[MODEL], f"{MODEL} differs between main and backup cost maps"


def test_azure_ai_gpt_6_1_terra_global_standard_pricing():
    """Global Standard pricing from Microsoft Foundry (USD/1M tokens):
    input $1.00 / output $5.00, long-context (>272k) input $2.00 / output $7.50,
    cached input $0.05 / cache write $1.25.
    """
    entry = _load_cost_map()[MODEL]

    assert entry["litellm_provider"] == "azure_ai"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == 1e-06
    assert entry["output_cost_per_token"] == 5e-06
    assert entry["input_cost_per_token_above_272k_tokens"] == 2e-06
    assert entry["output_cost_per_token_above_272k_tokens"] == 7.5e-06
    assert entry["cache_read_input_token_cost"] == 5e-08
    assert entry["cache_creation_input_token_cost"] == 1.25e-06
