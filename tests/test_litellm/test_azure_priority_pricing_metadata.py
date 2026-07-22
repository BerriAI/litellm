import json
from pathlib import Path

import pytest


# (model_name, input_priority, output_priority, cache_read_priority)
AZURE_PRIORITY_PRICING = [
    ("azure/gpt-4.1", 3.5e-06, 1.4e-05, 8.75e-07),
    ("azure/gpt-4.1-2025-04-14", 3.5e-06, 1.4e-05, 8.75e-07),
    ("azure/gpt-5.1", 2.5e-06, 2e-05, 2.5e-07),
    ("azure/gpt-5.2", 3.5e-06, 2.8e-05, 3.5e-07),
]


@pytest.mark.parametrize(
    "model, input_priority, output_priority, cache_read_priority",
    AZURE_PRIORITY_PRICING,
)
def test_azure_priority_pricing_keys_present(
    model, input_priority, output_priority, cache_read_priority
):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model catalog"
    assert info["litellm_provider"] == "azure"

    assert info["input_cost_per_token_priority"] == input_priority
    assert info["output_cost_per_token_priority"] == output_priority
    assert info["cache_read_input_token_cost_priority"] == cache_read_priority


# Aliases that should carry supports_service_tier to match their dated counterparts
# (azure/gpt-5.1-2025-11-13, azure/gpt-5.2-2025-12-11). PR #24924 covers the gpt-4.1 aliases.
AZURE_SUPPORTS_SERVICE_TIER_ALIASES = [
    "azure/gpt-5.1",
    "azure/gpt-5.2",
]


@pytest.mark.parametrize("model", AZURE_SUPPORTS_SERVICE_TIER_ALIASES)
def test_azure_undated_aliases_advertise_service_tier_support(model):
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(json_path) as f:
        model_cost = json.load(f)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model catalog"
    assert (
        info.get("supports_service_tier") is True
    ), f"{model} should advertise supports_service_tier=true to match its dated variant"


def test_azure_priority_pricing_backup_matches_main():
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_path = repo_root / "model_prices_and_context_window.json"
    backup_path = repo_root / "litellm" / "model_prices_and_context_window_backup.json"

    with open(main_path) as f:
        main_cost = json.load(f)
    with open(backup_path) as f:
        backup_cost = json.load(f)

    for model, *_ in AZURE_PRIORITY_PRICING:
        assert backup_cost.get(model) == main_cost.get(
            model
        ), f"{model} differs between main and backup model cost maps"
