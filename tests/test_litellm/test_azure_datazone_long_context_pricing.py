import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import cost_per_token


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    original = litellm.model_cost
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield
    finally:
        litellm.model_cost = original


REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# Data Zone token pricing transcribed from the Azure OpenAI pricing page
# (https://azure.microsoft.com/en-us/pricing/details/azure-openai/): Standard
# on-demand plus Priority Processing columns, and the Long Context tier only for
# the models where Azure actually publishes a Data Zone Long Context row.
GPT_54_DATA_ZONE = {
    "cache_read_input_token_cost": 2.8e-07,
    "cache_read_input_token_cost_priority": 5.5e-07,
    "input_cost_per_token": 2.75e-06,
    "input_cost_per_token_priority": 5.5e-06,
    "output_cost_per_token": 1.65e-05,
    "output_cost_per_token_priority": 3.3e-05,
}

GPT_55_DATA_ZONE = {
    "cache_read_input_token_cost": 5.5e-07,
    "cache_read_input_token_cost_above_272k_tokens": 1.1e-06,
    "cache_read_input_token_cost_priority": 1.38e-06,
    "input_cost_per_token": 5.5e-06,
    "input_cost_per_token_above_272k_tokens": 1.1e-05,
    "input_cost_per_token_priority": 1.375e-05,
    "output_cost_per_token": 3.3e-05,
    "output_cost_per_token_above_272k_tokens": 4.95e-05,
    "output_cost_per_token_priority": 8.25e-05,
}

EXPECTED = {
    "azure/gpt-5.4": GPT_54_DATA_ZONE,
    "azure/gpt-5.4-2026-03-05": GPT_54_DATA_ZONE,
    "azure/gpt-5.5": GPT_55_DATA_ZONE,
    "azure/gpt-5.5-2026-04-23": GPT_55_DATA_ZONE,
}

NO_DATA_ZONE_MODELS = (
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.4-pro",
    "gpt-5.4-pro-2026-03-05",
    "gpt-5.5-pro",
    "gpt-5.5-pro-2026-04-23",
)

NO_LONG_CONTEXT_MODELS = (
    "azure/gpt-5.4-mini",
    "azure/gpt-5.4-mini-2026-03-17",
    "azure/gpt-5.4-nano",
    "azure/gpt-5.4-nano-2026-03-17",
)

DATA_ZONE_KEYS = tuple(
    f"azure/{zone}/{model.split('/', 1)[1]}" for model in EXPECTED for zone in ("us", "eu")
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def main_cost():
    return _load(MAIN_PATH)


def _cost_fields(entry):
    return {k: v for k, v in entry.items() if "cost" in k and isinstance(v, (int, float)) and not isinstance(v, bool)}


@pytest.mark.parametrize("regional_base", tuple(EXPECTED))
@pytest.mark.parametrize("zone", ("us", "eu"))
def test_data_zone_entry_matches_azure_published(main_cost, regional_base, zone):
    zone_key = regional_base.replace("azure/", f"azure/{zone}/", 1)
    entry = main_cost.get(zone_key)
    assert entry is not None, f"{zone_key} not found in model cost map"
    assert _cost_fields(entry) == EXPECTED[regional_base], f"{zone_key} cost fields do not match the Azure pricing page"

    regional = main_cost[regional_base]
    for key, value in entry.items():
        if "cost" not in key:
            assert value == regional[key], f"{zone_key}.{key} metadata differs from {regional_base}"


@pytest.mark.parametrize("model", NO_DATA_ZONE_MODELS)
@pytest.mark.parametrize("zone", ("us", "eu"))
def test_data_zone_not_invented_for_global_only_models(main_cost, model, zone):
    assert f"azure/{zone}/{model}" not in main_cost


@pytest.mark.parametrize("model", NO_LONG_CONTEXT_MODELS)
def test_mini_nano_have_no_long_context_tier(main_cost, model):
    assert not any("above_272k" in key for key in main_cost[model]), f"{model} must not carry a long-context tier"


def test_gpt54_data_zone_resolves_and_is_premium_over_global():
    prompt, completion = 1_000, 500
    base_in, base_out = cost_per_token(
        model="azure/gpt-5.4", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    assert base_in == pytest.approx(prompt * 2.5e-06, rel=1e-9)
    for zone in ("us", "eu"):
        z_in, z_out = cost_per_token(
            model=f"azure/{zone}/gpt-5.4",
            custom_llm_provider="azure",
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        assert z_in == pytest.approx(prompt * 2.75e-06, rel=1e-9)
        assert z_out == pytest.approx(completion * 1.65e-05, rel=1e-9)
        assert z_in > base_in


def test_gpt55_data_zone_long_context_surcharge_applies_above_threshold():
    below_in, _ = cost_per_token(
        model="azure/us/gpt-5.5", custom_llm_provider="azure", prompt_tokens=1_000, completion_tokens=10
    )
    assert below_in == pytest.approx(1_000 * 5.5e-06, rel=1e-9)

    prompt, completion = 300_000, 1_000  # prompt_tokens > 272_000
    for zone in ("us", "eu"):
        z_in, z_out = cost_per_token(
            model=f"azure/{zone}/gpt-5.5",
            custom_llm_provider="azure",
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        assert z_in == pytest.approx(prompt * 1.1e-05, rel=1e-9)
        assert z_out == pytest.approx(completion * 4.95e-05, rel=1e-9)


def test_gpt54_data_zone_has_no_long_context_surcharge():
    prompt, completion = 300_000, 1_000
    z_in, z_out = cost_per_token(
        model="azure/us/gpt-5.4", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    assert z_in == pytest.approx(prompt * 2.75e-06, rel=1e-9)
    assert z_out == pytest.approx(completion * 1.65e-05, rel=1e-9)


@pytest.mark.parametrize("key", DATA_ZONE_KEYS)
def test_backup_matches_main(key):
    main = _load(MAIN_PATH)
    backup = _load(BACKUP_PATH)
    assert backup.get(key) == main.get(key), f"{key} differs between main and backup model cost maps"
