import json
from decimal import Decimal
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import cost_per_token


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    """cost_per_token reads litellm.model_cost, which is fetched from the live
    upstream file by default. Force the bundled backup so these tests exercise
    the entries added in this change set rather than whatever is deployed."""
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

DATA_ZONE_MULTIPLIER = Decimal("1.1")

FAMILY = (
    "gpt-5.4",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.4-pro",
    "gpt-5.4-pro-2026-03-05",
    "gpt-5.5",
    "gpt-5.5-2026-04-23",
    "gpt-5.5-pro",
    "gpt-5.5-pro-2026-04-23",
)

DATA_ZONE_KEYS = tuple(f"azure/{zone}/{model}" for model in FAMILY for zone in ("us", "eu"))

LONG_CONTEXT_MODELS = (
    "azure/gpt-5.4-mini",
    "azure/gpt-5.4-mini-2026-03-17",
    "azure/gpt-5.4-nano",
    "azure/gpt-5.4-nano-2026-03-17",
)


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def main_cost():
    return _load(MAIN_PATH)


def _is_cost_field(key, value):
    return "cost" in key and isinstance(value, (int, float)) and not isinstance(value, bool)


@pytest.mark.parametrize("data_zone_key", DATA_ZONE_KEYS)
def test_data_zone_entry_is_110pct_of_regional(main_cost, data_zone_key):
    regional_key = data_zone_key.replace("/us/", "/").replace("/eu/", "/")
    regional = main_cost.get(regional_key)
    entry = main_cost.get(data_zone_key)
    assert regional is not None, f"regional base {regional_key} missing"
    assert entry is not None, f"{data_zone_key} not found in model cost map"

    for key, value in entry.items():
        if _is_cost_field(key, value):
            expected = float(Decimal(repr(regional[key])) * DATA_ZONE_MULTIPLIER)
            assert value == expected, f"{data_zone_key}.{key} expected {expected}, got {value}"
        else:
            assert value == regional[key], f"{data_zone_key}.{key} metadata differs from regional"

    # every regional cost field must be carried over (no silently dropped tier)
    for key, value in regional.items():
        if _is_cost_field(key, value):
            assert key in entry, f"{data_zone_key} missing cost field {key} present on regional"


@pytest.mark.parametrize(
    "model, input_above, output_above, cache_above",
    [
        ("azure/gpt-5.4-mini", 1.5e-06, 6.75e-06, 1.5e-07),
        ("azure/gpt-5.4-nano", 4e-07, 1.875e-06, 4e-08),
    ],
)
def test_regional_long_context_surcharge_present(main_cost, model, input_above, output_above, cache_above):
    entry = main_cost[model]
    assert entry["input_cost_per_token_above_272k_tokens"] == input_above
    assert entry["output_cost_per_token_above_272k_tokens"] == output_above
    assert entry["cache_read_input_token_cost_above_272k_tokens"] == cache_above
    # long-context tier is only meaningful when the window exceeds the threshold
    assert entry["max_input_tokens"] > 272000


def test_cost_per_token_data_zone_is_110pct_of_regional():
    """The cost calculator must resolve the azure/us and azure/eu keys directly
    (a fallback to the regional/base entry would drop the data-zone premium)."""
    prompt, completion = 1000, 500
    reg_in, reg_out = cost_per_token(
        model="azure/gpt-5.5", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    for zone in ("us", "eu"):
        z_in, z_out = cost_per_token(
            model=f"azure/{zone}/gpt-5.5",
            custom_llm_provider="azure",
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
        assert z_in == pytest.approx(reg_in * 1.1, rel=1e-9)
        assert z_out == pytest.approx(reg_out * 1.1, rel=1e-9)
        assert z_in > reg_in


def test_cost_per_token_long_context_surcharge_applies_above_threshold():
    """azure/gpt-5.4-mini must bill the above-272k rate once the prompt crosses
    the threshold, and its data-zone variants must charge 1.1x that."""
    prompt, completion = 300_000, 1_000  # prompt_tokens > 272_000
    reg_in, reg_out = cost_per_token(
        model="azure/gpt-5.4-mini", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    assert reg_in == pytest.approx(prompt * 1.5e-06, rel=1e-9)
    assert reg_out == pytest.approx(completion * 6.75e-06, rel=1e-9)

    us_in, us_out = cost_per_token(
        model="azure/us/gpt-5.4-mini", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    assert us_in == pytest.approx(prompt * 1.65e-06, rel=1e-9)
    assert us_out == pytest.approx(completion * 7.425e-06, rel=1e-9)


def test_cost_per_token_below_threshold_uses_base_rate():
    prompt, completion = 1_000, 100  # prompt_tokens < 272_000
    reg_in, reg_out = cost_per_token(
        model="azure/gpt-5.4-mini", custom_llm_provider="azure", prompt_tokens=prompt, completion_tokens=completion
    )
    assert reg_in == pytest.approx(prompt * 7.5e-07, rel=1e-9)
    assert reg_out == pytest.approx(completion * 4.5e-06, rel=1e-9)


@pytest.mark.parametrize("key", DATA_ZONE_KEYS + LONG_CONTEXT_MODELS)
def test_backup_matches_main(key):
    main = _load(MAIN_PATH)
    backup = _load(BACKUP_PATH)
    assert backup.get(key) == main.get(key), f"{key} differs between main and backup model cost maps"
