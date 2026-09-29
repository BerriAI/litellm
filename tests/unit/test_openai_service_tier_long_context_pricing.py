import json
from functools import lru_cache
from pathlib import Path

import pytest

import litellm

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

FLEX_LONG_CONTEXT = {
    "gpt-5.4": {
        "input_cost_per_token_above_272k_tokens_flex": 2.5e-06,
        "output_cost_per_token_above_272k_tokens_flex": 1.125e-05,
        "cache_read_input_token_cost_above_272k_tokens_flex": 2.5e-07,
    },
    "gpt-5.4-pro": {
        "input_cost_per_token_above_272k_tokens_flex": 3e-05,
        "output_cost_per_token_above_272k_tokens_flex": 0.000135,
    },
    "gpt-5.5": {
        "input_cost_per_token_above_272k_tokens_flex": 5e-06,
        "output_cost_per_token_above_272k_tokens_flex": 2.25e-05,
        "cache_read_input_token_cost_above_272k_tokens_flex": 5e-07,
    },
}

PRIORITY_LONG_CONTEXT = {
    "gpt-5.6": {
        "input_cost_per_token_above_272k_tokens_priority": 1.6e-05,
        "output_cost_per_token_above_272k_tokens_priority": 6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 1.6e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 2e-05,
    },
    "gpt-5.6-sol": {
        "input_cost_per_token_above_272k_tokens_priority": 1.6e-05,
        "output_cost_per_token_above_272k_tokens_priority": 6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 1.6e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 2e-05,
    },
    "gpt-5.6-terra": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-06,
        "output_cost_per_token_above_272k_tokens_priority": 3.6e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-07,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-05,
    },
    "gpt-5.6-luna": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-07,
        "output_cost_per_token_above_272k_tokens_priority": 3.6e-06,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-08,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-06,
    },
    "gpt-6-astra": {
        "input_cost_per_token_above_272k_tokens_priority": 4e-05,
        "output_cost_per_token_above_272k_tokens_priority": 0.00015,
        "cache_read_input_token_cost_above_272k_tokens_priority": 4e-06,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 5e-05,
        "input_cost_per_token_above_272k_tokens_ultrafast": 0.00012,
        "output_cost_per_token_above_272k_tokens_ultrafast": 0.00045,
        "cache_read_input_token_cost_above_272k_tokens_ultrafast": 1.2e-05,
        "cache_creation_input_token_cost_above_272k_tokens_ultrafast": 0.00015,
        "input_cost_per_token_ultrafast": 6e-05,
        "output_cost_per_token_ultrafast": 0.0003,
        "cache_read_input_token_cost_ultrafast": 6e-06,
        "cache_creation_input_token_cost_ultrafast": 7.5e-05,
    },
    "gpt-6-sol": {
        "input_cost_per_token_above_272k_tokens_priority": 8e-06,
        "output_cost_per_token_above_272k_tokens_priority": 3e-05,
        "cache_read_input_token_cost_above_272k_tokens_priority": 8e-07,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 1e-05,
    },
    "gpt-6-luna": {
        "input_cost_per_token_above_272k_tokens_priority": 4e-07,
        "output_cost_per_token_above_272k_tokens_priority": 1.5e-06,
        "cache_read_input_token_cost_above_272k_tokens_priority": 4e-08,
        "cache_creation_input_token_cost_above_272k_tokens_priority": 5e-07,
    },
}

EXPECTED = {**FLEX_LONG_CONTEXT, **PRIORITY_LONG_CONTEXT}

NO_PUBLISHED_PRIORITY_LONG_CONTEXT = ("gpt-5.4", "gpt-5.5")


@pytest.fixture(autouse=True)
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.add_known_models()


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return json.load(f)


LONG_CONTEXT_PROMPT_TOKENS = 300_000
COMPLETION_TOKENS = 1_000

TIERED_COST_CASES = [
    ("gpt-5.4", "flex", 2.5e-06, 1.125e-05),
    ("gpt-5.4-pro", "flex", 3e-05, 0.000135),
    ("gpt-5.5", "flex", 5e-06, 2.25e-05),
    ("gpt-5.6", "priority", 1.6e-05, 6e-05),
    ("gpt-5.6-sol", "priority", 1.6e-05, 6e-05),
    ("gpt-5.6-terra", "priority", 8e-06, 3.6e-05),
    ("gpt-5.6-luna", "priority", 8e-07, 3.6e-06),
    ("gpt-6-astra", "priority", 4e-05, 0.00015),
    ("gpt-6-sol", "priority", 8e-06, 3e-05),
    ("gpt-6-luna", "priority", 4e-07, 1.5e-06),
    ("gpt-6-astra", "ultrafast", 0.00012, 0.00045),
]


@pytest.mark.parametrize(("model", "service_tier", "input_rate", "output_rate"), TIERED_COST_CASES)
def test_long_context_service_tier_charges_tiered_rates(
    model: str, service_tier: str, input_rate: float, output_rate: float
) -> None:
    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model,
        prompt_tokens=LONG_CONTEXT_PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        service_tier=service_tier,
    )
    assert prompt_cost == pytest.approx(LONG_CONTEXT_PROMPT_TOKENS * input_rate)
    assert completion_cost == pytest.approx(COMPLETION_TOKENS * output_rate)


@pytest.mark.parametrize("path", [MAIN_PATH, BACKUP_PATH], ids=["main", "backup"])
def test_catalog_service_tier_long_context_rates_match(path: Path) -> None:
    catalog = _load(path)
    for model, fields in EXPECTED.items():
        for key, expected in fields.items():
            assert catalog[model][key] == pytest.approx(expected), f"{path.name}: {model}.{key}"


def test_short_context_ultrafast_charges_short_context_rates() -> None:
    entry = _load(MAIN_PATH)["gpt-6-astra"]
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="gpt-6-astra",
        prompt_tokens=COMPLETION_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        service_tier="ultrafast",
    )
    standard_prompt_cost, standard_completion_cost = litellm.cost_per_token(
        model="gpt-6-astra",
        prompt_tokens=COMPLETION_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
    )
    assert prompt_cost == pytest.approx(COMPLETION_TOKENS * entry["input_cost_per_token_ultrafast"])
    assert completion_cost == pytest.approx(COMPLETION_TOKENS * entry["output_cost_per_token_ultrafast"])
    assert prompt_cost != pytest.approx(standard_prompt_cost)
    assert completion_cost != pytest.approx(standard_completion_cost)
